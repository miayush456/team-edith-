from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from racing_event.pygame_renderer import PygameReplayRenderer, run_pygame
from racing_event.replay import sample_replay
from racing_event.replay_group import create_replay_group
from test_replay_group import ROOT, recording


class PygameGroupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = [recording(self.root / f"{i}.json", f"ghost-{i}", steps)
                      for i, steps in enumerate((30, 60, 90, 120))]
        self.manifest = self.root / "group.json"
        self.group = create_replay_group(self.manifest, self.paths, labels=["Alpha", "Beta", "Gamma", "Delta"])
        pygame.init()
        self.addCleanup(pygame.quit)
        self.surface = pygame.display.set_mode((640, 480), pygame.RESIZABLE)
        self.renderer = PygameReplayRenderer(self.group, self.surface)
        pygame.event.clear()

    def key(self, key, mod=0):
        # Real SDL event queue, surfaces, fonts and drawing; no Pygame doubles.
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod))
        for event in pygame.event.get():
            self.assertTrue(self.renderer.handle_event(event))
        self.renderer.draw()

    def lines(self):
        with mock.patch.object(self.renderer, "_text", wraps=self.renderer._text) as text:
            self.renderer.draw()
        return [call.args[0] for call in text.call_args_list]

    def test_unequal_lengths_share_clock_hold_final_without_blending_or_restarting(self):
        clock = self.renderer.clock
        self.assertEqual(clock.final_tick, 120)
        clock.playing = True
        clock.advance(30.5 / clock.tick_hz)
        short = sample_replay(self.group.participants[0].data, clock)
        self.assertEqual(short["frame"], self.group.participants[0].data["frames"][-1])
        self.assertIs(short["frame"], short["next_frame"])
        self.assertEqual(short["alpha"], 0)
        self.assertEqual(short["tick_position"], 30)
        self.assertTrue(clock.playing)
        self.assertIn("HOLD FINAL: FINISHED", self.lines())
        self.assertIn("Ghost 1 Events @ tick 30: Lap complete", self.lines())
        short_pose = short["pose"]
        clock.advance(2.0)
        self.assertEqual(sample_replay(self.group.participants[0].data, clock)["pose"], short_pose)
        self.key(pygame.K_END)
        self.assertEqual(clock.tick_position, 120)
        self.assertFalse(clock.playing)
        self.key(pygame.K_SPACE)
        self.assertEqual(clock.tick_position, 0)
        self.assertTrue(clock.playing)
        for participant in self.group.participants:
            self.assertEqual(sample_replay(participant.data, clock)["frame"]["tick"], 0)

    def test_selected_ghost_exact_telemetry_events_and_final_seek_on_shared_timeline(self):
        self.renderer.clock.seek_seconds(6.5 / self.group.tick_hz)
        self.key(pygame.K_2)
        data = self.group.participants[1].data
        sample = sample_replay(self.renderer.data, self.renderer.clock)
        self.assertEqual(sample["frame"], data["frames"][6])
        self.assertAlmostEqual(sample["pose"]["x"], data["frames"][6]["x"] + 0.05)
        for field in ("speed", "lateral_velocity", "yaw_rate", "steer", "steering_command", "throttle_brake"):
            self.assertEqual(sample["pose"][field], data["frames"][6][field])
        self.assertEqual(sample["frame"]["events"], [])
        self.assertIn("Speed    3.00 m/s", self.lines())
        self.assertIn("Command +0.098", self.lines())
        self.assertIn("Drive       -0.098", self.lines())
        self.assertNotIn("Command +0.194", self.lines())  # Ghost 1 at the same clock tick.
        self.assertIn("Ghost 2 Events @ tick 6: none", self.lines())
        self.key(pygame.K_n)
        self.assertEqual(self.renderer.clock.tick_position, 7)
        self.assertIn("OFF TRACK", self.lines())
        self.key(pygame.K_n)
        self.key(pygame.K_n)
        self.assertEqual(self.renderer.clock.tick_position, 60)
        self.assertIn("GHOST 2 | TICK 60/60", self.lines())
        self.assertEqual(self.renderer._tick_x(60), self.renderer.scrubber_rect.centerx)
        self.key(pygame.K_TAB)
        self.assertEqual(self.renderer.selected_index, 2)
        self.assertEqual(self.renderer.clock.tick_position, 60)
        self.assertIn("IN PROGRESS", self.lines())
        self.key(pygame.K_TAB, pygame.KMOD_SHIFT)
        self.assertEqual(self.renderer.selected_index, 1)

    def test_toggle_visibility_paths_and_selection_leave_clock_and_bytes_unchanged(self):
        data_before = copy.deepcopy(self.group)
        bytes_before = [p.read_bytes() for p in [self.manifest, *self.paths]]
        self.renderer.clock.seek_tick(12)
        self.renderer.clock.playing = True
        self.renderer.draw()
        pixels = pygame.image.tostring(self.surface, "RGB")
        for key in (pygame.K_v, pygame.K_t):
            self.key(key)
            self.assertEqual(self.renderer.clock.tick_position, 12)
            self.assertTrue(self.renderer.clock.playing)
        self.assertFalse(self.renderer.visible[0])
        self.assertFalse(self.renderer.show_paths[0])
        self.assertNotEqual(pixels, pygame.image.tostring(self.surface, "RGB"))
        self.assertIn("Speed    6.00 m/s", self.lines())  # Hidden ghosts retain exact telemetry.
        self.key(pygame.K_v)
        with mock.patch.object(self.renderer, "_polyline", side_effect=AssertionError("uncached track")), \
                mock.patch.object(self.renderer, "_world_bounds", side_effect=AssertionError("uncached bounds")), \
                mock.patch("pygame.draw.lines", side_effect=AssertionError("uncached path")):
            for key in (pygame.K_t, pygame.K_4, pygame.K_p, pygame.K_h, pygame.K_h, pygame.K_HOME):
                self.key(key)
        self.assertEqual(data_before, self.group)
        self.assertEqual(bytes_before, [p.read_bytes() for p in [self.manifest, *self.paths]])

    def test_four_ghost_layout_help_and_metric_rectangles_at_both_sizes(self):
        for size in ((640, 480), (1280, 800)):
            with self.subTest(size=size):
                self.renderer.resize(pygame.Surface(size))
                self.renderer.clock.seek_tick(23)
                self.renderer.draw()
                for region in (self.renderer.header_rect, self.renderer.footer_rect, self.renderer.panel_rect):
                    self.assertFalse(self.renderer.track_rect.colliderect(region))
                for path in self.renderer.paths:
                    self.assertTrue(all(self.renderer.track_rect.collidepoint(p) for p in path))
                for rect in self.renderer.buttons.values():
                    self.assertTrue(self.renderer.footer_rect.contains(rect))
                self.key(pygame.K_h)
                lines = self.lines()
                for line in ("1-4: select ghost telemetry", "V: selected ghost on / hidden", "T: selected path on / off",
                             "Short ghosts HOLD FINAL", "No interpolation past end"):
                    self.assertIn(line, lines)
                self.key(pygame.K_h)
                with mock.patch("pygame.draw.polygon", wraps=pygame.draw.polygon) as polygon:
                    self.renderer.draw()
                cars = [call for call in polygon.call_args_list if len(call.args[2]) == 4]
                self.assertEqual(len(cars), 4)
                self.assertEqual({call.args[1] for call in cars}, {p.color for p in self.group.participants})

    def test_cli_create_play_selected_event_help_is_deterministic_and_inputs_unchanged(self):
        output = self.root / "cli-group.json"
        args = [sys.executable, str(ROOT / "tools/build_replay_group.py"), "--output", str(output),
                "--replays", *map(str, self.paths), "--labels", "Alpha", "Beta", "Gamma", "Delta"]
        result = subprocess.run(args, cwd=self.root, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        before = [p.read_bytes() for p in [output, *self.paths]]
        for size in ("640x480", "1280x800"):
            images = [self.root / f"{size}-{i}.png" for i in range(2)]
            for image in images:
                args = [sys.executable, str(ROOT / "tools/run_pygame.py"), "--group", str(output),
                        "--select-ghost", "2", "--start-tick", "8", "--seek-event", "next", "--show-help",
                        "--size", size, "--screenshot", str(image)]
                result = subprocess.run(args, cwd=self.root, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(images[0].read_bytes(), images[1].read_bytes())
            self.renderer.resize(pygame.Surface(tuple(map(int, size.split("x")))))
            self.renderer.select_ghost(1)
            self.renderer.clock.seek_tick(19)
            self.renderer.show_help = True
            self.renderer.draw()
            loaded = pygame.image.load(str(images[0]))
            self.assertEqual(pygame.image.tostring(loaded, "RGB"), pygame.image.tostring(self.renderer.surface, "RGB"))
        self.assertEqual(before, [p.read_bytes() for p in [output, *self.paths]])

    def test_no_event_or_invalid_selected_ghost_is_an_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "selected ghost"):
            run_pygame(group_path=self.manifest, headless=True, selected_ghost=5)
        with self.assertRaisesRegex(ValueError, "no recorded event"):
            run_pygame(group_path=self.manifest, headless=True, start_tick=120, seek_event="next")

    def test_long_labels_do_not_hide_results_or_compact_crash_hold_status(self):
        crash = recording(self.root / "crash.json", "crash", 30, status="dnf_crash")
        group = create_replay_group(self.root / "long-labels.json", [crash, *self.paths[:3]], labels=["W" * 48] * 4)
        self.renderer = PygameReplayRenderer(group, self.surface)
        self.renderer.clock.seek_tick(group.final_tick)
        with mock.patch.object(self.renderer, "_text", wraps=self.renderer._text) as text:
            self.renderer.draw()
        summaries = [call for call in text.call_args_list if call.args[0].startswith("final:")]
        self.assertEqual(len(summaries), 4)
        for call in summaries:
            self.assertLessEqual(self.renderer.small_font.size(call.args[0])[0], call.args[1].width)
        status = next(call for call in text.call_args_list if call.args[0] == "HOLD FINAL: DNF crash")
        self.assertLessEqual(self.renderer.small_font.size(status.args[0])[0], status.args[1].width)

    def test_screenshot_cannot_overwrite_group_replay_or_hardlink_alias(self):
        png_source = recording(self.root / "source.png", "source", 30)
        path = self.root / "png-source-group.json"
        create_replay_group(path, [png_source])
        before = png_source.read_bytes()
        alias = self.root / "alias.png"
        os.link(png_source, alias)
        for target in (png_source, alias):
            with self.assertRaisesRegex(ValueError, "distinct"):
                run_pygame(group_path=path, screenshot=target)
        self.assertEqual(png_source.read_bytes(), before)

    def test_switching_event_rails_clears_endpoint_marker_pixels(self):
        # Ghost 4 has the group's endpoint event; ghost 1 ends earlier. The
        # endpoint diamond extends outside the horizontal scrubber's rectangle.
        self.renderer.draw()
        before = pygame.image.tostring(self.renderer._background, "RGB")
        self.key(pygame.K_4)
        self.assertNotEqual(before, pygame.image.tostring(self.renderer._background, "RGB"))
        self.key(pygame.K_1)
        self.assertEqual(before, pygame.image.tostring(self.renderer._background, "RGB"))


if __name__ == "__main__":
    unittest.main()
