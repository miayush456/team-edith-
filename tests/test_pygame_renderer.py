from __future__ import annotations

import copy
import math
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

from racing_event.config import SimulationConfig
from racing_event.pygame_renderer import ACCENT, BACKGROUND, PygameReplayRenderer, RESULT_LABELS, run_pygame
from racing_event.replay import ReplayClock, sample_replay
from racing_event.track import TrackGeometry
from racing_event.trajectory import ReplayWriter, load_replay, replay_frame, replay_header


ROOT = Path(__file__).resolve().parents[1]
TRACK = TrackGeometry.from_file(ROOT / "tests/fixtures/tracks/synthetic_oval.json")
CONFIG = SimulationConfig.from_file(ROOT / "config/simulation.json")


class PygameRendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        pygame.init()
        self.addCleanup(pygame.quit)
        self.surface = pygame.display.set_mode((640, 480), pygame.RESIZABLE)
        self.path, self.data = self.recording()
        self.renderer = PygameReplayRenderer(self.data, self.surface)

    def recording(self, status="dnf_crash", eventful=True):
        """Valid schema-1.1 fixtures, without running physics in viewer tests."""
        path = Path(self.temp.name) / f"{status}-{eventful}.json"
        header = replay_header(TRACK, CONFIG, participant_id="pygame-test", mode="evaluation")
        with ReplayWriter(path, header) as writer:
            for tick in range(61):
                state = {"x": float(TRACK.centerline[0][0]) + tick / 10,
                         "y": float(TRACK.centerline[0][1]), "heading": 0.0, "speed": float(tick),
                         "lateral_velocity": tick / 100, "yaw_rate": tick / 500, "steer": tick / 300}
                events = []
                if eventful:
                    if tick == 7:
                        events = ["off_track_start"]
                    elif tick == 19:
                        events = ["off_track_end"]
                if tick == 60:
                    if status == "finished":
                        events = ["lap_complete"]
                    elif status == "dnf_crash":
                        # Both events can be recorded; the Python crash result wins.
                        events = ["barrier_contact", "lap_complete"]
                writer.append(replay_frame(
                    tick, state, steering_command=tick / 100, throttle_brake=-tick / 100,
                    off_track=eventful and 7 <= tick < 19, events=events,
                    terminated=tick == 60 and status in ("finished", "dnf_crash", "dnf_backward"),
                    truncated=tick == 60 and status in ("dnf_stuck", "dnf_timeout"),
                ))
            writer.finish({"status": status, "steps": 60, "elapsed_s": 60 / CONFIG.tick_hz,
                           "lap_time_s": 60 / CONFIG.tick_hz if status == "finished" else None,
                           "trajectory_hash": "a" * 64})
        return path, load_replay(path)

    def click(self, position):
        self.assertTrue(self.renderer.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=position, button=1)))
        self.assertTrue(self.renderer.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=position, button=1)))

    def button(self, action):
        self.click(self.renderer.buttons[action].center)

    def key(self, key, mod=0):
        return self.renderer.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod))

    def drawn_text(self):
        with mock.patch.object(self.renderer, "_text", wraps=self.renderer._text) as text:
            self.renderer.draw()
        return [call.args[0] for call in text.call_args_list]

    def test_clickable_play_pause_restart_and_speed_limits(self) -> None:
        self.renderer.draw()
        self.button("play")
        self.assertTrue(self.renderer.clock.playing)
        self.renderer.clock.advance(0.5)
        self.button("play")
        self.assertFalse(self.renderer.clock.playing)
        paused = self.renderer.clock.seconds
        self.renderer.clock.advance(0.5)
        self.assertEqual(self.renderer.clock.seconds, paused)
        self.button("restart")
        self.assertEqual(self.renderer.clock.tick_position, 0)
        self.assertTrue(self.renderer.clock.playing)
        for _ in range(10):
            self.button("faster")
        self.assertEqual(self.renderer.clock.speed, 16)
        self.assertFalse(self.renderer._enabled("faster"))
        for _ in range(20):
            self.button("slower")
        self.assertEqual(self.renderer.clock.speed, 0.125)
        self.assertFalse(self.renderer._enabled("slower"))
        self.assertLessEqual(self.renderer.small_font.size("0.125x")[0], self.renderer.buttons["speed"].width - 4)
        self.renderer.draw()

    def test_arrow_keys_are_exact_ticks_shift_is_one_second_and_seeks_pause(self) -> None:
        clock = self.renderer.clock
        clock.seek_tick(10)
        clock.playing = True
        self.key(pygame.K_RIGHT)
        self.assertEqual(clock.tick_position, 11)
        self.assertFalse(clock.playing)
        self.key(pygame.K_RIGHT, pygame.KMOD_SHIFT)
        self.assertEqual(clock.tick_position, 11 + CONFIG.tick_hz)
        self.key(pygame.K_LEFT)
        self.key(pygame.K_LEFT, pygame.KMOD_SHIFT)
        self.assertEqual(clock.tick_position, 10)
        clock.seek_seconds(11.5 / CONFIG.tick_hz)
        self.key(pygame.K_RIGHT)
        self.assertEqual(sample_replay(self.data, clock)["frame"]["tick"], 12)
        self.key(pygame.K_HOME)
        self.key(pygame.K_LEFT)
        self.assertEqual(clock.tick_position, 0)
        self.key(pygame.K_END)
        self.key(pygame.K_RIGHT)
        self.assertEqual(clock.tick_position, 60)
        self.assertFalse(clock.playing)
        self.key(pygame.K_SPACE)
        self.assertTrue(clock.playing)
        self.assertEqual(clock.tick_position, 0)  # Play from the end restarts.
        self.key(pygame.K_RIGHTBRACKET)
        self.assertEqual(clock.speed, 2)
        self.key(pygame.K_MINUS)
        self.assertEqual(clock.speed, 1)
        self.key(pygame.K_r)
        self.assertEqual(clock.tick_position, 0)

    def test_recorded_event_jumps_markers_and_hover_reach_exact_ticks(self) -> None:
        self.assertEqual(self.renderer.event_ticks, (7, 19, 60))
        self.assertFalse(self.renderer._enabled("previous_event"))
        self.button("previous_event")
        self.assertEqual(self.renderer.clock.tick_position, 0)
        self.renderer.clock.playing = True
        self.button("next_event")
        self.assertEqual(self.renderer.clock.tick_position, 7)
        self.assertFalse(self.renderer.clock.playing)
        self.key(pygame.K_n)
        self.assertEqual(self.renderer.clock.tick_position, 19)
        self.button("next_event")
        self.assertEqual(self.renderer.clock.tick_position, 60)
        self.assertFalse(self.renderer._enabled("next_event"))
        self.key(pygame.K_n)
        self.assertEqual(self.renderer.clock.tick_position, 60)
        self.key(pygame.K_p)
        self.assertEqual(self.renderer.clock.tick_position, 19)
        self.click(self.renderer.marker_rects[7].center)
        self.assertEqual(sample_replay(self.data, self.renderer.clock)["frame"]["events"], ["off_track_start"])
        self.renderer.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=self.renderer.marker_rects[60].center))
        self.assertIn("Marker @ tick 60: Barrier contact, Lap complete", self.drawn_text())
        self.assertEqual(self.renderer.clock.tick_position, 7)  # Hover never seeks.
        self.click(self.renderer.marker_rects[60].center)
        self.assertEqual(self.renderer.clock.tick_position, 60)
        # Actual marker symbols occupy the rail, rather than just labels.
        for rect in self.renderer.marker_rects.values():
            colors = {tuple(self.surface.get_at((x, y))[:3]) for x in range(rect.left, rect.right)
                      for y in range(rect.top, rect.bottom)}
            self.assertTrue(colors - {BACKGROUND})

    def test_event_navigation_from_fractional_time_and_empty_recording(self) -> None:
        self.renderer.clock.seek_seconds(7.5 / CONFIG.tick_hz)
        self.button("previous_event")
        self.assertEqual(self.renderer.clock.tick_position, 7)
        self.renderer.clock.seek_seconds(7.5 / CONFIG.tick_hz)
        self.button("next_event")
        self.assertEqual(self.renderer.clock.tick_position, 19)
        _, data = self.recording("dnf_timeout", eventful=False)
        self.renderer = PygameReplayRenderer(data, self.surface)
        self.assertEqual(self.renderer.event_ticks, ())
        self.button("next_event")
        self.key(pygame.K_p)
        self.assertEqual(self.renderer.clock.tick_position, 0)
        self.assertFalse(self.renderer._enabled("next_event"))
        self.assertIn("Events @ tick 0: none in this recording", self.drawn_text())

    def test_scrubber_click_drag_release_and_focus_loss(self) -> None:
        rect = self.renderer.scrubber_rect
        self.renderer.clock.playing = True
        self.click((rect.centerx, rect.top + 8))
        self.assertEqual(self.renderer.clock.tick_position, 30)
        self.assertFalse(self.renderer.clock.playing)
        self.renderer.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(rect.left, rect.top + 8), button=1))
        self.renderer.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(rect.right + 200, -20)))
        self.assertEqual(self.renderer.clock.tick_position, 60)
        self.renderer.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(rect.left - 200, -20), button=1))
        self.assertEqual(self.renderer.clock.tick_position, 0)
        self.assertFalse(self.renderer.dragging)
        self.renderer.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=rect.center, button=1))
        self.renderer.handle_event(pygame.event.Event(pygame.WINDOWFOCUSLOST))
        self.renderer.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(rect.left - 100, 0)))
        self.assertEqual(self.renderer.clock.tick_position, 30)

    def test_help_is_clickable_keyboard_operable_and_does_not_cover_track(self) -> None:
        self.renderer.draw()
        track_before = pygame.image.tostring(self.surface.subsurface(self.renderer.track_rect), "RGB")
        self.button("help")
        self.assertTrue(self.renderer.show_help)
        lines = self.drawn_text()
        self.assertIn("Shift + Left / Right: 1 s", lines)
        self.assertIn("P / N: prev / next event", lines)
        self.assertEqual(track_before, pygame.image.tostring(self.surface.subsurface(self.renderer.track_rect), "RGB"))
        self.key(pygame.K_h)
        self.assertFalse(self.renderer.show_help)
        self.key(pygame.K_QUESTION)
        self.assertTrue(self.renderer.show_help)
        self.assertFalse(self.key(pygame.K_ESCAPE))
        self.assertFalse(self.key(pygame.K_q))
        self.assertFalse(self.renderer.handle_event(pygame.event.Event(pygame.QUIT)))

    def test_exact_telemetry_and_surface_state_do_not_leak_future_events(self) -> None:
        self.renderer.clock.seek_seconds(6.5 / CONFIG.tick_hz)
        sampled = sample_replay(self.data, self.renderer.clock)
        self.assertAlmostEqual(sampled["pose"]["x"], self.data["frames"][6]["x"] + 0.05)
        lines = self.drawn_text()
        self.assertIn("ON TRACK", lines)
        self.assertIn("IN PROGRESS", lines)
        self.assertIn("Speed    6.00 m/s", lines)
        self.assertIn("Events @ tick 6: none", lines)
        panel_before = pygame.image.tostring(self.surface.subsurface(self.renderer.panel_rect), "RGB")
        self.click(self.renderer.marker_rects[7].center)
        lines = self.drawn_text()
        self.assertIn("OFF TRACK", lines)
        self.assertIn("Speed    7.00 m/s", lines)
        self.assertNotEqual(panel_before, pygame.image.tostring(self.surface.subsurface(self.renderer.panel_rect), "RGB"))

    def test_final_results_are_labeled_separately_from_current_tick(self) -> None:
        for status, label in RESULT_LABELS.items():
            with self.subTest(status=status):
                _, data = self.recording(status)
                self.renderer = PygameReplayRenderer(data, self.surface)
                self.renderer.clock.seek_seconds(59.5 / CONFIG.tick_hz)
                before = self.drawn_text()
                self.assertIn("IN PROGRESS", before)
                self.assertNotIn(label, before)
                summary = next(line for line in before if line.startswith("Recorded final:"))
                self.assertIn(label, summary)
                self.assertIn("at tick 60", summary)
                self.assertIn("lap 2.000 s" if status == "finished" else "no lap", summary)
                self.key(pygame.K_END)
                after = self.drawn_text()
                self.assertIn(label, after)
                self.assertNotIn("IN PROGRESS", after)
                self.assertIn("End of recording | Tick 60 | 2.000 / 2.000 s", after)

    def test_resized_layout_fits_track_path_and_controls_in_reserved_regions(self) -> None:
        data = copy.deepcopy(self.data)
        # Fit includes an extreme recorded pose, not just the track boundaries.
        data["frames"][-1]["x"] = 500
        data["frames"][-1]["y"] = -400
        for size in ((640, 480), (800, 600), (960, 540), (1280, 800)):
            with self.subTest(size=size):
                surface = pygame.Surface(size)
                renderer = PygameReplayRenderer(data, surface)
                renderer.clock.seek_tick(60)
                renderer.draw()
                self.assertGreater(renderer._scale, 0)
                for region in (renderer.header_rect, renderer.footer_rect, renderer.panel_rect):
                    self.assertFalse(renderer.track_rect.colliderect(region))
                self.assertTrue(renderer.footer_rect.contains(renderer.scrubber_rect))
                for rect in renderer.buttons.values():
                    self.assertTrue(renderer.footer_rect.contains(rect))
                for point in renderer.path:
                    self.assertTrue(renderer.track_rect.collidepoint(point))
                for name in ("left_boundary", "right_boundary", "barrier_left", "barrier_right"):
                    for point in renderer._polyline(data["track"][name]):
                        self.assertTrue(renderer.track_rect.collidepoint(point))
                radius = math.hypot(data["car"]["length_m"], data["car"]["width_m"]) / 2
                for x, y in ((500 - radius, -400 - radius), (500 + radius, -400 + radius)):
                    self.assertTrue(renderer.track_rect.collidepoint(renderer._screen_point(x, y)))
                # Real pixels from geometry and path are in the map viewport.
                colors = {tuple(surface.get_at(point)[:3]) for point in renderer.path}
                self.assertNotEqual(colors, {BACKGROUND})
                self.assertEqual(tuple(surface.get_at((0, 90))[:3]), BACKGROUND)
        self.renderer.handle_event(pygame.event.Event(pygame.VIDEORESIZE, w=300, h=200))
        self.assertEqual(self.renderer.surface.get_size(), (640, 480))
        self.renderer.handle_event(pygame.event.Event(pygame.VIDEORESIZE, w=900, h=650))
        self.assertEqual(self.renderer.surface.get_size(), (900, 650))
        self.renderer.draw()

    def test_static_track_path_and_markers_are_cached_until_resize(self) -> None:
        background = self.renderer._background
        with mock.patch.object(self.renderer, "_polyline", side_effect=AssertionError("track recomputed")), \
                mock.patch.object(self.renderer, "_world_bounds", side_effect=AssertionError("recording rescanned")), \
                mock.patch("pygame.draw.lines", wraps=pygame.draw.lines) as lines:
            for tick in (0, 7, 19, 60):
                self.renderer.clock.seek_tick(tick)
                self.renderer.draw()
            self.renderer.resize(pygame.Surface((640, 480)))
            self.renderer.draw()
            lines.assert_not_called()
        self.assertIs(self.renderer._background, background)
        with mock.patch("pygame.draw.lines", wraps=pygame.draw.lines) as lines:
            self.renderer.resize(pygame.Surface((800, 600)))
            self.assertEqual(lines.call_count, 6)  # five track rings + recorded path
        self.assertIsNot(self.renderer._background, background)

    def test_car_render_preserves_metric_footprint_heading_and_y_up(self) -> None:
        renderer = self.renderer
        renderer._origin_x, renderer._origin_y, renderer._scale = 320, 240, 20
        length, width = self.data["car"]["length_m"], self.data["car"]["width_m"]
        for heading in (0, math.pi / 2, math.pi / 4):
            with self.subTest(heading=heading):
                self.surface.fill(BACKGROUND)
                pose = {"x": 0, "y": 0, "heading": heading}
                with mock.patch("pygame.draw.polygon", wraps=pygame.draw.polygon) as polygon:
                    renderer._draw_car(pygame, pose, False)
                corners = polygon.call_args.args[2]
                self.assertAlmostEqual(math.dist(corners[0], corners[1]), width * 20, delta=1.5)
                self.assertAlmostEqual(math.dist(corners[1], corners[2]), length * 20, delta=1.5)
                front = ((corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2)
                self.assertAlmostEqual(front[0], 320 + length * 10 * math.cos(heading), delta=1)
                self.assertAlmostEqual(front[1], 240 - length * 10 * math.sin(heading), delta=1)
                self.assertEqual(tuple(self.surface.get_at(corners[0])[:3]), ACCENT)
                footprint = pygame.Rect(min(x for x, _ in corners), min(y for _, y in corners),
                                        max(x for x, _ in corners) - min(x for x, _ in corners) + 1,
                                        max(y for _, y in corners) - min(y for _, y in corners) + 1)
                # No visual nose, shadow or oversized sprite escapes the footprint.
                mask = pygame.mask.from_threshold(self.surface, ACCENT, (1, 1, 1, 255))
                for rect in mask.get_bounding_rects():
                    self.assertTrue(footprint.contains(rect))

    def test_render_and_navigation_leave_recording_unchanged(self) -> None:
        before = copy.deepcopy(self.data)
        encoded = self.path.read_bytes()
        for action in ("play", "next_event", "previous_event", "restart", "help"):
            self.button(action)
            self.renderer.draw()
        self.assertEqual(self.data, before)
        self.assertEqual(self.path.read_bytes(), encoded)
        self.assertEqual(self.data["schema_version"], "1.1")

    def test_real_loop_renders_restart_and_end_without_advancing_after_input(self) -> None:
        drawn = []
        draw = PygameReplayRenderer.draw

        def capture(renderer):
            drawn.append((renderer.clock.tick_position, renderer.clock.playing, renderer.surface.get_size()))
            draw(renderer)

        batches = [[pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)],
                   [pygame.event.Event(pygame.VIDEORESIZE, w=800, h=600),
                    pygame.event.Event(pygame.KEYDOWN, key=pygame.K_END)]]
        wall_clock = mock.Mock()
        wall_clock.tick.return_value = 500
        with mock.patch("pygame.event.get", side_effect=batches), \
                mock.patch("pygame.time.Clock", return_value=wall_clock), \
                mock.patch.object(PygameReplayRenderer, "draw", capture):
            run_pygame(self.path, size=(640, 480), max_frames=2)
        self.assertEqual(drawn, [(0, True, (640, 480)), (60, False, (800, 600))])

    def test_cli_screenshot_is_deterministic_headless_and_matches_selected_pose(self) -> None:
        first = Path(self.temp.name) / "first.png"
        second = Path(self.temp.name) / "second.png"
        encoded = self.path.read_bytes()
        for output in (first, second):
            result = subprocess.run(
                [sys.executable, "tools/run_pygame.py", "--replay", str(self.path), "--size", "640x480",
                 "--start-seconds", "0.25", "--speed", "2", "--screenshot", str(output)],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        image = pygame.image.load(str(first))
        self.assertEqual(image.get_size(), (640, 480))
        self.renderer.clock.seek_seconds(0.25)
        self.renderer.clock.set_speed(2)
        self.renderer.draw()
        self.assertEqual(pygame.image.tostring(image, "RGB"), pygame.image.tostring(self.surface, "RGB"))
        self.assertEqual(self.path.read_bytes(), encoded)

    def test_headless_smoke_and_screenshot_reject_recording_overwrite(self) -> None:
        run_pygame(self.path, size=(640, 480), headless=True)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "distinct from the recording"):
            run_pygame(self.path, screenshot=self.path)
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
