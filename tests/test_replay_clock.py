from __future__ import annotations

import copy
import math
import unittest

from racing_event.replay import ReplayClock, sample_replay


def replay_data() -> dict[str, object]:
    def frame(tick: int, x: float, heading: float) -> dict[str, object]:
        return {
            "tick": tick,
            "x": x,
            "y": 0.0,
            "heading": heading,
            "speed": 2.0,
            "lateral_velocity": 0.0,
            "yaw_rate": 0.0,
            "steer": 0.0,
            "steering_command": 0.0,
            "throttle_brake": 0.5,
            "off_track": False,
            "events": [],
            "terminated": False,
            "truncated": False,
        }

    return {"tick_hz": 10, "frames": [frame(0, 0.0, math.radians(179)), frame(1, 10.0, math.radians(-179)), frame(2, 20.0, math.radians(-170))]}


class ReplayClockTests(unittest.TestCase):
    def test_seek_roundtrip_is_exact_across_bounded_recording(self) -> None:
        for hz in (30, 60):
            clock = ReplayClock(hz, 20_000)
            for tick in range(20_001):
                clock.seek_tick(tick)
                self.assertEqual(clock.tick_position, tick, (hz, tick))
            clock.seek_tick(-5)
            self.assertEqual(clock.tick_position, 0)
            clock.seek_tick(30_000)
            self.assertEqual(clock.tick_position, 20_000)

    def test_only_position_and_shortest_angle_interpolate_not_telemetry(self) -> None:
        data = replay_data()
        first, second, _ = data["frames"]
        for index, name in enumerate(("speed", "lateral_velocity", "yaw_rate", "steer", "steering_command", "throttle_brake")):
            first[name], second[name] = index / 10, (index + 1) / 10
        second["off_track"] = True
        second["events"] = ["off_track_start"]
        second["terminated"] = True
        second["truncated"] = True
        before = copy.deepcopy(data)
        clock = ReplayClock(10, 2, seconds=0.05)
        sampled = sample_replay(data, clock)
        self.assertEqual(sampled["pose"]["x"], 5)
        self.assertAlmostEqual(sampled["pose"]["heading"], math.pi)
        self.assertIs(sampled["frame"], first)
        for name in ("speed", "lateral_velocity", "yaw_rate", "steer", "steering_command", "throttle_brake"):
            self.assertEqual(sampled["pose"][name], first[name])
            self.assertEqual(sampled["frame"][name], first[name])
        self.assertFalse(sampled["frame"]["off_track"])
        self.assertFalse(sampled["frame"]["terminated"])
        self.assertFalse(sampled["frame"]["truncated"])
        self.assertEqual(sampled["frame"]["events"], [])
        clock.seek_seconds(0.1 - 1e-8)
        self.assertEqual(sample_replay(data, clock)["frame"]["tick"], 0)
        clock.seek_tick(1)
        self.assertIs(sample_replay(data, clock)["frame"], second)
        self.assertEqual(data, before)

    def test_pause_restart_speed_and_end_of_recording(self) -> None:
        clock = ReplayClock(30, 60)
        clock.advance(1)
        self.assertEqual(clock.seconds, 0)
        clock.restart()
        clock.set_speed(2)
        clock.advance(0.25)
        self.assertEqual(clock.tick_position, 15)
        clock.pause()
        clock.advance(1)
        self.assertEqual(clock.tick_position, 15)
        self.assertTrue(clock.toggle())
        clock.advance(100)
        self.assertEqual(clock.tick_position, 60)
        self.assertFalse(clock.playing)
        clock.restart()
        self.assertEqual(clock.tick_position, 0)
        self.assertTrue(clock.playing)

    def test_exact_ticks_and_final_frame_clamp(self) -> None:
        data = replay_data()
        clock = ReplayClock(10, 2)
        self.assertEqual(sample_replay(data, clock)["frame_index"], 0)
        clock.seek_tick(1)
        self.assertEqual(sample_replay(data, clock)["frame_index"], 1)
        clock.seek_seconds(99)
        sampled = sample_replay(data, clock)
        self.assertEqual(sampled["frame_index"], 2)
        self.assertEqual(sampled["alpha"], 0.0)

    def test_shortest_heading_interpolation_and_playback_speed(self) -> None:
        data = replay_data()
        clock = ReplayClock(10, 2, seconds=0.05)
        sampled = sample_replay(data, clock)
        self.assertGreater(abs(sampled["pose"]["heading"]), math.radians(178))
        clock.playing = True
        clock.set_speed(2.0)
        clock.advance(0.1)
        self.assertAlmostEqual(clock.seconds, 0.2)

    def test_invalid_clock_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ReplayClock(0, 1)
        clock = ReplayClock(10, 2)
        with self.assertRaises(ValueError):
            clock.set_speed(0)
        with self.assertRaises(ValueError):
            clock.seek_seconds(float("nan"))
        for invalid in (float("inf"), float("nan"), -1):
            with self.assertRaises(ValueError):
                clock.advance(invalid)
            with self.assertRaises(ValueError):
                clock.set_speed(invalid)
        for invalid in (1.5, True):
            with self.assertRaises(ValueError):
                clock.seek_tick(invalid)


if __name__ == "__main__":
    unittest.main()
