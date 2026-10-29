from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from racing_event.config import SimulationConfig
from racing_event.env import RacingEnv
from racing_event.reward import RewardConfig, SIGNAL_NAMES
from racing_event.track import TrackGeometry


ROOT = Path(__file__).resolve().parents[1]
TRACK = TrackGeometry.from_file(ROOT / "tests" / "fixtures" / "tracks" / "synthetic_oval.json")
CONFIG = SimulationConfig.from_file(ROOT / "config" / "simulation.json")


class RacingEnvironmentTests(unittest.TestCase):
    def make_env(self) -> RacingEnv:
        return RacingEnv(TRACK, CONFIG)

    def test_reset_and_fixed_action_sequence_are_deterministic(self) -> None:
        actions = [np.asarray([0.1, 0.6]), np.asarray([-0.1, 0.3])] * 15
        first = self.make_env()
        second = self.make_env()
        first_obs, _ = first.reset(seed=123)
        second_obs, _ = second.reset(seed=123)
        np.testing.assert_array_equal(first_obs, second_obs)
        for action in actions:
            left = first.step(action)
            right = second.step(action)
            np.testing.assert_array_equal(left[0], right[0])
            self.assertEqual(left[1:], right[1:])

    def test_observation_contract_and_signal_contract(self) -> None:
        env = self.make_env()
        observation, reset_info = env.reset(
            seed=7,
            spawn_s=0.0,
            lateral_offset=0.0,
            initial_speed=0.0,
            heading_error=0.0,
        )
        self.assertEqual(observation.shape, (23,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertEqual(reset_info["sector_index"], 0)
        self.assertEqual(reset_info["sector_coverage_count"], 0)
        _, reward, _, _, info = env.step([0.0, 0.5])
        self.assertEqual(set(info["signals"]), set(SIGNAL_NAMES))
        expected = sum(env.reward_config.weights[name] * info["signals"][name] for name in SIGNAL_NAMES)
        self.assertAlmostEqual(reward, expected)
        self.assertEqual(set(info["state"]), {"x", "y", "heading", "speed", "lateral_velocity", "yaw_rate", "steer"})
        self.assertEqual(info["sector_coverage_count"], 1)
        self.assertEqual(info["signals"]["checkpoint"], 1.0)
        self.assertGreaterEqual(info["sector_index"], 0)
        self.assertGreaterEqual(info["boundary_excess_m"], 0.0)
        self.assertGreaterEqual(info["off_track_progress_m"], 0.0)

    def test_reward_weights_can_be_replaced_without_changing_physics(self) -> None:
        reward = RewardConfig({"progress": 1.0})
        env = RacingEnv(TRACK, CONFIG, reward)
        env.reset(seed=2, spawn_s=0.0, lateral_offset=0.0, initial_speed=0.0, heading_error=0.0)
        _, actual, _, _, info = env.step([0.0, 0.5])
        self.assertAlmostEqual(actual, info["signals"]["progress"])

    def test_evaluation_reset_uses_fixed_start_and_zero_noise(self) -> None:
        env = self.make_env()
        _, info = env.reset(seed=99, evaluation=True)
        self.assertEqual(info["spawn_s"], TRACK.start_line_s)
        self.assertTrue(info["evaluation"])
        self.assertEqual(env.state.speed, 0.0)
        point, heading, _ = TRACK.point_at_s(TRACK.start_line_s)
        self.assertEqual(env.state.x, point[0])
        self.assertEqual(env.state.y, point[1])
        self.assertEqual(env.state.heading, heading)

    def test_stuck_signal_is_emitted_once_for_stuck_truncation(self) -> None:
        config = replace(
            CONFIG,
            episode=replace(CONFIG.episode, max_steps=20, stuck_steps=1),
        )
        env = RacingEnv(TRACK, config)
        env.reset(seed=2, evaluation=True)
        _, _, terminated, truncated, info = env.step([0.0, 0.0])
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["signals"]["stuck"], 1.0)

    def test_training_spawn_can_match_evaluation_standing_start(self) -> None:
        config = replace(
            CONFIG,
            episode=replace(CONFIG.episode, standing_start_probability=1.0),
        )
        env = RacingEnv(TRACK, config)
        env.reset(seed=123)
        self.assertEqual(env.state.speed, 0.0)


if __name__ == "__main__":
    unittest.main()
