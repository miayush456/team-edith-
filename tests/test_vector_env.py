from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from racing_event.config import SensorConfig, SimulationConfig
from racing_event.env import RacingEnv
from racing_event.vector_env import VectorizedRacingEnv
from racing_event.track import TrackGeometry


ROOT = Path(__file__).resolve().parents[1]
TRACK = TrackGeometry.from_file(ROOT / "tests" / "fixtures" / "tracks" / "synthetic_oval.json")
CONFIG = SimulationConfig.from_file(ROOT / "config" / "simulation.json")


class VectorizedEnvironmentTests(unittest.TestCase):
    def make_vector(self, n_envs: int = 1) -> VectorizedRacingEnv:
        return VectorizedRacingEnv(TRACK, CONFIG, n_envs=n_envs)

    def test_single_environment_matches_scalar_for_fixed_start(self) -> None:
        scalar = RacingEnv(TRACK, CONFIG)
        vector = self.make_vector()
        scalar.reset(seed=4, spawn_s=15.0, lateral_offset=0.0, initial_speed=2.0, heading_error=0.01)
        vector.reset(seed=4, spawn_s=15.0, lateral_offset=0.0, initial_speed=2.0, heading_error=0.01)
        actions = [np.asarray([0.1, 0.6]), np.asarray([-0.2, 0.4]), np.asarray([0.0, -0.3])] * 8
        for action in actions:
            scalar_result = scalar.step(action)
            vector_result = vector.step(action.reshape(1, 2))
            np.testing.assert_array_equal(scalar_result[0], vector_result[0][0])
            self.assertEqual(scalar_result[1], vector_result[1][0])
            self.assertEqual(scalar_result[2], vector_result[2][0])
            self.assertEqual(scalar_result[3], vector_result[3][0])
            for name in ("s", "distance_since_spawn", "off_track", "crash", "lap_complete"):
                self.assertEqual(scalar_result[4][name], vector_result[4][name][0])
            for name in ("sector_index", "sector_coverage_count", "boundary_excess_m", "off_track_progress_m"):
                self.assertEqual(scalar_result[4][name], vector_result[4][name][0])
            self.assertEqual(
                scalar_result[4]["sector_coverage_mask"],
                int(vector_result[4]["sector_coverage_mask"][0]),
            )
            for name, value in scalar_result[4]["signals"].items():
                self.assertEqual(value, vector_result[4]["signals"][name][0])
            for name, value in scalar_result[4]["state"].items():
                self.assertEqual(value, vector_result[4]["state"][name][0])

    def test_random_training_spawn_matches_scalar_for_one_lane(self) -> None:
        scalar = RacingEnv(TRACK, CONFIG)
        vector = self.make_vector()
        scalar_observation, scalar_info = scalar.reset(seed=321)
        vector_observation, vector_info = vector.reset(seed=321)
        np.testing.assert_array_equal(scalar_observation, vector_observation[0])
        self.assertEqual(scalar_info["track_id"], vector_info["track_id"][0])
        self.assertEqual(scalar_info["spawn_s"], vector_info["spawn_s"][0])
        self.assertEqual(scalar.state.__dict__, vector.state.item(0).__dict__)

    def test_custom_ray_layout_preserves_scalar_vector_parity(self) -> None:
        config = SimulationConfig(sensors=SensorConfig(ray_angles_deg=(-30.0, 0.0, 30.0)))
        scalar = RacingEnv(TRACK, config)
        vector = VectorizedRacingEnv(TRACK, config, n_envs=1)
        scalar_observation, _ = scalar.reset(seed=12, evaluation=True)
        vector_observation, _ = vector.reset(seed=12, evaluation=True)
        self.assertEqual(scalar_observation.shape, (17,))
        np.testing.assert_array_equal(scalar_observation, vector_observation[0])
        scalar_step = scalar.step([0.1, 0.5])
        vector_step = vector.step([[0.1, 0.5]])
        np.testing.assert_array_equal(scalar_step[0], vector_step[0][0])

    def test_batch_seed_and_actions_are_deterministic(self) -> None:
        actions = np.asarray([[0.1, 0.6], [-0.2, 0.4], [0.0, -0.3], [0.4, 0.2]])
        first = self.make_vector(4)
        second = self.make_vector(4)
        first_obs, _ = first.reset(seed=123)
        second_obs, _ = second.reset(seed=123)
        np.testing.assert_array_equal(first_obs, second_obs)
        for _ in range(20):
            left = first.step(actions)
            right = second.step(actions)
            for left_value, right_value in zip(left[:4], right[:4]):
                np.testing.assert_array_equal(left_value, right_value)
            np.testing.assert_array_equal(left[4]["s"], right[4]["s"])
            np.testing.assert_array_equal(left[4]["distance_since_spawn"], right[4]["distance_since_spawn"])

    def test_evaluation_reset_is_fixed_and_batched(self) -> None:
        env = self.make_vector(3)
        observations, info = env.reset(seed=8, evaluation=True)
        self.assertEqual(observations.shape, (3, env.observation_size))
        self.assertEqual(observations.dtype, np.float32)
        np.testing.assert_array_equal(info["spawn_s"], np.full(3, TRACK.start_line_s))
        np.testing.assert_array_equal(env.state.speed, np.zeros(3))
        np.testing.assert_array_equal(env.state.x, np.full(3, TRACK.point_at_s(TRACK.start_line_s)[0][0]))
        np.testing.assert_array_equal(info["sector_coverage_count"], np.zeros(3, dtype=np.int64))

    def test_non_finite_actions_are_rejected(self) -> None:
        scalar = RacingEnv(TRACK, CONFIG)
        scalar.reset(seed=1)
        with self.assertRaises(ValueError):
            scalar.step([np.nan, 0.0])
        vector = self.make_vector(2)
        vector.reset(seed=1)
        with self.assertRaises(ValueError):
            vector.step([[0.0, 0.0], [np.inf, 0.0]])

    def test_selective_reset_preserves_nonselected_episode_state(self) -> None:
        env = self.make_vector(3)
        env.reset(
            seed=5,
            spawn_s=np.asarray([10.0, 20.0, 30.0]),
            lateral_offset=np.zeros(3),
            initial_speed=np.asarray([1.0, 2.0, 3.0]),
            heading_error=np.zeros(3),
        )
        env.step(np.asarray([[0.1, 0.5], [-0.2, 0.4], [0.3, 0.2]]))
        before_state = env.state_to_dict()
        before_track_indices = env.track_indices.copy()
        before_previous_s = env.previous_s.copy()
        before_distance = env.distance_since_spawn.copy()
        before_steps = env.steps.copy()
        before_stuck = env.stuck_ticks.copy()
        before_done = env.done

        observations, info = env.reset(
            indices=[1],
            evaluation=False,
            spawn_s=np.asarray([40.0, 50.0, 60.0]),
            lateral_offset=np.zeros(3),
            initial_speed=np.asarray([7.0, 0.0, 9.0]),
            heading_error=np.zeros(3),
        )

        self.assertEqual(observations.shape, (3, env.observation_size))
        self.assertEqual(info["spawn_s"].shape, (3,))
        np.testing.assert_array_equal(env.state.x[[0, 2]], before_state["x"][[0, 2]])
        np.testing.assert_array_equal(env.state.y[[0, 2]], before_state["y"][[0, 2]])
        np.testing.assert_array_equal(env.state.heading[[0, 2]], before_state["heading"][[0, 2]])
        np.testing.assert_array_equal(env.state.speed[[0, 2]], before_state["speed"][[0, 2]])
        np.testing.assert_array_equal(env.track_indices[[0, 2]], before_track_indices[[0, 2]])
        np.testing.assert_array_equal(env.previous_s[[0, 2]], before_previous_s[[0, 2]])
        np.testing.assert_array_equal(env.distance_since_spawn[[0, 2]], before_distance[[0, 2]])
        np.testing.assert_array_equal(env.steps[[0, 2]], before_steps[[0, 2]])
        np.testing.assert_array_equal(env.stuck_ticks[[0, 2]], before_stuck[[0, 2]])
        np.testing.assert_array_equal(env.done[[0, 2]], before_done[[0, 2]])
        self.assertEqual(env.steps[1], 0)
        self.assertFalse(env.done[1])
        self.assertEqual(env.previous_s[1], 50.0)
        self.assertEqual(env.state.speed[1], 0.0)

    def test_selective_reset_is_deterministic_and_empty_is_a_noop(self) -> None:
        first = self.make_vector(3)
        second = self.make_vector(3)
        for env in (first, second):
            env.reset(seed=11)
            env.step(np.asarray([[0.1, 0.3], [-0.1, 0.4], [0.2, 0.5]]))

        first.reset(indices=[0, 2], seed=99)
        second.reset(indices=[0, 2], seed=99)
        np.testing.assert_array_equal(first._observations(), second._observations())
        np.testing.assert_array_equal(first.track_indices, second.track_indices)
        np.testing.assert_array_equal(first.previous_s, second.previous_s)
        np.testing.assert_array_equal(first.steps, second.steps)

        before_state = first.state_to_dict()
        before_steps = first.steps.copy()
        before_done = first.done
        observations, info = first.reset(indices=[])
        np.testing.assert_array_equal(observations, first._observations())
        np.testing.assert_array_equal(first.steps, before_steps)
        np.testing.assert_array_equal(first.done, before_done)
        for name, values in before_state.items():
            np.testing.assert_array_equal(values, first.state_to_dict()[name])
        self.assertEqual(info["spawn_s"].shape, (3,))

    def test_masked_step_advances_only_active_lanes_and_allows_inactive_done(self) -> None:
        env = self.make_vector(3)
        env.reset(seed=12, evaluation=True)
        env.step(np.asarray([[0.1, 0.5], [0.0, 0.5], [-0.1, 0.5]]))
        env._done[2] = True
        before_state = env.state_to_dict()
        before_track_indices = env.track_indices.copy()
        before_previous_s = env.previous_s.copy()
        before_distance = env.distance_since_spawn.copy()
        before_steps = env.steps.copy()
        before_stuck = env.stuck_ticks.copy()

        observations, rewards, terminated, truncated, info = env.step(
            np.asarray([[0.2, 0.6], [-0.3, 0.7], [0.9, 0.9]]),
            active_mask=np.asarray([True, False, False]),
        )

        self.assertEqual(observations.shape, (3, env.observation_size))
        np.testing.assert_array_equal(info["active_mask"], np.asarray([True, False, False]))
        np.testing.assert_array_equal(rewards[1:], np.zeros(2))
        np.testing.assert_array_equal(terminated, np.zeros(3, dtype=bool))
        np.testing.assert_array_equal(truncated, np.zeros(3, dtype=bool))
        for name, values in before_state.items():
            np.testing.assert_array_equal(values[1:], env.state_to_dict()[name][1:])
        np.testing.assert_array_equal(env.track_indices[1:], before_track_indices[1:])
        np.testing.assert_array_equal(env.previous_s[1:], before_previous_s[1:])
        np.testing.assert_array_equal(env.distance_since_spawn[1:], before_distance[1:])
        np.testing.assert_array_equal(env.steps[1:], before_steps[1:])
        np.testing.assert_array_equal(env.stuck_ticks[1:], before_stuck[1:])
        self.assertTrue(env.done[2])
        for values in info["signals"].values():
            np.testing.assert_array_equal(values[1:], np.zeros(2))
        self.assertEqual(info["events"][1], [])
        self.assertEqual(info["events"][2], [])

        with self.assertRaises(RuntimeError):
            env.step(np.zeros((3, 2)), active_mask=np.asarray([False, False, True]))

    def test_invalid_reset_indices_and_active_masks_are_rejected(self) -> None:
        env = self.make_vector(3)
        env.reset(seed=13)
        invalid_indices = ([0, 0], [-1], [3], [True], [1.5])
        for indices in invalid_indices:
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                env.reset(indices=indices)

        invalid_masks = (
            np.asarray([True, False]),
            np.asarray([1, 0, 1]),
            np.asarray([1.0, 0.0, 1.0]),
            np.asarray([[True, False, True]]),
        )
        actions = np.zeros((3, 2))
        for active_mask in invalid_masks:
            with self.subTest(active_mask=active_mask), self.assertRaises(ValueError):
                env.step(actions, active_mask=active_mask)

        with self.assertRaises(ValueError):
            env.step(np.asarray([[np.nan, 0.0], [0.0, 0.0], [0.0, 0.0]]), active_mask=np.zeros(3, dtype=bool))


if __name__ == "__main__":
    unittest.main()
