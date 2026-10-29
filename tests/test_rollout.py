from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from racing_event.budget import RunBudget, StepLimitExceeded
from racing_event.config import EpisodeConfig, SimulationConfig
from racing_event.observation import RunningObservationNormalizer, TrainingContract
from racing_event.rollout import PPOHyperparameters, RolloutCollector
from racing_event.track import TrackGeometry
from racing_event.vector_env import VectorizedRacingEnv


ROOT = Path(__file__).resolve().parents[1]
TRACK = TrackGeometry.from_file(ROOT / "tests" / "fixtures" / "tracks" / "synthetic_oval.json")
CONFIG = SimulationConfig.from_file(ROOT / "config" / "simulation.json")


class ZeroPolicy:
    def __call__(self, observations: np.ndarray) -> np.ndarray:
        return np.zeros((observations.shape[0], 2), dtype=np.float64)


class RolloutCollectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.budget_path = Path(self.temp_dir.name) / "budget.json"
        self.budget = RunBudget.create(self.budget_path, max_runs=1, max_steps_per_run=8)
        self.run_id = self.budget.start_run()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_collector(self, *, n_envs: int = 3, config: SimulationConfig = CONFIG) -> RolloutCollector:
        env = VectorizedRacingEnv(TRACK, config, n_envs=n_envs)
        normalizer = RunningObservationNormalizer(env.observation_size)
        return RolloutCollector(
            env,
            ZeroPolicy(),
            self.budget,
            self.run_id,
            normalizer=normalizer,
            seed=7,
        )

    def test_exact_transition_budget_and_partial_final_batch(self) -> None:
        collector = self.make_collector()
        result = collector.collect(5)
        self.assertEqual(result.batch.transitions, 5)
        self.assertEqual(result.batch.active_mask.tolist(), [[True, True, True], [True, True, False]])
        self.assertEqual(result.batch.observations.shape, (2, 3, collector.observation_size))
        self.assertEqual(result.batch.next_observations.shape, result.batch.observations.shape)
        self.assertEqual(self.budget.status(self.run_id)["used_steps"], 5)
        self.assertEqual(result.metrics["env_steps"], 5)

        with self.assertRaises(StepLimitExceeded):
            collector.collect(4)
        self.assertEqual(self.budget.status(self.run_id)["used_steps"], 5)

    def test_normalizer_updates_in_training_and_contract_can_be_saved_frozen(self) -> None:
        collector = self.make_collector(n_envs=2)
        collector.collect(4)
        assert collector.normalizer is not None
        self.assertGreater(collector.normalizer.count, 0)
        collector.normalizer.freeze()
        contract_path = Path(self.temp_dir.name) / "contract.json"
        collector.env.training_contract.save(contract_path, normalizer=collector.normalizer)
        loaded_contract, loaded_normalizer = TrainingContract.load(contract_path)
        self.assertEqual(loaded_contract.observation.names, collector.env.observation_schema.names)
        assert loaded_normalizer is not None
        self.assertFalse(loaded_normalizer.training)
        before = loaded_normalizer.count
        loaded_normalizer.transform(np.zeros((2, collector.observation_size), dtype=np.float32), training=False)
        self.assertEqual(loaded_normalizer.count, before)

    def test_done_lanes_are_reset_only_when_more_transitions_are_needed(self) -> None:
        short_config = replace(CONFIG, episode=EpisodeConfig(max_steps=1, stuck_steps=90, backward_limit_m=25.0))
        collector = self.make_collector(n_envs=2, config=short_config)
        first = collector.collect(2)
        self.assertEqual(first.metrics["episodes"], 2)
        self.assertTrue(np.all(collector.env.done))

        second = collector.collect(2)
        self.assertEqual(second.batch.transitions, 2)
        self.assertEqual(second.metrics["episodes"], 2)
        self.assertTrue(np.all(collector.env.done))

    def test_invalid_policy_output_is_rejected_before_budget_reservation(self) -> None:
        env = VectorizedRacingEnv(TRACK, CONFIG, n_envs=2)

        def bad_policy(_: np.ndarray) -> np.ndarray:
            return np.asarray([[np.nan, 0.0], [0.0, 0.0]])

        collector = RolloutCollector(env, bad_policy, self.budget, self.run_id, seed=1)
        with self.assertRaises(ValueError):
            collector.collect(1)
        self.assertEqual(self.budget.status(self.run_id)["used_steps"], 0)

    def test_participant_reward_runs_outside_environment(self) -> None:
        env = VectorizedRacingEnv(TRACK, CONFIG, reward=None, n_envs=2)
        calls: list[tuple[dict[str, float], bool, bool]] = []

        def participant_reward(observation, action, signals, terminated, truncated):
            observation[0] = 999.0
            action[0] = 999.0
            calls.append((signals, terminated, truncated))
            return 2.5

        collector = RolloutCollector(
            env,
            ZeroPolicy(),
            self.budget,
            self.run_id,
            reward_function=participant_reward,
            seed=7,
        )
        result = collector.collect(3)

        self.assertIsNone(env.reward_config)
        self.assertEqual(len(calls), 3)
        np.testing.assert_allclose(result.batch.rewards[result.batch.active_mask], 2.5)
        self.assertEqual(set(calls[0][0]), {"progress", "speed", "curvature_matched_speed", "off_track", "crash", "stuck", "lap_complete", "time", "steer_smoothness", "checkpoint"})

    def test_participant_reward_sees_applied_bounded_action(self) -> None:
        env = VectorizedRacingEnv(TRACK, CONFIG, reward=None, n_envs=1)
        seen: list[np.ndarray] = []

        def policy(observations: np.ndarray) -> np.ndarray:
            return np.asarray([[5.0, -5.0]])

        def participant_reward(observation, action, signals, terminated, truncated):
            del observation, signals, terminated, truncated
            seen.append(action)
            return 0.0

        collector = RolloutCollector(
            env,
            policy,
            self.budget,
            self.run_id,
            reward_function=participant_reward,
            seed=7,
        )
        result = collector.collect(1)

        np.testing.assert_array_equal(seen[0], [1.0, -1.0])
        np.testing.assert_array_equal(result.batch.actions[0, 0], [5.0, -5.0])
        np.testing.assert_array_equal(env.state.previous_longitudinal_command, [-1.0])

    def test_reward_failure_consumes_transition_and_blocks_stale_retry(self) -> None:
        env = VectorizedRacingEnv(TRACK, CONFIG, reward=None, n_envs=1)

        def participant_reward(observation, action, signals, terminated, truncated):
            del observation, action, signals, terminated, truncated
            return np.nan

        collector = RolloutCollector(
            env,
            ZeroPolicy(),
            self.budget,
            self.run_id,
            reward_function=participant_reward,
            seed=7,
        )
        with self.assertRaises(ValueError):
            collector.collect(1)
        self.assertEqual(self.budget.status(self.run_id)["used_steps"], 1)
        with self.assertRaisesRegex(RuntimeError, "cannot be reused"):
            collector.collect(1)

    def test_collector_rejects_unbounded_batch_request(self) -> None:
        collector = self.make_collector()
        with self.assertRaisesRegex(ValueError, "max_batch_steps"):
            collector.collect(8_193)
        self.assertEqual(self.budget.status(self.run_id)["used_steps"], 0)

    def test_crash_reset_is_random_new_episode_and_budget_keeps_counting(self) -> None:
        class CrashProbe(VectorizedRacingEnv):
            def __init__(self):
                super().__init__(TRACK, CONFIG, n_envs=1)
                self.spawns = []

            def reset(self, *args, **kwargs):
                observation, info = super().reset(*args, **kwargs)
                self.spawns.append(float(info["spawn_s"][0]))
                return observation, info

        env = CrashProbe()

        def barrier_impact(observation):
            # Force a barrier-contact scenario before each real engine step;
            # the collector must preserve its failure then reset normally.
            barrier = env.tracks[int(env.track_indices[0])].barrier_left[0]
            env.state.x[0], env.state.y[0] = barrier
            return np.zeros((1, 2))

        collector = RolloutCollector(env, barrier_impact, self.budget, self.run_id, seed=17)
        result = collector.collect(2)
        self.assertEqual(len(env.spawns), 2)
        self.assertNotEqual(env.spawns[0], env.spawns[1])
        self.assertTrue(np.all(result.batch.terminated))
        np.testing.assert_array_equal(result.batch.signals["crash"], [[1.0], [1.0]])
        self.assertEqual(self.budget.status(self.run_id)["used_steps"], 2)
        # The next episode starts with a new distance target, not free progress
        # from the old spawn; no reset itself is stored as a PPO transition.
        self.assertEqual(result.batch.transitions, 2)
        self.assertTrue(np.all(result.batch.signals["lap_complete"] == 0))
        self.assertEqual(result.training_position[0]["end_reason"], "barrier contact")

    def test_runoff_is_a_recoverable_position_and_not_an_automatic_reset(self) -> None:
        class RunoffProbe(VectorizedRacingEnv):
            def __init__(self):
                super().__init__(TRACK, CONFIG, n_envs=1)
                self.reset_count = 0

            def reset(self, *args, **kwargs):
                self.reset_count += 1
                kwargs.update(spawn_s=0.0, lateral_offset=4.0, initial_speed=0.0, heading_error=0.0)
                return super().reset(*args, **kwargs)

        env = RunoffProbe()
        collector = RolloutCollector(env, ZeroPolicy(), self.budget, self.run_id, seed=17)
        result = collector.collect(2)
        self.assertEqual(env.reset_count, 1)
        self.assertFalse(np.any(result.batch.terminated))
        self.assertFalse(np.any(result.batch.truncated))
        self.assertTrue(result.training_position[0]["off_track"])
        self.assertIsNone(result.training_position[0]["end_reason"])
        self.assertAlmostEqual(result.training_position[0]["episode_time_s"], 2 * CONFIG.dt)
        self.assertEqual(result.training_position[0]["speed_mps"], 0.0)

    def test_hyperparameters_are_explicit_and_serializable(self) -> None:
        hyperparameters = PPOHyperparameters()
        payload = hyperparameters.to_dict()
        self.assertEqual(payload["rollout_length"], 128)
        self.assertEqual(payload["hidden_sizes"], [64, 64])
        self.assertAlmostEqual(payload["gamma"], 0.99)
        self.assertEqual(payload["update_epochs"], 10)
        self.assertEqual(payload["minibatch_size"], 64)
        self.assertAlmostEqual(payload["learning_rate"], 3.0e-4)
        self.assertAlmostEqual(payload["entropy_coefficient"], 0.01)


if __name__ == "__main__":
    unittest.main()
