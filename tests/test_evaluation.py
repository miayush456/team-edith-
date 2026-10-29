from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from racing_event.evaluation import evaluate_policy, validate_evaluation_contract
from racing_event.env import RacingEnv
from racing_event.config import SimulationConfig
from racing_event.observation import RunningObservationNormalizer
from racing_event.track import TrackGeometry
from racing_event.vector_env import VectorizedRacingEnv
from racing_event.evaluation_audit import EvaluationAudit
from tools.run_scripted import scripted_action


ROOT = Path(__file__).resolve().parents[1]
TRACK = TrackGeometry.from_file(ROOT / "tests" / "fixtures" / "tracks" / "synthetic_oval.json")
CONFIG = SimulationConfig.from_file(ROOT / "config" / "simulation.json")


class ZeroPolicy:
    def __call__(self, observations: np.ndarray) -> np.ndarray:
        return np.zeros((observations.shape[0], 2), dtype=np.float64)


class EvaluationTests(unittest.TestCase):
    def make_env(self, *, max_steps: int | None = None) -> VectorizedRacingEnv:
        config = CONFIG
        if max_steps is not None:
            config = replace(
                CONFIG,
                episode=replace(CONFIG.episode, max_steps=max_steps),
            )
        return VectorizedRacingEnv(TRACK, config, n_envs=1)

    def test_evaluation_is_deterministic_and_does_not_update_normalizer(self) -> None:
        env = self.make_env(max_steps=3)
        normalizer = RunningObservationNormalizer(env.observation_size)
        before = normalizer.state_dict()
        first = evaluate_policy(env, ZeroPolicy(), normalizer=normalizer)
        after_first = normalizer.state_dict()
        second = evaluate_policy(env, ZeroPolicy(), normalizer=normalizer)

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.status, "dnf_timeout")
        self.assertEqual(first.steps, 3)
        self.assertEqual(first.lap_time_s, None)
        self.assertEqual(before, after_first)
        self.assertEqual(len(first.trajectory_hash), 64)

    def test_contract_validation_rejects_changed_config_or_track(self) -> None:
        env = self.make_env()
        validate_evaluation_contract(env, env.training_contract)
        with self.assertRaises(ValueError):
            validate_evaluation_contract(
                env,
                replace(env.training_contract, config_hash="0" * 64),
            )
        with self.assertRaises(ValueError):
            validate_evaluation_contract(
                env,
                replace(env.training_contract, track_hashes=("0" * 64,)),
            )

    def test_evaluation_reset_rejects_noncanonical_spawn(self) -> None:
        scalar = RacingEnv(TRACK, CONFIG)
        with self.assertRaises(ValueError):
            scalar.reset(evaluation=True, initial_speed=1.0)
        with self.assertRaises(ValueError):
            self.make_env().reset(evaluation=True, lateral_offset=0.1)

    def test_stuck_outcome_is_distinct_from_timeout(self) -> None:
        env = self.make_env()
        result = evaluate_policy(env, ZeroPolicy(), max_steps=CONFIG.episode.stuck_steps + 1)
        self.assertEqual(result.status, "dnf_stuck")
        self.assertGreaterEqual(result.steps, CONFIG.episode.stuck_steps)

    def test_audit_preserves_preexisting_trajectory_digest_and_replay_contract(self) -> None:
        result = evaluate_policy(self.make_env(), ZeroPolicy(), max_steps=3)
        self.assertEqual(result.trajectory_hash, "020827c2735b4b88a71c69865eef23cf39f69f918dde0d6de2f81d39f7c915b2")
        audit = result.to_dict()["diagnostics"]
        self.assertEqual(audit["stalled_steps"], 3)
        self.assertEqual(audit["max_contiguous_stalled_steps"], 3)
        self.assertEqual(audit["elapsed_s"], 3 * CONFIG.dt)
        self.assertEqual(audit["positive_progress_m"], 0.0)
        self.assertIsNone(audit["positive_off_track_progress_fraction"])
        self.assertEqual(audit["sector_first_crossing_ticks"], [None] * 16)

    def test_clean_scripted_lap_has_ordered_fixed_tick_splits_and_no_runoff(self) -> None:
        env = self.make_env()

        def policy(_observations):
            snapshot = SimpleNamespace(track=TRACK, config=CONFIG, state=env.state.item(0))
            return scripted_action(snapshot)[None, :]

        result = evaluate_policy(env, policy)
        self.assertEqual(result.status, "finished")
        audit = result.diagnostics
        self.assertEqual(audit["off_track_duration_s"], 0.0)
        self.assertEqual(audit["max_boundary_excess_m"], 0.0)
        self.assertEqual(audit["sector_coverage_count"], 16)
        splits = audit["sector_first_crossing_ticks"]
        self.assertTrue(all(isinstance(tick, int) for tick in splits))
        self.assertEqual(splits, sorted(set(splits)))
        self.assertEqual(splits[-1], result.steps)

    def test_evaluation_rejects_random_track_pool_before_reset(self) -> None:
        second = TrackGeometry.from_file(ROOT / "tests/fixtures/tracks/synthetic_flowing.json")
        env = VectorizedRacingEnv((TRACK, second), CONFIG, n_envs=1)
        with self.assertRaisesRegex(ValueError, "one explicit track"):
            evaluate_policy(env, ZeroPolicy(), max_steps=1)

    def test_contract_validation_checks_semantics_not_only_names(self) -> None:
        env = self.make_env()
        original = env.training_contract
        for schema in (
            replace(original.observation, dtype="float64"),
            replace(original.observation, normalization="participant_changed"),
            replace(original.observation, fields=(
                replace(original.observation.fields[0], high=2.0),
                *original.observation.fields[1:],
            )),
        ):
            with self.subTest(schema=schema):
                with self.assertRaisesRegex(ValueError, "observation semantics"):
                    validate_evaluation_contract(env, replace(original, observation=schema))


class EvaluationAuditTests(unittest.TestCase):
    def test_first_forward_splits_backward_recrossing_and_partial_run(self) -> None:
        audit = EvaluationAudit(160.0, 0.1)
        previous = 0.0
        for distance in (25.0, 5.0, 35.0, 35.0):
            audit.update(previous_distance=previous, distance=distance, off_track=False,
                         boundary_excess=0.0, coverage_mask=1, raw_action=np.zeros(2))
            previous = distance
        result = audit.snapshot()
        self.assertEqual(result["sector_first_crossing_ticks"], [1, 1, 3] + [None] * 13)
        self.assertEqual(result["backward_progress_m"], 20.0)
        self.assertEqual(result["positive_progress_m"], 55.0)
        self.assertEqual(result["stalled_steps"], 1)

    def test_runoff_streaks_positive_progress_and_applied_action_deltas(self) -> None:
        audit = EvaluationAudit(160.0, 0.1)
        for tick, (distance, off_track) in enumerate(((2, True), (4, True), (6, False), (8, True))):
            audit.update(previous_distance=tick * 2, distance=distance, off_track=off_track,
                         boundary_excess=0.4 if off_track else 0.0, coverage_mask=1,
                         raw_action=np.asarray([2.0, 0.5]))
        result = audit.snapshot()
        self.assertAlmostEqual(result["off_track_duration_s"], 0.3)
        self.assertEqual(result["max_contiguous_off_track_steps"], 2)
        self.assertEqual(result["positive_off_track_progress_m"], 6.0)
        self.assertEqual(result["positive_off_track_progress_fraction"], 0.75)
        self.assertEqual(result["max_boundary_excess_m"], 0.4)
        self.assertEqual(result["action_saturation_rate"], {"steering": 1.0, "longitudinal": 0.0})
        self.assertEqual(result["mean_abs_applied_action_delta"], {"steering": 0.25, "longitudinal": 0.125})
        result["sector_first_crossing_ticks"][0] = 1000
        self.assertIsNone(audit.snapshot()["sector_first_crossing_ticks"][0])


if __name__ == "__main__":
    unittest.main()
