from __future__ import annotations

import copy
import json
import unittest

import numpy as np

from racing_event.metrics import RolloutMetrics
from racing_event.reward import SIGNAL_NAMES


def make_info(
    n_envs: int,
    *,
    track_ids: tuple[str, ...] | None = None,
    signals: dict[str, np.ndarray] | None = None,
    speed: np.ndarray | None = None,
    off_track: np.ndarray | None = None,
    crash: np.ndarray | None = None,
    lap_complete: np.ndarray | None = None,
    boundary_excess_m: np.ndarray | None = None,
    off_track_progress_m: np.ndarray | None = None,
    sector_coverage_count: np.ndarray | None = None,
) -> dict:
    info = {
        "track_id": track_ids or tuple("track_a" for _ in range(n_envs)),
        "signals": signals
        or {name: np.zeros(n_envs, dtype=np.float64) for name in SIGNAL_NAMES},
        "state": {"speed": speed if speed is not None else np.zeros(n_envs)},
        "off_track": off_track if off_track is not None else np.zeros(n_envs, dtype=bool),
        "crash": crash if crash is not None else np.zeros(n_envs, dtype=bool),
        "lap_complete": lap_complete
        if lap_complete is not None
        else np.zeros(n_envs, dtype=bool),
    }
    if boundary_excess_m is not None:
        info["boundary_excess_m"] = boundary_excess_m
    if off_track_progress_m is not None:
        info["off_track_progress_m"] = off_track_progress_m
    if sector_coverage_count is not None:
        info["sector_coverage_count"] = sector_coverage_count
    return info


class RolloutMetricsTests(unittest.TestCase):
    def test_denominators_and_simultaneous_crash_lap(self) -> None:
        metrics = RolloutMetrics(2, 1 / 30, {"progress": 2.0, "crash": -3.0})
        signals = {name: np.asarray([1.0, 2.0]) for name in SIGNAL_NAMES}
        metrics.update(
            rewards=np.asarray([2.0, 3.0]),
            terminated=np.asarray([False, True]),
            truncated=np.asarray([False, False]),
            actions=np.asarray([[1.2, -0.5], [0.0, 1.0]]),
            info=make_info(
                2,
                track_ids=("oval", "tight"),
                signals=signals,
                speed=np.asarray([4.0, 8.0]),
                off_track=np.asarray([True, False]),
                crash=np.asarray([False, True]),
                lap_complete=np.asarray([False, True]),
            ),
        )

        snapshot = metrics.snapshot(wall_time_s=2.0)
        self.assertEqual(snapshot["env_steps"], 2)
        self.assertEqual(snapshot["episodes"], 1)
        self.assertEqual(snapshot["reward_sum"], 5.0)
        self.assertEqual(snapshot["mean_episode_return"], 3.0)
        self.assertEqual(snapshot["crashes"], 1)
        self.assertEqual(snapshot["lap_completions"], 0)
        self.assertEqual(snapshot["terminated_episodes"], 1)
        self.assertEqual(snapshot["truncated_episodes"], 0)
        self.assertEqual(snapshot["crash_rate"], 1.0)
        self.assertEqual(snapshot["lap_completion_rate"], 0.0)
        self.assertEqual(snapshot["off_track_rate"], 0.5)
        self.assertEqual(snapshot["mean_speed_mps"], 6.0)
        self.assertEqual(snapshot["action_saturation_rate"], 0.5)
        self.assertEqual(snapshot["mean_abs_action_delta"], 0.625)
        self.assertEqual(snapshot["signal_sums"]["progress"], 3.0)
        self.assertEqual(snapshot["signal_means"]["progress"], 1.5)
        self.assertEqual(snapshot["weighted_signal_sums"]["progress"], 6.0)
        self.assertEqual(snapshot["weighted_signal_sums"]["crash"], -9.0)
        self.assertEqual(snapshot["env_steps_per_second"], 1.0)

    def test_episode_and_action_history_cross_calls_and_reset(self) -> None:
        metrics = RolloutMetrics(1, 0.1, {})
        info = make_info(1)
        metrics.update(
            rewards=np.asarray([2.0]),
            terminated=np.asarray([False]),
            truncated=np.asarray([False]),
            actions=np.asarray([[0.5, 0.5]]),
            info=info,
        )
        metrics.update(
            rewards=np.asarray([3.0]),
            terminated=np.asarray([True]),
            truncated=np.asarray([False]),
            actions=np.asarray([[1.5, -1.5]]),
            info=info,
        )
        metrics.update(
            rewards=np.asarray([7.0]),
            terminated=np.asarray([False]),
            truncated=np.asarray([False]),
            actions=np.asarray([[0.2, 0.0]]),
            info=info,
        )

        snapshot = metrics.snapshot(wall_time_s=1.0)
        self.assertEqual(snapshot["env_steps"], 3)
        self.assertEqual(snapshot["episodes"], 1)
        self.assertEqual(snapshot["reward_sum"], 12.0)
        self.assertEqual(snapshot["mean_episode_return"], 5.0)
        self.assertAlmostEqual(snapshot["mean_abs_action_delta"], 3.2 / 6.0)
        self.assertEqual(metrics._episode_returns[0], 7.0)
        self.assertEqual(metrics._episode_lengths[0], 1)

    def test_masked_lanes_do_not_count_or_reset(self) -> None:
        metrics = RolloutMetrics(2, 0.1, {})
        metrics.update(
            rewards=np.asarray([1.0, 5.0]),
            terminated=np.asarray([False, False]),
            truncated=np.asarray([False, False]),
            actions=np.asarray([[0.1, 0.0], [0.8, 0.8]]),
            active_mask=np.asarray([True, True]),
            info=make_info(
                2,
                track_ids=("used", "ignored"),
                speed=np.asarray([3.0, 99.0]),
                boundary_excess_m=np.asarray([1.0, 100.0]),
                off_track_progress_m=np.asarray([2.0, 100.0]),
                sector_coverage_count=np.asarray([2, 9]),
            ),
        )
        metrics.update(
            rewards=np.asarray([2.0, 99.0]),
            terminated=np.asarray([False, True]),
            truncated=np.asarray([False, False]),
            actions=np.asarray([[0.2, 0.0], [1.0, 1.0]]),
            active_mask=np.asarray([True, False]),
            info=make_info(
                2,
                track_ids=("used", "ignored"),
                speed=np.asarray([4.0, 99.0]),
                boundary_excess_m=np.asarray([3.0, 999.0]),
                off_track_progress_m=np.asarray([4.0, 999.0]),
                sector_coverage_count=np.asarray([3, 99]),
            ),
        )
        snapshot = metrics.snapshot(wall_time_s=1.0)
        self.assertEqual(snapshot["env_steps"], 3)
        self.assertEqual(snapshot["reward_sum"], 8.0)
        self.assertEqual(snapshot["boundary_excess_m_sum"], 104.0)
        self.assertEqual(snapshot["positive_off_track_progress_m"], 106.0)
        self.assertEqual(set(snapshot["per_track"]), {"used", "ignored"})
        self.assertEqual(metrics._episode_returns[1], 0.0)
        self.assertEqual(metrics._episode_lengths[1], 0)

        metrics.update(
            rewards=np.asarray([3.0, 7.0]),
            terminated=np.asarray([False, False]),
            truncated=np.asarray([False, False]),
            actions=np.asarray([[0.3, 0.0], [0.2, 0.0]]),
            active_mask=np.asarray([False, True]),
            info=make_info(2, track_ids=("used", "ignored"), speed=np.asarray([4.0, 1.0])),
        )
        self.assertAlmostEqual(metrics.snapshot(wall_time_s=1.0)["mean_abs_action_delta"], 0.25)

    def test_raw_signals_and_weighted_terms_are_kept_per_track(self) -> None:
        weights = {name: float(index + 1) for index, name in enumerate(SIGNAL_NAMES)}
        metrics = RolloutMetrics(2, 0.1, weights)
        signal_values = {
            name: np.asarray([float(index + 1), float(2 * (index + 1))])
            for index, name in enumerate(SIGNAL_NAMES)
        }
        metrics.update(
            rewards=np.asarray([0.0, 0.0]),
            terminated=np.asarray([False, False]),
            truncated=np.asarray([False, False]),
            actions=np.zeros((2, 2)),
            info=make_info(2, track_ids=("a", "b"), signals=signal_values),
        )
        snapshot = metrics.snapshot(wall_time_s=1.0)
        for index, name in enumerate(SIGNAL_NAMES):
            self.assertEqual(snapshot["signal_sums"][name], float(index + 1 + 2 * (index + 1)))
            self.assertEqual(
                snapshot["weighted_signal_sums"][name],
                float((index + 1) * (index + 1 + 2 * (index + 1))),
            )
        self.assertEqual(snapshot["per_track"]["a"]["signal_sums"]["progress"], 1.0)
        self.assertEqual(snapshot["per_track"]["b"]["signal_sums"]["progress"], 2.0)

    def test_empty_snapshot_has_null_denominators_and_is_json_safe(self) -> None:
        snapshot = RolloutMetrics(2, 0.1, {}).snapshot(wall_time_s=0.0)
        self.assertIsNone(snapshot["mean_episode_return"])
        self.assertIsNone(snapshot["lap_completion_rate"])
        self.assertIsNone(snapshot["crash_rate"])
        self.assertIsNone(snapshot["off_track_rate"])
        self.assertIsNone(snapshot["mean_speed_mps"])
        self.assertIsNone(snapshot["action_saturation_rate"])
        self.assertIsNone(snapshot["mean_abs_action_delta"])
        self.assertIsNone(snapshot["env_steps_per_second"])
        json.dumps(snapshot, allow_nan=False)

    def test_anti_grind_and_sector_metrics_are_aggregated(self) -> None:
        metrics = RolloutMetrics(2, 0.1, {})
        metrics.update(
            rewards=np.asarray([1.0, 2.0]),
            terminated=np.asarray([True, False]),
            truncated=np.asarray([False, False]),
            actions=np.zeros((2, 2)),
            info=make_info(
                2,
                track_ids=("a", "b"),
                off_track=np.asarray([True, False]),
                boundary_excess_m=np.asarray([0.5, 0.0]),
                off_track_progress_m=np.asarray([1.25, 0.0]),
                sector_coverage_count=np.asarray([3, 4]),
            ),
        )

        snapshot = metrics.snapshot(wall_time_s=1.0)
        self.assertEqual(snapshot["positive_off_track_progress_m"], 1.25)
        self.assertEqual(snapshot["boundary_excess_m_sum"], 0.5)
        self.assertEqual(snapshot["boundary_excess_step_rate"], 0.5)
        self.assertEqual(snapshot["mean_sector_coverage"], 3.0)
        self.assertEqual(snapshot["per_track"]["a"]["mean_sector_coverage"], 3.0)
        self.assertIsNone(snapshot["per_track"]["b"]["mean_sector_coverage"])

    def test_invalid_update_is_rejected_atomically(self) -> None:
        metrics = RolloutMetrics(1, 0.1, {})
        valid_info = make_info(1)
        metrics.update(
            rewards=np.asarray([1.0]),
            terminated=np.asarray([False]),
            truncated=np.asarray([False]),
            actions=np.asarray([[0.25, -0.25]]),
            info=valid_info,
        )
        before = copy.deepcopy(metrics.snapshot(wall_time_s=1.0))

        invalid_updates = (
            {"rewards": np.asarray([np.nan])},
            {"actions": np.zeros((1, 1))},
            {"terminated": np.asarray([1])},
            {"info": {**valid_info, "signals": {"progress": np.zeros(1)}}},
            {"info": make_info(1, speed=np.asarray([np.inf]))},
            {"info": {**valid_info, "track_id": ["track_a"]}},
        )
        for replacement in invalid_updates:
            kwargs = {
                "rewards": np.asarray([1.0]),
                "terminated": np.asarray([False]),
                "truncated": np.asarray([False]),
                "actions": np.asarray([[0.25, -0.25]]),
                "info": valid_info,
            }
            kwargs.update(replacement)
            with self.assertRaises(ValueError):
                metrics.update(**kwargs)
            self.assertEqual(metrics.snapshot(wall_time_s=1.0), before)


if __name__ == "__main__":
    unittest.main()
