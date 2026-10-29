"""Small, batch-oriented rollout metrics for the NumPy racing trainer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .reward import SIGNAL_NAMES


class _Summary:
    """Mutable counters shared by the global and per-track summaries."""

    def __init__(self) -> None:
        self.env_steps = 0
        self.episodes = 0
        self.reward_sum = 0.0
        self.completed_return_sum = 0.0
        self.lap_completions = 0
        self.crashes = 0
        self.terminated_episodes = 0
        self.truncated_episodes = 0
        self.off_track_steps = 0
        self.off_track_progress_m = 0.0
        self.boundary_excess_m_sum = 0.0
        self.boundary_excess_steps = 0
        self.sector_coverage_sum = 0
        self.sector_coverage_episodes = 0
        self.sector_coverage_step_sum = 0
        self.sector_coverage_steps = 0
        self.speed_sum = 0.0
        self.action_saturation_components = 0
        self.action_components = 0
        self.action_delta_sum = 0.0
        self.signal_sums = {name: 0.0 for name in SIGNAL_NAMES}

    def add_step(
        self,
        *,
        reward: float,
        signals: Mapping[str, float],
        speed: float,
        off_track: bool,
        crash: bool,
        lap_complete: bool,
        action_saturated: int,
        action_delta: float,
        boundary_excess_m: float,
        off_track_progress_m: float,
        sector_coverage_count: int | None,
    ) -> None:
        self.env_steps += 1
        self.reward_sum += reward
        self.off_track_steps += int(off_track)
        self.off_track_progress_m += off_track_progress_m
        if sector_coverage_count is not None:
            self.sector_coverage_step_sum += sector_coverage_count
            self.sector_coverage_steps += 1
        self.boundary_excess_m_sum += boundary_excess_m
        self.boundary_excess_steps += int(boundary_excess_m > 0.0)
        self.speed_sum += speed
        self.action_saturation_components += action_saturated
        self.action_components += 2
        self.action_delta_sum += action_delta
        self.crashes += int(crash)
        self.lap_completions += int(lap_complete and not crash)
        for name in SIGNAL_NAMES:
            self.signal_sums[name] += signals[name]

    def add_episode(
        self,
        *,
        episode_return: float,
        terminated: bool,
        truncated: bool,
        sector_coverage_count: int | None,
    ) -> None:
        self.episodes += 1
        self.completed_return_sum += episode_return
        self.terminated_episodes += int(terminated)
        self.truncated_episodes += int(truncated)
        if sector_coverage_count is not None:
            self.sector_coverage_sum += sector_coverage_count
            self.sector_coverage_episodes += 1


class RolloutMetrics:
    """Accumulate trainer-facing metrics from full vector-environment batches.

    The environment owns lane resets.  Consequently, a lane's unfinished
    return, length, and previous applied action stay here between calls and
    are cleared only after that lane reports ``terminated`` or ``truncated``.
    """

    def __init__(self, n_envs: int, dt: float, weights: Mapping[str, float]):
        if isinstance(n_envs, (bool, np.bool_)) or not isinstance(n_envs, (int, np.integer)):
            raise ValueError("n_envs must be a positive integer")
        if int(n_envs) <= 0:
            raise ValueError("n_envs must be a positive integer")
        try:
            dt_value = float(dt)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("dt must be a positive finite number") from exc
        if not np.isfinite(dt_value) or dt_value <= 0.0:
            raise ValueError("dt must be a positive finite number")
        if not isinstance(weights, Mapping):
            raise ValueError("weights must be a mapping")
        unknown = set(weights) - set(SIGNAL_NAMES)
        if unknown:
            raise ValueError(f"unknown reward signals: {sorted(unknown)}")

        normalized_weights: dict[str, float] = {}
        for name in SIGNAL_NAMES:
            try:
                value = float(weights.get(name, 0.0))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"weight for {name!r} must be finite") from exc
            if not np.isfinite(value):
                raise ValueError(f"weight for {name!r} must be finite")
            normalized_weights[name] = value

        self.n_envs = int(n_envs)
        self.dt = dt_value
        self.weights = normalized_weights
        self._summary = _Summary()
        self._per_track: dict[str, _Summary] = {}
        self._episode_returns = np.zeros(self.n_envs, dtype=np.float64)
        self._episode_lengths = np.zeros(self.n_envs, dtype=np.int64)
        self._episode_sector_coverage_available = np.zeros(self.n_envs, dtype=bool)
        self._last_applied_actions = np.zeros((self.n_envs, 2), dtype=np.float64)

    @staticmethod
    def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite numeric array") from exc
        if raw.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if np.issubdtype(raw.dtype, np.complexfloating) or not np.issubdtype(raw.dtype, np.number):
            raise ValueError(f"{name} must be a finite numeric array")
        try:
            result = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite numeric array") from exc
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain only finite values")
        return result.copy()

    @staticmethod
    def _bool_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
        try:
            result = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a boolean array") from exc
        if result.shape != shape or result.dtype != np.dtype(bool):
            raise ValueError(f"{name} must be a boolean array with shape {shape}")
        return result.copy()

    def _validate_update(
        self,
        *,
        rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        info: dict,
        actions: np.ndarray,
        active_mask: np.ndarray | None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[str, ...],
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        bool,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        shape = (self.n_envs,)
        reward_values = self._finite_array(rewards, shape, "rewards")
        terminated_values = self._bool_array(terminated, shape, "terminated")
        truncated_values = self._bool_array(truncated, shape, "truncated")
        action_values = self._finite_array(actions, (self.n_envs, 2), "actions")
        if active_mask is None:
            active_values = np.ones(self.n_envs, dtype=bool)
        else:
            active_values = self._bool_array(active_mask, shape, "active_mask")

        if not isinstance(info, dict):
            raise ValueError("info must be a dict")
        track_ids = info.get("track_id")
        if not isinstance(track_ids, tuple) or len(track_ids) != self.n_envs or any(
            not isinstance(track_id, str) for track_id in track_ids
        ):
            raise ValueError("info['track_id'] must be a tuple of n_envs strings")

        signal_data = info.get("signals")
        if not isinstance(signal_data, dict) or set(signal_data) != set(SIGNAL_NAMES):
            raise ValueError("info['signals'] must contain exactly all reward signals")
        signal_values = {
            name: self._finite_array(signal_data[name], shape, f"signals[{name!r}]")
            for name in SIGNAL_NAMES
        }

        state = info.get("state")
        if not isinstance(state, dict) or "speed" not in state:
            raise ValueError("info['state'] must contain speed")
        speed_values = self._finite_array(state["speed"], shape, "state['speed']")
        off_track_values = self._bool_array(info.get("off_track"), shape, "info['off_track']")
        crash_values = self._bool_array(info.get("crash"), shape, "info['crash']")
        lap_values = self._bool_array(info.get("lap_complete"), shape, "info['lap_complete']")
        boundary_excess_values = self._optional_nonnegative_array(
            info.get("boundary_excess_m"), shape, "info['boundary_excess_m']"
        )
        off_track_progress_values = self._optional_nonnegative_array(
            info.get("off_track_progress_m"), shape, "info['off_track_progress_m']"
        )
        coverage_present = "sector_coverage_count" in info
        coverage_values = (
            self._optional_nonnegative_integer_array(
                info.get("sector_coverage_count"), shape, "info['sector_coverage_count']"
            )
            if coverage_present
            else np.zeros(self.n_envs, dtype=np.float64)
        )
        return (
            reward_values,
            terminated_values,
            truncated_values,
            track_ids,
            signal_values,
            speed_values,
            off_track_values,
            crash_values,
            lap_values,
            active_values,
            action_values,
            boundary_excess_values,
            off_track_progress_values,
            coverage_present,
            coverage_values,
        )

    def _optional_nonnegative_array(
        self, value: Any, shape: tuple[int, ...], name: str
    ) -> np.ndarray:
        if value is None:
            return np.zeros(shape, dtype=np.float64)
        result = self._finite_array(value, shape, name)
        if np.any(result < 0.0):
            raise ValueError(f"{name} must contain only non-negative values")
        return result

    def _optional_nonnegative_integer_array(
        self, value: Any, shape: tuple[int, ...], name: str
    ) -> np.ndarray:
        result = self._optional_nonnegative_array(value, shape, name)
        if np.any(result != np.floor(result)):
            raise ValueError(f"{name} must contain only integer values")
        return result

    def update(
        self,
        *,
        rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        info: dict,
        actions: np.ndarray,
        active_mask: np.ndarray | None = None,
    ) -> None:
        """Consume one full batch, counting only lanes selected by ``active_mask``."""
        (
            reward_values,
            terminated_values,
            truncated_values,
            track_ids,
            signal_values,
            speed_values,
            off_track_values,
            crash_values,
            lap_values,
            active_values,
            action_values,
            boundary_excess_values,
            off_track_progress_values,
            coverage_present,
            coverage_values,
        ) = self._validate_update(
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info=info,
            actions=actions,
            active_mask=active_mask,
        )

        applied_actions = np.clip(action_values, -1.0, 1.0)
        active_indices = np.flatnonzero(active_values)
        for index in active_indices:
            lane = int(index)
            track_summary = self._per_track.setdefault(track_ids[lane], _Summary())
            previous_action = self._last_applied_actions[lane]
            action_delta = float(np.abs(applied_actions[lane] - previous_action).sum())
            action_saturated = int(np.count_nonzero(np.abs(action_values[lane]) >= 1.0))
            row_signals = {name: float(signal_values[name][lane]) for name in SIGNAL_NAMES}
            step_args = {
                "reward": float(reward_values[lane]),
                "signals": row_signals,
                "speed": float(speed_values[lane]),
                "off_track": bool(off_track_values[lane]),
                "crash": bool(crash_values[lane]),
                "lap_complete": bool(lap_values[lane]),
                "action_saturated": action_saturated,
                "action_delta": action_delta,
                "boundary_excess_m": float(boundary_excess_values[lane]),
                "off_track_progress_m": float(off_track_progress_values[lane]),
                "sector_coverage_count": int(coverage_values[lane]) if coverage_present else None,
            }
            self._summary.add_step(**step_args)
            track_summary.add_step(**step_args)
            if coverage_present:
                self._episode_sector_coverage_available[lane] = True

            episode_return = float(self._episode_returns[lane] + reward_values[lane])
            self._episode_returns[lane] = episode_return
            self._episode_lengths[lane] += 1
            done = bool(terminated_values[lane] or truncated_values[lane])
            if done:
                episode_args = {
                    "episode_return": episode_return,
                    "terminated": bool(terminated_values[lane]),
                    "truncated": bool(truncated_values[lane]),
                    "sector_coverage_count": (
                        int(coverage_values[lane])
                        if self._episode_sector_coverage_available[lane]
                        else None
                    ),
                }
                self._summary.add_episode(**episode_args)
                track_summary.add_episode(**episode_args)
                self._episode_returns[lane] = 0.0
                self._episode_lengths[lane] = 0
                self._episode_sector_coverage_available[lane] = False
                self._last_applied_actions[lane] = 0.0
            else:
                self._last_applied_actions[lane] = applied_actions[lane]

        # A collector may report an ended lane while masking it out of the
        # counters.  Its reset still must not inherit the old episode/action.
        masked_done = (~active_values) & (terminated_values | truncated_values)
        self._episode_returns[masked_done] = 0.0
        self._episode_lengths[masked_done] = 0
        self._episode_sector_coverage_available[masked_done] = False
        self._last_applied_actions[masked_done] = 0.0

    def _summary_dict(self, summary: _Summary) -> dict[str, Any]:
        steps = summary.env_steps
        episodes = summary.episodes
        weighted = {
            name: float(summary.signal_sums[name] * self.weights[name])
            for name in SIGNAL_NAMES
        }
        weighted_total = float(sum(weighted.values()))
        contribution_denominator = sum(abs(value) for value in weighted.values())
        return {
            "env_steps": int(steps),
            "episodes": int(episodes),
            "reward_sum": float(summary.reward_sum),
            "mean_episode_return": (
                float(summary.completed_return_sum / episodes) if episodes else None
            ),
            "lap_completions": int(summary.lap_completions),
            "crashes": int(summary.crashes),
            "terminated_episodes": int(summary.terminated_episodes),
            "truncated_episodes": int(summary.truncated_episodes),
            "lap_completion_rate": (
                float(summary.lap_completions / episodes) if episodes else None
            ),
            "crash_rate": float(summary.crashes / episodes) if episodes else None,
            "off_track_rate": (
                float(summary.off_track_steps / steps) if steps else None
            ),
            "positive_off_track_progress_m": float(summary.off_track_progress_m),
            "boundary_excess_m_sum": float(summary.boundary_excess_m_sum),
            "boundary_excess_step_rate": (
                float(summary.boundary_excess_steps / steps) if steps else None
            ),
            "mean_sector_coverage": (
                float(summary.sector_coverage_sum / summary.sector_coverage_episodes)
                if summary.sector_coverage_episodes
                else None
            ),
            "mean_sector_coverage_during_update": (
                float(summary.sector_coverage_step_sum / summary.sector_coverage_steps)
                if summary.sector_coverage_steps else None
            ),
            "mean_speed_mps": float(summary.speed_sum / steps) if steps else None,
            "action_saturation_rate": (
                float(summary.action_saturation_components / summary.action_components)
                if summary.action_components
                else None
            ),
            "mean_abs_action_delta": (
                float(summary.action_delta_sum / summary.action_components)
                if summary.action_components
                else None
            ),
            "signal_sums": {
                name: float(summary.signal_sums[name]) for name in SIGNAL_NAMES
            },
            "signal_means": {
                name: (float(summary.signal_sums[name] / steps) if steps else None)
                for name in SIGNAL_NAMES
            },
            "weighted_signal_sums": weighted,
            "signal_contributions": {
                name: (float(value / contribution_denominator) if contribution_denominator else 0.0)
                for name, value in weighted.items()
            },
            "weighted_signal_total": weighted_total,
        }

    def snapshot(self, *, wall_time_s: float) -> dict[str, Any]:
        """Return a JSON-safe global and per-track metrics snapshot."""
        if isinstance(wall_time_s, (bool, np.bool_)):
            raise ValueError("wall_time_s must be a non-negative finite number")
        try:
            wall_time = float(wall_time_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("wall_time_s must be a non-negative finite number") from exc
        if not np.isfinite(wall_time) or wall_time < 0.0:
            raise ValueError("wall_time_s must be a non-negative finite number")

        result = self._summary_dict(self._summary)
        result["per_track"] = {
            track_id: self._summary_dict(summary)
            for track_id, summary in self._per_track.items()
        }
        result["wall_time_s"] = wall_time
        result["env_steps_per_second"] = (
            float(self._summary.env_steps / wall_time) if wall_time > 0.0 else None
        )
        return result
