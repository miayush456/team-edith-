"""Budgeted rollout collection for the SYNAPSE environment boundary.

This module owns environment transitions and metrics.  The optional reference
PPO optimizer lives in :mod:`racing_event.ppo`, which consumes the immutable
batch produced here; participant algorithms remain outside this package.
The separation keeps budget accounting independent from model updates:
policy observations, exact environment-step accounting, selective lane
stepping/resetting, saved observation normalization, and training metrics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral
import time
from typing import Any

import numpy as np

from .budget import RunBudget, StepLimitExceeded
from .metrics import RolloutMetrics
from .observation import RunningObservationNormalizer
from .reward import RewardFunction, SIGNAL_NAMES, evaluate_reward_function
from .vector_env import VectorizedRacingEnv


@dataclass(frozen=True)
class PPOHyperparameters:
    """Explicit provisional PPO settings for the next trainer slice.

    These values are a trainer contract, not a measured event decision.  E2
    must validate them before they appear in a public participant release.
    """

    rollout_length: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    update_epochs: int = 10
    minibatch_size: int = 64
    learning_rate: float = 3.0e-4
    entropy_coefficient: float = 0.01
    value_loss_coefficient: float = 0.5
    hidden_sizes: tuple[int, ...] = (64, 64)
    max_grad_norm: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.rollout_length, bool) or not isinstance(self.rollout_length, Integral) or self.rollout_length <= 0:
            raise ValueError("rollout_length must be a positive integer")
        if isinstance(self.update_epochs, bool) or not isinstance(self.update_epochs, Integral) or self.update_epochs <= 0:
            raise ValueError("update_epochs must be a positive integer")
        if isinstance(self.minibatch_size, bool) or not isinstance(self.minibatch_size, Integral) or self.minibatch_size <= 0:
            raise ValueError("minibatch_size must be a positive integer")
        if not self.hidden_sizes or any(
            isinstance(size, bool) or not isinstance(size, Integral) or size <= 0
            for size in self.hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain positive integers")
        for name in (
            "gamma",
            "gae_lambda",
            "clip_range",
            "learning_rate",
            "entropy_coefficient",
            "value_loss_coefficient",
            "max_grad_norm",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.gamma > 1.0 or self.gae_lambda > 1.0 or self.clip_range <= 0.0:
            raise ValueError("gamma and gae_lambda must be at most 1, and clip_range must be positive")
        if self.learning_rate <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("learning_rate and max_grad_norm must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_length": int(self.rollout_length),
            "gamma": float(self.gamma),
            "gae_lambda": float(self.gae_lambda),
            "clip_range": float(self.clip_range),
            "update_epochs": int(self.update_epochs),
            "minibatch_size": int(self.minibatch_size),
            "learning_rate": float(self.learning_rate),
            "entropy_coefficient": float(self.entropy_coefficient),
            "value_loss_coefficient": float(self.value_loss_coefficient),
            "hidden_sizes": [int(size) for size in self.hidden_sizes],
            "max_grad_norm": float(self.max_grad_norm),
        }


@dataclass(frozen=True)
class RolloutBatch:
    """A full-lane rollout batch with an explicit transition mask.

    The first dimension is the number of calls to ``env.step``.  The second
    dimension is the fixed environment batch.  Consumers must filter by
    ``active_mask`` before constructing PPO minibatches.  Keeping inactive
    lanes in the arrays preserves lane alignment and makes partial final
    budget reservations auditable.
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    active_mask: np.ndarray
    signals: dict[str, np.ndarray]
    track_ids: tuple[tuple[str, ...], ...]

    @property
    def transitions(self) -> int:
        return int(np.count_nonzero(self.active_mask))


@dataclass(frozen=True)
class RolloutResult:
    """Collected transitions, metrics, and the end-of-rollout state."""

    batch: RolloutBatch
    metrics: dict[str, Any]
    wall_time_s: float
    training_position: list[dict[str, Any]] | None = None


Policy = Callable[[np.ndarray], np.ndarray]


class RolloutCollector:
    """Collect exact environment transitions from a vectorized racing env.

    ``policy`` receives a float32 array shaped ``(n_envs, observation_size)``
    and returns raw action commands shaped ``(n_envs, 2)``.  The collector
    accepts out-of-range finite actions so saturation can be logged; the
    environment applies the authoritative ``[-1, 1]`` clipping. A participant
    reward callback receives the clipped action that was actually applied.

    A ``RunBudget`` reservation is made immediately before every environment
    step.  The reservation counts active lanes, not calls to ``step``.  If the
    requested collection ends in the middle of a vector step, only the needed
    lanes are active and the remaining lanes are not advanced.

    ``max_batch_steps`` bounds the retained batch memory. A failed participant
    reward callback consumes the already-reserved transition and permanently
    invalidates this collector; create a fresh collector rather than retrying
    with potentially stale state.
    """

    def __init__(
        self,
        env: VectorizedRacingEnv,
        policy: Policy,
        budget: RunBudget,
        run_id: str,
        *,
        normalizer: RunningObservationNormalizer | None = None,
        reward_function: RewardFunction | None = None,
        reward_weights: Mapping[str, float] | None = None,
        max_batch_steps: int = 8_192,
        training: bool = True,
        seed: int | None = None,
    ) -> None:
        if not isinstance(env, VectorizedRacingEnv) and not getattr(env, "budget_managed", False):
            raise TypeError("env must be a VectorizedRacingEnv")
        if not callable(policy):
            raise TypeError("policy must be callable")
        if not isinstance(budget, RunBudget):
            raise TypeError("budget must be a RunBudget")
        if not isinstance(run_id, str):
            raise TypeError("run_id must be a string")
        if normalizer is not None and normalizer.size != env.observation_size:
            raise ValueError("normalizer size does not match environment observations")
        if isinstance(max_batch_steps, (bool, np.bool_)) or not isinstance(max_batch_steps, (int, np.integer)) or int(max_batch_steps) <= 0:
            raise ValueError("max_batch_steps must be a positive integer")
        self.env = env
        self.policy = policy
        self.budget = budget
        self.run_id = run_id
        self.normalizer = normalizer
        self.reward_function = reward_function
        if reward_weights is None and env.reward_config is not None:
            reward_weights = env.reward_config.weights
        self.reward_weights = {
            name: float((reward_weights or {}).get(name, 0.0)) for name in SIGNAL_NAMES
        }
        self.max_batch_steps = int(max_batch_steps)
        self.training = bool(training)
        self.seed = seed
        self._budget_managed = bool(getattr(env, "budget_managed", False))
        self._raw_observation: np.ndarray | None = None
        self._invalid_reason: str | None = None

    @property
    def observation_size(self) -> int:
        return self.env.observation_size

    def _normalize(self, observations: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
        values = np.asarray(observations, dtype=np.float32)
        if self.normalizer is None:
            return values.copy()
        if self.training and np.any(active_mask):
            self.normalizer.update(values[active_mask])
        return self.normalizer.transform(values, training=False)

    def _validate_policy_action(self, action: Any) -> np.ndarray:
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (self.env.n_envs, self.env.action_size):
            raise ValueError(
                f"policy must return shape ({self.env.n_envs}, {self.env.action_size})"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("policy actions must contain only finite values")
        return values

    def _initial_observation(self) -> np.ndarray:
        if self._raw_observation is None:
            raw_observation, _ = self.env.reset(seed=self.seed)
            self._raw_observation = raw_observation
        done = self.env.done
        if np.any(done):
            indices = np.flatnonzero(done).tolist()
            raw_observation, _ = self.env.reset(indices=indices)
            self._raw_observation = raw_observation
        ready_mask = ~self.env.done
        return self._normalize(self._raw_observation, ready_mask)

    def _empty_result(self, wall_time_s: float) -> RolloutResult:
        batch = RolloutBatch(
            observations=np.empty((0, self.env.n_envs, self.observation_size), dtype=np.float32),
            next_observations=np.empty((0, self.env.n_envs, self.observation_size), dtype=np.float32),
            actions=np.empty((0, self.env.n_envs, self.env.action_size), dtype=np.float64),
            rewards=np.empty((0, self.env.n_envs), dtype=np.float64),
            terminated=np.empty((0, self.env.n_envs), dtype=bool),
            truncated=np.empty((0, self.env.n_envs), dtype=bool),
            active_mask=np.empty((0, self.env.n_envs), dtype=bool),
            signals={name: np.empty((0, self.env.n_envs), dtype=np.float64) for name in SIGNAL_NAMES},
            track_ids=(),
        )
        return RolloutResult(batch=batch, metrics=RolloutMetrics(self.env.n_envs, self.env.config.dt, self.reward_weights).snapshot(wall_time_s=wall_time_s), wall_time_s=wall_time_s, training_position=[])

    def collect(self, steps: int, *, metrics: RolloutMetrics | None = None) -> RolloutResult:
        """Collect exactly ``steps`` transitions or raise before stepping.

        ``steps`` is a count of individual environment transitions.  It may
        be smaller than ``n_envs``; the final call then uses a partial active
        mask.  The budget is not changed if ``steps`` exceeds its remainder.
        """
        if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or int(steps) <= 0:
            raise ValueError("steps must be a positive integer")
        steps = int(steps)
        if self._invalid_reason is not None:
            raise RuntimeError(
                "this rollout collector cannot be reused after a participant "
                f"reward failure: {self._invalid_reason}; create a fresh collector"
            )
        if steps > self.max_batch_steps:
            raise ValueError(
                f"steps exceeds max_batch_steps={self.max_batch_steps}; "
                "collect smaller bounded batches to keep memory predictable"
            )
        remaining = self.budget.remaining(self.run_id)
        if steps > remaining:
            raise StepLimitExceeded(
                f"requested {steps} transitions exceeds {remaining} remaining steps for {self.run_id}"
            )
        if metrics is None:
            metrics = RolloutMetrics(self.env.n_envs, self.env.config.dt, self.reward_weights)
        elif metrics.n_envs != self.env.n_envs:
            raise ValueError("metrics n_envs does not match environment")

        observation = self._initial_observation()
        observations: list[np.ndarray] = []
        next_observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        terminated: list[np.ndarray] = []
        truncated: list[np.ndarray] = []
        active_masks: list[np.ndarray] = []
        signal_history: dict[str, list[np.ndarray]] = {
            name: [] for name in SIGNAL_NAMES
        }
        track_history: list[tuple[str, ...]] = []
        latest_positions: dict[int, dict[str, Any]] = {}
        collected = 0
        started = time.perf_counter()

        while collected < steps:
            available = ~self.env.done
            if not np.any(available):
                self._raw_observation = None
                observation = self._initial_observation()
                available = ~self.env.done
            count = min(int(np.count_nonzero(available)), steps - collected)
            active_indices = np.flatnonzero(available)[:count]
            active_mask = np.zeros(self.env.n_envs, dtype=bool)
            active_mask[active_indices] = True

            raw_observation = self._raw_observation.copy() if self._raw_observation is not None else observation.copy()
            raw_action = self._validate_policy_action(self.policy(observation.copy()))
            applied_action = np.clip(raw_action, -1.0, 1.0)
            if not self._budget_managed:
                self.budget.consume(self.run_id, count)
            if self.reward_function is None:
                next_raw, reward, term, trunc, info = self.env.step(raw_action, active_mask=active_mask)
            else:
                next_raw, term, trunc, info = self.env.step_raw(raw_action, active_mask=active_mask)
                try:
                    reward = evaluate_reward_function(
                        self.reward_function,
                        raw_observation,
                        applied_action,
                        info["signals"],
                        term,
                        trunc,
                        active_mask,
                    )
                except Exception as exc:
                    # The reservation and environment transition are already
                    # committed. Do not allow a caller to retry with a stale
                    # observation after participant code failed.
                    self._raw_observation = next_raw
                    self._invalid_reason = str(exc)
                    raise
            next_observation = self._normalize(next_raw, active_mask)
            metrics.update(
                rewards=reward,
                terminated=term,
                truncated=trunc,
                info=info,
                actions=raw_action,
                active_mask=active_mask,
            )

            state = info["state"]
            for lane in active_indices:
                lane = int(lane)
                track = self.env.tracks[int(self.env.track_indices[lane])]
                reason = None
                if term[lane]:
                    if info["crash"][lane]:
                        reason = "barrier contact"
                    elif info["lap_complete"][lane]:
                        reason = "lap completed"
                    else:
                        reason = "backward limit"
                elif trunc[lane]:
                    reason = "stuck" if info["signals"]["stuck"][lane] else "episode time limit"
                latest_positions[lane] = {
                    "lane": lane,
                    "track_id": info["track_id"][lane],
                    "x": float(state["x"][lane]),
                    "y": float(state["y"][lane]),
                    "heading": float(state["heading"][lane]),
                    "progress_fraction": float(info["distance_since_spawn"][lane] / track.length),
                    "zones_reached": int(info["sector_coverage_count"][lane]),
                    "speed_mps": float(state["speed"][lane]),
                    "episode_time_s": float(info["step"][lane]) * self.env.config.dt,
                    "off_track": bool(info["off_track"][lane]),
                    "end_reason": reason,
                }

            observations.append(observation.copy())
            next_observations.append(next_observation.copy())
            actions.append(raw_action.copy())
            rewards.append(np.asarray(reward, dtype=np.float64).copy())
            terminated.append(np.asarray(term, dtype=bool).copy())
            truncated.append(np.asarray(trunc, dtype=bool).copy())
            active_masks.append(active_mask.copy())
            track_history.append(tuple(info["track_id"]))
            for name in signal_history:
                signal_history[name].append(np.asarray(info["signals"][name], dtype=np.float64).copy())

            collected += count
            self._raw_observation = next_raw
            done_indices = np.flatnonzero((term | trunc) & active_mask).tolist()
            if collected < steps and done_indices:
                self._raw_observation, _ = self.env.reset(indices=done_indices)
                reset_mask = np.zeros(self.env.n_envs, dtype=bool)
                reset_mask[done_indices] = True
                observation = self._normalize(self._raw_observation, reset_mask)
            else:
                observation = next_observation

        wall_time_s = time.perf_counter() - started
        batch = RolloutBatch(
            observations=np.stack(observations).astype(np.float32, copy=False),
            next_observations=np.stack(next_observations).astype(np.float32, copy=False),
            actions=np.stack(actions).astype(np.float64, copy=False),
            rewards=np.stack(rewards).astype(np.float64, copy=False),
            terminated=np.stack(terminated),
            truncated=np.stack(truncated),
            active_mask=np.stack(active_masks),
            signals={name: np.stack(values).astype(np.float64, copy=False) for name, values in signal_history.items()},
            track_ids=tuple(track_history),
        )
        return RolloutResult(
            batch=batch, metrics=metrics.snapshot(wall_time_s=wall_time_s),
            wall_time_s=wall_time_s,
            training_position=[latest_positions[lane] for lane in sorted(latest_positions)],
        )


__all__ = ["PPOHyperparameters", "RolloutBatch", "RolloutCollector", "RolloutResult"]
