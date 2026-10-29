"""Reward helpers and the participant-owned reward boundary.

The simulator publishes raw transition signals.  ``RewardConfig`` remains as
an optional compatibility adapter for the local reference baseline, but a
participant may supply any callable implementing ``RewardFunction`` to the
rollout collector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np


SIGNAL_NAMES = (
    "progress",
    "speed",
    "curvature_matched_speed",
    "off_track",
    "crash",
    "stuck",
    "lap_complete",
    "checkpoint",
    "time",
    "steer_smoothness",
)
REWARD_SCHEMA_VERSION = 1
REWARD_WEIGHT_LIMIT = 100.0


class RewardFunction(Protocol):
    """Participant-owned reward callback for one transition.

    Inputs are copies and the signal mapping contains plain Python floats, so
    the callback cannot mutate simulator state through its arguments.  The
    callback may use any model-independent reward logic; SYNAPSE only owns the
    raw signals and transition outcome.
    """

    def __call__(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        signals: Mapping[str, float],
        terminated: bool,
        truncated: bool,
    ) -> float: ...


def evaluate_reward_function(
    reward_function: RewardFunction,
    observations: np.ndarray,
    actions: np.ndarray,
    signals: Mapping[str, np.ndarray],
    terminated: np.ndarray,
    truncated: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    """Evaluate a participant reward callback over active vector lanes.

    The function is deliberately called lane-by-lane.  This keeps the public
    callback simple and permits arbitrary participant reward code while the
    collector still returns a bounded NumPy batch.
    """

    observation_values = np.asarray(observations, dtype=np.float32)
    action_values = np.asarray(actions, dtype=np.float64)
    terminated_values = np.asarray(terminated, dtype=bool)
    truncated_values = np.asarray(truncated, dtype=bool)
    active_values = np.asarray(active_mask, dtype=bool)
    if observation_values.ndim != 2 or action_values.ndim != 2:
        raise ValueError("observations and actions must be two-dimensional arrays")
    n_envs = observation_values.shape[0]
    if action_values.shape[0] != n_envs or terminated_values.shape != (n_envs,) or truncated_values.shape != (n_envs,) or active_values.shape != (n_envs,):
        raise ValueError("reward inputs must have matching vector-lane shapes")
    if set(signals) != set(SIGNAL_NAMES):
        raise ValueError("signals must contain exactly the published signal names")
    signal_values = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in signals.items()
    }
    if any(values.shape != (n_envs,) for values in signal_values.values()):
        raise ValueError("each signal must have one value per environment lane")

    rewards = np.zeros(n_envs, dtype=np.float64)
    for index in np.flatnonzero(active_values):
        lane = int(index)
        lane_signals = {name: float(values[lane]) for name, values in signal_values.items()}
        value = reward_function(
            observation_values[lane].copy(),
            action_values[lane].copy(),
            lane_signals,
            bool(terminated_values[lane]),
            bool(truncated_values[lane]),
        )
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("participant reward must return one finite scalar per transition") from exc
        if not np.isfinite(scalar):
            raise ValueError("participant reward must return one finite scalar per transition")
        rewards[lane] = scalar
    return rewards


@dataclass(frozen=True)
class RewardConfig:
    weights: Mapping[str, float] = field(default_factory=lambda: {
        "progress": 12.0,
        "speed": 0.5,
        "curvature_matched_speed": 0.25,
        "off_track": -1.5,
        "crash": -10.0,
        "stuck": -8.0,
        "lap_complete": 20.0,
        "checkpoint": 1.0,
        "time": -0.15,
        "steer_smoothness": -0.05,
    })

    def __post_init__(self) -> None:
        if not isinstance(self.weights, Mapping):
            raise ValueError("weights must be a mapping")
        unknown = set(self.weights) - set(SIGNAL_NAMES)
        if unknown:
            raise ValueError(f"unknown reward signals: {sorted(unknown)}")
        normalized = {}
        for name in SIGNAL_NAMES:
            if isinstance(self.weights.get(name, 0.0), bool):
                raise ValueError(f"weight for {name!r} must be numeric, not bool")
            try:
                value = float(self.weights.get(name, 0.0))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"weight for {name!r} must be finite") from exc
            if not np.isfinite(value):
                raise ValueError(f"weight for {name!r} must be finite")
            if abs(value) > REWARD_WEIGHT_LIMIT:
                raise ValueError(f"weight for {name!r} must be within +/-{REWARD_WEIGHT_LIMIT:g}")
            normalized[name] = value
        object.__setattr__(self, "weights", normalized)

    @classmethod
    def from_file(cls, path: str | Path) -> "RewardConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("reward config must be a JSON object")
        if isinstance(data, dict) and "schema_version" in data and data["schema_version"] != REWARD_SCHEMA_VERSION:
            raise ValueError(f"unsupported reward config schema: {data['schema_version']!r}")
        weights = data.get("weights", data)
        if not isinstance(weights, dict):
            raise ValueError("reward config must contain a weights object")
        return cls(weights=weights)

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": REWARD_SCHEMA_VERSION, "weights": dict(self.weights)}

    @property
    def content_hash(self) -> str:
        payload = self.to_dict()
        # Preserve run identities for pre-checkpoint files that implicitly
        # disabled this signal, while still writing the explicit zero weight.
        if payload["weights"]["checkpoint"] == 0.0:
            del payload["weights"]["checkpoint"]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    def calculate(self, signals: Mapping[str, float]) -> float:
        missing = set(SIGNAL_NAMES) - set(signals)
        if missing:
            raise ValueError(f"missing reward signals: {sorted(missing)}")
        result = float(sum(self.weights[name] * float(signals[name]) for name in SIGNAL_NAMES))
        if not np.isfinite(result):
            raise ValueError("weighted reward must be finite")
        return result
