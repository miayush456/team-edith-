"""Versioned observation and evaluation contract for the racing environment.

The environment emits physically normalized observations.  This module keeps
the field order and bounds explicit so a trainer can add a running normalizer
without making the public environment shape implicit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np

from .config import SimulationConfig


OBSERVATION_SCHEMA_VERSION = "1.0"
ACTION_SCHEMA_VERSION = "1.0"
TRAINING_CONTRACT_VERSION = "1.0"


def _optional_sha256(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest or null")
    return value


@dataclass(frozen=True)
class ObservationField:
    """One field in the fixed observation vector."""

    name: str
    low: float
    high: float
    unit: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "low": self.low,
            "high": self.high,
            "unit": self.unit,
            "description": self.description,
        }


@dataclass(frozen=True)
class ObservationSchema:
    """Names, order, bounds, and normalization policy for observations."""

    fields: tuple[ObservationField, ...]
    version: str = OBSERVATION_SCHEMA_VERSION
    dtype: str = "float32"
    normalization: str = "environment_static_bounds"
    ray_angles_deg: tuple[float, ...] = ()

    @property
    def size(self) -> int:
        return len(self.fields)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    @property
    def lows(self) -> np.ndarray:
        return np.asarray([field.low for field in self.fields], dtype=np.float32)

    @property
    def highs(self) -> np.ndarray:
        return np.asarray([field.high for field in self.fields], dtype=np.float32)

    def validate(self, observations: np.ndarray | Iterable[float]) -> np.ndarray:
        values = np.asarray(observations)
        if values.shape[-1:] != (self.size,):
            raise ValueError(f"observations must end with shape ({self.size},), got {values.shape}")
        if not np.issubdtype(values.dtype, np.floating):
            raise ValueError("observations must contain floating-point values")
        if not np.all(np.isfinite(values)):
            raise ValueError("observations must contain only finite values")
        lows = self.lows
        highs = self.highs
        if np.any(values < lows - 1.0e-6) or np.any(values > highs + 1.0e-6):
            raise ValueError("observations contain values outside the versioned bounds")
        return values

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "dtype": self.dtype,
            "normalization": self.normalization,
            "ray_angles_deg": list(self.ray_angles_deg),
            "ray_count": len(self.ray_angles_deg),
            "fields": [field.to_dict() for field in self.fields],
        }


def build_observation_schema(config: SimulationConfig) -> ObservationSchema:
    """Build the schema for a sensor configuration without inspecting state."""

    fields: list[ObservationField] = [
        ObservationField("speed", 0.0, 1.0, "normalized", "forward speed divided by maximum speed"),
        ObservationField(
            "longitudinal_velocity",
            -1.0,
            1.0,
            "normalized",
            "signed forward velocity divided by maximum speed; reverse is disabled in v1",
        ),
        ObservationField("lateral_velocity", -1.0, 1.0, "normalized", "car-frame lateral velocity divided by maximum speed"),
        ObservationField("yaw_rate", -1.0, 1.0, "normalized", "yaw rate clipped to five radians per second"),
        ObservationField("lateral_offset", -1.0, 1.0, "normalized", "signed offset divided by twice the local half-width"),
        ObservationField("heading_error", -1.0, 1.0, "normalized", "wrapped heading error divided by pi"),
        ObservationField("previous_steering", -1.0, 1.0, "normalized", "previous steering command"),
        ObservationField("previous_longitudinal", -1.0, 1.0, "normalized", "previous signed throttle/brake command"),
        ObservationField("steering_angle", -1.0, 1.0, "normalized", "current steering angle divided by its configured maximum"),
    ]
    fields.extend(
        ObservationField(
            f"ray_{index}",
            0.0,
            1.0,
            "normalized",
            f"distance to the track boundary for sensor ray {index}, divided by ray maximum range",
        )
        for index in range(len(config.sensors.ray_angles_deg))
    )
    fields.extend(
        ObservationField(
            f"curvature_lookahead_{index}",
            -1.0,
            1.0,
            "normalized",
            f"signed curvature feature at lookahead {distance:g} metres",
        )
        for index, distance in enumerate(config.sensors.lookahead_distances_m)
    )
    fields.append(
        ObservationField(
            "local_width",
            0.0,
            1.0,
            "normalized",
            "minimum local half-width divided by five times car width",
        )
    )
    return ObservationSchema(tuple(fields), ray_angles_deg=tuple(config.sensors.ray_angles_deg))


class RunningObservationNormalizer:
    """Streaming mean/variance normalizer whose state is safe to serialize.

    The environment's static bounds remain the public schema.  This optional
    trainer-side transform is deliberately separate and must be frozen by
    calling ``freeze`` (or by using ``training=False`` in ``transform``) for
    deterministic evaluation.
    """

    def __init__(self, size: int, *, epsilon: float = 1.0e-8, clip: float = 10.0):
        if int(size) <= 0:
            raise ValueError("normalizer size must be positive")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("normalizer epsilon must be positive and finite")
        if not np.isfinite(clip) or clip <= 0:
            raise ValueError("normalizer clip must be positive and finite")
        self.size = int(size)
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.count = 0
        self.mean = np.zeros(self.size, dtype=np.float64)
        self.m2 = np.zeros(self.size, dtype=np.float64)
        self.training = True

    @property
    def variance(self) -> np.ndarray:
        if self.count < 2:
            return np.ones(self.size, dtype=np.float64)
        return np.maximum(self.m2 / self.count, 0.0)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.variance + self.epsilon)

    def update(self, observations: np.ndarray | Iterable[float]) -> None:
        if not self.training:
            return
        values = np.asarray(observations, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.size:
            raise ValueError(f"observations must have shape (batch, {self.size})")
        if not np.all(np.isfinite(values)):
            raise ValueError("observations must contain only finite values")
        batch_count = values.shape[0]
        if batch_count == 0:
            return
        batch_mean = np.mean(values, axis=0)
        centered = values - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + delta * delta * (self.count * batch_count / total)
        self.mean += delta * (batch_count / total)
        self.count = total

    def transform(self, observations: np.ndarray | Iterable[float], *, training: bool | None = None) -> np.ndarray:
        values = np.asarray(observations, dtype=np.float64)
        if values.shape[-1:] != (self.size,):
            raise ValueError(f"observations must end with shape ({self.size},)")
        if not np.all(np.isfinite(values)):
            raise ValueError("observations must contain only finite values")
        if training is None:
            training = self.training
        if training:
            self.update(values)
        transformed = (values - self.mean) / self.std
        return np.clip(transformed, -self.clip, self.clip).astype(np.float32)

    def freeze(self) -> None:
        self.training = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "epsilon": self.epsilon,
            "clip": self.clip,
            "count": self.count,
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
            "training": self.training,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "RunningObservationNormalizer":
        normalizer = cls(int(state["size"]), epsilon=float(state["epsilon"]), clip=float(state["clip"]))
        normalizer.count = int(state["count"])
        normalizer.mean = np.asarray(state["mean"], dtype=np.float64)
        normalizer.m2 = np.asarray(state["m2"], dtype=np.float64)
        normalizer.training = bool(state.get("training", False))
        if normalizer.mean.shape != (normalizer.size,) or normalizer.m2.shape != (normalizer.size,):
            raise ValueError("normalizer statistics have the wrong shape")
        if normalizer.count < 0 or not np.all(np.isfinite(normalizer.mean)) or not np.all(np.isfinite(normalizer.m2)):
            raise ValueError("normalizer statistics are invalid")
        return normalizer

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RunningObservationNormalizer":
        return cls.from_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class TrainingContract:
    """Serializable model/environment boundary used by training and evaluation."""

    observation: ObservationSchema
    track_ids: tuple[str, ...] = ()
    track_hashes: tuple[str, ...] = ()
    config_hash: str = ""
    action_distribution: str = "participant_defined"
    evaluation_spawn: str = "fixed_start_line_standing_zero_noise"
    deterministic_evaluation: bool = True
    reward_config_hash: str | None = None
    reward_sha256: str | None = None
    version: str = TRAINING_CONTRACT_VERSION

    @staticmethod
    def config_hash_for(config: SimulationConfig) -> str:
        payload = {
            "tick_hz": config.tick_hz,
            "car": config.car.__dict__,
            "episode": config.episode.__dict__,
            "sensors": config.sensors.__dict__,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def for_config(
        cls,
        config: SimulationConfig,
        *,
        track_ids: Iterable[str] = (),
        track_hashes: Iterable[str] = (),
    ) -> "TrainingContract":
        ids = tuple(track_ids)
        hashes = tuple(track_hashes)
        if hashes and len(ids) != len(hashes):
            raise ValueError("track_ids and track_hashes must have the same length")
        return cls(
            observation=build_observation_schema(config),
            track_ids=ids,
            track_hashes=hashes,
            config_hash=cls.config_hash_for(config),
        )

    @property
    def action_names(self) -> tuple[str, str]:
        return ("steering", "throttle_brake")

    def to_dict(self, normalizer: RunningObservationNormalizer | None = None) -> dict[str, Any]:
        if normalizer is not None and normalizer.size != self.observation.size:
            raise ValueError("normalizer size does not match observation schema")
        return {
            "version": self.version,
            "action": {
                "version": ACTION_SCHEMA_VERSION,
                "names": list(self.action_names),
                "shape": [2],
                "dtype": "float32",
                "bounds": [[-1.0, 1.0], [-1.0, 1.0]],
                "semantics": ["steering", "positive throttle, zero coast, negative brake"],
                "finite_inputs_required": True,
            },
            "observation": self.observation.to_dict(),
            "tracks": {"ids": list(self.track_ids), "hashes": list(self.track_hashes)},
            "config_hash": self.config_hash,
            "training_provenance": {
                "reward_config_hash": self.reward_config_hash,
                "reward_sha256": self.reward_sha256,
            },
            "trainer": {
                "action_distribution": self.action_distribution,
                "normalizer": None if normalizer is None else normalizer.state_dict(),
            },
            "evaluation": {
                "deterministic": self.deterministic_evaluation,
                "spawn": self.evaluation_spawn,
                "policy_action": "mean",
                "normalizer_updates": False,
            },
        }

    def save(self, path: str | Path, *, normalizer: RunningObservationNormalizer | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(self.to_dict(normalizer), sort_keys=True, indent=2) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> tuple["TrainingContract", RunningObservationNormalizer | None]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != TRAINING_CONTRACT_VERSION:
            raise ValueError(f"unsupported training contract version: {data.get('version')!r}")
        observation_data = data.get("observation", {})
        fields = tuple(ObservationField(**field) for field in observation_data.get("fields", []))
        ray_angles = tuple(float(value) for value in observation_data.get("ray_angles_deg", ()))
        if "ray_count" in observation_data and observation_data["ray_count"] != len(ray_angles):
            raise ValueError("observation ray_count does not match ray_angles_deg")
        schema = ObservationSchema(
            fields=fields,
            version=observation_data.get("version", ""),
            dtype=observation_data.get("dtype", ""),
            normalization=observation_data.get("normalization", ""),
            ray_angles_deg=ray_angles,
        )
        if schema.version != OBSERVATION_SCHEMA_VERSION or schema.dtype != "float32":
            raise ValueError("unsupported observation schema")
        tracks = data.get("tracks", {})
        provenance = data.get("training_provenance", {})
        trainer = data.get("trainer", {})
        evaluation = data.get("evaluation", {})
        action = data.get("action", {})
        if action.get("version") != ACTION_SCHEMA_VERSION or action.get("shape") != [2]:
            raise ValueError("unsupported action schema")
        contract = cls(
            observation=schema,
            track_ids=tuple(tracks.get("ids", [])),
            track_hashes=tuple(tracks.get("hashes", [])),
            config_hash=str(data.get("config_hash", "")),
            action_distribution=str(trainer.get("action_distribution", "")),
            evaluation_spawn=str(evaluation.get("spawn", "")),
            deterministic_evaluation=bool(evaluation.get("deterministic", False)),
            reward_config_hash=_optional_sha256(provenance.get("reward_config_hash"), "training_provenance.reward_config_hash"),
            reward_sha256=_optional_sha256(provenance.get("reward_sha256"), "training_provenance.reward_sha256"),
            version=str(data["version"]),
        )
        if (contract.reward_config_hash is None) != (contract.reward_sha256 is None):
            raise ValueError("training provenance reward hashes must be both present or both null")
        normalizer_data = trainer.get("normalizer")
        normalizer = None if normalizer_data is None else RunningObservationNormalizer.from_state_dict(normalizer_data)
        if normalizer is not None and normalizer.size != schema.size:
            raise ValueError("saved normalizer does not match observation schema")
        return contract, normalizer
