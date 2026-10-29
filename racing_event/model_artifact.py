"""Safe, declarative NumPy participant model artifacts.

The artifact format deliberately supports a small sequential MLP rather than
loading participant Python, pickle, or arbitrary operator graphs. It is an
organizer-side evaluation format: participant reward code is never imported or
executed here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import numpy as np

from .observation import RunningObservationNormalizer, TrainingContract


ARTIFACT_FORMAT = "synapse_numpy_mlp"
ARTIFACT_FORMAT_VERSION = 1
_REQUIRED_FILES = {"manifest.json", "training_contract.json", "weights.npz"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_KEYS = {
    "format",
    "format_version",
    "environment_code_hash",
    "observation",
    "action",
    "preprocessing",
    "layers",
    "weights_sha256",
    "training_contract_sha256",
}
_OBSERVATION_KEYS = {"schema_version", "size", "dtype"}
_OBSERVATION_OPTIONAL_KEYS = {"ray_angles_deg", "ray_count"}
_ACTION_KEYS = {"schema_version", "size", "dtype", "bounds", "output_activation"}
_PREPROCESSING_KEYS = {"kind", "mean", "scale", "clip"}
_LAYER_KEYS = {"name", "weights", "bias", "activation"}
_ACTIVATIONS = {"identity", "relu", "tanh"}


@dataclass(frozen=True)
class ArtifactLimits:
    """Resource limits applied before model bytes become executable arrays."""

    max_layers: int = 8
    max_layer_width: int = 1024
    max_parameters: int = 1_000_000
    max_artifact_bytes: int = 64 * 1024 * 1024
    max_batch_size: int = 64

    def validate(self) -> None:
        for name in (
            "max_layers",
            "max_layer_width",
            "max_parameters",
            "max_artifact_bytes",
            "max_batch_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class _Layer:
    name: str
    weights: np.ndarray
    bias: np.ndarray
    activation: str


@dataclass(frozen=True)
class ParticipantModelArtifact:
    """A validated participant-owned model with deterministic NumPy inference."""

    manifest: dict[str, Any]
    contract: TrainingContract
    normalizer: RunningObservationNormalizer | None
    layers: tuple[_Layer, ...]
    artifact_sha256: str
    limits: ArtifactLimits
    _affine_mean: np.ndarray | None = None
    _affine_scale: np.ndarray | None = None
    _affine_clip: np.ndarray | None = None

    @property
    def observation_size(self) -> int:
        return self.contract.observation.size

    @property
    def action_size(self) -> int:
        return 2

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        values = np.asarray(observations, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.observation_size:
            raise ValueError(
                f"artifact policy expects shape (batch, {self.observation_size})"
            )
        if values.shape[0] > self.limits.max_batch_size:
            raise ValueError(
                f"artifact policy batch exceeds max_batch_size={self.limits.max_batch_size}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("artifact policy input must contain only finite values")

        if self.normalizer is not None:
            values = self.normalizer.transform(values, training=False)
        elif self._affine_mean is not None:
            assert self._affine_scale is not None and self._affine_clip is not None
            values = np.clip(
                (values - self._affine_mean) / self._affine_scale,
                -self._affine_clip,
                self._affine_clip,
            ).astype(np.float32, copy=False)

        for layer in self.layers:
            values = values @ layer.weights.T + layer.bias
            if layer.activation == "relu":
                values = np.maximum(values, 0.0)
            elif layer.activation == "tanh":
                values = np.tanh(values)
            values = np.asarray(values, dtype=np.float32)
            if not np.all(np.isfinite(values)):
                raise ValueError("artifact policy produced non-finite values")

        if values.shape[1] != self.action_size or np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("artifact policy output must be two finite actions in [-1, 1]")
        return values


def _reject_constant(value: str) -> None:
    raise ValueError(f"JSON does not allow non-finite constant {value}")


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"artifact JSON is invalid: {path.name}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"artifact JSON must be an object: {path.name}")
    return data


def _strict_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise ValueError(f"artifact.{name} has unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"artifact.{name} is missing fields: {sorted(missing)}")


def _required_keys(value: dict[str, Any], required: set[str], allowed: set[str], name: str) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise ValueError(f"artifact.{name} has unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"artifact.{name} is missing fields: {sorted(missing)}")


def _sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return array


def _validate_paths(root: Path, limits: ArtifactLimits) -> dict[str, Path]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("artifact path must be a real directory")
    entries = list(root.iterdir())
    names = {entry.name for entry in entries}
    if names != _REQUIRED_FILES:
        raise ValueError(
            f"artifact must contain exactly {sorted(_REQUIRED_FILES)}; got {sorted(names)}"
        )
    result: dict[str, Path] = {}
    total_bytes = 0
    for name in sorted(_REQUIRED_FILES):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"artifact file must be a regular non-symlink file: {name}")
        size = path.stat().st_size
        total_bytes += size
        result[name] = path
    if total_bytes > limits.max_artifact_bytes:
        raise ValueError("artifact.resource_limit: artifact is too large")
    return result


def _validate_preprocessing(
    preprocessing: dict[str, Any], observation_size: int, normalizer: RunningObservationNormalizer | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    unknown = set(preprocessing) - _PREPROCESSING_KEYS
    if unknown or "kind" not in preprocessing:
        raise ValueError(f"artifact.preprocessing has invalid fields: {sorted(unknown)}")
    kind = preprocessing["kind"]
    if kind == "identity":
        if any(key in preprocessing for key in ("mean", "scale", "clip")):
            raise ValueError("identity preprocessing cannot include affine parameters")
        if normalizer is not None:
            raise ValueError("identity preprocessing cannot include a contract normalizer")
        return None, None, None
    if kind == "contract_normalizer":
        if normalizer is None or normalizer.training:
            raise ValueError("contract_normalizer requires a frozen normalizer")
        if any(key in preprocessing for key in ("mean", "scale", "clip")):
            raise ValueError("contract_normalizer cannot include affine parameters")
        return None, None, None
    if kind != "affine":
        raise ValueError(f"unsupported artifact preprocessing kind: {kind!r}")
    if not {"mean", "scale", "clip"}.issubset(preprocessing):
        raise ValueError("affine preprocessing requires mean, scale, and clip")
    if normalizer is not None:
        raise ValueError("affine preprocessing cannot include a contract normalizer")
    mean = _require_finite_vector(preprocessing["mean"], observation_size, "preprocessing.mean")
    scale = _require_finite_vector(preprocessing["scale"], observation_size, "preprocessing.scale")
    clip = _require_finite_vector(preprocessing["clip"], observation_size, "preprocessing.clip")
    if np.any(scale <= 0.0) or np.any(clip <= 0.0):
        raise ValueError("affine preprocessing scale and clip must be positive")
    return mean, scale, clip


def _load_layers(
    manifest_layers: Any,
    weights_file: Path,
    observation_size: int,
    limits: ArtifactLimits,
) -> tuple[_Layer, ...]:
    if not isinstance(manifest_layers, list) or not manifest_layers:
        raise ValueError("artifact.layers must be a non-empty list")
    if len(manifest_layers) > limits.max_layers:
        raise ValueError("artifact.resource_limit: too many layers")

    try:
        archive = np.load(weights_file, allow_pickle=False)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("artifact.weights: weights.npz could not be loaded safely") from exc

    layers: list[_Layer] = []
    expected_keys: set[str] = set()
    input_size = observation_size
    parameter_count = 0
    with archive:
        for index, raw_layer in enumerate(manifest_layers):
            if not isinstance(raw_layer, dict):
                raise ValueError(f"artifact.layers[{index}] must be an object")
            _strict_keys(raw_layer, _LAYER_KEYS, f"layers[{index}]")
            name = raw_layer["name"]
            weights_name = raw_layer["weights"]
            bias_name = raw_layer["bias"]
            activation = raw_layer["activation"]
            if not all(isinstance(value, str) and value for value in (name, weights_name, bias_name)):
                raise ValueError(f"artifact.layers[{index}] names must be non-empty strings")
            if activation not in _ACTIVATIONS:
                raise ValueError(f"unsupported artifact activation: {activation!r}")
            if weights_name in expected_keys or bias_name in expected_keys:
                raise ValueError("artifact layer weight keys must be unique")
            expected_keys.update((weights_name, bias_name))
            try:
                weights = np.asarray(archive[weights_name])
                bias = np.asarray(archive[bias_name])
            except KeyError as exc:
                raise ValueError(f"artifact.weights: missing declared array {exc.args[0]!r}") from exc
            if weights.dtype != np.dtype("float32") or bias.dtype != np.dtype("float32"):
                raise ValueError("artifact.weights: arrays must have dtype float32")
            if weights.ndim != 2 or bias.ndim != 1:
                raise ValueError("artifact.weights: dense arrays have invalid rank")
            if weights.shape[1] != input_size or bias.shape != (weights.shape[0],):
                raise ValueError("artifact.weights: dense array shapes do not chain")
            output_size = int(weights.shape[0])
            if output_size <= 0 or output_size > limits.max_layer_width:
                raise ValueError("artifact.resource_limit: layer width is too large")
            if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(bias)):
                raise ValueError("artifact.weights: arrays must be finite")
            parameter_count += int(weights.size + bias.size)
            if parameter_count > limits.max_parameters:
                raise ValueError("artifact.resource_limit: too many model parameters")
            layers.append(_Layer(name, weights.copy(), bias.copy(), activation))
            input_size = output_size

        if set(archive.files) != expected_keys:
            raise ValueError("artifact.weights: archive keys do not match the manifest")

    if input_size != 2:
        raise ValueError("artifact model must end with two action outputs")
    if manifest_layers[-1]["activation"] != "tanh":
        raise ValueError("artifact final activation must be tanh")
    return tuple(layers)


def _artifact_digest(paths: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for name in sorted(paths):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(paths[name].read_bytes())
    return digest.hexdigest()


def load_participant_artifact(
    path: str | Path,
    *,
    limits: ArtifactLimits | None = None,
) -> ParticipantModelArtifact:
    """Validate and load a declarative artifact without executing participant code."""

    limits = limits or ArtifactLimits()
    limits.validate()
    root = Path(path).resolve()
    files = _validate_paths(root, limits)
    manifest = _load_json(files["manifest.json"])
    _strict_keys(manifest, _MANIFEST_KEYS, "manifest")
    if manifest["format"] != ARTIFACT_FORMAT or manifest["format_version"] != ARTIFACT_FORMAT_VERSION:
        raise ValueError("artifact.schema: unsupported model artifact format")
    _require_hash(manifest["environment_code_hash"], "manifest.environment_code_hash")
    weights_hash = _require_hash(manifest["weights_sha256"], "manifest.weights_sha256")
    contract_hash = _require_hash(manifest["training_contract_sha256"], "manifest.training_contract_sha256")
    if _sha256_bytes(files["weights.npz"]) != weights_hash:
        raise ValueError("artifact.hash: weights.npz hash mismatch")
    if _sha256_bytes(files["training_contract.json"]) != contract_hash:
        raise ValueError("artifact.hash: training_contract.json hash mismatch")

    observation = manifest["observation"]
    if not isinstance(observation, dict):
        raise ValueError("artifact.observation must be an object")
    _required_keys(observation, _OBSERVATION_KEYS, _OBSERVATION_KEYS | _OBSERVATION_OPTIONAL_KEYS, "observation")
    if observation["schema_version"] != "1.0" or observation["dtype"] != "float32":
        raise ValueError("artifact.contract: unsupported observation schema")
    if type(observation["size"]) is not int or observation["size"] <= 0:
        raise ValueError("artifact.observation.size must be a positive integer")

    action = manifest["action"]
    if not isinstance(action, dict):
        raise ValueError("artifact.action must be an object")
    _strict_keys(action, _ACTION_KEYS, "action")
    if action["schema_version"] != "1.0" or action["size"] != 2 or action["dtype"] != "float32":
        raise ValueError("artifact.contract: unsupported action schema")
    if action["bounds"] != [[-1.0, 1.0], [-1.0, 1.0]] or action["output_activation"] != "tanh":
        raise ValueError("artifact.contract: action bounds or output activation mismatch")

    contract, normalizer = TrainingContract.load(files["training_contract.json"])
    if contract.observation.size != observation["size"] or contract.observation.version != observation["schema_version"]:
        raise ValueError("artifact.contract: manifest and training contract disagree")
    if "ray_angles_deg" in observation and tuple(float(value) for value in observation["ray_angles_deg"]) != contract.observation.ray_angles_deg:
        raise ValueError("artifact.contract: manifest and training contract ray angles disagree")
    if "ray_count" in observation and observation["ray_count"] != len(contract.observation.ray_angles_deg):
        raise ValueError("artifact.contract: manifest ray count is inconsistent")
    if not contract.track_ids or len(contract.track_ids) != len(contract.track_hashes):
        raise ValueError("artifact.contract: training track IDs and hashes must be paired")
    if len(contract.track_hashes) != len(set(contract.track_hashes)):
        raise ValueError("artifact.contract: training track hashes must be unique")
    if normalizer is not None and normalizer.training:
        raise ValueError("artifact.contract: normalizer must be frozen")

    preprocessing = manifest["preprocessing"]
    if not isinstance(preprocessing, dict):
        raise ValueError("artifact.preprocessing must be an object")
    affine_mean, affine_scale, affine_clip = _validate_preprocessing(
        preprocessing, observation["size"], normalizer
    )
    layers = _load_layers(manifest["layers"], files["weights.npz"], observation["size"], limits)
    return ParticipantModelArtifact(
        manifest=manifest,
        contract=contract,
        normalizer=normalizer,
        layers=layers,
        artifact_sha256=_artifact_digest(files),
        limits=limits,
        _affine_mean=affine_mean,
        _affine_scale=affine_scale,
        _affine_clip=affine_clip,
    )


def save_layers(
    layers: list[tuple[np.ndarray, np.ndarray, str]] | tuple[tuple[np.ndarray, np.ndarray, str], ...],
    output: str | Path,
    contract: TrainingContract,
    *,
    preprocessing: dict[str, Any] | None = None,
    environment_code_hash: str | None = None,
) -> Path:
    """Write the safe declarative artifact for NumPy/sklearn-style models."""
    from .trajectory import engine_code_hash

    destination = Path(output).resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError("artifact output must be a new directory")
    if not isinstance(contract, TrainingContract):
        raise TypeError("contract must be a TrainingContract")
    if not layers:
        raise ValueError("layers must contain at least one dense layer")
    processing = preprocessing or {"kind": "identity"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        arrays: dict[str, np.ndarray] = {}
        manifest_layers: list[dict[str, str]] = []
        input_size = contract.observation.size
        for index, (raw_weights, raw_bias, activation) in enumerate(layers):
            weights = np.asarray(raw_weights, dtype=np.float32)
            bias = np.asarray(raw_bias, dtype=np.float32)
            if weights.ndim != 2 or bias.ndim != 1 or weights.shape[1] != input_size or bias.shape != (weights.shape[0],):
                raise ValueError(f"layer {index} has incompatible dense shapes")
            if activation not in _ACTIVATIONS:
                raise ValueError(f"unsupported artifact activation: {activation!r}")
            if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(bias)):
                raise ValueError("layer weights and biases must be finite")
            weights_name, bias_name = f"layer_{index}_weights", f"layer_{index}_bias"
            arrays[weights_name] = weights
            arrays[bias_name] = bias
            manifest_layers.append({"name": f"layer_{index}", "weights": weights_name, "bias": bias_name, "activation": activation})
            input_size = weights.shape[0]
        if input_size != 2 or manifest_layers[-1]["activation"] != "tanh":
            raise ValueError("artifact layers must end in two tanh actions")
        np.savez(temporary / "weights.npz", **arrays)
        contract.save(temporary / "training_contract.json")
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "environment_code_hash": environment_code_hash or engine_code_hash(),
            "observation": {
                "schema_version": contract.observation.version,
                "size": contract.observation.size,
                "dtype": contract.observation.dtype,
                "ray_angles_deg": list(contract.observation.ray_angles_deg),
                "ray_count": len(contract.observation.ray_angles_deg),
            },
            "action": {"schema_version": "1.0", "size": 2, "dtype": "float32", "bounds": [[-1.0, 1.0], [-1.0, 1.0]], "output_activation": "tanh"},
            "preprocessing": processing,
            "layers": manifest_layers,
            "weights_sha256": _sha256_bytes(temporary / "weights.npz"),
            "training_contract_sha256": _sha256_bytes(temporary / "training_contract.json"),
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        load_participant_artifact(temporary)
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


__all__ = ["ARTIFACT_FORMAT", "ARTIFACT_FORMAT_VERSION", "ArtifactLimits", "ParticipantModelArtifact", "load_participant_artifact", "save_layers"]
