"""Run a bounded ONNX ghost race through the authoritative Python engine.

This module is orchestration only.  Each entrant gets an independent
fixed-start evaluation on the same explicit track.  There is no second
physics implementation and there is no car-to-car interaction: the resulting
recordings are comparable ghosts for the read-only renderers.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Sequence

import numpy as np

from .evaluation import evaluate_participant_artifact, validate_evaluation_contract
from .isolated_policy import IsolatedPolicy
from .observation import TrainingContract
from .onnx_artifact import FORMAT, RUNTIME_VERSION, MAX_JSON_BYTES
from .replay_group import create_replay_group
from .submission import load_submission
from .track import TrackGeometry
from .trajectory import ReplayWriter, engine_code_hash, replay_header, replay_result
from .vector_env import VectorizedRacingEnv


RACE_KIND = "synapse_onnx_race"
RACE_VERSION = "1.0"
MAX_ENTRANTS = 4
MAX_TRACKS = 32
MAX_REPEATS = 8
MAX_TOTAL_STEPS = 2_000_000
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class RaceEntrant:
    """One ONNX artifact and its display identity."""

    participant_id: str
    artifact_dir: Path
    label: str | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.participant_id, str) or not 1 <= len(self.participant_id) <= 128
                or not all(char.isprintable() for char in self.participant_id)):
            raise ValueError("participant_id must contain 1 to 128 printable characters")
        if not isinstance(self.artifact_dir, (str, Path)):
            raise TypeError("artifact_dir must be a path")
        label = self.label if self.label is not None else self.participant_id
        if (not isinstance(label, str) or not 1 <= len(label) <= 48
                or not all(char.isprintable() for char in label)):
            raise ValueError("entrant label must contain 1 to 48 printable characters")

    @property
    def display_label(self) -> str:
        return self.label if self.label is not None else self.participant_id


@dataclass(frozen=True)
class RaceReport:
    """Versioned machine-readable report for one ONNX ghost race."""

    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.data, allow_nan=False))


def _safe_name(value: str, fallback: str) -> str:
    name = _SAFE_NAME.sub("-", value).strip(".-")
    return name[:64] or fallback


def _require_onnx_manifest(path: Path) -> dict[str, Any]:
    """Check the declared format without parsing model bytes.

    The isolated path uses this preflight so a NumPy artifact is rejected
    before a worker is started, while model parsing remains inside the worker.
    """

    if path.is_symlink() or not path.is_dir():
        raise ValueError("ONNX entrant must be a regular artifact directory")
    manifest_path = path / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("ONNX entrant manifest must be a bounded regular file")
    try:
        from .model_artifact import _load_json
        manifest = _load_json(manifest_path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid ONNX entrant manifest: {path}") from exc
    if manifest.get("format") != FORMAT:
        raise ValueError("ONNX race accepts only synapse_onnx_policy artifacts")
    return manifest


def _validate_entrants(entrants: Sequence[RaceEntrant]) -> tuple[RaceEntrant, ...]:
    if not isinstance(entrants, Sequence) or isinstance(entrants, (str, bytes)):
        raise TypeError("entrants must be a sequence of RaceEntrant values")
    values = tuple(entrants)
    if not 1 <= len(values) <= MAX_ENTRANTS:
        raise ValueError(f"ONNX races support 1 to {MAX_ENTRANTS} entrants")
    if any(not isinstance(value, RaceEntrant) for value in values):
        raise TypeError("entrants must contain RaceEntrant values")
    ids = [value.participant_id for value in values]
    if len(set(ids)) != len(ids):
        raise ValueError("participant IDs must be unique")
    return values


def _validate_tracks(tracks: Sequence[TrackGeometry]) -> tuple[TrackGeometry, ...]:
    if not isinstance(tracks, Sequence) or isinstance(tracks, (str, bytes)):
        raise TypeError("tracks must be a sequence of TrackGeometry values")
    values = tuple(tracks)
    if not 1 <= len(values) <= MAX_TRACKS:
        raise ValueError(f"provide 1 to {MAX_TRACKS} explicit tracks")
    if any(not isinstance(value, TrackGeometry) for value in values):
        raise TypeError("tracks must contain TrackGeometry values")
    ids = [value.track_id for value in values]
    hashes = [value.content_hash for value in values]
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
        raise ValueError("race tracks must have unique IDs and content hashes")
    return values


def _validate_options(config: Any, repeats: int, max_steps: int | None) -> int:
    if type(repeats) is not int or not 1 <= repeats <= MAX_REPEATS:
        raise ValueError(f"repeats must be an integer from 1 to {MAX_REPEATS}")
    if max_steps is None:
        selected = int(config.episode.max_steps)
    elif isinstance(max_steps, (bool, np.bool_)) or not isinstance(max_steps, (int, np.integer)):
        raise ValueError("max_steps must be a positive integer")
    else:
        selected = int(max_steps)
    if selected <= 0 or selected > config.episode.max_steps:
        raise ValueError("max_steps must be no greater than the environment episode limit")
    return selected


def _paths_overlap(output_dir: Path, artifact_dir: Path) -> bool:
    output = output_dir.resolve()
    artifact = artifact_dir.resolve()
    return output == artifact or output.is_relative_to(artifact) or artifact.is_relative_to(output)


def _open_output_dir(path: Path, artifacts: Sequence[Path]) -> Path:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"race output directory already exists: {path}")
    if any(_paths_overlap(path, artifact) for artifact in artifacts):
        raise ValueError("race output directory must not overlap an entrant artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    return path


def _write_report(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _artifact_summary(entrant: RaceEntrant, artifact: Any) -> dict[str, Any]:
    manifest = artifact.manifest
    return {
        "participant_id": entrant.participant_id,
        "label": entrant.display_label,
        "artifact_sha256": artifact.artifact_sha256,
        "model_sha256": manifest["model_sha256"],
        "training_contract_sha256": manifest["training_contract_sha256"],
    }


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def run_onnx_race(
    entrants: Sequence[RaceEntrant],
    tracks: Sequence[TrackGeometry],
    config: Any,
    *,
    output_dir: str | Path | None = None,
    emit_replays: bool = False,
    max_steps: int | None = None,
    repeats: int = 2,
    isolate: bool = False,
    worker_image: str = "synapse-evaluator:v1",
    worker_memory_mb: int = 512,
    startup_timeout: float = 30,
    action_timeout: float = 2,
) -> RaceReport:
    """Evaluate ONNX entrants on identical fixed-start tracks.

    Entrants are evaluated sequentially and independently.  ``isolate=True``
    uses the existing model-only worker; the default is a trusted local
    preview for development.  Repeated evaluations must have identical
    trajectory hashes.  If ``emit_replays`` is true, compatible schema-1.1
    recordings and replay-group manifests are emitted below ``output_dir``.
    """

    entrants = _validate_entrants(entrants)
    tracks = _validate_tracks(tracks)
    max_steps = _validate_options(config, repeats, max_steps)
    total_steps = len(entrants) * len(tracks) * repeats * max_steps
    if total_steps > MAX_TOTAL_STEPS:
        raise ValueError("race exceeds the bounded total evaluation step limit")
    if emit_replays and output_dir is None:
        raise ValueError("emit_replays requires output_dir")

    output_root: Path | None = None
    if output_dir is not None:
        output_root = Path(output_dir)
        output_root = _open_output_dir(output_root, [Path(entrant.artifact_dir) for entrant in entrants])

    try:
        with ExitStack() as stack:
            loaded: list[tuple[RaceEntrant, Any]] = []
            seen_artifacts: set[str] = set()
            for entrant in entrants:
                source = Path(entrant.artifact_dir)
                _require_onnx_manifest(source)
                artifact = (
                    stack.enter_context(IsolatedPolicy(
                        source,
                        image=worker_image,
                        memory_mb=worker_memory_mb,
                        startup_timeout=startup_timeout,
                        action_timeout=action_timeout,
                    ))
                    if isolate else load_submission(source)
                )
                if artifact.manifest.get("format") != FORMAT:
                    raise ValueError("ONNX race accepts only synapse_onnx_policy artifacts")
                if artifact.artifact_sha256 in seen_artifacts:
                    raise ValueError("entrant artifacts must be unique")
                seen_artifacts.add(artifact.artifact_sha256)
                loaded.append((entrant, artifact))

            # Validate every contract and every track before the first model
            # transition. This prevents a later entrant from producing a
            # partial race after a contract mismatch is discovered.
            environments: list[VectorizedRacingEnv] = []
            for track in tracks:
                env = VectorizedRacingEnv(track, config, reward=None, n_envs=1)
                environments.append(env)
                for _, artifact in loaded:
                    if artifact.manifest["environment_code_hash"] != engine_code_hash():
                        raise ValueError("entrant environment code hash does not match evaluator")
                    validate_evaluation_contract(
                        env,
                        artifact.contract,
                        artifact.normalizer,
                        require_track_match=False,
                    )

            summaries = [_artifact_summary(entrant, artifact) for entrant, artifact in loaded]
            track_reports: list[dict[str, Any]] = []
            for track_index, (track, env) in enumerate(zip(tracks, environments)):
                results: list[dict[str, Any]] = []
                replay_paths: list[Path] = []
                replay_dir: Path | None = None
                if emit_replays:
                    assert output_root is not None
                    replay_dir = output_root / "tracks" / f"{track_index:02d}-{_safe_name(track.track_id, 'track')}"
                    replay_dir.mkdir(parents=True, exist_ok=True)

                for entrant_index, (entrant, artifact) in enumerate(loaded):
                    replay_path: Path | None = None
                    writer: ReplayWriter | None = None
                    if replay_dir is not None:
                        replay_path = replay_dir / f"{entrant_index:02d}-{_safe_name(entrant.participant_id, 'entrant')}.json"
                        writer = ReplayWriter(
                            replay_path,
                            replay_header(track, config, participant_id=entrant.participant_id, mode="evaluation"),
                        )
                    if writer is None:
                        first = evaluate_participant_artifact(env, artifact, max_steps=max_steps)
                        repeated_reports = [first.to_dict()]
                        for _ in range(1, repeats):
                            repeated = evaluate_participant_artifact(env, artifact, max_steps=max_steps)
                            if repeated.trajectory_hash != first.trajectory_hash:
                                raise RuntimeError(f"repeated evaluation differed for {entrant.participant_id} on {track.track_id}")
                            repeated_reports.append(repeated.to_dict())
                    else:
                        with writer:
                            first = evaluate_participant_artifact(
                                env, artifact, max_steps=max_steps, frame_sink=writer.append,
                            )
                            repeated_reports = [first.to_dict()]
                            for _ in range(1, repeats):
                                repeated = evaluate_participant_artifact(env, artifact, max_steps=max_steps)
                                if repeated.trajectory_hash != first.trajectory_hash:
                                    raise RuntimeError(f"repeated evaluation differed for {entrant.participant_id} on {track.track_id}")
                                repeated_reports.append(repeated.to_dict())
                            writer.finish(replay_result(first, config.tick_hz))
                        replay_paths.append(replay_path)
                    results.append({
                        "participant_id": entrant.participant_id,
                        "label": entrant.display_label,
                        "artifact_sha256": artifact.artifact_sha256,
                        "model_sha256": artifact.manifest["model_sha256"],
                        "training_contract_sha256": artifact.manifest["training_contract_sha256"],
                        "result": first.to_dict(),
                        "repeats": repeated_reports,
                        "replay": _relative(replay_path, output_root) if replay_path is not None and output_root else None,
                    })

                group_path: Path | None = None
                if replay_dir is not None:
                    group_path = replay_dir / "race.group.json"
                    create_replay_group(group_path, replay_paths, labels=[entrant.display_label for entrant, _ in loaded])
                track_reports.append({
                    "track_id": track.track_id,
                    "track_hash": track.content_hash,
                    "results": results,
                    "replay_group": _relative(group_path, output_root) if group_path is not None and output_root else None,
                })

            data = {
                "kind": RACE_KIND,
                "schema_version": RACE_VERSION,
                "engine_code_hash": engine_code_hash(),
                "config_hash": TrainingContract.config_hash_for(config),
                "tick_hz": config.tick_hz,
                "race_type": "independent_ghosts",
                "evaluation": {
                    "max_steps": max_steps,
                    "repeats": repeats,
                    "start": "fixed_start_line_standing_zero_noise",
                    "policy_action": "deterministic_mean",
                },
                "inference": {
                    "mode": "isolated_docker" if isolate else "trusted_local_preview",
                    "runtime": f"onnxruntime=={RUNTIME_VERSION}",
                    "profile": "static_vector_cpu_opset17_v1",
                    **({
                        "image": worker_image,
                        "memory_mb": worker_memory_mb,
                        "cpus": 1,
                        "startup_timeout_s": startup_timeout,
                        "action_timeout_s": action_timeout,
                    } if isolate else {}),
                },
                "entrants": summaries,
                "tracks": track_reports,
                "ranking_policy": None,
            }
            report = RaceReport(data)
            if output_root is not None:
                _write_report(output_root / "race.json", data)
            return report
    except BaseException:
        if output_root is not None:
            shutil.rmtree(output_root, ignore_errors=True)
        raise


__all__ = ["RaceEntrant", "RaceReport", "run_onnx_race"]
