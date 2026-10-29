"""Evaluate a declarative participant model artifact on organizer tracks."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing_event.config import SimulationConfig
from racing_event.evaluation import evaluate_participant_artifact
from racing_event.isolated_policy import IsolatedPolicy
from racing_event.submission import load_submission
from racing_event.track import TrackGeometry
from racing_event.trajectory import engine_code_hash
from racing_event.vector_env import VectorizedRacingEnv


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if not args.tracks:
        raise ValueError("at least one --track is required")
    isolated = getattr(args, "isolate", False)
    context = (
        IsolatedPolicy(
            args.artifact_dir, image=args.worker_image, memory_mb=args.worker_memory_mb,
            startup_timeout=args.startup_timeout, action_timeout=args.action_timeout,
        ) if isolated else nullcontext(load_submission(args.artifact_dir))
    )
    with context as artifact:
        return _evaluate(args, artifact, isolated=isolated)


def _evaluate(args: argparse.Namespace, artifact, *, isolated: bool) -> dict[str, object]:
    config = SimulationConfig.from_file(args.config)
    results: list[dict[str, object]] = []
    for track_path in args.tracks:
        track = TrackGeometry.from_file(track_path)
        env = VectorizedRacingEnv(track, config, reward=None, n_envs=1)
        repeated: list[dict[str, object]] = []
        expected_hash: str | None = None
        for _ in range(args.repeats):
            result = evaluate_participant_artifact(
                env,
                artifact,
                max_steps=args.max_steps,
            )
            if expected_hash is None:
                expected_hash = result.trajectory_hash
            elif result.trajectory_hash != expected_hash:
                raise RuntimeError(
                    f"repeated evaluation differed for track {track.track_id}"
                )
            repeated.append(result.to_dict())
        results.append({
            "track_id": track.track_id,
            "track_hash": track.content_hash,
            "repeats": repeated,
        })
    return {
        "artifact_sha256": artifact.artifact_sha256,
        "environment_code_hash": engine_code_hash(),
        "config_hash": artifact.contract.config_hash,
        "observation_size": artifact.observation_size,
        "ray_angles_deg": list(artifact.contract.observation.ray_angles_deg),
        "deterministic": True,
        "inference": ({
            "mode": "isolated_docker", "image": args.worker_image,
            "memory_mb": args.worker_memory_mb, "cpus": 1,
            "startup_timeout_s": args.startup_timeout, "action_timeout_s": args.action_timeout,
        } if isolated else {"mode": "trusted_local_preview"}),
        "tracks": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--track", dest="tracks", action="append", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--isolate", action="store_true", help="required for untrusted submissions; run a model-only Docker worker")
    parser.add_argument("--worker-image", default="synapse-evaluator:v1", help="already-built organizer-controlled image; prefer an immutable digest")
    parser.add_argument("--worker-memory-mb", type=int, default=512)
    parser.add_argument("--startup-timeout", type=float, default=30)
    parser.add_argument("--action-timeout", type=float, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = evaluate(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        raise SystemExit(f"participant evaluation failed: {exc}") from exc
    serialized = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
