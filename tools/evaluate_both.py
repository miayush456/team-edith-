"""Evaluate one saved model on every public development track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing_event.config import SimulationConfig
from racing_event.evaluation import evaluate_participant_artifact, evaluate_policy, validate_evaluation_contract
from racing_event.observation import TrainingContract
from racing_event.submission import load_submission
from racing_event.participant_model import build_policy_from_model, load_policy_state
from racing_event.track import TrackGeometry
from racing_event.vector_env import VectorizedRacingEnv

ROOT = Path(__file__).resolve().parents[1]


def evaluate_both(source: Path, config_path: Path, track_dir: Path, repeats: int) -> dict[str, object]:
    config = SimulationConfig.from_file(config_path)
    reference = all((source / name).is_file() for name in ("policy.pt", "policy.json", "training_contract.json"))
    artifact = None
    normalizer = None
    if reference:
        contract, normalizer = TrainingContract.load(source / "training_contract.json")
        metadata = json.loads((source / "policy.json").read_text(encoding="utf-8"))
        if metadata.get("format") == "racing_event_actor_critic":
            model_path = source / "model_source.py"
            if not model_path.is_file():
                raise ValueError("custom actor run is missing model_source.py")
            policy, _ = build_policy_from_model(model_path, contract.observation.size)
            load_policy_state(policy, source / "policy.pt")
        else:
            from racing_event.ppo import FixedMLPActorCritic
            policy, _ = FixedMLPActorCritic.load_artifact(source / "policy.pt", source / "policy.json")
        if normalizer is not None:
            normalizer.freeze()
    else:
        artifact = load_submission(source)
        contract = artifact.contract
    results = []
    for path in sorted(track_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        track = TrackGeometry.from_file(path)
        env = VectorizedRacingEnv(track, config, reward=None, n_envs=1)
        if reference:
            validate_evaluation_contract(env, contract, normalizer, require_track_match=False)
            reports = [evaluate_policy(env, policy.as_policy(deterministic=True), normalizer=normalizer).to_dict() for _ in range(repeats)]
        else:
            reports = [evaluate_participant_artifact(env, artifact).to_dict() for _ in range(repeats)]
        if len({report["trajectory_hash"] for report in reports}) != 1:
            raise RuntimeError(f"repeated evaluation differed on {track.track_id}")
        results.append({"track_id": track.track_id, "track_hash": track.content_hash, "repeats": reports})
    return {"source": str(source), "tracks": results, "deterministic": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/simulation.json")
    parser.add_argument("--track-dir", type=Path, default=ROOT / "tracks/train")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = evaluate_both(args.source, args.config, args.track_dir, args.repeats)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"both-map evaluation failed: {exc}\n")
    serialized = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
