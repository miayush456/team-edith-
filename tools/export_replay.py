"""Export a Python-authored scripted or saved-policy run for a renderer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from racing_event.config import SimulationConfig
from racing_event.evaluation import evaluate_participant_artifact, evaluate_policy, validate_evaluation_contract
from racing_event.submission import load_submission
from racing_event.observation import TrainingContract
from racing_event.track import TrackGeometry
from racing_event.trajectory import ReplayWriter, replay_header, replay_result
from racing_event.vector_env import VectorizedRacingEnv
from tools.run_scripted import scripted_action


def export(args: argparse.Namespace) -> dict:
    artifact_dir = getattr(args, "artifact_dir", None)
    sources = (args.config, args.track, args.weights, args.metadata, args.contract, artifact_dir)
    if any(p is not None and Path(p).resolve() == args.output.resolve() for p in sources):
        raise ValueError("replay output must not overwrite a source artifact")
    if artifact_dir is not None:
        artifact_root = Path(artifact_dir).resolve()
        try:
            args.output.resolve().relative_to(artifact_root)
        except ValueError:
            pass
        else:
            raise ValueError("replay output must stay outside the participant artifact directory")
        if any(value is not None for value in (args.weights, args.metadata, args.contract)):
            raise ValueError("--artifact-dir cannot be combined with reference PPO artifact options")
    config = SimulationConfig.from_file(args.config)
    track = TrackGeometry.from_file(args.track)
    env = (
        VectorizedRacingEnv(track, config, reward=None, n_envs=1)
        if artifact_dir is not None
        else VectorizedRacingEnv(track, config, n_envs=1)
    )
    normalizer = None
    mode = "scripted" if args.weights is None and artifact_dir is None else "evaluation"
    artifact = None
    if artifact_dir is not None:
        artifact = load_submission(artifact_dir)
    elif args.weights is None:
        if args.contract is not None or args.metadata is not None:
            raise ValueError("--contract and --metadata require --weights")

        def policy(_observations: np.ndarray) -> np.ndarray:
            # The smoke controller reads a scalar snapshot, never steps a
            # second simulator. All transitions still go through evaluation.
            snapshot = SimpleNamespace(track=track, config=config, state=env.state.item(0))
            return scripted_action(snapshot)[None, :]
    else:
        if args.contract is None:
            raise ValueError("--weights requires --contract with saved normalizer")
        contract, normalizer = TrainingContract.load(args.contract)
        if normalizer is not None:
            # Trusted reference checkpoints retain resumable training state.
            # Freeze only this loaded evaluation copy, never rewrite statistics.
            normalizer.freeze()
        # Evaluation may use one public or held-out track from a multi-track
        # training contract; preserve strict physics/observation checks.
        validate_evaluation_contract(env, contract, normalizer, require_track_match=False)
        metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8")) if args.metadata else {}
        if metadata.get("format") == "racing_event_actor_critic":
            from racing_event.participant_model import build_policy_from_model, load_policy_state

            model_path = getattr(args, "model", None)
            if model_path is None:
                candidate = Path(args.weights).parent / "model_source.py"
                if candidate.is_file():
                    model_path = candidate
            if model_path is None:
                raise ValueError("custom actor replay requires its saved model_source.py")
            model, _ = build_policy_from_model(model_path, env.observation_size)
            load_policy_state(model, args.weights)
        else:
            from racing_event.ppo import FixedMLPActorCritic

            model, _ = FixedMLPActorCritic.load_artifact(
                args.weights, args.metadata, expected_observation_size=env.observation_size,
            )
        policy = model.as_policy(deterministic=True)

    header = replay_header(track, config, mode=mode, participant_id=args.participant_id)
    with ReplayWriter(args.output, header) as writer:
        if artifact is not None:
            result = evaluate_participant_artifact(
                env, artifact, max_steps=args.steps, frame_sink=writer.append,
            )
        else:
            result = evaluate_policy(
                env, policy, normalizer=normalizer, max_steps=args.steps, frame_sink=writer.append,
            )
        writer.finish(replay_result(result, config.tick_hz))
    return result.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument("--scripted", action="store_true", help="smoke controller, not a trained policy")
    policy_group.add_argument("--weights", type=Path, help="saved reference-PPO state dictionary")
    policy_group.add_argument("--artifact-dir", type=Path, help="declarative participant model artifact directory")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--model", type=Path, help="optional custom actor model.py; auto-detected beside saved weights")
    parser.add_argument("--track", type=Path, default=Path("tracks/train/monaco.json"))
    parser.add_argument("--config", type=Path, default=Path("config/simulation.json"))
    parser.add_argument("--steps", type=int, default=None, help="optional preview horizon; never extends engine limits")
    parser.add_argument("--participant-id", default="local_driver")
    parser.add_argument("--output", type=Path, required=True, help="JSON output; parent directory must exist")
    args = parser.parse_args()
    try:
        result = export(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Replay export failed: {exc}\n")
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
