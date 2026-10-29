"""Evaluate the optional saved reference PPO artifact with deterministic mean actions.

The evaluator uses one environment lane and streams a transition digest.  It
therefore reports reproducible outcomes without retaining a trajectory in
memory.  Run it twice with the same artifact to verify identical hashes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing_event.evaluation import evaluate_policy, validate_evaluation_contract
from racing_event.observation import TrainingContract
from racing_event.ppo import FixedMLPActorCritic
from racing_event.config import SimulationConfig
from racing_event.track import TrackGeometry
from racing_event.vector_env import VectorizedRacingEnv


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    policy, metadata = FixedMLPActorCritic.load_artifact(args.weights, args.metadata)
    contract, normalizer = TrainingContract.load(args.contract)
    config = SimulationConfig.from_file(args.config)
    track = TrackGeometry.from_file(args.track)
    env = VectorizedRacingEnv(track, config, n_envs=1)
    validate_evaluation_contract(env, contract, normalizer)
    if metadata["observation_size"] != env.observation_size:
        raise ValueError("model metadata does not match the environment observation size")

    results: list[dict[str, object]] = []
    expected_hash: str | None = None
    for _ in range(args.repeats):
        result = evaluate_policy(
            env,
            policy.as_policy(deterministic=True),
            normalizer=normalizer,
            max_steps=args.max_steps,
        )
        if expected_hash is None:
            expected_hash = result.trajectory_hash
        elif result.trajectory_hash != expected_hash:
            raise RuntimeError("repeated deterministic evaluation produced different trajectory hashes")
        results.append(result.to_dict())

    return {
        "contract": str(args.contract),
        "weights": str(args.weights),
        "metadata": str(args.metadata or args.weights.with_suffix(".json")),
        "deterministic": True,
        "repeats": results,
    }


def main() -> None:
    args = parse_args()
    report = evaluate(args)
    serialized = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
