"""Train the reference actor to imitate the deterministic racing teacher.

This is a serious baseline improvement experiment, not an event submission
contract.  The teacher is the existing Python scripted controller, so every
sample still comes from the authoritative environment.  The resulting PPO
actor artifact can be evaluated with the same deterministic evaluator.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from racing_event.config import SimulationConfig
from racing_event.observation import TrainingContract
from racing_event.ppo import FixedMLPActorCritic, PPOHyperparameters
from racing_event.participant_model import build_policy_from_model
from racing_event.track import TrackGeometry
from racing_event.env import RacingEnv
from tools.run_scripted import scripted_action
import shutil


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-dir", type=Path, default=ROOT / "tracks/train")
    parser.add_argument("--config", type=Path, default=ROOT / "config/simulation.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--samples-per-track", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", type=Path, default=None, help="participant model.py to train via imitation")
    return parser.parse_args()


def collect_teacher_data(track_paths: list[Path], config: SimulationConfig, samples_per_track: int) -> tuple[np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for track_path in track_paths:
        track = TrackGeometry.from_file(track_path)
        env = RacingEnv(track, config)
        observation, _ = env.reset(seed=0, spawn_s=track.start_line_s, lateral_offset=0.0, initial_speed=0.0, heading_error=0.0)
        for _ in range(samples_per_track):
            action = scripted_action(env)
            observations.append(np.asarray(observation, dtype=np.float32))
            actions.append(np.asarray(action, dtype=np.float32))
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                observation, _ = env.reset(seed=0, spawn_s=track.start_line_s, lateral_offset=0.0, initial_speed=0.0, heading_error=0.0)
    if not observations:
        raise ValueError("teacher produced no samples")
    return np.stack(observations), np.stack(actions)


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.epochs <= 0 or args.samples_per_track <= 0:
        raise ValueError("epochs and samples-per-track must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = SimulationConfig.from_file(args.config)
    track_paths = sorted(path for path in args.track_dir.glob("*.json") if path.name != "manifest.json")
    if not track_paths:
        raise ValueError("no training tracks found")
    tracks = tuple(TrackGeometry.from_file(path) for path in track_paths)
    observations, actions = collect_teacher_data(track_paths, config, args.samples_per_track)
    if args.model:
        policy, _ = build_policy_from_model(args.model, observations.shape[1])
    else:
        policy = FixedMLPActorCritic(observations.shape[1])
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    inputs = torch.as_tensor(observations, dtype=torch.float32)
    target = torch.atanh(torch.as_tensor(actions, dtype=torch.float32).clamp(-0.995, 0.995))
    batch_size = min(256, len(observations))
    history: list[dict[str, float | int]] = []
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(inputs), generator=generator)
        losses: list[float] = []
        for start in range(0, len(inputs), batch_size):
            indices = order[start:start + batch_size]
            means, _ = policy(inputs[indices])
            loss = torch.nn.functional.mse_loss(means, target[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = TrainingContract.for_config(
        config,
        track_ids=(track.track_id for track in tracks),
        track_hashes=(track.content_hash for track in tracks),
    )
    hyperparameters = PPOHyperparameters(rollout_length=64)
    policy.save_artifact(output_dir / "policy.pt", output_dir / "policy.json", hyperparameters=hyperparameters)
    contract.save(output_dir / "training_contract.json")
    if args.model:
        shutil.copyfile(args.model, output_dir / "model_source.py")
    shutil.copyfile(args.config, output_dir / "run_config.json")
    report = {
        "architecture": "participant_model" if args.model else "fixed_mlp_actor_critic",
        "teacher": "tools.run_scripted.scripted_action",
        "tracks": [track.track_id for track in tracks],
        "samples": len(observations),
        "epochs": args.epochs,
        "seed": args.seed,
        "final_loss": history[-1]["loss"],
        "history": history,
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = train(args)
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
