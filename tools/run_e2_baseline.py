"""Run the measured baseline PPO convergence experiment (E2).

This tool intentionally writes a small Markdown report instead of changing
public event settings.  The current repository contains one synthetic
training track, so the report labels its result as a local fixture baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import torch

from racing_event.budget import RunBudget
from racing_event.config import SimulationConfig
from racing_event.observation import RunningObservationNormalizer
from racing_event.ppo import FixedMLPActorCritic, PPOTrainer
from racing_event.reward import RewardConfig
from racing_event.rollout import PPOHyperparameters, RolloutCollector
from racing_event.track import TrackGeometry
from racing_event.vector_env import VectorizedRacingEnv


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--rollout-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="optional directory for the final policy, metadata, and frozen contract",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "e2_baseline_convergence.md",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> str:
    if args.updates <= 0 or args.n_envs <= 0 or args.rollout_length <= 0:
        raise ValueError("updates, n-envs, and rollout-length must be positive")
    torch.manual_seed(args.seed)
    config = SimulationConfig.from_file(ROOT / "config" / "simulation.json")
    track = TrackGeometry.from_file(ROOT / "tracks" / "train" / "monaco.json")
    reward = RewardConfig.from_file(ROOT / "examples" / "baseline_reward.json")
    env = VectorizedRacingEnv(track, config, reward, n_envs=args.n_envs)
    hyperparameters = PPOHyperparameters(
        rollout_length=args.rollout_length,
    )
    policy = FixedMLPActorCritic(env.observation_size)
    trainer = PPOTrainer(policy, hyperparameters)
    transitions_per_update = args.rollout_length * args.n_envs

    rows: list[dict[str, float | int | None]] = []
    with tempfile.TemporaryDirectory(prefix="racing-e2-") as directory:
        budget = RunBudget.create(
            Path(directory) / "budget.json",
            max_runs=1,
            max_steps_per_run=args.updates * transitions_per_update,
        )
        run_id = budget.start_run()
        normalizer = RunningObservationNormalizer(env.observation_size)
        collector = RolloutCollector(
            env,
            policy.as_policy(),
            budget,
            run_id,
            normalizer=normalizer,
            seed=args.seed,
        )
        for update in range(1, args.updates + 1):
            rollout = collector.collect(transitions_per_update)
            ppo_metrics = trainer.update(rollout.batch)
            rows.append(
                {
                    "update": update,
                    "steps": update * transitions_per_update,
                    "lap_completion_rate": rollout.metrics["lap_completion_rate"],
                    "mean_speed_mps": rollout.metrics["mean_speed_mps"],
                    "crash_rate": rollout.metrics["crash_rate"],
                    "off_track_rate": rollout.metrics["off_track_rate"],
                    "mean_episode_return": rollout.metrics["mean_episode_return"],
                    "policy_loss": ppo_metrics.policy_loss,
                    "value_loss": ppo_metrics.value_loss,
                    "approx_kl": ppo_metrics.approx_kl,
                    "clip_fraction": ppo_metrics.clip_fraction,
                    "explained_variance": ppo_metrics.explained_variance,
                    "environment_steps_per_second": rollout.metrics["env_steps_per_second"],
                    "update_steps_per_second": ppo_metrics.update_transitions_per_second,
                }
            )

        if args.artifact_dir is not None:
            args.artifact_dir.mkdir(parents=True, exist_ok=True)
            normalizer.freeze()
            policy.save_artifact(
                args.artifact_dir / "policy.pt",
                args.artifact_dir / "policy.json",
                hyperparameters=hyperparameters,
            )
            env.training_contract.save(
                args.artifact_dir / "training_contract.json",
                normalizer=normalizer,
            )

    lines = [
        "# E2 baseline convergence",
        "",
        "This is a measured local experiment, not a public step-budget claim.",
        "The current participant-facing repository has one synthetic training fixture,",
        "so this run does not establish multi-track generalization or final event settings.",
        "",
        "## Configuration",
        "",
        f"- Python package: `racing-event` local checkout",
        f"- Track: `monaco`",
        f"- Seed: `{args.seed}`",
        f"- Environments: `{args.n_envs}`",
        f"- Rollout length per environment: `{args.rollout_length}`",
        f"- Transitions per update: `{transitions_per_update}`",
        f"- PPO hyperparameters: `{hyperparameters.to_dict()}`",
        "- Reward: `examples/baseline_reward.json`",
        "- Observation normalizer: updated during training; not evaluated in this experiment",
        *(
            [
                f"- Saved artifact directory name: `{args.artifact_dir.name}`",
                "- The saved normalizer is frozen for deterministic evaluation.",
            ]
            if args.artifact_dir is not None
            else []
        ),
        "",
        "## Measurements",
        "",
        "| Update | Steps | Lap completion | Mean speed (m/s) | Crash rate | Off-track rate | Mean episode return | Policy loss | Value loss | Approx. KL | Clip fraction | Explained variance | Environment steps/s | Update transitions/s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def cell(name: str) -> str:
            value = row[name]
            return "n/a" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)

        lines.append(
            "| "
            + " | ".join(
                [
                    cell("update"),
                    cell("steps"),
                    cell("lap_completion_rate"),
                    cell("mean_speed_mps"),
                    cell("crash_rate"),
                    cell("off_track_rate"),
                    cell("mean_episode_return"),
                    cell("policy_loss"),
                    cell("value_loss"),
                    cell("approx_kl"),
                    cell("clip_fraction"),
                    cell("explained_variance"),
                    cell("environment_steps_per_second"),
                    cell("update_steps_per_second"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Use lap completion, speed, crash rate, and off-track rate together.",
            "Training reward alone is not an evaluation result. Repeat E2 after the",
            "training track set and final event decisions are frozen.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
