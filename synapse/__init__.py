"""Small participant SDK facade for custom SYNAPSE algorithms."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import torch
from torch import nn

from racing_event.budget import RunBudget
from racing_event.budgeted_env import BudgetedEnv
from racing_event.config import SimulationConfig
from racing_event.observation import TrainingContract
from racing_event.ppo import ActorCritic, build_actor_critic
from racing_event.reward import RewardConfig
from racing_event.track import TrackGeometry
from racing_event.vector_env import VectorizedRacingEnv

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SYNAPSE participant SDK")
    parser.add_argument("--reward", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--hparams", type=Path, default=None)
    parser.add_argument("--track-dir", type=Path, default=Path("tracks/train"))
    parser.add_argument("--config", type=Path, default=Path("config/simulation.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--budget-steps", type=int, required=True)
    parser.add_argument("--n-envs", type=int, required=True)
    parser.add_argument("--rollout-length", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args(argv)


def make_env(args: argparse.Namespace) -> BudgetedEnv:
    config = SimulationConfig.from_file(args.config)
    track_paths = sorted(path for path in Path(args.track_dir).glob("*.json") if path.name != "manifest.json")
    if not track_paths:
        raise ValueError("track-dir contains no tracks")
    tracks = tuple(TrackGeometry.from_file(path) for path in track_paths)
    reward_path = getattr(args, "reward", None)
    reward = None if reward_path is None else RewardConfig.from_file(reward_path)
    env = VectorizedRacingEnv(tracks, config, reward, n_envs=args.n_envs)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    budget_path = output / "budget.json"
    budget = RunBudget.open(budget_path, max_runs=1, max_steps_per_run=args.budget_steps) if budget_path.exists() else RunBudget.create(budget_path, max_runs=1, max_steps_per_run=args.budget_steps)
    status = budget.status()
    run_id = status["runs"][0]["run_id"] if status["runs"] else budget.start_run()
    args._synapse_env = env
    args._synapse_budget = budget
    args._synapse_run_id = run_id
    args._synapse_reward_path = None if reward_path is None else Path(reward_path).resolve()
    return BudgetedEnv(env, budget, run_id)


def save(actor: ActorCritic | nn.Module, output_dir: str | Path, args: argparse.Namespace | None = None) -> Path:
    """Export a trained actor into the same validated ONNX flow as the UI."""
    requested_output = Path(output_dir).expanduser()
    if requested_output.is_symlink():
        raise ValueError("SDK output path must not be a symlink")
    output = requested_output.resolve()
    if output.exists() and not output.is_dir():
        raise ValueError("SDK output path must be a regular directory")
    if output.exists():
        # make_env owns this file in the run directory. It is expected to be
        # present when save(actor, args.output_dir) is called after training.
        sdk_files = {"budget.json", "budget.json.lock"}
        unexpected = [entry.name for entry in output.iterdir() if entry.name not in sdk_files]
        if unexpected:
            raise ValueError(
                "SDK output directory may contain only make_env's budget files; "
                "choose a new output directory"
            )
    output.mkdir(parents=True, exist_ok=True)
    env = getattr(args, "_synapse_env", None) if args is not None else None
    if env is None:
        raise ValueError("call synapse.make_env(args) before synapse.save")
    base_env = getattr(env, "env", env)
    policy = actor if isinstance(actor, ActorCritic) else build_actor_critic(actor, base_env.observation_size)
    contract = base_env.training_contract
    if base_env.reward_config is not None:
        reward_path = getattr(args, "_synapse_reward_path", None)
        if reward_path is None or not reward_path.is_file():
            raise ValueError("the reward config used by make_env is unavailable for artifact provenance")
        contract = replace(
            contract,
            reward_config_hash=base_env.reward_config.content_hash,
            reward_sha256=hashlib.sha256(reward_path.read_bytes()).hexdigest(),
        )
    from tools.export_onnx import export_torch_policy
    submission = output / "submission"
    export_torch_policy(policy, contract, None, submission, source_kind="synapse")
    shutil.copyfile(args.config, output / "run_config.json")
    hparams = getattr(args, "hparams", None)
    if hparams is not None:
        shutil.copyfile(hparams, output / "hparams.json")
    budget = getattr(args, "_synapse_budget", None)
    if budget is not None:
        (output / "budget_status.json").write_text(json.dumps(budget.status(getattr(args, "_synapse_run_id", None)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "training_report.md").write_text(
        "# SYNAPSE SDK run\n\nThis run was saved by the participant SDK. The model-only ONNX submission is in `submission/`.\n",
        encoding="utf-8",
    )
    return submission


__all__ = ["make_env", "parse_args", "save"]
