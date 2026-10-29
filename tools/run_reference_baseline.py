"""Canonical local reference PPO lifecycle for every participant actor.

Exact CPU process resume uses reference_checkpoint.json, never the dashboard
mirrors. Legacy prototype artifacts can only be explicit non-exact warm starts
in a new output directory. Fine-tuning preserves source preprocessing, uses a
fresh optimizer, and selects on public development tracks (not event scoring).

Participants may replace the actor architecture through ``--model``. The
reference baseline remains the training and reward path so the ten published
reward weights are applied uniformly.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from racing_event.budget import RunBudget, RunBudgetError
from racing_event.config import SimulationConfig
from racing_event.evaluation import evaluate_policy, validate_evaluation_contract
from racing_event.observation import RunningObservationNormalizer, TrainingContract
from racing_event.ppo import FixedMLPActorCritic, PPOTrainer
from racing_event.participant_model import build_policy_from_model, load_policy_state, sha256_file
from racing_event import reference_checkpoint as checkpoint
from racing_event.reward import RewardConfig
from racing_event.rollout import PPOHyperparameters, RolloutCollector
from racing_event.track import TrackGeometry
from racing_event.training_replay import BestTrainingReplay
from racing_event.vector_env import VectorizedRacingEnv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACK_DIR = ROOT / "tracks" / "train"
DEFAULT_CONFIG = ROOT / "config" / "simulation.json"
DEFAULT_REWARD = ROOT / "examples" / "baseline_reward.json"
SELECTION_RULE = (
    "reference-only public-development selection v1; not event scoring: "
    "lexicographically maximize finish count, minimize total off-track seconds, "
    "then minimize total physical lap seconds over finished tracks; "
    "strict improvement only (ties/regressions retain best); training reward excluded"
)


def _resolve_tracks(track_dir: Path, explicit: list[Path] | None) -> list[Path]:
    paths = explicit or sorted(track_dir.glob("*.json"))
    paths = [Path(path) for path in paths if Path(path).name != "manifest.json"]
    if not paths:
        raise ValueError("no training track JSON files were found")
    return paths


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward", type=Path, default=DEFAULT_REWARD)
    parser.add_argument("--track-dir", type=Path, default=DEFAULT_TRACK_DIR)
    parser.add_argument("--tracks", type=Path, nargs="+", default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("participant_run"))
    parser.add_argument("--budget-steps", type=int, default=1_000_000)
    parser.add_argument("--steps", type=int, default=1_000_000, help="maximum training transitions in this invocation")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="PPO training processor")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--rollout-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warm-start", type=Path, help="imitation/reference artifact directory; new optimizer, NOT exact source resume")
    parser.add_argument("--init-from", type=Path, help="fork a run/artifact with a fresh optimizer and current reward")
    parser.add_argument("--model", type=Path, help="participant model.py exposing build_actor(obs_size) and optional build_critic")
    parser.add_argument("--hparams", type=Path, help="JSON object overriding PPO hyperparameters")
    parser.add_argument("--fine-tune", action="store_true", help="bounded optional reference fine-tuning with public-track best selection")
    parser.add_argument("--max-updates", type=int, help="updates per invocation; fine-tuning default 4, maximum 32")
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=None, help="optional fine-tuning evaluation cadence; 0 chooses automatically")
    parser.add_argument("--eval-max-steps", type=int, default=None, help="per-public-track validation cap for optional fine-tuning")
    parser.add_argument("--eval-track-dir", type=Path, default=DEFAULT_TRACK_DIR, help="public development tracks only (no private evaluator)")
    return parser.parse_args(argv)


def _load_hparams(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read hparams JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("hparams must contain a JSON object")
    allowed = {"rollout_length", "gamma", "gae_lambda", "clip_range", "update_epochs", "minibatch_size", "learning_rate", "entropy_coefficient", "value_loss_coefficient", "hidden_sizes", "max_grad_norm"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"hparams contains unknown fields: {sorted(unknown)}")
    return value


def _open_budget(path: Path, budget_steps: int) -> tuple[RunBudget, str]:
    if budget_steps <= 0:
        raise ValueError("--budget-steps must be positive")
    if path.exists():
        budget = RunBudget.open(path, max_runs=1, max_steps_per_run=budget_steps)
        status = budget.status()
        if status["runs"]:
            return budget, status["runs"][0]["run_id"]
    else:
        budget = RunBudget.create(path, max_runs=1, max_steps_per_run=budget_steps)
    return budget, budget.start_run()


def load_warm_start(source: Path, env: VectorizedRacingEnv) -> tuple[Any, Any, dict[str, Any]]:
    """Safe declarative reference artifacts only; no import of source Python."""
    source = source.resolve()
    if (source / checkpoint.POINTER).exists() or (source / checkpoint.POINTER).is_symlink():
        loaded = checkpoint.load_current(source)
        assert loaded is not None
        source = loaded[0]
    policy, _ = FixedMLPActorCritic.load_artifact(
        source / "policy.pt", source / "policy.json", expected_observation_size=env.observation_size,
    )
    contract, normalizer = TrainingContract.load(source / "training_contract.json")
    validate_evaluation_contract(env, contract, normalizer)
    if checkpoint.read_json(source / "training_contract.json") != contract.to_dict(normalizer):
        raise ValueError("warm-start contract contains unsupported semantics")
    if normalizer is not None:
        if np.any(normalizer.m2 < 0):
            raise ValueError("invalid warm-start normalizer variance")
        normalizer.freeze()  # Preserve its mapping, rather than changing source preprocessing.
    origin = {
        "kind": "non-exact artifact warm start; optimizer and collector newly initialized",
        "files": {name: checkpoint.sha256(source / name) for name in ("policy.pt", "policy.json", "training_contract.json")},
        "preprocessing": "environment_only" if normalizer is None else "frozen_source_normalizer",
    }
    return policy, normalizer, origin


def evaluate_candidate(policy: Any, normalizer: Any, tracks: tuple[Any, ...],
                       config: SimulationConfig, max_steps: int) -> dict[str, Any]:
    """One deterministic fixed start per public track; no retries or rewards."""
    reports = []
    for track in tracks:
        env = VectorizedRacingEnv(track, config, reward=None, n_envs=1)
        report = evaluate_policy(
            env, policy.as_policy(deterministic=True), normalizer=normalizer, max_steps=max_steps,
        ).to_dict()
        # The evaluator's reward is zero (reward=None); do not expose it as a score.
        report.pop("total_reward")
        reports.append(report)
    return {
        "tracks": reports,
        "finish_count": sum(row["status"] == "finished" for row in reports),
        "off_track_seconds": sum(row["off_track_steps"] * config.dt for row in reports),
        "finished_lap_seconds": sum(row["lap_time_s"] for row in reports if row["status"] == "finished"),
    }


def selection_key(report: dict[str, Any]) -> tuple[int, float, float]:
    return (report["finish_count"], -report["off_track_seconds"], -report["finished_lap_seconds"])


def select_candidate(previous: Any, candidate: dict[str, Any], generation: str,
                     update: int) -> dict[str, Any]:
    improved = previous is None or selection_key(candidate) > selection_key(previous["best_report"])
    history = [] if previous is None else list(previous["history"])
    history.append({"generation": generation, "update": update, "selected": improved, "report": candidate})
    return {
        "rule": SELECTION_RULE, "history": history,
        "best_generation": generation if improved else previous["best_generation"],
        "best_manifest_sha256": None if improved else previous["best_manifest_sha256"],
        "best_report": candidate if improved else previous["best_report"],
        "best_artifact_directory": "best",
        "evaluation_budget": "separate bounded diagnostics; no training RunBudget reservations",
    }


def _selection_with_best_hash(output: Path, selection: Any) -> Any:
    if selection is not None and selection["best_manifest_sha256"] is None:
        # A generation cannot contain its own manifest hash. On the next commit,
        # bind a retained best to its hash before referencing it from that commit.
        selection = dict(selection)
        best, _ = checkpoint.validate_generation(output, selection["best_generation"])
        selection["best_manifest_sha256"] = checkpoint.sha256(best / "manifest.json")
    return selection


def _write_report(output: Path, state: dict[str, Any]) -> None:
    text = (
        "# Optional reference baseline\n\nParticipants own their models and training algorithms.\n\n"
        f"- Updates: {state['updates']}\n"
        f"- Training budget: {state['budget_used']} / {state['budget_steps']}\n"
        f"- Resume: {state['resume_note']}\n"
        f"- Authoritative checkpoint: `{checkpoint.POINTER}`\n"
        "- Root policy files: latest committed dashboard mirrors, not necessarily best.\n"
    )
    if state["fine_tune"]:
        text += f"- Selection: {SELECTION_RULE}\n- Best artifact: `best/`; report: `selection_report.json`\n"
    if state.get("updates", 0):
        text += "- Best deterministic training replay: `best_replay.json`\n"
    checkpoint.atomic_bytes(output / "training_report.md", text.encode())


def _training_position_payload(cars: list[dict[str, Any]], update: int) -> dict[str, Any]:
    """Expose each lane's latest driven position, before any autoreset."""
    return {"kind": "training_position", "update": update, "cars": cars}


def _public_validation_payload(selection: dict[str, Any] | None, dt: float) -> dict[str, Any] | None:
    """Summarize only the latest and retained-best public-track diagnostics."""
    if selection is None or not selection["history"]:
        return None

    def summarize(report: dict[str, Any]) -> dict[str, Any]:
        tracks = [{
            "track_id": row["track_id"], "status": row["status"],
            "lap_time_s": row["lap_time_s"], "mean_speed_mps": row["mean_speed_mps"],
            "off_track_seconds": row["off_track_steps"] * dt,
        } for row in report["tracks"]]
        return {
            "finish_count": report["finish_count"], "track_count": len(tracks),
            "off_track_seconds": report["off_track_seconds"],
            "finished_lap_seconds": report["finished_lap_seconds"],
            "tracks": tracks,
        }

    latest = selection["history"][-1]
    best = next((row for row in selection["history"] if row["generation"] == selection["best_generation"]), None)
    return {
        "latest_update": latest["update"], "latest_improved": latest["selected"],
        "latest": summarize(latest["report"]),
        "best_update": None if best is None else best["update"],
        "best": summarize(selection["best_report"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps <= 0 or args.n_envs <= 0 or args.rollout_length <= 0:
        raise ValueError("--steps, --n-envs, and --rollout-length must be positive")
    if args.n_envs * args.rollout_length > 8_192:
        raise ValueError("n_envs * rollout_length must be at most 8192 for the bounded reference trainer")
    requested_device = getattr(args, "device", "cpu")
    if requested_device == "cuda":
        if torch.version.cuda is None:
            raise ValueError(
                f"GPU selected, but this PyTorch build ({torch.__version__}) has no CUDA support"
            )
        if not torch.cuda.is_available():
            raise ValueError(
                f"GPU selected, but CUDA is unavailable (PyTorch CUDA build {torch.version.cuda}, "
                f"visible devices {torch.cuda.device_count()}); check GPU driver/device visibility"
            )
        args._resolved_device = torch.device("cuda:0")
    elif requested_device == "cpu":
        args._resolved_device = torch.device("cpu")
    else:
        raise ValueError("--device must be cpu or cuda")
    output = args.output_dir.resolve()
    with checkpoint.writer_lock(output):
        return _run_locked(args, output)


def _run_locked(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    # The reference model has no need for a large CPU thread pool; fixing this
    # keeps its host-side simulator and data processing consistent on either device.
    torch.set_num_threads(1)
    device = getattr(args, "_resolved_device", torch.device("cpu"))
    current = checkpoint.load_current(output)
    saved_manifest = None if current is None else current[1]
    saved_identity = None if current is None else saved_manifest["identity"]
    requested_warm_start = getattr(args, "warm_start", None) or getattr(args, "init_from", None)
    model_path = getattr(args, "model", None)
    if model_path is None and saved_identity is not None and saved_identity.get("model", {}).get("model_sha256"):
        candidate = output / "model_source.py"
        if candidate.is_file():
            model_path = candidate
    fine_tune = bool(getattr(args, "fine_tune", False) or requested_warm_start)
    if saved_identity is not None and saved_identity["selection"] is not None:
        fine_tune = True  # Preserve legacy fine-tune checkpoint behavior.
    selection_enabled = fine_tune
    max_updates = getattr(args, "max_updates", None)
    if max_updates is None:
        max_updates = 4 if fine_tune else args.steps
    if max_updates <= 0 or (fine_tune and (max_updates > 32 or args.steps > 65_536)):
        raise ValueError("fine-tuning requires 1..32 updates and at most 65536 transitions per invocation")
    hparams = _load_hparams(getattr(args, "hparams", None))
    hparams.setdefault("rollout_length", args.rollout_length)
    hparams.setdefault("learning_rate", getattr(args, "learning_rate", 3.0e-4))
    hparams.setdefault("update_epochs", getattr(args, "update_epochs", 10))
    hparams.setdefault("minibatch_size", 64)
    hp = PPOHyperparameters(**hparams)
    if hp.update_epochs > 16 or hp.learning_rate > 0.01:
        raise ValueError("reference updates require at most 16 epochs and learning-rate <= 0.01")
    config = SimulationConfig.from_file(args.config)
    reward = RewardConfig.from_file(args.reward)
    tracks = tuple(TrackGeometry.from_file(path) for path in _resolve_tracks(args.track_dir, args.tracks))
    env = VectorizedRacingEnv(tracks, config, reward, n_envs=args.n_envs)
    env.rng = np.random.default_rng(args.seed)
    contract = replace(
        env.training_contract, action_distribution="tanh_normal",
        reward_config_hash=reward.content_hash, reward_sha256=checkpoint.sha256(args.reward),
    )
    evaluation_tracks: tuple[Any, ...] = ()
    selection_settings = None
    if selection_enabled:
        requested_eval_every = getattr(args, "eval_every", None)
        eval_max_steps = getattr(args, "eval_max_steps", None)
        if eval_max_steps is None:
            eval_max_steps = 2400
        evaluation_tracks = tuple(TrackGeometry.from_file(path) for path in _resolve_tracks(
            getattr(args, "eval_track_dir", DEFAULT_TRACK_DIR), None,
        ))
        if not isinstance(eval_max_steps, int) or isinstance(eval_max_steps, bool) or not 1 <= eval_max_steps <= 20_000 or not 1 <= len(evaluation_tracks) <= 16 or len(evaluation_tracks) * eval_max_steps > 100_000:
            raise ValueError("evaluation cap: 1..20000 steps/track, <=16 tracks and <=100000 total steps/evaluation")
        public = {track.content_hash for track in (
            TrackGeometry.from_file(path) for path in _resolve_tracks(DEFAULT_TRACK_DIR, None)
        )}
        if any(track.content_hash not in public for track in evaluation_tracks):
            raise ValueError("reference selection requires public train/development tracks")
        if len({track.track_id for track in evaluation_tracks}) != len(evaluation_tracks):
            raise ValueError("selection tracks must be unique")
        rollout_size = args.n_envs * args.rollout_length
        possible_updates = min(max_updates, (args.steps + rollout_size - 1) // rollout_size)
        auto_eval_every = requested_eval_every in (None, 0)
        eval_every = 1 if auto_eval_every else requested_eval_every
        if isinstance(eval_every, bool) or not isinstance(eval_every, int) or eval_every <= 0 or (not auto_eval_every and eval_every > 32):
            raise ValueError("--eval-every must be 1..32 (or 0 for an automatic interval)")
        evaluations_bound = int(current is None) + (possible_updates + eval_every - 1) // eval_every
        if evaluations_bound * len(evaluation_tracks) * eval_max_steps > 100_000:
            raise ValueError("evaluation cap: at most 100000 diagnostic steps per invocation; reduce eval-max-steps or increase eval-every")
        selection_settings = {
            "rule": SELECTION_RULE, "every": eval_every, "max_steps": eval_max_steps,
            "tracks": [[track.track_id, track.content_hash] for track in evaluation_tracks],
        }
        # Keep the saved validation cadence stable when a participant spends a
        # new allocation to extend the same exact-resume checkpoint.
        if (current is not None and saved_identity is not None
                and args.budget_steps > int(saved_identity["budget_steps"])):
            selection_settings = saved_identity["selection"]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    warm_start = requested_warm_start
    origin = None if saved_identity is None else saved_identity["warm_start"]
    if warm_start is not None:
        if model_path is not None and (warm_start / "model_source.py").is_file() and getattr(args, "model", None) is None:
            model_path = warm_start / "model_source.py"
        if model_path is None:
            policy, normalizer, requested_origin = load_warm_start(warm_start, env)
        else:
            source = warm_start / "reference_generations" if (warm_start / "reference_checkpoint.json").exists() else warm_start
            source_current = checkpoint.load_current(warm_start)[0] if (warm_start / "reference_checkpoint.json").exists() else source
            policy, model_info = build_policy_from_model(model_path, env.observation_size)
            load_policy_state(policy, source_current / "policy.pt")
            normalizer = None
            source_contract_path = source_current / "training_contract.json"
            if source_contract_path.exists():
                _, normalizer = TrainingContract.load(source_contract_path)
                if normalizer is not None:
                    normalizer.freeze()
            requested_origin = {
                "kind": "participant model fork; optimizer and collector newly initialized",
                "files": {name: checkpoint.sha256(source_current / name) for name in ("policy.pt", "policy.json", "training_contract.json")},
                "preprocessing": "environment_only" if normalizer is None else "frozen_source_normalizer",
                "model_sha256": model_info["model_sha256"],
            }
        if current is not None and requested_origin != origin:
            raise ValueError("warm-start origin does not match this run")
        origin = requested_origin
    elif current is None:
        if fine_tune:
            raise ValueError("new fine-tuning runs require --warm-start and a new output directory")
        if model_path is None:
            policy = FixedMLPActorCritic(env.observation_size)
            model_info = None
        else:
            policy, model_info = build_policy_from_model(model_path, env.observation_size)
        normalizer = RunningObservationNormalizer(env.observation_size)
    model_identity = None
    if model_path is not None:
        _, inspected_model = build_policy_from_model(model_path, env.observation_size)
        model_identity = {
            "model_sha256": sha256_file(model_path),
            "actor_parameters": inspected_model["actor_parameters"],
            "parameters": inspected_model["parameters"],
        }
    hparams_path = getattr(args, "hparams", None)
    identity = {
        "runtime": checkpoint.runtime_identity(str(device)), "contract": contract.to_dict(),
        "ppo": hp.to_dict(), "n_envs": args.n_envs, "seed": args.seed,
        "budget_steps": args.budget_steps, "warm_start": origin,
        "preprocessing": "running_reference_normalizer" if origin is None else origin["preprocessing"],
        "selection": selection_settings,
    }
    if hparams_path is not None:
        identity["hparams"] = hp.to_dict()
    if model_identity is not None:
        identity["model"] = model_identity
    extended_budget = bool(
        current is not None and saved_identity is not None
        and args.budget_steps > int(saved_identity["budget_steps"])
    )
    if current is not None:
        comparable_identity = dict(identity)
        if extended_budget:
            comparable_identity["budget_steps"] = saved_identity["budget_steps"]
        if comparable_identity != saved_identity:
            raise ValueError("saved reference checkpoint identity does not match this invocation (config, tracks, seed, PPO, budget, preprocessing, selection or runtime)")
        directory, manifest = current
        if not (output / "budget.json").exists():
            raise ValueError("external budget missing; exact recovery refused")
        if model_path is None:
            policy, metadata = FixedMLPActorCritic.load_artifact(directory / "policy.pt", directory / "policy.json", expected_observation_size=env.observation_size)
        else:
            policy, model_info = build_policy_from_model(model_path, env.observation_size)
            load_policy_state(policy, directory / "policy.pt")
            metadata = checkpoint.read_json(directory / "policy.json")
        saved_contract, normalizer = TrainingContract.load(directory / "training_contract.json")
        if saved_contract.to_dict() != contract.to_dict() or metadata.get("ppo") != hp.to_dict():
            raise ValueError("checkpoint policy/contract mismatch")
        if checkpoint.read_json(directory / "training_contract.json") != saved_contract.to_dict(normalizer):
            raise ValueError("unsupported checkpoint contract semantics")
        mode = identity["preprocessing"]
        if (mode == "environment_only") != (normalizer is None) or (normalizer is not None and normalizer.training != (mode == "running_reference_normalizer")):
            raise ValueError("checkpoint preprocessing mismatch")
        if normalizer is not None and np.any(normalizer.m2 < 0):
            raise ValueError("invalid checkpoint normalizer variance")
        state = checkpoint.read_json(directory / "training_state.json")
        metrics = [json.loads(line) for line in (directory / "training_metrics.jsonl").read_text().splitlines()]
        selection = checkpoint.read_json(directory / "selection_report.json")
        if not isinstance(state, dict) or not {"updates", "run_id", "budget_used", "budget_remaining", "budget_steps", "fine_tune"}.issubset(state):
            raise ValueError("incomplete checkpoint training state")
        if type(state["updates"]) is not int or state["updates"] < 0 or len(metrics) != state["updates"]:
            raise ValueError("checkpoint update/metrics mismatch")
        if (type(state["budget_used"]) is not int or state["budget_used"] < 0
                or state["budget_steps"] > args.budget_steps
                or state["budget_remaining"] != state["budget_steps"] - state["budget_used"]
                or (state["budget_steps"] != args.budget_steps and not extended_budget)
                or state["fine_tune"] != fine_tune):
            raise ValueError("checkpoint training state/budget mismatch")
        budget = RunBudget.open(output / "budget.json", max_runs=1, max_steps_per_run=args.budget_steps)
        runs = budget.status()["runs"]
        if len(runs) != 1:
            raise ValueError("external budget has no matching run; exact recovery refused")
        run_id = runs[0]["run_id"]
    else:
        budget, run_id = _open_budget(output / "budget.json", args.budget_steps)
    status = budget.status(run_id)
    head = budget.status()["head_hash"]
    if current is not None:
        budget_extension_is_authorized = False
        if extended_budget and head != manifest["budget_head"]:
            records = checkpoint.read_json(output / "budget.json")["records"]
            pointer = next((index for index, row in enumerate(records) if row.get("hash") == manifest["budget_head"]), None)
            tail = [] if pointer is None else records[pointer + 1:]
            budget_extension_is_authorized = (
                pointer is not None and bool(tail)
                and all(row.get("type") == "budget_extended" for row in tail)
                and sum(row.get("steps", 0) for row in tail) == args.budget_steps - state["budget_steps"]
            )
        if (state["run_id"] != run_id or status["used_steps"] != state["budget_used"]
                or (head != manifest["budget_head"] and not budget_extension_is_authorized)):
            raise ValueError("external budget differs from committed boundary; spent steps retained, exact recovery refused (interrupted update)")
    elif status["used_steps"]:
        raise ValueError("external budget already spent without an authoritative checkpoint; exact recovery refused")
    policy.to(device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    trainer = PPOTrainer(policy, hp)
    collector = RolloutCollector(env, policy.as_policy(), budget, run_id, normalizer=normalizer, seed=args.seed)
    if current is not None:
        checkpoint.restore_local_state(directory / "local_state.pt", trainer)
        optimizer_steps = sum(int(row.get("optimizer_steps", hp.update_epochs)) for row in metrics)
        values = trainer.optimizer.state_dict()["state"]
        if bool(values) != bool(optimizer_steps) or any(float(item["step"]) != optimizer_steps for item in values.values()):
            raise ValueError("optimizer step count does not match completed updates")
        checkpoint.restore_collector(collector, checkpoint.read_json(directory / "collector.json"))
        if selection_enabled != (selection is not None):
            raise ValueError("checkpoint selection state mismatch")
        checkpoint.repair_mirrors(output, directory, selection)
        selection = _selection_with_best_hash(output, selection)
    else:
        state = {
            "release_id": "synapse-local-reference-v1", "run_id": run_id,
            "budget_steps": args.budget_steps, "budget_used": 0,
            "budget_remaining": status["remaining_steps"], "updates": 0,
            "track_ids": [track.track_id for track in tracks], "track_hashes": [track.content_hash for track in tracks],
            "config_hash": contract.config_hash, "reward_sha256": contract.reward_sha256,
            "reward_config_hash": reward.content_hash, "ray_angles_deg": list(config.sensors.ray_angles_deg),
            "observation_size": env.observation_size, "hidden_sizes": [64, 64],
            "n_envs": args.n_envs, "rollout_length": args.rollout_length, "seed": args.seed,
            "fine_tune": fine_tune, "preprocessing": identity["preprocessing"],
            "resume_note": "exact completed-rollout CPU state via atomic generation pointer; external budget never rolled back",
            "warm_start": origin,
        }
        metrics, selection = [], None

    if model_path is not None:
        model_source = output / "model_source.py"
        if not model_source.exists():
            model_source.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(model_path).resolve(), model_source)
    if hparams_path is not None:
        destination = output / "hparams.json"
        if not destination.exists():
            shutil.copyfile(Path(hparams_path).resolve(), destination)

    def save(*, evaluate: bool) -> None:
        nonlocal selection
        generation = checkpoint.new_generation()
        if evaluate:
            candidate = evaluate_candidate(policy, normalizer, evaluation_tracks, config, selection_settings["max_steps"])
            selection = select_candidate(selection, candidate, generation, state["updates"])
        checkpoint.publish(
            output, generation, policy=policy, contract=contract, normalizer=normalizer,
            trainer=trainer, collector=collector, state=state, metrics=metrics,
            selection=selection, identity=identity, budget_head=budget.status()["head_hash"],
        )
        selection = _selection_with_best_hash(output, selection)
        _write_report(output, state)

    if extended_budget:
        state["budget_steps"] = args.budget_steps
        state["budget_remaining"] = status["remaining_steps"]
        save(evaluate=False)

    if current is None:
        save(evaluate=selection_enabled)
    print(f"training processor: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""), flush=True)
    target = min(args.steps, status["remaining_steps"])
    invocation_updates = 0
    replay_worker = BestTrainingReplay(output, tracks[0], config)
    replay_worker.submit(policy, normalizer, state["updates"])
    try:
        while target > 0 and invocation_updates < max_updates:
            batch_steps = min(args.n_envs * args.rollout_length, target)
            rollout = collector.collect(batch_steps)
            ppo_metrics = trainer.update(rollout.batch)
            replay_worker.submit(policy, normalizer, state["updates"] + 1)
            status = budget.status(run_id)
            live_budget = budget.status()
            if live_budget["max_steps_per_run"] > state["budget_steps"]:
                extension = live_budget["max_steps_per_run"] - state["budget_steps"]
                target += extension
                state["budget_steps"] = live_budget["max_steps_per_run"]
                identity["budget_steps"] = live_budget["max_steps_per_run"]
                args.budget_steps = live_budget["max_steps_per_run"]
            state["updates"] += 1
            state["budget_used"], state["budget_remaining"] = status["used_steps"], status["remaining_steps"]
            metrics.append({
                "update": state["updates"], "budget_used": status["used_steps"],
                "budget_remaining": status["remaining_steps"],
                **{name: rollout.metrics[name] for name in ("mean_episode_return", "lap_completion_rate", "mean_speed_mps", "crash_rate", "off_track_rate")},
                "mean_sector_coverage_during_update": rollout.metrics.get("mean_sector_coverage_during_update"),
                "signal_contributions": rollout.metrics.get("signal_contributions", {}),
                "weighted_signal_sums": rollout.metrics.get("weighted_signal_sums", {}),
                **ppo_metrics.to_dict(),
            })
            invocation_updates += 1
            target -= batch_steps
            evaluate = selection_enabled and state["updates"] % selection_settings["every"] == 0
            save(evaluate=evaluate)
            row = metrics[-1]
            def display(name: str) -> str:
                value = row[name]
                return "n/a" if value is None else f"{float(value):.3f}"
            print(
                f"update {state['updates']}: steps {state['budget_used']}/{args.budget_steps} "
                f"return={display('mean_episode_return')} lap={display('lap_completion_rate')} "
                f"speed={display('mean_speed_mps')} m/s crash={display('crash_rate')} "
                f"off_track={display('off_track_rate')} zones={display('mean_sector_coverage_during_update')}", flush=True,
            )
            print("SYNAPSE_METRIC " + json.dumps({
                "kind": "training_progress",
                "training_position": _training_position_payload(rollout.training_position or [], state["updates"]),
                "public_validation": _public_validation_payload(selection, config.dt),
                "best_replay": str(output / "best_replay.json"),
            },
                sort_keys=True, separators=(",", ":"), allow_nan=False,
            ), flush=True)
    finally:
        replay_worker.close()
    print(f"best training replay: {output / 'best_replay.json'}", flush=True)
    _write_report(output, state)
    return state


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        state = run(args)
    except (OSError, ValueError, FloatingPointError, RunBudgetError) as exc:
        raise SystemExit(f"reference baseline could not continue: {exc}") from exc
    print(f"saved reference run in {args.output_dir}; budget remaining: {state['budget_remaining']}")


if __name__ == "__main__":
    main()
