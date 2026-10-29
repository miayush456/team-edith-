"""Launch the canonical SYNAPSE reference PPO training path.

The actor architecture is participant-selectable through ``--model``. The
    training algorithm and published-signal reward application stay in
``tools/run_reference_baseline.py`` so the dashboard cannot accidentally run a
trainer that ignores its weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing_event.client_budget import (
    ClientTrainingBudgetError,
    DEFAULT_MAX_RETRAINS,
    DEFAULT_MAX_STEPS_PER_RETRAIN,
    reserve_training_allocation,
    release_training_allocation,
)


ROOT = Path(__file__).resolve().parents[1]
CLIENT_BUDGET_PATH = ROOT / ".synapse" / "client_training_budget.json"
REFERENCE_TRAINER = ROOT / "tools" / "run_reference_baseline.py"
REFERENCE_TRAINER_ALIAS = ROOT / "examples" / "participant_train.py"
DEFAULT_REWARD = ROOT / "examples" / "baseline_reward.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_public_training_tracks(track_dir: Path) -> tuple[tuple[str, str], ...]:
    """Require the participant launcher to train on the complete public set."""

    candidates = sorted(path for path in track_dir.glob("*.json") if path.name != "manifest.json")
    if not candidates:
        raise ValueError(f"track_dir contains no JSON tracks: {track_dir}")
    try:
        manifest = json.loads((ROOT / "tracks" / "train_manifest.json").read_text(encoding="utf-8"))
        expected = {
            (str(item["map_id"]), str(item["canonical_content_hash"]), str(item["delivered_file_sha256"]))
            for item in manifest["maps"]
        }
        selected = set()
        for path in candidates:
            data = json.loads(path.read_text(encoding="utf-8"))
            meta = data["meta"]
            selected.add((str(meta["id"]), str(meta["content_hash"]), _sha256(path)))
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"could not validate public training maps in {track_dir}") from exc
    if selected != expected:
        expected_ids = ", ".join(sorted(item[0] for item in expected))
        received = ", ".join(sorted(item[0] for item in selected)) or "none"
        raise ValueError(
            "participant training must use all public training maps "
            f"({expected_ids}); received {received}"
        )
    return tuple(sorted((item[0], item[1]) for item in selected))


def _workspace_path(value: str, *, label: str) -> Path:
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the project workspace") from exc
    return resolved


def canonical_reference_trainer(value: str | Path) -> Path:
    """Accept the reference entrypoint and its compatibility wrapper only.

    Architecture changes belong in ``--model``. Keeping one launcher here is
    what guarantees that dashboard reward weights reach the same PPO rollout
    for fixed and custom actors alike.
    """
    trainer = _workspace_path(str(value), label="trainer")
    if not trainer.is_file():
        raise ValueError(
            f"trainer does not exist: {trainer}. The supported training path is "
            "tools/run_reference_baseline.py."
        )
    if trainer not in (REFERENCE_TRAINER, REFERENCE_TRAINER_ALIAS):
        raise ValueError(
            "participant training always uses tools/run_reference_baseline.py; "
            "change that file to modify PPO and use --model for a custom architecture"
        )
    return REFERENCE_TRAINER


def _positive(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{name} must be a positive integer")
    return parsed


def _nonnegative(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"{name} must be a non-negative integer")
    return parsed


def _positive_value(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trainer",
        type=Path,
        default=Path("tools/run_reference_baseline.py"),
        help="compatibility alias; training always runs tools/run_reference_baseline.py",
    )
    parser.add_argument("--reward", type=Path, default=Path("examples/baseline_reward.json"), help="participant reward signal weights")
    parser.add_argument("--track-dir", type=Path, default=Path("tracks/train"))
    parser.add_argument("--config", type=Path, default=Path("config/simulation.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/run-001"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="PPO training processor")
    parser.add_argument("--model", type=Path, default=None, help="optional participant model.py exposing build_actor(obs_size)")
    parser.add_argument("--hparams", type=Path, default=None, help="optional JSON PPO hyperparameter snapshot")
    parser.add_argument("--init-from", type=Path, default=None, help="optional run/artifact to warm-start in a new output directory")
    parser.add_argument("--steps", type=lambda value: _positive(value, "--steps"), default=1_000_000)
    parser.add_argument("--budget-steps", type=lambda value: _positive(value, "--budget-steps"), default=1_000_000)
    parser.add_argument("--n-envs", type=lambda value: _positive(value, "--n-envs"), default=2)
    parser.add_argument("--rollout-length", type=lambda value: _positive(value, "--rollout-length"), default=32)
    parser.add_argument("--seed", type=lambda value: _positive(value, "--seed"), default=17)
    return parser.parse_args()


def build_training_command(args: argparse.Namespace) -> list[str]:
    trainer = canonical_reference_trainer(getattr(args, "trainer", REFERENCE_TRAINER))
    track_dir = _workspace_path(str(args.track_dir), label="track_dir")
    config = _workspace_path(str(args.config), label="config")
    output_dir = _workspace_path(str(args.output_dir), label="output_dir")
    if not config.is_file():
        raise ValueError(f"config does not exist: {config}")
    if not track_dir.is_dir():
        raise ValueError(f"track_dir is not a directory: {track_dir}")
    validate_public_training_tracks(track_dir)
    reward_value = getattr(args, "reward", None) or DEFAULT_REWARD
    reward = _workspace_path(str(reward_value), label="reward")
    if not reward.is_file():
        raise ValueError(f"reward does not exist: {reward}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output_dir must be a directory path: {output_dir}")
    if output_dir.parent.exists() and not output_dir.parent.is_dir():
        raise ValueError(f"output_dir parent must be a directory: {output_dir.parent}")
    steps = _positive_value(args.steps, "--steps")
    budget_steps = _positive_value(args.budget_steps, "--budget-steps")
    if budget_steps > DEFAULT_MAX_RETRAINS * DEFAULT_MAX_STEPS_PER_RETRAIN:
        raise ValueError("--budget-steps cannot exceed the combined 12-iteration transition cap")
    n_envs = _positive_value(args.n_envs, "--n-envs")
    rollout_length = _positive_value(args.rollout_length, "--rollout-length")
    seed = _positive_value(args.seed, "--seed")
    command = [
        sys.executable,
        str(trainer),
        "--track-dir", str(track_dir),
        "--config", str(config),
        "--output-dir", str(output_dir),
        "--steps", str(steps),
        "--budget-steps", str(budget_steps),
        "--device", getattr(args, "device", "cpu"),
        "--n-envs", str(n_envs),
        "--rollout-length", str(rollout_length),
        "--seed", str(seed),
    ]
    if reward is not None:
        command[2:2] = ["--reward", str(reward)]
    for flag, value in (("--model", getattr(args, "model", None)), ("--hparams", getattr(args, "hparams", None)), ("--init-from", getattr(args, "init_from", None))):
        if value is not None:
            command.extend([flag, str(_workspace_path(str(value), label=flag))])
    return command


def main() -> None:
    args = parse_args()
    try:
        command = build_training_command(args)
        allocation = reserve_training_allocation(
            CLIENT_BUDGET_PATH,
            _workspace_path(str(args.output_dir), label="output_dir"),
            _positive_value(args.budget_steps, "--budget-steps"),
            max_retrains=DEFAULT_MAX_RETRAINS,
            max_steps_per_retrain=DEFAULT_MAX_STEPS_PER_RETRAIN,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"client training allocation {allocation.slot}/{allocation.max_retrains} "
        f"({allocation.remaining_retrains} new retrains remaining)"
        + ("; resuming existing allocation" if allocation.reused else ""),
        flush=True,
    )
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        output_dir = _workspace_path(str(args.output_dir), label="output_dir")
        checkpoint_markers = (
            output_dir / "policy.pt", output_dir / "reference_checkpoint.json",
            output_dir / "artifact" / "manifest.json", output_dir / "submission" / "manifest.json",
        )
        if not any(path.exists() for path in checkpoint_markers):
            try:
                release_training_allocation(CLIENT_BUDGET_PATH, output_dir)
                print("released client training allocation after failure before checkpoint", flush=True)
            except ClientTrainingBudgetError as exc:
                print(f"could not release failed client training allocation: {exc}", file=sys.stderr, flush=True)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
