"""Validate, export, parity-check and time a participant model.py."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing_event.config import SimulationConfig
from racing_event.participant_model import build_policy_from_model
from racing_event.onnx_artifact import load_onnx_artifact
from racing_event.track import TrackGeometry
from racing_event.vector_env import VectorizedRacingEnv
from tools.export_onnx import export_torch_policy

ROOT = Path(__file__).resolve().parents[1]


def _probe_import(model: Path, timeout: float) -> None:
    code = "from racing_event.participant_model import load_model_module; load_model_module(__import__('pathlib').Path(__import__('sys').argv[1]))"
    try:
        result = subprocess.run([sys.executable, "-c", code, str(model.resolve())], cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"model import exceeded {timeout:g}s timeout") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise ValueError(f"model import failed: {detail[-1] if detail else 'unknown error'}")


def check_model(model: Path, config_path: Path = ROOT / "config/simulation.json", *, timeout: float = 10.0, budget_ms: float = 20.0) -> dict[str, object]:
    if not math.isfinite(float(budget_ms)) or float(budget_ms) <= 0:
        raise ValueError("budget_ms must be a positive finite number")
    _probe_import(model, timeout)
    config = SimulationConfig.from_file(config_path)
    track = TrackGeometry.from_file(ROOT / "tracks/train/monaco.json")
    env = VectorizedRacingEnv(track, config, reward=None, n_envs=1)
    policy, info = build_policy_from_model(model, env.observation_size)
    observations = np.random.default_rng(17).uniform(-1, 1, size=(256, env.observation_size)).astype(np.float32)
    zero = np.zeros((2, env.observation_size), dtype=np.float32)
    mean, value = policy(torch_tensor(observations))
    if mean.shape != (256, 2) or value.shape != (256,) or not np.isfinite(mean.detach().numpy()).all():
        raise ValueError("actor/critic forward probe returned invalid values")
    with tempfile.TemporaryDirectory(prefix="synapse-model-check-") as directory:
        submission = Path(directory) / "submission"
        export_torch_policy(policy, env.training_contract, None, submission)
        artifact = load_onnx_artifact(submission)
        torch_actions = policy.act(observations, deterministic=True).astype(np.float32)
        onnx_actions = artifact(observations[:1])
        # Compare in small batches because the live profile intentionally has a
        # batch-one boundary; parity is still checked over 256 observations.
        onnx_rows = np.vstack([artifact(row[None, :]) for row in observations])
        max_diff = float(np.max(np.abs(torch_actions - onnx_rows)))
        if max_diff > 1.0e-5:
            raise ValueError(f"Torch/ONNX parity exceeded 1e-5: {max_diff:g}")
        started = time.perf_counter()
        for row in observations[:64]:
            artifact(row[None, :])
        latency_ms = (time.perf_counter() - started) * 1000.0 / 64.0
        import onnx
        graph = onnx.load(submission / "model.onnx").graph
        nodes = len(graph.node)
    return {
        "model": str(model),
        "obs_size": env.observation_size,
        "ray_count": len(config.sensors.ray_angles_deg),
        "parameters": info["parameters"],
        "actor_parameters": info["actor_parameters"],
        "nodes": nodes,
        "parity_max_abs_diff": max_diff,
        "latency_ms": latency_ms,
        "budget_ms": budget_ms,
        "latency_ok": latency_ms <= budget_ms,
        "initial_action": policy.act(np.zeros((1, env.observation_size), dtype=np.float32), deterministic=True)[0].tolist(),
        "hint": "final-layer throttle bias is low; consider a mild positive initial throttle bias" if abs(float(policy.act(np.zeros((1, env.observation_size), dtype=np.float32), deterministic=True)[0, 1])) < 0.1 else None,
    }


def torch_tensor(value):
    import torch
    return torch.as_tensor(value, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config/simulation.json")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--budget-ms", type=float, default=20.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = check_model(args.model, args.config, timeout=args.timeout, budget_ms=args.budget_ms)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"model check failed: {exc}\n")
    serialized = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
