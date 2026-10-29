"""Measure the first vectorized engine slice for E1.

This benchmark is deliberately small and reproducible.  It reports full
environment transitions, a NumPy policy plus environment, the batched physics
kernel, and the remaining geometry/reward/observation overhead.  The latter
is a proxy for sensor/projection cost, not a claim that every overhead is a
raycast.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing_event.config import SimulationConfig
from racing_event.physics import CarStateBatch, step_car_batch
from racing_event.track import TrackGeometry
from racing_event.vector_env import VectorizedRacingEnv


ROOT = Path(__file__).resolve().parents[1]
TRACK_PATH = ROOT / "tracks" / "train" / "monaco.json"
CONFIG_PATH = ROOT / "config" / "simulation.json"


def _step_count(env: VectorizedRacingEnv, actions: np.ndarray, steps: int) -> None:
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(actions)
        if np.any(terminated | truncated):
            env.reset(seed=1234)


def _measure_environment(n_envs: int, actions: np.ndarray, steps: int, config: SimulationConfig, track: TrackGeometry) -> float:
    env = VectorizedRacingEnv(track, config, n_envs=n_envs)
    env.reset(seed=1234)
    _step_count(env, actions, 1)
    env.reset(seed=1234)
    start = time.perf_counter()
    _step_count(env, actions, steps)
    return time.perf_counter() - start


def _measure_policy_plus_environment(n_envs: int, steps: int, config: SimulationConfig, track: TrackGeometry) -> float:
    env = VectorizedRacingEnv(track, config, n_envs=n_envs)
    observations, _ = env.reset(seed=1234)
    rng = np.random.default_rng(99)
    first = rng.normal(0.0, 0.15, size=(observations.shape[1], 32))
    second = rng.normal(0.0, 0.15, size=(32, 2))
    start = time.perf_counter()
    for _ in range(steps):
        hidden = np.tanh(observations @ first)
        actions = np.tanh(hidden @ second).astype(np.float64)
        observations, _, terminated, truncated, _ = env.step(actions)
        if np.any(terminated | truncated):
            observations, _ = env.reset(seed=1234)
    return time.perf_counter() - start


def _measure_physics(n_envs: int, actions: np.ndarray, steps: int, config: SimulationConfig) -> float:
    zeros = np.zeros(n_envs, dtype=np.float64)
    state = CarStateBatch(
        x=zeros.copy(),
        y=zeros.copy(),
        heading=zeros.copy(),
        speed=zeros.copy(),
        lateral_velocity=zeros.copy(),
        yaw_rate=zeros.copy(),
        steering_angle=zeros.copy(),
        previous_steering_command=zeros.copy(),
        previous_longitudinal_command=zeros.copy(),
    )
    for _ in range(1):
        state = step_car_batch(state, actions, config.car, config.dt)
    state = CarStateBatch(*(np.zeros(n_envs, dtype=np.float64) for _ in range(9)))
    start = time.perf_counter()
    for _ in range(steps):
        state = step_car_batch(state, actions, config.car, config.dt)
    return time.perf_counter() - start


def benchmark(n_envs_values: tuple[int, ...], steps: int) -> str:
    config = SimulationConfig.from_file(CONFIG_PATH)
    track = TrackGeometry.from_file(TRACK_PATH)
    lines = [
        "# E1 Vectorized Environment Benchmark",
        "",
        f"- Date (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Host: {platform.platform()}",
        f"- Python: {platform.python_version()}",
        f"- NumPy: {np.__version__}",
        f"- Track: `{track.track_id}` ({track.content_hash})",
        f"- Tick rate: {config.tick_hz} Hz",
        f"- Measured transitions per row: {steps}",
        "- Sensor/projection column is full environment time minus batched physics-only time; it includes geometry, reward, and observation work.",
        "",
        "| n_envs | environment steps/s | policy + environment steps/s | physics-only steps/s | geometry/reward/observation overhead ms/env-step |",
        "|---:|---:|---:|---:|---:|",
    ]
    for n_envs in n_envs_values:
        actions = np.zeros((n_envs, 2), dtype=np.float64)
        environment_seconds = _measure_environment(n_envs, actions, steps, config, track)
        policy_seconds = _measure_policy_plus_environment(n_envs, steps, config, track)
        physics_seconds = _measure_physics(n_envs, actions, steps, config)
        environment_steps_per_second = n_envs * steps / environment_seconds
        policy_steps_per_second = n_envs * steps / policy_seconds
        physics_steps_per_second = n_envs * steps / physics_seconds
        overhead_ms = max(0.0, (environment_seconds - physics_seconds) / (steps * n_envs) * 1000.0)
        lines.append(
            f"| {n_envs} | {environment_steps_per_second:.2f} | {policy_steps_per_second:.2f} | "
            f"{physics_steps_per_second:.2f} | {overhead_ms:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These are local measurements for the current Python process and fixture track, not a portable performance guarantee. The full environment currently retains per-car geometry calls for scalar parity. Optimize raycasts/projection only after comparing this baseline with the required training workload.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5, help="timed transitions per environment")
    parser.add_argument("--n-envs", type=int, nargs="+", default=[1, 2, 4, 8], help="batch sizes to measure")
    parser.add_argument("--output", type=Path, default=None, help="optional Markdown report path")
    args = parser.parse_args()
    if args.steps <= 0 or any(value <= 0 for value in args.n_envs):
        parser.error("--steps and --n-envs values must be positive")
    report = benchmark(tuple(args.n_envs), args.steps)
    if args.output is None:
        print(report)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
