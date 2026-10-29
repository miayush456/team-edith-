"""Fixed-rate live ONNX control loop for one Python-authoritative heat.

The live path deliberately does not use replay frames to drive the cars. On
each control tick it obtains fresh policy actions, advances the real vector
environment once, and publishes the resulting state to a read-only client.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Sequence
from uuid import uuid4
from contextlib import ExitStack
import shutil

import numpy as np

from .evaluation import _update_trajectory_hash, validate_evaluation_contract
from .isolated_policy import IsolatedPolicy
from .live_protocol import LIVE_SCHEMA_VERSION, LiveStatePublisher
from .observation import TrainingContract
from .onnx_artifact import FORMAT, RUNTIME_VERSION
from .race import (
    RaceEntrant,
    _artifact_summary,
    _open_output_dir,
    _require_onnx_manifest,
    _validate_entrants,
)
from .submission import load_submission
from .track import TrackGeometry
from .trajectory import engine_code_hash, replay_header
from .vector_env import VectorizedRacingEnv


LIVE_RACE_KIND = "synapse_live_onnx_race"
LIVE_RACE_VERSION = "1.0"
LIVE_PROFILES = {"strict_scored", "trusted_live"}


@dataclass(frozen=True)
class LiveRaceOptions:
    """Explicit real-time policy and pacing choices."""

    control_hz: int = 30
    policy_budget_ms: float = 20.0
    max_steps: int | None = None
    late_policy: str = "strict_scored"
    pace: bool = True
    action_hold_ticks: int | None = None
    startup_timeout: float = 30.0
    action_timeout: float | None = None

    def resolve(self, config: Any) -> dict[str, Any]:
        physics_hz = int(config.tick_hz)
        if type(self.control_hz) is not int or not 1 <= self.control_hz <= physics_hz:
            raise ValueError("control_hz must be an integer from 1 through physics tick_hz")
        if physics_hz % self.control_hz:
            raise ValueError("physics_hz must be divisible by control_hz")
        expected_hold = physics_hz // self.control_hz
        hold = expected_hold if self.action_hold_ticks is None else self.action_hold_ticks
        if type(hold) is not int or hold != expected_hold or hold <= 0:
            raise ValueError("action_hold_ticks must equal physics_hz // control_hz")
        if not math.isfinite(self.policy_budget_ms) or not 0 < self.policy_budget_ms < 1000:
            raise ValueError("policy_budget_ms must be finite and in (0,1000)")
        if self.late_policy not in LIVE_PROFILES:
            raise ValueError(f"late_policy must be one of {sorted(LIVE_PROFILES)}")
        max_steps = int(config.episode.max_steps) if self.max_steps is None else self.max_steps
        if isinstance(max_steps, (bool, np.bool_)) or not isinstance(max_steps, (int, np.integer)):
            raise ValueError("max_steps must be a positive integer")
        if not 0 < int(max_steps) <= config.episode.max_steps:
            raise ValueError("max_steps must be no greater than the environment episode limit")
        startup_timeout = float(self.startup_timeout)
        action_timeout = self.policy_budget_ms / 1000.0 if self.action_timeout is None else float(self.action_timeout)
        if not math.isfinite(startup_timeout) or not 0 < startup_timeout <= 300:
            raise ValueError("startup_timeout must be in (0,300]")
        if not math.isfinite(action_timeout) or not 0 < action_timeout <= 300:
            raise ValueError("action_timeout must be in (0,300]")
        if self.late_policy == "strict_scored" and action_timeout < self.policy_budget_ms / 1000:
            raise ValueError("action_timeout cannot be shorter than policy_budget_ms")
        if action_timeout > self.policy_budget_ms / 1000:
            raise ValueError("action_timeout must not exceed policy_budget_ms")
        return {
            "physics_hz": physics_hz,
            "control_hz": self.control_hz,
            "action_hold_ticks": hold,
            "policy_budget_ms": float(self.policy_budget_ms),
            "max_steps": int(max_steps),
            "late_policy": self.late_policy,
            "pace": bool(self.pace),
            "startup_timeout": startup_timeout,
            "action_timeout": action_timeout,
        }


@dataclass(frozen=True)
class LiveRaceReport:
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.data, allow_nan=False))


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
        return float(ordered[index])

    return {
        "count": len(ordered),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": float(max(ordered)),
    }


def _car_snapshot(
    index: int,
    entrant: RaceEntrant,
    state: dict[str, np.ndarray],
    *,
    status: str,
    off_track: bool,
    events: Sequence[str],
    action: np.ndarray | None,
    action_tick: int | None,
    action_age_ticks: int | None,
    stale_action: bool,
    terminated: bool,
    truncated: bool,
    dnf_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "participant_id": entrant.participant_id,
        "label": entrant.display_label,
        "status": status,
        "x": float(state["x"][index]),
        "y": float(state["y"][index]),
        "heading": float(state["heading"][index]),
        "speed": float(state["speed"][index]),
        "lateral_velocity": float(state["lateral_velocity"][index]),
        "yaw_rate": float(state["yaw_rate"][index]),
        "steer": float(state["steer"][index]),
        "steering_command": None if action is None else float(action[index, 0]),
        "throttle_brake": None if action is None else float(action[index, 1]),
        "off_track": bool(off_track),
        "events": list(events),
        "action_tick": action_tick,
        "action_age_ticks": action_age_ticks,
        "stale_action": bool(stale_action),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "dnf_reason": dnf_reason,
    }


def _hello(
    track: TrackGeometry,
    config: Any,
    entrants: Sequence[RaceEntrant],
    options: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    header = replay_header(track, config, participant_id=entrants[0].participant_id, mode="evaluation")
    return {
        "kind": "synapse.live.hello",
        "schema_version": LIVE_SCHEMA_VERSION,
        "run_id": run_id,
        "physics_hz": options["physics_hz"],
        "control_hz": options["control_hz"],
        "action_hold_ticks": options["action_hold_ticks"],
        "profile": options["late_policy"],
        "engine": header["engine"],
        "car": header["car"],
        "track": header["track"],
        "entrants": [
            {"participant_id": entrant.participant_id, "label": entrant.display_label}
            for entrant in entrants
        ],
    }


def _result_status(env: VectorizedRacingEnv, index: int, info: dict[str, Any], terminated: bool, truncated: bool) -> str:
    if bool(info["crash"][index]):
        return "dnf_crash"
    if bool(info["lap_complete"][index]):
        return "finished"
    if env.distance_since_spawn[index] < -env.config.episode.backward_limit_m:
        return "dnf_backward"
    if truncated and env.stuck_ticks[index] >= env.config.episode.stuck_steps:
        return "dnf_stuck"
    if terminated or truncated:
        return "dnf_timeout"
    return "running"


def run_live_onnx_race(
    entrants: Sequence[RaceEntrant],
    track: TrackGeometry,
    config: Any,
    *,
    publisher: LiveStatePublisher | Any | None = None,
    options: LiveRaceOptions | None = None,
    isolate: bool = False,
    worker_image: str = "synapse-evaluator:v1",
    worker_memory_mb: int = 512,
    output_dir: str | Path | None = None,
    on_started: Callable[[str], None] | None = None,
    on_telemetry: Callable[[dict[str, Any]], None] | None = None,
) -> LiveRaceReport:
    """Run one live heat and publish state after every real physics tick."""

    entrants = _validate_entrants(entrants)
    if not isinstance(track, TrackGeometry):
        raise TypeError("track must be a TrackGeometry")
    options = options or LiveRaceOptions()
    resolved = options.resolve(config)
    output_root: Path | None = None
    if output_dir is not None:
        output_root = _open_output_dir(Path(output_dir), [Path(entrant.artifact_dir) for entrant in entrants])

    run_id = f"live-{uuid4().hex}"
    env = VectorizedRacingEnv(track, config, reward=None, n_envs=len(entrants))
    stack = ExitStack()
    executor = ThreadPoolExecutor(max_workers=len(entrants), thread_name_prefix="synapse-live-policy")
    policies: list[Any] = []
    artifact_summaries: list[dict[str, Any]] = []
    try:
        for entrant in entrants:
            source = Path(entrant.artifact_dir)
            _require_onnx_manifest(source)
            policy = (
                stack.enter_context(IsolatedPolicy(
                    source,
                    image=worker_image,
                    memory_mb=worker_memory_mb,
                    startup_timeout=resolved["startup_timeout"],
                    action_timeout=resolved["action_timeout"],
                ))
                if isolate else load_submission(source)
            )
            if policy.manifest.get("format") != FORMAT:
                raise ValueError("live race accepts only synapse_onnx_policy artifacts")
            if policy.manifest["environment_code_hash"] != engine_code_hash():
                raise ValueError("entrant environment code hash does not match evaluator")
            validate_evaluation_contract(env, policy.contract, policy.normalizer, require_track_match=False)
            policies.append(policy)
            summary = _artifact_summary(entrant, policy)
            summary["artifact_bytes"] = sum(path.stat().st_size for path in source.iterdir() if path.is_file())
            summary["model_bytes"] = (source / "model.onnx").stat().st_size
            artifact_summaries.append(summary)

        # Model/session construction and one representative inference happen
        # before the measured live clock starts.
        warmup = np.zeros((1, env.observation_size), dtype=np.float32)
        for policy in policies:
            result = np.asarray(policy(warmup), dtype=np.float32)
            if result.shape != (1, 2) or not np.isfinite(result).all():
                raise ValueError("ONNX warm-up produced invalid actions")

        hello = _hello(track, config, entrants, resolved, run_id)
        if publisher is not None:
            endpoint = publisher.start(hello)
            if on_started is not None:
                on_started(endpoint)
        observations, _ = env.reset(evaluation=True)
        state = env.state_to_dict()
        active = np.ones(len(entrants), dtype=bool)
        statuses = ["running"] * len(entrants)
        dnf_reasons: list[str | None] = [None] * len(entrants)
        last_action = np.zeros((len(entrants), 2), dtype=np.float64)
        last_action_tick: list[int | None] = [None] * len(entrants)
        stale_counts = [0] * len(entrants)
        off_track = np.zeros(len(entrants), dtype=bool)
        terminal_flags = [(False, False)] * len(entrants)
        event_lists: list[list[str]] = [[] for _ in entrants]
        seq = 0
        sim_tick = 0
        speed_totals = np.zeros(len(entrants), dtype=np.float64)
        latency_values: list[list[float]] = [[] for _ in entrants]
        cycle_latency_values: list[float] = []
        release_jitter_values: list[float] = []
        deadline_misses = [0] * len(entrants)
        inference_failures = [0] * len(entrants)
        trajectory_digests = [hashlib.sha256() for _ in entrants]
        starting_ns = time.monotonic_ns()
        tick_period_ns = round(1_000_000_000 / resolved["physics_hz"])

        def publish_state() -> None:
            nonlocal seq
            seq += 1
            cars = [
                _car_snapshot(
                    index,
                    entrant,
                    state,
                    status=statuses[index],
                    off_track=bool(off_track[index]),
                    events=event_lists[index],
                    action=last_action,
                    action_tick=last_action_tick[index],
                    action_age_ticks=None if last_action_tick[index] is None else sim_tick - last_action_tick[index],
                    stale_action=stale_counts[index] > 0,
                    terminated=terminal_flags[index][0],
                    truncated=terminal_flags[index][1],
                    dnf_reason=dnf_reasons[index],
                )
                for index, entrant in enumerate(entrants)
            ]
            message = {
                "kind": "synapse.live.state",
                "schema_version": LIVE_SCHEMA_VERSION,
                "run_id": run_id,
                "seq": seq,
                "tick": sim_tick,
                "sim_time_s": sim_tick / resolved["physics_hz"],
                "cars": cars,
            }
            if publisher is not None:
                publisher.publish_state(message)

        publish_state()

        def collect_actions(release_ns: int, observations_now: np.ndarray, tick: int) -> np.ndarray:
            actions = last_action.copy()
            indices = [index for index, is_active in enumerate(active) if is_active]
            if not indices:
                return actions
            deadline_ns = release_ns + round(resolved["policy_budget_ms"] * 1_000_000)
            submitted_ns = time.monotonic_ns()
            futures = {
                executor.submit(policies[index], observations_now[index:index + 1].copy()): index
                for index in indices
            }
            pending = set(futures)
            outcomes: dict[int, tuple[np.ndarray | None, str | None, float]] = {}
            while pending:
                remaining_ns = deadline_ns - time.monotonic_ns()
                if remaining_ns <= 0:
                    break
                done, pending = wait(pending, timeout=remaining_ns / 1_000_000_000)
                for future in done:
                    index = futures[future]
                    completed_ns = time.monotonic_ns()
                    latency_ms = (completed_ns - submitted_ns) / 1_000_000.0
                    try:
                        raw = np.asarray(future.result(), dtype=np.float64)
                        if raw.shape != (1, 2) or not np.isfinite(raw).all() or np.any(raw < -1) or np.any(raw > 1):
                            raise ValueError("policy action must be finite shape (1,2) in [-1,1]")
                        outcomes[index] = (raw[0], None if completed_ns <= deadline_ns else "late", latency_ms)
                    except BaseException as exc:
                        outcomes[index] = (None, type(exc).__name__, latency_ms)
            for future in pending:
                future.cancel()
            for index in indices:
                action, error, latency_ms = outcomes.get(index, (None, "deadline", (time.monotonic_ns() - submitted_ns) / 1_000_000.0))
                latency_values[index].append(float(latency_ms))
                if error is None:
                    actions[index] = action
                    last_action[index] = action
                    last_action_tick[index] = tick
                    stale_counts[index] = 0
                    continue
                if error == "late" or error == "deadline":
                    deadline_misses[index] += 1
                else:
                    inference_failures[index] += 1
                if resolved["late_policy"] == "trusted_live" and stale_counts[index] == 0:
                    stale_counts[index] = 1
                    continue
                active[index] = False
                statuses[index] = "dnf_inference"
                dnf_reasons[index] = "late_action" if error in {"late", "deadline"} else "invalid_inference"
                terminal_flags[index] = (True, False)
            cycle_latency_values.append((time.monotonic_ns() - submitted_ns) / 1_000_000.0)
            return actions

        for tick in range(resolved["max_steps"]):
            if not np.any(active):
                break
            release_ns = starting_ns + tick * tick_period_ns
            if resolved["pace"]:
                remaining = release_ns - time.monotonic_ns()
                if remaining > 0:
                    time.sleep(remaining / 1_000_000_000)
            release_jitter_values.append((time.monotonic_ns() - release_ns) / 1_000_000.0)
            if tick % resolved["action_hold_ticks"] == 0:
                actions = collect_actions(release_ns, observations, tick)
            else:
                actions = last_action.copy()

            if not np.any(active):
                publish_state()
                break
            previous_observations = observations
            observations, _rewards, terminated, truncated, info = env.step(actions, active_mask=active)
            sim_tick += 1
            state = info["state"]
            for index in range(len(entrants)):
                if not active[index] and statuses[index] == "dnf_inference":
                    continue
                if active[index]:
                    _update_trajectory_hash(
                        trajectory_digests[index],
                        previous_observations[index],
                        actions[index],
                        observations[index],
                        0.0,
                        bool(terminated[index]),
                        bool(truncated[index]),
                        {name: np.asarray([values[index]]) for name, values in state.items()},
                        bool(info["off_track"][index]),
                    )
                    speed_totals[index] += float(state["speed"][index])
                    off_track[index] = bool(info["off_track"][index])
                    event_lists[index] = list(info["events"][index])
                    terminal_flags[index] = (bool(terminated[index]), bool(truncated[index]))
                    if terminated[index] or truncated[index]:
                        statuses[index] = _result_status(env, index, info, bool(terminated[index]), bool(truncated[index]))
                        active[index] = False
            publish_state()
            if on_telemetry is not None:
                on_telemetry({
                    "kind": "live_tick",
                    "tick": sim_tick,
                    "sim_time_s": sim_tick / resolved["physics_hz"],
                    "entrants": [
                        {
                            "participant_id": entrants[index].participant_id,
                            "label": entrants[index].display_label,
                            "status": statuses[index],
                            "latency_ms": latency_values[index][-1] if latency_values[index] else None,
                            "deadline_misses": deadline_misses[index],
                            "inference_failures": inference_failures[index],
                            "model_mb": round(artifact_summaries[index]["model_bytes"] / (1024 * 1024), 3),
                        }
                        for index in range(len(entrants))
                    ],
                })

        for index in range(len(entrants)):
            if statuses[index] == "running":
                statuses[index] = "dnf_timeout"
                dnf_reasons[index] = "step_limit"
                terminal_flags[index] = (False, True)
        if any(status == "dnf_timeout" and reason == "step_limit" for status, reason in zip(statuses, dnf_reasons)):
            publish_state()

        results = []
        for index, entrant in enumerate(entrants):
            steps = int(env.steps[index])
            status = statuses[index]
            results.append({
                "participant_id": entrant.participant_id,
                "label": entrant.display_label,
                "artifact_sha256": artifact_summaries[index]["artifact_sha256"],
                "status": status,
                "dnf_reason": dnf_reasons[index],
                "steps": steps,
                "lap_time_s": steps / resolved["physics_hz"] if status == "finished" else None,
                "mean_speed_mps": float(speed_totals[index] / steps) if steps else 0.0,
                "trajectory_hash": trajectory_digests[index].hexdigest(),
                "deadline_misses": deadline_misses[index],
                "inference_failures": inference_failures[index],
                "latency_ms": _latency_summary(latency_values[index]),
            })
        end = {
            "kind": "synapse.live.end",
            "schema_version": LIVE_SCHEMA_VERSION,
            "run_id": run_id,
            "seq": seq + 1,
            "tick": sim_tick,
            "sim_time_s": sim_tick / resolved["physics_hz"],
            "results": results,
        }
        if publisher is not None:
            publisher.publish_end(end)
        data = {
            "kind": LIVE_RACE_KIND,
            "schema_version": LIVE_RACE_VERSION,
            "run_id": run_id,
            "engine_code_hash": engine_code_hash(),
            "config_hash": TrainingContract.config_hash_for(config),
            "track_id": track.track_id,
            "track_hash": track.content_hash,
            "race_type": "independent_live_ghosts",
            "tick_rate": {
                "physics_hz": resolved["physics_hz"],
                "control_hz": resolved["control_hz"],
                "action_hold_ticks": resolved["action_hold_ticks"],
            },
            "deadline": {
                "policy_budget_ms": resolved["policy_budget_ms"],
                "late_policy": resolved["late_policy"],
                "pace": resolved["pace"],
            },
            "inference": {
                "mode": "isolated_docker" if isolate else "trusted_local_preview",
                "runtime": f"onnxruntime=={RUNTIME_VERSION}",
                "profile": "static_vector_cpu_opset17_v1",
                **({"image": worker_image, "memory_mb": worker_memory_mb, "cpus": 1} if isolate else {}),
            },
            "timing": {
                "action_cycle_latency_ms": _latency_summary(cycle_latency_values),
                "release_jitter_ms": _latency_summary([abs(value) for value in release_jitter_values]),
                "scheduled_period_ms": 1000.0 / resolved["physics_hz"],
            },
            "entrants": artifact_summaries,
            "results": results,
            "ranking_policy": None,
        }
        report = LiveRaceReport(data)
        if output_root is not None:
            (output_root / "live-race.json").write_text(
                json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8",
            )
        return report
    except BaseException:
        if output_root is not None:
            shutil.rmtree(output_root, ignore_errors=True)
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        stack.close()


__all__ = ["LiveRaceOptions", "LiveRaceReport", "run_live_onnx_race"]
