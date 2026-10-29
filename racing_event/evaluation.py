"""Deterministic, low-memory policy evaluation for the racing engine.

Evaluation uses the same Python-authoritative environment as training, but it
always starts at the fixed start line and asks the policy for its deterministic
mean action.  The evaluator keeps only running counters and a streaming
trajectory digest.  It does not retain a frame list, which keeps a long lap
small on memory-constrained machines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Callable

import numpy as np

from .observation import RunningObservationNormalizer, TrainingContract
from .evaluation_audit import EvaluationAudit
from .trajectory import engine_code_hash, replay_frame
from .vector_env import VectorizedRacingEnv


EvaluationPolicy = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class EvaluationResult:
    """Outcome and compact metrics for one deterministic evaluation run."""

    status: str
    steps: int
    lap_time_s: float | None
    total_reward: float
    mean_speed_mps: float
    off_track_steps: int
    crash: bool
    final_distance_m: float
    trajectory_hash: str
    track_id: str
    track_hash: str
    config_hash: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "steps": self.steps,
            "lap_time_s": self.lap_time_s,
            "total_reward": self.total_reward,
            "mean_speed_mps": self.mean_speed_mps,
            "off_track_steps": self.off_track_steps,
            "crash": self.crash,
            "final_distance_m": self.final_distance_m,
            "trajectory_hash": self.trajectory_hash,
            "track_id": self.track_id,
            "track_hash": self.track_hash,
            "config_hash": self.config_hash,
            "diagnostics": self.diagnostics,
        }


def validate_evaluation_contract(
    env: VectorizedRacingEnv,
    contract: TrainingContract,
    normalizer: RunningObservationNormalizer | None = None,
    *,
    require_track_match: bool = True,
) -> None:
    """Reject an artifact that does not belong to ``env``.

    Track and config hashes are checked before the first transition.  This is
    intentionally strict: silently evaluating a policy against changed
    geometry or physics would make the resulting time incomparable.
    """

    if not isinstance(contract, TrainingContract):
        raise TypeError("contract must be a TrainingContract")
    if contract.observation.names != env.observation_schema.names:
        raise ValueError("saved observation schema does not match the environment")
    if contract.observation.size != env.observation_size:
        raise ValueError("saved observation size does not match the environment")
    saved_schema, live_schema = contract.observation, env.observation_schema
    if (saved_schema.version, saved_schema.dtype, saved_schema.normalization) != (
        live_schema.version, live_schema.dtype, live_schema.normalization,
    ) or any(
        (saved.low, saved.high, saved.unit) != (live.low, live.high, live.unit)
        for saved, live in zip(saved_schema.fields, live_schema.fields)
    ):
        raise ValueError("saved observation semantics do not match the environment")
    if contract.observation.ray_angles_deg and contract.observation.ray_angles_deg != tuple(env.config.sensors.ray_angles_deg):
        raise ValueError("saved ray angles do not match the environment")
    if contract.config_hash != env.training_contract.config_hash:
        raise ValueError("saved config hash does not match the environment")
    if not contract.track_ids or len(contract.track_ids) != len(contract.track_hashes):
        raise ValueError("saved training track IDs and hashes must be paired")
    if require_track_match:
        if contract.track_ids != tuple(track.track_id for track in env.tracks):
            raise ValueError("saved track IDs do not match the environment")
        if contract.track_hashes != tuple(track.content_hash for track in env.tracks):
            raise ValueError("saved track hashes do not match the environment")
    if contract.action_distribution not in {"participant_defined", "tanh_normal"}:
        raise ValueError("saved contract uses an unsupported action distribution")
    if not contract.deterministic_evaluation or contract.evaluation_spawn != "fixed_start_line_standing_zero_noise":
        raise ValueError("saved contract does not describe deterministic evaluation")
    if normalizer is not None and normalizer.size != env.observation_size:
        raise ValueError("saved normalizer size does not match the environment")
    if not require_track_match and normalizer is not None and normalizer.training:
        raise ValueError("saved normalizer must be frozen for evaluation")


def evaluate_participant_artifact(
    env: VectorizedRacingEnv,
    artifact: Any,
    *,
    max_steps: int | None = None,
    frame_sink: Callable[[dict[str, Any]], None] | None = None,
) -> EvaluationResult:
    """Evaluate a validated model artifact without executing participant code."""

    from .model_artifact import ParticipantModelArtifact
    from .onnx_artifact import OnnxPolicyArtifact
    from .isolated_policy import IsolatedPolicy

    if not isinstance(artifact, (ParticipantModelArtifact, OnnxPolicyArtifact, IsolatedPolicy)):
        raise TypeError("artifact must be a validated participant model artifact")
    if env.reward_config is not None:
        raise ValueError("participant artifact evaluation requires an environment with reward=None")
    if artifact.manifest["environment_code_hash"] != engine_code_hash():
        raise ValueError("artifact.environment: environment code hash does not match evaluator")
    validate_evaluation_contract(
        env,
        artifact.contract,
        artifact.normalizer,
        require_track_match=False,
    )
    return evaluate_policy(env, artifact, max_steps=max_steps, frame_sink=frame_sink)


def _hash_array(digest: "hashlib._Hash", name: str, value: Any, dtype: str) -> None:
    """Add a shape-delimited, little-endian array to a streaming digest."""

    array = np.asarray(value, dtype=dtype)
    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
    digest.update(array.tobytes())


def _update_trajectory_hash(
    digest: "hashlib._Hash",
    observation: np.ndarray,
    action: np.ndarray,
    next_observation: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    state: dict[str, np.ndarray],
    off_track: bool,
) -> None:
    """Hash one transition without storing it for later."""

    _hash_array(digest, "observation", observation, "<f4")
    _hash_array(digest, "action", action, "<f8")
    _hash_array(digest, "next_observation", next_observation, "<f4")
    _hash_array(digest, "reward", np.asarray([reward]), "<f8")
    _hash_array(digest, "terminated", np.asarray([terminated]), "<u1")
    _hash_array(digest, "truncated", np.asarray([truncated]), "<u1")
    for name in ("x", "y", "heading", "speed", "lateral_velocity", "yaw_rate", "steer"):
        _hash_array(digest, name, np.asarray([state[name][0]]), "<f8")
    _hash_array(digest, "off_track", np.asarray([off_track]), "<u1")


def _classify_outcome(
    env: VectorizedRacingEnv,
    info: dict[str, Any],
    truncated: bool,
) -> str:
    """Map engine flags to the evaluation result vocabulary."""

    if bool(info["crash"][0]):
        return "dnf_crash"
    if bool(info["lap_complete"][0]):
        return "finished"
    if env.distance_since_spawn[0] < -env.config.episode.backward_limit_m:
        return "dnf_backward"
    if truncated and env.stuck_ticks[0] >= env.config.episode.stuck_steps:
        return "dnf_stuck"
    if truncated:
        return "dnf_timeout"
    return "dnf_timeout"


def evaluate_policy(
    env: VectorizedRacingEnv,
    policy: EvaluationPolicy,
    *,
    normalizer: RunningObservationNormalizer | None = None,
    max_steps: int | None = None,
    frame_sink: Callable[[dict[str, Any]], None] | None = None,
) -> EvaluationResult:
    """Run one deterministic fixed-start evaluation with bounded memory.

    ``env`` must have exactly one lane.  The policy is called with a batch of
    one normalized observation and must return a finite ``(1, 2)`` action.  A
    normalizer is always invoked with ``training=False`` so evaluation cannot
    change its saved statistics, even if the loaded object was not frozen.
    An optional ``frame_sink`` receives independent snapshots at tick zero and
    after each step. Its return value is ignored; renderers cannot send actions
    or decisions back through this boundary.
    """

    if not isinstance(env, VectorizedRacingEnv) or env.n_envs != 1:
        raise ValueError("evaluation requires a VectorizedRacingEnv with n_envs=1")
    if len(env.tracks) != 1:
        raise ValueError("evaluation requires exactly one explicit track; evaluate track pools separately")
    if not callable(policy):
        raise TypeError("policy must be callable")
    if max_steps is None:
        max_steps = env.config.episode.max_steps
    if isinstance(max_steps, (bool, np.bool_)) or not isinstance(max_steps, (int, np.integer)) or int(max_steps) <= 0:
        raise ValueError("max_steps must be a positive integer")
    max_steps = int(max_steps)
    if normalizer is not None and normalizer.size != env.observation_size:
        raise ValueError("normalizer size does not match the environment")
    if frame_sink is not None and not callable(frame_sink):
        raise TypeError("frame_sink must be callable")

    observation, _ = env.reset(evaluation=True)
    track = env.tracks[0]
    audit = EvaluationAudit(track.length, env.config.dt)
    if frame_sink is not None:
        track = env.tracks[int(env.track_indices[0])]
        edge = track.edge_status(
            np.asarray([env.state.x[0], env.state.y[0]]), float(env.state.heading[0]),
            env.config.car.length_m, env.config.car.width_m,
        )
        frame_sink(replay_frame(
            0, {name: float(values[0]) for name, values in env.state_to_dict().items()},
            off_track=bool(edge.off_track),
        ))
    digest = hashlib.sha256()
    steps = 0
    total_reward = 0.0
    speed_total = 0.0
    off_track_steps = 0
    status = "dnf_timeout"
    crash = False
    final_distance = 0.0

    while steps < max_steps and not bool(env.done[0]):
        policy_observation = (
            observation
            if normalizer is None
            else normalizer.transform(observation, training=False)
        )
        raw_action = np.asarray(policy(policy_observation.copy()), dtype=np.float64)
        if raw_action.shape != (1, env.action_size):
            raise ValueError(f"policy must return shape (1, {env.action_size})")
        if not np.all(np.isfinite(raw_action)):
            raise ValueError("policy actions must contain only finite values")

        next_observation, rewards, terminated, truncated, info = env.step(raw_action)
        step_terminated = bool(terminated[0])
        step_truncated = bool(truncated[0])
        state = info["state"]
        _update_trajectory_hash(
            digest,
            observation[0],
            raw_action[0],
            next_observation[0],
            float(rewards[0]),
            step_terminated,
            step_truncated,
            state,
            bool(info["off_track"][0]),
        )
        steps += 1
        total_reward += float(rewards[0])
        speed_total += float(state["speed"][0])
        off_track_steps += int(bool(info["off_track"][0]))
        crash = crash or bool(info["crash"][0])
        final_distance = float(info["distance_since_spawn"][0])
        audit.update(
            previous_distance=float(info["previous_distance_since_spawn"][0]),
            distance=final_distance,
            off_track=bool(info["off_track"][0]),
            boundary_excess=float(info["boundary_excess_m"][0]),
            coverage_mask=int(info["sector_coverage_mask"][0]),
            raw_action=raw_action[0],
        )
        if frame_sink is not None:
            frame_sink(replay_frame(
                steps, {name: float(values[0]) for name, values in state.items()},
                steering_command=float(env.state.previous_steering_command[0]),
                throttle_brake=float(env.state.previous_longitudinal_command[0]),
                off_track=bool(info["off_track"][0]), events=info["events"][0],
                terminated=step_terminated, truncated=step_truncated,
            ))
        observation = next_observation

        if step_terminated or step_truncated:
            status = _classify_outcome(
                env,
                info,
                step_truncated,
            )

    if not env.done[0] and steps >= max_steps:
        status = "dnf_timeout"

    lap_time_s = float(steps * env.config.dt) if status == "finished" else None
    track = env.tracks[int(env.track_indices[0])]
    return EvaluationResult(
        status=status,
        steps=steps,
        lap_time_s=lap_time_s,
        total_reward=float(total_reward),
        mean_speed_mps=float(speed_total / steps) if steps else 0.0,
        off_track_steps=off_track_steps,
        crash=crash,
        final_distance_m=final_distance,
        trajectory_hash=digest.hexdigest(),
        track_id=track.track_id,
        track_hash=track.content_hash,
        config_hash=env.training_contract.config_hash,
        diagnostics=audit.snapshot(),
    )


__all__ = [
    "EvaluationResult",
    "evaluate_policy",
    "evaluate_participant_artifact",
    "validate_evaluation_contract",
]
