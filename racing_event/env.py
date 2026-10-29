"""Gymnasium-style single-car racing environment without a Gym dependency."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import SimulationConfig
from .physics import CarState, step_car
from .reward import RewardConfig, SIGNAL_NAMES
from .signals import compute_signals
from .track import TrackGeometry
from .observation import TrainingContract, build_observation_schema
from .telemetry import DEFAULT_SECTOR_COUNT, sector_bit, sector_coverage_count, sector_index


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


_DEFAULT_REWARD = object()


class RacingEnv:
    """A deterministic environment with reset/step_raw semantics.

    Observation length is fixed at ``10 + len(rays) + len(lookaheads)``:
    self state (4), track relation (2), controls (3), raycasts, lookaheads,
    and local width (1). All values are normalized or clipped.
    """

    def __init__(
        self,
        track: TrackGeometry,
        config: SimulationConfig,
        reward: RewardConfig | None | object = _DEFAULT_REWARD,
    ):
        config.validate()
        self.track = track
        self.config = config
        # Omitting ``reward`` preserves the old scalar API for the reference
        # baseline. Passing ``reward=None`` creates an environment-only
        # instance; participants then use ``step_raw`` and own the reward.
        self.reward_config = RewardConfig() if reward is _DEFAULT_REWARD else reward
        if self.reward_config is not None and not isinstance(self.reward_config, RewardConfig):
            raise TypeError("reward must be a RewardConfig or None")
        self.observation_schema = build_observation_schema(config)
        self.training_contract = TrainingContract.for_config(
            config,
            track_ids=(track.track_id,),
            track_hashes=(track.content_hash,),
        )
        self.rng = np.random.default_rng()
        self.state = CarState(float(self.track.centerline[0, 0]), float(self.track.centerline[0, 1]), 0.0)
        self.previous_s = 0.0
        self.distance_since_spawn = 0.0
        self.steps = 0
        self.stuck_ticks = 0
        self.sector_index = 0
        self.sector_coverage_mask = 0
        self._done = False
        self.last_info: dict[str, Any] = {}

    @classmethod
    def from_files(cls, track_path: str | Path, config_path: str | Path, reward_path: str | Path | None = None) -> "RacingEnv":
        reward = RewardConfig.from_file(reward_path) if reward_path is not None else None
        return cls(TrackGeometry.from_file(track_path), SimulationConfig.from_file(config_path), reward)

    @property
    def observation_size(self) -> int:
        return self.observation_schema.size

    def reset(
        self,
        seed: int | None = None,
        *,
        evaluation: bool = False,
        spawn_s: float | None = None,
        lateral_offset: float | None = None,
        initial_speed: float | None = None,
        heading_error: float | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if evaluation and any(
            value is not None
            for value in (spawn_s, lateral_offset, initial_speed, heading_error)
        ):
            raise ValueError(
                "evaluation reset owns spawn_s, lateral_offset, initial_speed, and heading_error"
            )
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if spawn_s is None:
            spawn_s = self.track.start_line_s if evaluation else float(self.rng.uniform(0.0, self.track.length))
        if not np.isfinite(spawn_s):
            raise ValueError("spawn_s must be finite")
        point, track_heading, _ = self.track.point_at_s(spawn_s)
        left_width, right_width = self.track.width_at_s(spawn_s)
        if lateral_offset is None:
            lateral_offset = 0.0 if evaluation else float(self.rng.uniform(-0.3 * right_width, 0.3 * left_width))
        if not np.isfinite(lateral_offset):
            raise ValueError("lateral_offset must be finite")
        if initial_speed is None:
            if evaluation:
                initial_speed = 0.0
            else:
                standing_start = self.rng.random() < self.config.episode.standing_start_probability
                random_speed = self.rng.uniform(0.0, 0.5 * self.config.car.max_speed_mps)
                initial_speed = 0.0 if standing_start else float(random_speed)
        if not np.isfinite(initial_speed) or initial_speed < 0.0:
            raise ValueError("initial_speed must be finite and non-negative")
        if heading_error is None:
            heading_error = 0.0 if evaluation else float(self.rng.uniform(-0.08, 0.08))
        if not np.isfinite(heading_error):
            raise ValueError("heading_error must be finite")
        normal_left = np.array([-math.sin(track_heading), math.cos(track_heading)], dtype=np.float64)
        start = point + lateral_offset * normal_left
        self.state = CarState(float(start[0]), float(start[1]), track_heading + heading_error, max(0.0, initial_speed))
        self.previous_s = float(spawn_s % self.track.length)
        self.distance_since_spawn = 0.0
        self.steps = 0
        self.stuck_ticks = 0
        self.sector_index = 0
        # The spawn sector is reported, but coverage counts only sectors the
        # car reaches on a transition. A reset/teleport is not driven coverage.
        self.sector_coverage_mask = 0
        self._done = False
        observation = self._observation()
        info = {
            "track_id": self.track.track_id,
            "spawn_s": self.previous_s,
            "signals": {name: 0.0 for name in SIGNAL_NAMES},
            "evaluation": bool(evaluation),
            "sector_index": self.sector_index,
            "sector_count": DEFAULT_SECTOR_COUNT,
            "sector_coverage_mask": self.sector_coverage_mask,
            "sector_coverage_count": sector_coverage_count(self.sector_coverage_mask),
            "boundary_excess_m": 0.0,
            "off_track_progress_m": 0.0,
        }
        self.last_info = info
        return observation, info

    def step(self, action: np.ndarray | list[float] | tuple[float, float]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done or self.steps >= self.config.episode.max_steps:
            raise RuntimeError("episode has ended; call reset()")
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_array.size != 2:
            raise ValueError("action must contain steering and throttle/brake")
        if not np.all(np.isfinite(action_array)):
            raise ValueError("action must contain only finite values")
        action_array = np.clip(action_array, -1.0, 1.0)
        previous_state = self.state.copy()
        previous_projection = self.track.project(np.array([previous_state.x, previous_state.y]))
        previous_s = previous_projection.s
        previous_steering = previous_state.previous_steering_command
        previous_distance = self.distance_since_spawn
        previous_off_track = self.track.edge_status(
            np.array([previous_state.x, previous_state.y]), previous_state.heading, self.config.car.length_m, self.config.car.width_m
        ).off_track
        grip_factor = self.config.car.runoff_grip_factor if previous_off_track else 1.0
        self.state = step_car(self.state, action_array, self.config.car, self.config.dt, grip_factor)
        projection = self.track.project(np.array([self.state.x, self.state.y]))
        delta_s = projection.s - previous_s
        if delta_s < -self.track.length / 2.0:
            delta_s += self.track.length
        elif delta_s > self.track.length / 2.0:
            delta_s -= self.track.length
        self.distance_since_spawn += delta_s
        edge = self.track.edge_status(
            np.array([self.state.x, self.state.y]), self.state.heading, self.config.car.length_m, self.config.car.width_m
        )
        lap_complete = self.distance_since_spawn >= self.track.length
        crash = edge.barrier_contact
        if abs(delta_s) < 0.01:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = 0
        self.steps += 1
        backward_too_far = self.distance_since_spawn < -self.config.episode.backward_limit_m
        terminated = bool(crash or lap_complete or backward_too_far)
        stuck = bool(self.stuck_ticks >= self.config.episode.stuck_steps)
        truncated = bool(not terminated and (self.steps >= self.config.episode.max_steps or stuck))
        self._done = terminated or truncated
        self.sector_index = sector_index(self.distance_since_spawn, self.track.length)
        checkpoint_bit = sector_bit(self.sector_index)
        checkpoint = not (self.sector_coverage_mask & checkpoint_bit)
        self.sector_coverage_mask |= checkpoint_bit
        boundary_excess_m = max(edge.maximum_left_excess, edge.maximum_right_excess)
        off_track_progress_m = max(delta_s, 0.0) if edge.off_track else 0.0
        events: list[str] = []
        if edge.off_track and not previous_off_track:
            events.append("off_track_start")
        if not edge.off_track and previous_off_track:
            events.append("off_track_end")
        if crash:
            events.append("barrier_contact")
        if lap_complete:
            events.append("lap_complete")
        if checkpoint:
            events.append("checkpoint")
        curvature_ahead = self.track.curvature_at_s(projection.s + max(self.config.sensors.lookahead_distances_m[0], self.state.speed * 0.75))
        signals = compute_signals(
            track=self.track,
            car=self.config.car,
            dt=self.config.dt,
            previous_s=previous_s,
            current_s=projection.s,
            speed=self.state.speed,
            previous_steering_command=previous_steering,
            steering_command=float(action_array[0]),
            curvature_ahead=curvature_ahead,
            off_track=edge.off_track,
            crash=crash,
            stuck=stuck,
            lap_complete=lap_complete,
            checkpoint=checkpoint,
        )
        reward = 0.0 if self.reward_config is None else self.reward_config.calculate(signals)
        observation = self._observation()
        info = {
            "track_id": self.track.track_id,
            "signals": signals,
            "events": events,
            "state": self.state_to_dict(),
            "s": projection.s,
            "distance_since_spawn": self.distance_since_spawn,
            "off_track": edge.off_track,
            "crash": crash,
            "lap_complete": lap_complete,
            "step": self.steps,
            "previous_distance_since_spawn": previous_distance,
            "sector_index": self.sector_index,
            "sector_count": DEFAULT_SECTOR_COUNT,
            "sector_coverage_mask": self.sector_coverage_mask,
            "sector_coverage_count": sector_coverage_count(self.sector_coverage_mask),
            "boundary_excess_m": boundary_excess_m,
            "off_track_progress_m": off_track_progress_m,
        }
        self.previous_s = projection.s
        self.last_info = info
        return observation, reward, terminated, truncated, info

    def step_raw(
        self, action: np.ndarray | list[float] | tuple[float, float]
    ) -> tuple[np.ndarray, bool, bool, dict[str, Any]]:
        """Advance one transition and return raw signals without reward.

        ``step`` remains available as a compatibility adapter for the local
        reference baseline. Participant training should use this method and
        calculate reward outside the SYNAPSE-owned environment.
        """

        observation, _legacy_reward, terminated, truncated, info = self.step(action)
        return observation, terminated, truncated, info

    def _observation(self) -> np.ndarray:
        position = np.array([self.state.x, self.state.y], dtype=np.float64)
        projection = self.track.project(position)
        left_width, right_width = self.track.width_at_s(projection.s)
        heading_error = self.track.observation_heading_error(self.state.heading, projection)
        ray_values = []
        for relative_angle in self.config.sensors.ray_angles_deg:
            ray_values.append(self.track.raycast(position, self.state.heading + math.radians(relative_angle), self.config.sensors.ray_max_range_m) / self.config.sensors.ray_max_range_m)
        lookahead_values = []
        for distance in self.config.sensors.lookahead_distances_m:
            lookahead_values.append(np.clip(self.track.curvature_at_s(projection.s + distance) * 100.0, -1.0, 1.0))
        values = [
            np.clip(self.state.speed / self.config.car.max_speed_mps, 0.0, 1.0),
            np.clip(self.state.longitudinal_velocity / self.config.car.max_speed_mps, -1.0, 1.0),
            np.clip(self.state.lateral_velocity / self.config.car.max_speed_mps, -1.0, 1.0),
            np.clip(self.state.yaw_rate / 5.0, -1.0, 1.0),
            np.clip(projection.lateral_offset / max(left_width, right_width), -2.0, 2.0) / 2.0,
            heading_error / math.pi,
            self.state.previous_steering_command,
            self.state.previous_longitudinal_command,
            self.state.steering_angle / self.config.car.max_steering_rad,
            *ray_values,
            *lookahead_values,
            np.clip(min(left_width, right_width) / (5.0 * self.config.car.width_m), 0.0, 1.0),
        ]
        observation = np.asarray(values, dtype=np.float32)
        if len(observation) != self.observation_size:
            raise RuntimeError(f"observation size changed: expected {self.observation_size}, got {len(observation)}")
        self.observation_schema.validate(observation)
        return observation

    def state_to_dict(self) -> dict[str, float]:
        return {
            "x": self.state.x,
            "y": self.state.y,
            "heading": self.state.heading,
            "speed": self.state.speed,
            "lateral_velocity": self.state.lateral_velocity,
            "yaw_rate": self.state.yaw_rate,
            "steer": self.state.steering_angle,
        }
