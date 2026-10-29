"""Batched NumPy racing environment.

The physics transition is vectorized over cars while the existing
``TrackGeometry`` implementation remains the single geometry authority.  The
small per-car geometry loops are intentional in this first slice: they make
scalar/vector parity easy to verify and leave ray/projection optimization to
the measured E1 benchmark.
"""

from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .config import SimulationConfig
from .env import _wrap_angle
from .observation import ObservationSchema, TrainingContract, build_observation_schema
from .physics import CarStateBatch, step_car_batch
from .reward import RewardConfig, SIGNAL_NAMES
from .signals import compute_signals
from .track import EdgeStatus, Projection, TrackGeometry
from .telemetry import DEFAULT_SECTOR_COUNT, sector_bit, sector_coverage_count, sector_index


_DEFAULT_REWARD = object()


class VectorizedRacingEnv:
    """A fixed-size batch of independent racing environments.

    ``reset`` returns ``(n_envs, observation_size)`` observations.  ``step_raw``
    accepts an ``(n_envs, 2)`` action array and returns raw signals and
    termination flags; the legacy ``step`` additionally returns reference
    weighted rewards. Tracks can be supplied as a pool; each environment
    samples one track at reset and keeps it until the next reset.  A reset can
    target selected lanes, and a step can advance only lanes selected by a
    boolean mask.  Explicit spawn arguments remain scalar or full-batch
    ``(n_envs,)`` values even when only some lanes are selected; selected
    lanes use the values at their corresponding batch positions.
    """

    def __init__(
        self,
        tracks: TrackGeometry | Sequence[TrackGeometry],
        config: SimulationConfig,
        reward: RewardConfig | None | object = _DEFAULT_REWARD,
        *,
        n_envs: int,
    ):
        config.validate()
        if int(n_envs) <= 0:
            raise ValueError("n_envs must be positive")
        if isinstance(tracks, TrackGeometry):
            track_list = (tracks,)
        else:
            track_list = tuple(tracks)
        if not track_list or any(not isinstance(track, TrackGeometry) for track in track_list):
            raise ValueError("tracks must contain at least one TrackGeometry")
        self.tracks = track_list
        self.config = config
        # Omitting ``reward`` preserves the old vector ``step`` API for the
        # reference baseline. Passing ``reward=None`` exposes raw signals so
        # participant code can own reward calculation.
        self.reward_config = RewardConfig() if reward is _DEFAULT_REWARD else reward
        if self.reward_config is not None and not isinstance(self.reward_config, RewardConfig):
            raise TypeError("reward must be a RewardConfig or None")
        self.n_envs = int(n_envs)
        self.observation_schema: ObservationSchema = build_observation_schema(config)
        self.training_contract = TrainingContract.for_config(
            config,
            track_ids=tuple(track.track_id for track in self.tracks),
            track_hashes=tuple(track.content_hash for track in self.tracks),
        )
        self.rng = np.random.default_rng()
        self.track_indices = np.zeros(self.n_envs, dtype=np.int64)
        self.state = self._blank_state()
        self.previous_s = np.zeros(self.n_envs, dtype=np.float64)
        self.distance_since_spawn = np.zeros(self.n_envs, dtype=np.float64)
        self.steps = np.zeros(self.n_envs, dtype=np.int64)
        self.stuck_ticks = np.zeros(self.n_envs, dtype=np.int64)
        self.sector_index = np.zeros(self.n_envs, dtype=np.int64)
        self.sector_coverage_mask = np.zeros(self.n_envs, dtype=np.uint64)
        self._done = np.zeros(self.n_envs, dtype=bool)
        self.last_info: dict[str, Any] = {}

    @classmethod
    def from_files(
        cls,
        track_paths: str | Path | Iterable[str | Path],
        config_path: str | Path,
        reward_path: str | Path | None = None,
        *,
        n_envs: int,
    ) -> "VectorizedRacingEnv":
        if isinstance(track_paths, (str, Path)):
            paths = (track_paths,)
        else:
            paths = tuple(track_paths)
        if not paths:
            raise ValueError("at least one track path is required")
        reward = RewardConfig.from_file(reward_path) if reward_path is not None else None
        return cls(tuple(TrackGeometry.from_file(path) for path in paths), SimulationConfig.from_file(config_path), reward, n_envs=n_envs)

    @property
    def observation_size(self) -> int:
        return self.observation_schema.size

    @property
    def action_size(self) -> int:
        return 2

    def _blank_state(self) -> CarStateBatch:
        zeros = lambda: np.zeros(self.n_envs, dtype=np.float64)
        return CarStateBatch(
            x=zeros(),
            y=zeros(),
            heading=zeros(),
            speed=zeros(),
            lateral_velocity=zeros(),
            yaw_rate=zeros(),
            steering_angle=zeros(),
            previous_steering_command=zeros(),
            previous_longitudinal_command=zeros(),
        )

    def _per_env_values(self, value: float | Iterable[float] | None, name: str) -> np.ndarray | None:
        if value is None:
            return None
        values = np.asarray(value, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(self.n_envs, float(values), dtype=np.float64)
        if values.shape != (self.n_envs,):
            raise ValueError(f"{name} must be a scalar or have shape ({self.n_envs},)")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
        return values

    def _validate_indices(self, indices: Iterable[int] | None) -> np.ndarray:
        if indices is None:
            return np.arange(self.n_envs, dtype=np.intp)
        try:
            values = list(indices)
        except TypeError as exc:
            raise ValueError("indices must be an iterable of unique integer lane indices") from exc
        if not values and isinstance(indices, np.ndarray) and indices.dtype.kind not in "iu":
            raise ValueError("indices must contain only integer values")
        validated: list[int] = []
        seen: set[int] = set()
        for value in values:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError("indices must contain only integer values, not bools or nonintegral values")
            index = int(value)
            if index < 0 or index >= self.n_envs:
                raise ValueError(f"index {index} is out of range for {self.n_envs} environments")
            if index in seen:
                raise ValueError(f"indices must be unique; {index} was repeated")
            seen.add(index)
            validated.append(index)
        return np.asarray(validated, dtype=np.intp)

    def _validate_active_mask(self, active_mask: np.ndarray | None) -> np.ndarray:
        if active_mask is None:
            return np.ones(self.n_envs, dtype=bool)
        values = np.asarray(active_mask)
        if values.shape != (self.n_envs,) or values.dtype != np.dtype(bool):
            raise ValueError(f"active_mask must be a boolean array with shape ({self.n_envs},)")
        return values.copy()

    def reset(
        self,
        seed: int | None = None,
        *,
        evaluation: bool = False,
        spawn_s: float | Iterable[float] | None = None,
        lateral_offset: float | Iterable[float] | None = None,
        initial_speed: float | Iterable[float] | None = None,
        heading_error: float | Iterable[float] | None = None,
        indices: Iterable[int] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset all lanes or only the unique lanes listed in ``indices``.

        ``spawn_s``, ``lateral_offset``, ``initial_speed``, and
        ``heading_error`` accept a scalar or a full ``(n_envs,)`` batch.  For
        a selective reset, the selected lanes use the corresponding entries;
        nonselected entries are ignored after full-batch validation.
        """

        if evaluation and any(
            value is not None
            for value in (spawn_s, lateral_offset, initial_speed, heading_error)
        ):
            raise ValueError(
                "evaluation reset owns spawn_s, lateral_offset, initial_speed, and heading_error"
            )

        selected = self._validate_indices(indices)
        if selected.size == 0:
            observation = self._observations()
            info = {
                "track_id": tuple(self.tracks[int(index)].track_id for index in self.track_indices),
                "spawn_s": self.previous_s.copy(),
                "signals": {name: np.zeros(self.n_envs, dtype=np.float64) for name in SIGNAL_NAMES},
                "evaluation": bool(evaluation),
                "sector_index": self.sector_index.copy(),
                "sector_count": DEFAULT_SECTOR_COUNT,
                "sector_coverage_mask": self.sector_coverage_mask.copy(),
                "sector_coverage_count": np.asarray(
                    [sector_coverage_count(int(mask)) for mask in self.sector_coverage_mask], dtype=np.int64
                ),
                "boundary_excess_m": np.zeros(self.n_envs, dtype=np.float64),
                "off_track_progress_m": np.zeros(self.n_envs, dtype=np.float64),
            }
            self.last_info = info
            return observation, info

        spawn_values_all = self._per_env_values(spawn_s, "spawn_s")
        lateral_values_all = self._per_env_values(lateral_offset, "lateral_offset")
        speed_values_all = self._per_env_values(initial_speed, "initial_speed")
        if speed_values_all is not None and np.any(speed_values_all < 0):
            raise ValueError("initial_speed must be non-negative")
        heading_values_all = self._per_env_values(heading_error, "heading_error")

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        track_indices = self.track_indices.copy()
        if len(self.tracks) == 1:
            track_indices[selected] = 0
        else:
            track_indices[selected] = self.rng.integers(0, len(self.tracks), size=selected.size, dtype=np.int64)
        tracks = [self.tracks[int(index)] for index in track_indices[selected]]
        lengths = np.asarray([track.length for track in tracks], dtype=np.float64)

        if spawn_values_all is None:
            if evaluation:
                spawn_values = np.asarray([track.start_line_s for track in tracks], dtype=np.float64)
            elif len(self.tracks) == 1:
                spawn_values = self.rng.uniform(0.0, float(lengths[0]), size=selected.size)
            else:
                spawn_values = self.rng.random(selected.size) * lengths
        else:
            spawn_values = spawn_values_all[selected]
        spawn_values = np.mod(spawn_values, lengths)

        if lateral_values_all is None:
            if evaluation:
                lateral_values = np.zeros(selected.size, dtype=np.float64)
            else:
                left = np.asarray([track.width_at_s(value)[0] for track, value in zip(tracks, spawn_values)], dtype=np.float64)
                right = np.asarray([track.width_at_s(value)[1] for track, value in zip(tracks, spawn_values)], dtype=np.float64)
                lateral_values = self.rng.uniform(-0.3 * right, 0.3 * left)
        else:
            lateral_values = lateral_values_all[selected]

        if speed_values_all is None:
            if evaluation:
                speed_values = np.zeros(selected.size, dtype=np.float64)
            else:
                standing_starts = self.rng.random(selected.size) < self.config.episode.standing_start_probability
                random_speeds = self.rng.uniform(
                    0.0,
                    0.5 * self.config.car.max_speed_mps,
                    size=selected.size,
                )
                speed_values = np.where(standing_starts, 0.0, random_speeds)
        else:
            speed_values = speed_values_all[selected]

        if heading_values_all is None:
            heading_values = np.zeros(selected.size, dtype=np.float64) if evaluation else self.rng.uniform(-0.08, 0.08, size=selected.size)
        else:
            heading_values = heading_values_all[selected]

        x = np.zeros(selected.size, dtype=np.float64)
        y = np.zeros(selected.size, dtype=np.float64)
        heading = np.zeros(selected.size, dtype=np.float64)
        for index, (track, value, lateral, error) in enumerate(zip(tracks, spawn_values, lateral_values, heading_values)):
            point, track_heading, _ = track.point_at_s(float(value))
            normal_left = np.array([-np.sin(track_heading), np.cos(track_heading)], dtype=np.float64)
            start = point + float(lateral) * normal_left
            x[index] = start[0]
            y[index] = start[1]
            heading[index] = track_heading + float(error)

        reset_fields = (
            x,
            y,
            heading,
            np.maximum(speed_values, 0.0),
            np.zeros(selected.size, dtype=np.float64),
            np.zeros(selected.size, dtype=np.float64),
            np.zeros(selected.size, dtype=np.float64),
            np.zeros(selected.size, dtype=np.float64),
            np.zeros(selected.size, dtype=np.float64),
        )
        for destination, values in zip(self.state.fields, reset_fields):
            destination[selected] = values
        self.track_indices[selected] = track_indices[selected]
        self.previous_s[selected] = spawn_values
        self.distance_since_spawn[selected] = 0.0
        self.steps[selected] = 0
        self.stuck_ticks[selected] = 0
        self.sector_index[selected] = 0
        # The spawn sector is reported, but coverage counts only sectors the
        # car reaches on a transition. A reset/teleport is not driven coverage.
        self.sector_coverage_mask[selected] = np.uint64(0)
        self._done[selected] = False
        observation = self._observations()
        info = {
            "track_id": tuple(self.tracks[int(index)].track_id for index in self.track_indices),
            "spawn_s": self.previous_s.copy(),
            "signals": {name: np.zeros(self.n_envs, dtype=np.float64) for name in SIGNAL_NAMES},
            "evaluation": bool(evaluation),
            "sector_index": self.sector_index.copy(),
            "sector_count": DEFAULT_SECTOR_COUNT,
            "sector_coverage_mask": self.sector_coverage_mask.copy(),
            "sector_coverage_count": np.asarray(
                [sector_coverage_count(int(mask)) for mask in self.sector_coverage_mask], dtype=np.int64
            ),
            "boundary_excess_m": np.zeros(self.n_envs, dtype=np.float64),
            "off_track_progress_m": np.zeros(self.n_envs, dtype=np.float64),
        }
        self.last_info = info
        return observation, info

    def _validate_actions(self, actions: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float64)
        if values.shape == (2,) and self.n_envs == 1:
            values = values.reshape(1, 2)
        if values.shape != (self.n_envs, self.action_size):
            raise ValueError(f"actions must have shape ({self.n_envs}, {self.action_size})")
        if not np.all(np.isfinite(values)):
            raise ValueError("action must contain only finite values")
        return np.clip(values, -1.0, 1.0)

    def _projections(self, indices: Iterable[int] | None = None) -> list[Projection]:
        selected = range(self.n_envs) if indices is None else indices
        return [
            self.tracks[int(track_index)].project(np.array([self.state.x[index], self.state.y[index]], dtype=np.float64))
            for index, track_index in ((index, self.track_indices[index]) for index in selected)
        ]

    def _edge_statuses(self, indices: Iterable[int] | None = None) -> list[EdgeStatus]:
        selected = range(self.n_envs) if indices is None else indices
        return [
            self.tracks[int(track_index)].edge_status(
                np.array([self.state.x[index], self.state.y[index]], dtype=np.float64),
                float(self.state.heading[index]),
                self.config.car.length_m,
                self.config.car.width_m,
            )
            for index, track_index in ((index, self.track_indices[index]) for index in selected)
        ]

    def step(
        self,
        actions: np.ndarray | Iterable[Iterable[float]],
        *,
        active_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        mask = self._validate_active_mask(active_mask)
        action_array = self._validate_actions(actions)
        active_indices = np.flatnonzero(mask)
        ended = np.flatnonzero(self._done & mask).tolist()
        if ended:
            raise RuntimeError(f"episodes have ended for environments {ended}; call reset()")
        previous_distance = self.distance_since_spawn.copy()
        previous_sector_coverage = self.sector_coverage_mask.copy()

        previous_projections = self._projections(active_indices)
        previous_s = np.asarray([projection.s for projection in previous_projections], dtype=np.float64)
        previous_steering = self.state.previous_steering_command[active_indices].copy()
        previous_edges = self._edge_statuses(active_indices)
        previous_off_track = np.asarray([edge.off_track for edge in previous_edges], dtype=bool)

        terminated = np.zeros(self.n_envs, dtype=bool)
        truncated = np.zeros(self.n_envs, dtype=bool)
        lap_complete = np.zeros(self.n_envs, dtype=bool)
        active_projections: list[Projection] = []
        active_edges: list[EdgeStatus] = []
        if active_indices.size:
            grip_factors = np.where(previous_off_track, self.config.car.runoff_grip_factor, 1.0)
            active_state = CarStateBatch(*(np.asarray(field[active_indices], dtype=np.float64).copy() for field in self.state.fields))
            updated_state = step_car_batch(
                active_state,
                action_array[active_indices],
                self.config.car,
                self.config.dt,
                grip_factors,
            )
            for destination, values in zip(self.state.fields, updated_state.fields):
                destination[active_indices] = values

            active_projections = self._projections(active_indices)
            current_s_active = np.asarray([projection.s for projection in active_projections], dtype=np.float64)
            lengths_active = np.asarray(
                [self.tracks[int(index)].length for index in self.track_indices[active_indices]], dtype=np.float64
            )
            delta_s = current_s_active - previous_s
            delta_s = np.where(delta_s < -lengths_active / 2.0, delta_s + lengths_active, delta_s)
            delta_s = np.where(delta_s > lengths_active / 2.0, delta_s - lengths_active, delta_s)
            self.distance_since_spawn[active_indices] += delta_s

            active_edges = self._edge_statuses(active_indices)
            crash_active = np.asarray([edge.barrier_contact for edge in active_edges], dtype=bool)
            lap_complete_active = self.distance_since_spawn[active_indices] >= lengths_active
            self.stuck_ticks[active_indices] = np.where(np.abs(delta_s) < 0.01, self.stuck_ticks[active_indices] + 1, 0)
            self.steps[active_indices] += 1
            backward_too_far = self.distance_since_spawn[active_indices] < -self.config.episode.backward_limit_m
            terminated_active = crash_active | lap_complete_active | backward_too_far
            stuck_active = self.stuck_ticks[active_indices] >= self.config.episode.stuck_steps
            truncated_active = (~terminated_active) & (
                (self.steps[active_indices] >= self.config.episode.max_steps)
                | stuck_active
            )
            terminated[active_indices] = terminated_active
            truncated[active_indices] = truncated_active
            lap_complete[active_indices] = lap_complete_active
            self._done[active_indices] = terminated_active | truncated_active

        projections = self._projections()
        current_s = np.asarray([projection.s for projection in projections], dtype=np.float64)
        lengths = np.asarray([self.tracks[int(index)].length for index in self.track_indices], dtype=np.float64)
        edges = self._edge_statuses()
        off_track = np.asarray([edge.off_track for edge in edges], dtype=bool)
        crash = np.asarray([edge.barrier_contact for edge in edges], dtype=bool)
        lap_complete |= self.distance_since_spawn >= lengths
        boundary_excess = np.asarray(
            [max(edge.maximum_left_excess, edge.maximum_right_excess) for edge in edges], dtype=np.float64
        )
        off_track_progress = np.zeros(self.n_envs, dtype=np.float64)
        if active_indices.size:
            delta_s_active = self.distance_since_spawn[active_indices] - previous_distance[active_indices]
            off_track_progress[active_indices] = np.where(
                off_track[active_indices], np.maximum(delta_s_active, 0.0), 0.0
            )
            sector_indices_active = np.asarray(
                [
                    sector_index(float(distance), float(length))
                    for distance, length in zip(self.distance_since_spawn[active_indices], lengths[active_indices])
                ],
                dtype=np.int64,
            )
            coverage_bits = np.asarray(
                [sector_bit(int(index)) for index in sector_indices_active], dtype=np.uint64
            )
            self.sector_index[active_indices] = sector_indices_active
            self.sector_coverage_mask[active_indices] |= coverage_bits

        signals: dict[str, np.ndarray] = {name: np.zeros(self.n_envs, dtype=np.float64) for name in SIGNAL_NAMES}
        rewards = np.zeros(self.n_envs, dtype=np.float64)
        events: list[list[str]] = [[] for _ in range(self.n_envs)]
        for position, (index, track, projection, edge) in enumerate(
            zip(
                active_indices,
                (self.tracks[int(value)] for value in self.track_indices[active_indices]),
                active_projections,
                active_edges,
            )
        ):
            index = int(index)
            curvature_ahead = track.curvature_at_s(
                projection.s + max(self.config.sensors.lookahead_distances_m[0], self.state.speed[index] * 0.75)
            )
            current_signals = compute_signals(
                track=track,
                car=self.config.car,
                dt=self.config.dt,
                previous_s=float(previous_s[position]),
                current_s=float(projection.s),
                speed=float(self.state.speed[index]),
                previous_steering_command=float(previous_steering[position]),
                steering_command=float(action_array[index, 0]),
                curvature_ahead=curvature_ahead,
                off_track=bool(edge.off_track),
                crash=bool(crash[index]),
                stuck=bool(
                    self.stuck_ticks[index] >= self.config.episode.stuck_steps
                ),
                lap_complete=bool(lap_complete[index]),
                checkpoint=not bool(
                    int(previous_sector_coverage[index])
                    & int(sector_bit(int(self.sector_index[index])))
                ),
            )
            for name, value in current_signals.items():
                signals[name][index] = value
            if self.reward_config is not None:
                rewards[index] = self.reward_config.calculate(current_signals)
            current_events: list[str] = []
            if edge.off_track and not previous_off_track[position]:
                current_events.append("off_track_start")
            if not edge.off_track and previous_off_track[position]:
                current_events.append("off_track_end")
            if crash[index]:
                current_events.append("barrier_contact")
            if lap_complete[index]:
                current_events.append("lap_complete")
            if current_signals["checkpoint"]:
                current_events.append("checkpoint")
            events[index] = current_events

        observation = self._observations()
        info = {
            "track_id": tuple(self.tracks[int(index)].track_id for index in self.track_indices),
            "signals": signals,
            "events": events,
            "state": self.state_to_dict(),
            "s": current_s.copy(),
            "distance_since_spawn": self.distance_since_spawn.copy(),
            "off_track": off_track,
            "crash": crash,
            "lap_complete": lap_complete,
            "step": self.steps.copy(),
            "previous_distance_since_spawn": previous_distance,
            "active_mask": mask.copy(),
            "sector_index": self.sector_index.copy(),
            "sector_count": DEFAULT_SECTOR_COUNT,
            "sector_coverage_mask": self.sector_coverage_mask.copy(),
            "sector_coverage_count": np.asarray(
                [sector_coverage_count(int(mask_value)) for mask_value in self.sector_coverage_mask], dtype=np.int64
            ),
            "boundary_excess_m": boundary_excess,
            "off_track_progress_m": off_track_progress,
        }
        self.previous_s[active_indices] = current_s[active_indices]
        self.last_info = info
        return observation, rewards, terminated, truncated, info

    def step_raw(
        self,
        actions: np.ndarray | Iterable[Iterable[float]],
        *,
        active_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Advance transitions and return raw signals without reward.

        The legacy five-value ``step`` method remains available for the local
        reference baseline. Participant-owned trainers should use this raw
        boundary and calculate rewards outside the simulator.
        """

        observation, _legacy_rewards, terminated, truncated, info = self.step(
            actions, active_mask=active_mask
        )
        return observation, terminated, truncated, info

    def _observation_at(self, index: int, projection: Projection | None = None) -> np.ndarray:
        track = self.tracks[int(self.track_indices[index])]
        position = np.array([self.state.x[index], self.state.y[index]], dtype=np.float64)
        projection = projection or track.project(position)
        left_width, right_width = track.width_at_s(projection.s)
        heading_error = track.observation_heading_error(float(self.state.heading[index]), projection)
        ray_values = [
            track.raycast(
                position,
                float(self.state.heading[index]) + np.deg2rad(relative_angle),
                self.config.sensors.ray_max_range_m,
            )
            / self.config.sensors.ray_max_range_m
            for relative_angle in self.config.sensors.ray_angles_deg
        ]
        lookahead_values = [
            np.clip(track.curvature_at_s(projection.s + distance) * 100.0, -1.0, 1.0)
            for distance in self.config.sensors.lookahead_distances_m
        ]
        values = [
            np.clip(self.state.speed[index] / self.config.car.max_speed_mps, 0.0, 1.0),
            np.clip(self.state.longitudinal_velocity[index] / self.config.car.max_speed_mps, -1.0, 1.0),
            np.clip(self.state.lateral_velocity[index] / self.config.car.max_speed_mps, -1.0, 1.0),
            np.clip(self.state.yaw_rate[index] / 5.0, -1.0, 1.0),
            np.clip(projection.lateral_offset / max(left_width, right_width), -2.0, 2.0) / 2.0,
            heading_error / np.pi,
            self.state.previous_steering_command[index],
            self.state.previous_longitudinal_command[index],
            self.state.steering_angle[index] / self.config.car.max_steering_rad,
            *ray_values,
            *lookahead_values,
            np.clip(min(left_width, right_width) / (5.0 * self.config.car.width_m), 0.0, 1.0),
        ]
        observation = np.asarray(values, dtype=np.float32)
        self.observation_schema.validate(observation)
        return observation

    def _observations(self) -> np.ndarray:
        projections = self._projections()
        observations = np.stack([self._observation_at(index, projection) for index, projection in enumerate(projections)])
        if observations.shape != (self.n_envs, self.observation_size):
            raise RuntimeError(f"observation shape changed: expected {(self.n_envs, self.observation_size)}, got {observations.shape}")
        return observations

    def state_to_dict(self) -> dict[str, np.ndarray]:
        return {
            "x": self.state.x.copy(),
            "y": self.state.y.copy(),
            "heading": self.state.heading.copy(),
            "speed": self.state.speed.copy(),
            "lateral_velocity": self.state.lateral_velocity.copy(),
            "yaw_rate": self.state.yaw_rate.copy(),
            "steer": self.state.steering_angle.copy(),
        }

    @property
    def done(self) -> np.ndarray:
        return self._done.copy()


# Keep the shorter name available for callers that use the usual vector-env
# terminology while retaining the explicit public class name in docs.
RacingVectorEnv = VectorizedRacingEnv
