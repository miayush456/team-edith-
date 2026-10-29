"""Versioned simulation configuration.

The simulation reads one configuration object. Renderers consume replay data
and must not duplicate or reinterpret these values.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class CarConfig:
    length_m: float = 4.8
    width_m: float = 2.0
    wheelbase_m: float = 2.8
    max_speed_mps: float = 80.0
    max_steering_rad: float = 0.55
    steering_rate_rad_s: float = 1.8
    max_accel_mps2: float = 8.0
    max_brake_mps2: float = 18.0
    drag_mps2_per_mps2: float = 0.008
    peak_lateral_accel_mps2: float = 28.0
    runoff_grip_factor: float = 0.55


@dataclass(frozen=True)
class EpisodeConfig:
    max_steps: int = 2400
    stuck_steps: int = 90
    backward_limit_m: float = 25.0
    standing_start_probability: float = 0.25


@dataclass(frozen=True)
class SensorConfig:
    ray_angles_deg: tuple[float, ...] = (0.0, -15.0, 15.0, -30.0, 30.0, -60.0, 60.0, -90.0, 90.0)
    ray_max_range_m: float = 100.0
    lookahead_distances_m: tuple[float, ...] = (15.0, 40.0, 80.0, 140.0)

    MAX_RAY_COUNT = 16
    MIN_RAY_ANGLE_DEG = -90.0
    MAX_RAY_ANGLE_DEG = 90.0

    def __post_init__(self) -> None:
        angles = tuple(0.0 if float(value) == 0.0 else float(value) for value in self.ray_angles_deg)
        object.__setattr__(self, "ray_angles_deg", angles)
        if not 1 <= len(angles) <= self.MAX_RAY_COUNT:
            raise ValueError(f"ray_angles_deg must contain 1..{self.MAX_RAY_COUNT} angles")
        if any(not math.isfinite(value) for value in angles):
            raise ValueError("ray_angles_deg must contain only finite values")
        if any(value < self.MIN_RAY_ANGLE_DEG or value > self.MAX_RAY_ANGLE_DEG for value in angles):
            raise ValueError("ray_angles_deg must stay within [-90, 90] degrees")
        if len(set(angles)) != len(angles):
            raise ValueError("ray_angles_deg must not contain duplicates")
        if 0.0 not in angles:
            raise ValueError("ray_angles_deg must include one forward center ray at 0 degrees")
        ray_range = float(self.ray_max_range_m)
        object.__setattr__(self, "ray_max_range_m", ray_range)
        if not math.isfinite(ray_range) or not 1.0 <= ray_range <= 500.0:
            raise ValueError("ray_max_range_m must be finite and in [1, 500] metres")
        lookaheads = tuple(float(value) for value in self.lookahead_distances_m)
        object.__setattr__(self, "lookahead_distances_m", lookaheads)
        if not lookaheads or any(not math.isfinite(value) or value <= 0.0 for value in lookaheads):
            raise ValueError("lookahead_distances_m must contain positive finite values")

    @property
    def ray_count(self) -> int:
        return len(self.ray_angles_deg)


@dataclass(frozen=True)
class SimulationConfig:
    tick_hz: int = 30
    car: CarConfig = CarConfig()
    episode: EpisodeConfig = EpisodeConfig()
    sensors: SensorConfig = SensorConfig()

    @property
    def dt(self) -> float:
        return 1.0 / self.tick_hz

    @classmethod
    def from_file(cls, path: str | Path) -> "SimulationConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        car = CarConfig(**data.get("car", {}))
        episode = EpisodeConfig(**data.get("episode", {}))
        sensors_data = data.get("sensors", {})
        sensors = SensorConfig(
            ray_angles_deg=tuple(float(v) for v in sensors_data.get("ray_angles_deg", SensorConfig.ray_angles_deg)),
            ray_max_range_m=float(sensors_data.get("ray_max_range_m", SensorConfig.ray_max_range_m)),
            lookahead_distances_m=tuple(float(v) for v in sensors_data.get("lookahead_distances_m", SensorConfig.lookahead_distances_m)),
        )
        config = cls(tick_hz=int(data.get("tick_hz", 30)), car=car, episode=episode, sensors=sensors)
        config.validate()
        return config

    def validate(self) -> None:
        if self.tick_hz <= 0:
            raise ValueError("tick_hz must be positive")
        if self.car.width_m <= 0 or self.car.length_m <= 0:
            raise ValueError("car dimensions must be positive")
        if self.car.max_speed_mps <= 0 or self.car.wheelbase_m <= 0:
            raise ValueError("car speed and wheelbase must be positive")
        if self.car.runoff_grip_factor <= 0 or self.car.runoff_grip_factor > 1:
            raise ValueError("runoff_grip_factor must be in (0, 1]")
        if self.episode.max_steps <= 0 or self.episode.stuck_steps <= 0:
            raise ValueError("episode limits must be positive")
        if not math.isfinite(self.episode.standing_start_probability) or not (
            0.0 <= self.episode.standing_start_probability <= 1.0
        ):
            raise ValueError("standing_start_probability must be in [0, 1]")
        if not self.sensors.ray_angles_deg or not self.sensors.lookahead_distances_m:
            raise ValueError("at least one sensor ray and lookahead are required")
