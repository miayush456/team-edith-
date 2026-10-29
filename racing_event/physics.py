"""Small deterministic car model used by the Python environment."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import CarConfig


@dataclass
class CarState:
    x: float
    y: float
    heading: float
    speed: float = 0.0
    lateral_velocity: float = 0.0
    yaw_rate: float = 0.0
    steering_angle: float = 0.0
    previous_steering_command: float = 0.0
    previous_longitudinal_command: float = 0.0

    @property
    def longitudinal_velocity(self) -> float:
        """Forward velocity exposed by the non-reversing v1 model.

        ``speed`` is retained as the public state name used by the first
        slice.  The current model has no reverse gear or independent body-
        frame longitudinal state, so the signed forward velocity is equal to
        speed.  Giving it an explicit name keeps the observation contract
        honest without inventing an unmodelled quantity.
        """

        return self.speed

    def copy(self) -> "CarState":
        return CarState(**self.__dict__)


@dataclass
class CarStateBatch:
    """Structure-of-arrays vehicle state for in-process vector stepping."""

    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    speed: np.ndarray
    lateral_velocity: np.ndarray
    yaw_rate: np.ndarray
    steering_angle: np.ndarray
    previous_steering_command: np.ndarray
    previous_longitudinal_command: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            self.x,
            self.y,
            self.heading,
            self.speed,
            self.lateral_velocity,
            self.yaw_rate,
            self.steering_angle,
            self.previous_steering_command,
            self.previous_longitudinal_command,
        )
        if not arrays or any(np.asarray(value).ndim != 1 for value in arrays):
            raise ValueError("batched car state fields must be one-dimensional arrays")
        if any(len(value) != len(self.x) for value in arrays):
            raise ValueError("batched car state fields must have equal length")
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("batched car state fields must be finite")

    def copy(self) -> "CarStateBatch":
        return CarStateBatch(*(np.asarray(value, dtype=np.float64).copy() for value in self.fields))

    @property
    def fields(self) -> tuple[np.ndarray, ...]:
        return (
            self.x,
            self.y,
            self.heading,
            self.speed,
            self.lateral_velocity,
            self.yaw_rate,
            self.steering_angle,
            self.previous_steering_command,
            self.previous_longitudinal_command,
        )

    @property
    def size(self) -> int:
        return len(self.x)

    @property
    def longitudinal_velocity(self) -> np.ndarray:
        return self.speed

    def item(self, index: int) -> CarState:
        return CarState(*(float(value[index]) for value in self.fields))


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def step_car(state: CarState, action: np.ndarray, config: CarConfig, dt: float, grip_factor: float = 1.0) -> CarState:
    """Advance one fixed timestep.

    This is intentionally kinematic and deterministic. It models steering-rate
    limits, power/drag, braking, and a lateral grip cap without introducing a
    hard collision or renderer dependency.
    """

    action_array = np.asarray(action, dtype=np.float64).reshape(-1)
    if action_array.size != 2:
        raise ValueError("action must contain steering and throttle/brake")
    if not np.all(np.isfinite(action_array)):
        raise ValueError("action must contain only finite values")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if not np.isfinite(grip_factor) or grip_factor <= 0.0:
        raise ValueError("grip_factor must be positive and finite")
    steering_command = _clip(float(action_array[0]), -1.0, 1.0)
    longitudinal_command = _clip(float(action_array[1]), -1.0, 1.0)
    target_steering = steering_command * config.max_steering_rad
    maximum_change = config.steering_rate_rad_s * dt
    steering_delta = _clip(target_steering - state.steering_angle, -maximum_change, maximum_change)
    steering_angle = state.steering_angle + steering_delta

    throttle = max(0.0, longitudinal_command)
    brake = max(0.0, -longitudinal_command)
    speed_factor = _clip(1.0 - state.speed / config.max_speed_mps, 0.0, 1.0)
    acceleration = throttle * config.max_accel_mps2 * speed_factor
    acceleration -= brake * config.max_brake_mps2
    acceleration -= config.drag_mps2_per_mps2 * state.speed * state.speed
    speed = _clip(state.speed + acceleration * dt, 0.0, config.max_speed_mps)

    grip = max(1e-6, config.peak_lateral_accel_mps2 * grip_factor)
    longitudinal_acceleration = (speed - state.speed) / dt
    usable_lateral_grip = float(
        np.sqrt(max(0.0, grip * grip - min(abs(longitudinal_acceleration), grip) ** 2))
    )
    requested_yaw_rate = speed * math.tan(steering_angle) / config.wheelbase_m
    requested_lateral_accel = abs(speed * requested_yaw_rate)
    if requested_lateral_accel > usable_lateral_grip:
        yaw_rate = math.copysign(usable_lateral_grip / max(speed, 1e-6), requested_yaw_rate)
    else:
        yaw_rate = requested_yaw_rate

    heading = state.heading + yaw_rate * dt
    average_heading = state.heading + 0.5 * yaw_rate * dt
    x = state.x + speed * math.cos(average_heading) * dt
    y = state.y + speed * math.sin(average_heading) * dt
    lateral_velocity = speed * math.sin(steering_angle) * 0.1
    return CarState(
        x=x,
        y=y,
        heading=heading,
        speed=speed,
        lateral_velocity=lateral_velocity,
        yaw_rate=yaw_rate,
        steering_angle=steering_angle,
        previous_steering_command=steering_command,
        previous_longitudinal_command=longitudinal_command,
    )


def step_car_batch(
    state: CarStateBatch,
    actions: np.ndarray,
    config: CarConfig,
    dt: float,
    grip_factors: np.ndarray | float = 1.0,
) -> CarStateBatch:
    """Advance all cars using the same equations as :func:`step_car`.

    Geometry queries and reward assembly remain environment responsibilities;
    this function contains only the hot, structure-of-arrays physics update.
    """

    action_array = np.asarray(actions, dtype=np.float64)
    if action_array.shape != (state.size, 2):
        raise ValueError(f"actions must have shape ({state.size}, 2)")
    if not np.all(np.isfinite(action_array)):
        raise ValueError("action must contain only finite values")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    grip = np.asarray(grip_factors, dtype=np.float64)
    if grip.ndim == 0:
        grip = np.full(state.size, float(grip), dtype=np.float64)
    if grip.shape != (state.size,) or not np.all(np.isfinite(grip)) or np.any(grip <= 0):
        raise ValueError(f"grip_factors must be positive and have shape ({state.size},)")

    steering_command = np.clip(action_array[:, 0], -1.0, 1.0)
    longitudinal_command = np.clip(action_array[:, 1], -1.0, 1.0)
    target_steering = steering_command * config.max_steering_rad
    maximum_change = config.steering_rate_rad_s * dt
    steering_delta = np.clip(target_steering - state.steering_angle, -maximum_change, maximum_change)
    steering_angle = state.steering_angle + steering_delta

    throttle = np.maximum(0.0, longitudinal_command)
    brake = np.maximum(0.0, -longitudinal_command)
    speed_factor = np.clip(1.0 - state.speed / config.max_speed_mps, 0.0, 1.0)
    acceleration = throttle * config.max_accel_mps2 * speed_factor
    acceleration -= brake * config.max_brake_mps2
    acceleration -= config.drag_mps2_per_mps2 * state.speed * state.speed
    speed = np.clip(state.speed + acceleration * dt, 0.0, config.max_speed_mps)

    peak_grip = np.maximum(1.0e-6, config.peak_lateral_accel_mps2 * grip)
    longitudinal_acceleration = (speed - state.speed) / dt
    usable_lateral_grip = np.sqrt(
        np.maximum(
            0.0,
            peak_grip * peak_grip
            - np.minimum(np.abs(longitudinal_acceleration), peak_grip) ** 2,
        )
    )
    requested_yaw_rate = speed * np.tan(steering_angle) / config.wheelbase_m
    requested_lateral_accel = np.abs(speed * requested_yaw_rate)
    limited_yaw_rate = np.sign(requested_yaw_rate) * usable_lateral_grip / np.maximum(speed, 1.0e-6)
    yaw_rate = np.where(requested_lateral_accel > usable_lateral_grip, limited_yaw_rate, requested_yaw_rate)

    heading = state.heading + yaw_rate * dt
    average_heading = state.heading + 0.5 * yaw_rate * dt
    x = state.x + speed * np.cos(average_heading) * dt
    y = state.y + speed * np.sin(average_heading) * dt
    lateral_velocity = speed * np.sin(steering_angle) * 0.1
    return CarStateBatch(
        x=x,
        y=y,
        heading=heading,
        speed=speed,
        lateral_velocity=lateral_velocity,
        yaw_rate=yaw_rate,
        steering_angle=steering_angle,
        previous_steering_command=steering_command,
        previous_longitudinal_command=longitudinal_command,
    )
