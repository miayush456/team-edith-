"""Raw reward signals. The environment exposes these before weighting."""

from __future__ import annotations

from typing import Any

from .config import CarConfig
from .track import TrackGeometry


def compute_signals(
    *,
    track: TrackGeometry,
    car: CarConfig,
    dt: float,
    previous_s: float,
    current_s: float,
    speed: float,
    previous_steering_command: float,
    steering_command: float,
    curvature_ahead: float,
    off_track: bool,
    crash: bool,
    stuck: bool,
    lap_complete: bool,
    checkpoint: bool = False,
) -> dict[str, float]:
    raw_delta = current_s - previous_s
    if raw_delta < -track.length / 2.0:
        raw_delta += track.length
    elif raw_delta > track.length / 2.0:
        raw_delta -= track.length

    v_lo = 5.0
    v_hi = min(car.max_speed_mps, 55.0)
    speed_signal = min(1.0, max(0.0, (speed - v_lo) / max(v_hi - v_lo, 1e-6))) * dt
    target_speed = min(car.max_speed_mps, (car.peak_lateral_accel_mps2 / max(abs(curvature_ahead), 1e-4)) ** 0.5)
    curvature_signal = -abs(speed - target_speed) / car.max_speed_mps * dt
    return {
        "progress": raw_delta / track.length,
        "speed": speed_signal,
        "curvature_matched_speed": curvature_signal,
        "off_track": float(off_track) * dt,
        "crash": float(crash),
        "stuck": float(stuck),
        "lap_complete": float(lap_complete),
        "checkpoint": float(checkpoint),
        "time": dt,
        "steer_smoothness": abs(steering_command - previous_steering_command) * (speed / car.max_speed_mps),
    }
