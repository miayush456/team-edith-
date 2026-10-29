"""Renderer-neutral playback helpers for validated replay documents.

The replay is already authoritative when it reaches this module.  Playback
only samples recorded frames; it never steps the racing environment or derives
events from geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _shortest_angle_delta(start: float, end: float) -> float:
    return (end - start + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class ReplayClock:
    """A visual clock whose position is expressed in replay seconds."""

    tick_hz: int
    final_tick: int
    seconds: float = 0.0
    speed: float = 1.0
    playing: bool = False

    def __post_init__(self) -> None:
        if type(self.tick_hz) is not int or self.tick_hz <= 0:
            raise ValueError("tick_hz must be a positive integer")
        if type(self.final_tick) is not int or self.final_tick < 1:
            raise ValueError("final_tick must be a positive integer")
        self.seek_seconds(self.seconds)
        self.set_speed(self.speed)

    @property
    def duration_seconds(self) -> float:
        return self.final_tick / self.tick_hz

    @property
    def tick_position(self) -> float:
        position = self.seconds * self.tick_hz
        # A seek to e.g. tick 123 at 30 Hz can round just below 123 when
        # converted back from seconds. Correct only floating-point roundoff,
        # so an exact seek never selects the preceding telemetry frame.
        nearest = round(position)
        return float(nearest) if abs(position - nearest) <= 2 * math.ulp(position) else position

    def set_speed(self, speed: float) -> None:
        speed = float(speed)
        if not math.isfinite(speed) or speed <= 0.0:
            raise ValueError("playback speed must be positive and finite")
        self.speed = speed

    def seek_seconds(self, seconds: float) -> None:
        seconds = float(seconds)
        if not math.isfinite(seconds):
            raise ValueError("playback position must be finite")
        self.seconds = _clamp(seconds, 0.0, self.duration_seconds)

    def seek_tick(self, tick: int) -> None:
        if type(tick) is not int:
            raise ValueError("replay tick must be an integer")
        self.seek_seconds(tick / self.tick_hz)

    def toggle(self) -> bool:
        self.playing = not self.playing
        return self.playing

    def pause(self) -> None:
        self.playing = False

    def restart(self) -> None:
        self.seek_tick(0)
        self.playing = True

    def advance(self, wall_seconds: float) -> None:
        wall_seconds = float(wall_seconds)
        if not math.isfinite(wall_seconds) or wall_seconds < 0.0:
            raise ValueError("wall_seconds must be finite and non-negative")
        if not self.playing:
            return
        self.seek_seconds(self.seconds + wall_seconds * self.speed)
        if self.seconds >= self.duration_seconds:
            self.playing = False


def sample_replay(data: Mapping[str, Any], clock: ReplayClock) -> dict[str, Any]:
    """Interpolate only position/heading; keep completed-tick telemetry exact."""

    frames = data["frames"]
    position = _clamp(clock.tick_position, 0.0, float(len(frames) - 1))
    first_index = min(int(math.floor(position)), len(frames) - 1)
    second_index = min(first_index + 1, len(frames) - 1)
    alpha = 0.0 if first_index == second_index else position - first_index
    first = frames[first_index]
    second = frames[second_index]

    pose: dict[str, float] = {}
    for name in ("x", "y"):
        pose[name] = float(first[name] + (second[name] - first[name]) * alpha)
    for name in ("speed", "lateral_velocity", "yaw_rate", "steer", "steering_command", "throttle_brake"):
        pose[name] = float(first[name])
    pose["heading"] = float(first["heading"] + _shortest_angle_delta(first["heading"], second["heading"]) * alpha)
    return {
        "pose": pose,
        "frame": first,
        "next_frame": second,
        "frame_index": first_index,
        "alpha": alpha,
        "tick_position": position,
    }


__all__ = ["ReplayClock", "sample_replay"]
