"""Versioned, Python-authored replay files. Renderers never step physics.

The writer streams frames to a temporary sibling file and publishes only a
complete replay. The reader has explicit size limits for local preview tools.
Checksums detect accidental changes, not malicious submissions or forgery.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from .config import SimulationConfig
from .observation import TrainingContract
from .track import TrackGeometry


REPLAY_VERSION = "1.1"
MAX_REPLAY_BYTES = 16 * 1024 * 1024
MAX_REPLAY_FRAMES = 20_001
MAX_TRACK_POINTS = 20_000
HEADER_FIELDS = {"schema_version", "kind", "engine", "source", "tick_hz", "car", "track"}
FRAME_FIELDS = {
    "tick", "x", "y", "heading", "speed", "lateral_velocity", "yaw_rate",
    "steer", "steering_command", "throttle_brake", "off_track", "events",
    "terminated", "truncated",
}
STATUSES = {"finished", "dnf_crash", "dnf_stuck", "dnf_backward", "dnf_timeout"}
EVENTS = {"off_track_start", "off_track_end", "barrier_contact", "lap_complete", "checkpoint"}
TRACK_ARRAYS = (
    "centerline", "s", "heading", "curvature", "half_width_left", "half_width_right",
    "left_boundary", "right_boundary", "barrier_left", "barrier_right",
)
ENGINE_FILES = (
    "config.py", "env.py", "vector_env.py", "physics.py", "track.py",
    "observation.py", "signals.py", "reward.py",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _hash(value: Any, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def engine_code_hash() -> str:
    """Hash named simulation sources, without machine paths or private files."""

    digest = hashlib.sha256()
    for name in ENGINE_FILES:
        digest.update(name.encode("ascii") + b"\0")
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def replay_header(
    track: TrackGeometry, config: SimulationConfig, *, participant_id: str, mode: str,
) -> dict[str, Any]:
    data = {name: getattr(track, name).tolist() for name in TRACK_ARRAYS}
    data.update({
        "version": "1.0", "L": track.length, "runoff_width": track.runoff_width,
        "start_line_s": track.start_line_s, "meta": dict(track.meta),
    })
    header = {
        "schema_version": REPLAY_VERSION, "kind": "racing_event_replay",
        "engine": {"code_hash": engine_code_hash(), "config_hash": TrainingContract.config_hash_for(config)},
        "source": {"mode": mode, "participant_id": participant_id},
        "tick_hz": config.tick_hz,
        "car": {"length_m": config.car.length_m, "width_m": config.car.width_m},
        "track": data,
    }
    _validate_header(header)
    return header


def replay_frame(
    tick: int, state: Mapping[str, float], *, steering_command: float = 0.0,
    throttle_brake: float = 0.0, off_track: bool = False,
    events: list[str] | None = None, terminated: bool = False, truncated: bool = False,
) -> dict[str, Any]:
    frame = {name: float(state[name]) for name in (
        "x", "y", "heading", "speed", "lateral_velocity", "yaw_rate", "steer",
    )}
    frame.update({
        "tick": tick, "steering_command": float(steering_command),
        "throttle_brake": float(throttle_brake), "off_track": off_track,
        "events": list(events or []), "terminated": terminated, "truncated": truncated,
    })
    _validate_frame(frame, tick)
    return frame


def _validate_header(header: Any) -> None:
    if not isinstance(header, dict) or set(header) != HEADER_FIELDS:
        raise ValueError("invalid replay header fields")
    if header["schema_version"] != REPLAY_VERSION:
        raise ValueError(f"unsupported replay schema version; expected {REPLAY_VERSION}")
    if header["kind"] != "racing_event_replay":
        raise ValueError("invalid replay kind")
    _integer(header["tick_hz"], "tick_hz", 1)
    if not isinstance(header["engine"], dict) or set(header["engine"]) != {"code_hash", "config_hash"}:
        raise ValueError("invalid engine identity")
    for key, value in header["engine"].items():
        _hash(value, key)
    source = header["source"]
    if not isinstance(source, dict) or set(source) != {"mode", "participant_id"}:
        raise ValueError("invalid replay source")
    if source["mode"] not in ("scripted", "evaluation"):
        raise ValueError("invalid replay mode")
    if not isinstance(source["participant_id"], str) or not 1 <= len(source["participant_id"]) <= 128:
        raise ValueError("participant_id must contain 1 to 128 characters")
    car = header["car"]
    if not isinstance(car, dict) or set(car) != {"length_m", "width_m"}:
        raise ValueError("invalid car dimensions")
    if any(_number(car[key], key) <= 0 for key in car):
        raise ValueError("car dimensions must be positive")
    track = header["track"]
    expected = set(TRACK_ARRAYS) | {"version", "L", "runoff_width", "start_line_s", "meta"}
    if not isinstance(track, dict) or set(track) != expected or track["version"] != "1.0":
        raise ValueError("invalid embedded track schema")
    meta = track["meta"]
    if not isinstance(meta, dict) or not isinstance(meta.get("id"), str) or not meta["id"]:
        raise ValueError("invalid track identity")
    if not isinstance(meta.get("display_name"), str) or not meta["display_name"]:
        raise ValueError("track display_name must be a nonempty string")
    _hash(meta.get("content_hash"), "track hash")
    if meta["content_hash"] != TrackGeometry.content_hash_for_data(track):
        raise ValueError("embedded track content hash mismatch")
    n = len(track["centerline"])
    if not 8 <= n <= MAX_TRACK_POINTS:
        raise ValueError("embedded track exceeds point limits")
    for key in TRACK_ARRAYS:
        values = np.asarray(track[key])
        shape = (n, 2) if key in ("centerline", "left_boundary", "right_boundary", "barrier_left", "barrier_right") else (n,)
        if values.shape != shape or values.dtype.kind not in "fi" or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid embedded track array: {key}")
    length = _number(track["L"], "track length")
    if length <= 0 or _number(track["runoff_width"], "runoff width") <= 0:
        raise ValueError("track length and runoff must be positive")
    if not 0 <= _number(track["start_line_s"], "start line") < length:
        raise ValueError("invalid start line")


def _validate_frame(frame: Any, tick: int) -> None:
    if not isinstance(frame, dict) or set(frame) != FRAME_FIELDS:
        raise ValueError("invalid replay frame fields")
    if _integer(frame["tick"], "tick") != tick:
        raise ValueError("replay ticks must be contiguous, starting at zero")
    for key in ("x", "y", "heading", "speed", "lateral_velocity", "yaw_rate", "steer", "steering_command", "throttle_brake"):
        _number(frame[key], key)
    if frame["speed"] < 0 or any(abs(frame[key]) > 1 for key in ("steering_command", "throttle_brake")):
        raise ValueError("invalid speed or applied control")
    for key in ("off_track", "terminated", "truncated"):
        if type(frame[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if frame["terminated"] and frame["truncated"]:
        raise ValueError("frame cannot be both terminated and truncated")
    events = frame["events"]
    if not isinstance(events, list) or any(not isinstance(e, str) or e not in EVENTS for e in events) or len(set(events)) != len(events):
        raise ValueError("invalid replay events")
    if tick == 0 and (events or frame["terminated"] or frame["truncated"]):
        raise ValueError("initial frame cannot contain terminal flags or events")
    if any(event in events for event in ("barrier_contact", "lap_complete")) and not frame["terminated"]:
        raise ValueError("terminal events require recorded termination")
    if "off_track_start" in events and (not frame["off_track"] or "off_track_end" in events):
        raise ValueError("off_track_start disagrees with recorded surface state")
    if "off_track_end" in events and frame["off_track"]:
        raise ValueError("off_track_end disagrees with recorded surface state")


def _validate_result(result: Any, last: dict[str, Any], tick_hz: int) -> None:
    fields = {"status", "steps", "lap_time_s", "elapsed_s", "trajectory_hash"}
    if not isinstance(result, dict) or set(result) != fields or result["status"] not in STATUSES:
        raise ValueError("invalid replay result")
    if _integer(result["steps"], "steps", 1) != last["tick"]:
        raise ValueError("replay result steps do not match final tick")
    duration = result["steps"] / tick_hz
    if not math.isclose(_number(result["elapsed_s"], "elapsed_s"), duration, abs_tol=1e-9, rel_tol=0):
        raise ValueError("elapsed_s must match Python fixed-step timing")
    _hash(result["trajectory_hash"], "trajectory hash")
    if result["status"] == "finished":
        if not last["terminated"] or "lap_complete" not in last["events"] or "barrier_contact" in last["events"]:
            raise ValueError("finish result disagrees with recorded terminal event")
        if not math.isclose(_number(result["lap_time_s"], "lap_time_s"), duration, abs_tol=1e-9, rel_tol=0):
            raise ValueError("lap time does not match Python fixed-step timing")
    elif result["lap_time_s"] is not None:
        raise ValueError("DNF lap_time_s must be null")
    if result["status"] == "dnf_crash" and ("barrier_contact" not in last["events"] or not last["terminated"]):
        raise ValueError("crash result requires recorded barrier contact")
    if result["status"] == "dnf_stuck" and not last["truncated"]:
        raise ValueError("stuck result requires recorded truncation")
    if result["status"] == "dnf_backward" and not last["terminated"]:
        raise ValueError("backward result requires recorded termination")
    if "barrier_contact" in last["events"] and result["status"] != "dnf_crash":
        raise ValueError("barrier contact must be a crash, not a finish")
    if "lap_complete" in last["events"] and result["status"] not in ("finished", "dnf_crash"):
        raise ValueError("recorded lap completion disagrees with result")
    if result["status"] == "dnf_timeout" and last["terminated"]:
        raise ValueError("timeout must not disguise a recorded termination")


class ReplayWriter:
    """Atomic streaming JSON writer; holds only the most recent frame.

    Use as a context manager. Without a successful ``finish``, no replay is
    published and an existing destination is preserved. IO/validation failures
    abort rather than quietly publishing a partial run.
    """

    def __init__(self, path: str | Path, header: dict[str, Any]):
        _validate_header(header)
        self.path = Path(path)
        self.header = json.loads(_canonical(header))
        self._digest = hashlib.sha256(_canonical(self.header) + b"\n")
        self._count = 0
        self._last: dict[str, Any] | None = None
        self._finished = False
        self._stream = None
        self._temporary: Path | None = None

    def __enter__(self) -> "ReplayWriter":
        if self._stream is not None or self._finished:
            raise RuntimeError("replay writer cannot be reopened")
        fd, name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        self._temporary = Path(name)
        self._stream = os.fdopen(fd, "wb")
        try:
            prefix = _canonical(self.header)[:-1] + b',"frames":['
            self._stream.write(prefix)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def append(self, frame: dict[str, Any]) -> None:
        if self._stream is None or self._finished:
            raise RuntimeError("replay writer is not open")
        if self._count >= MAX_REPLAY_FRAMES:
            raise ValueError("replay exceeds frame limit")
        if self._last is not None and (self._last["terminated"] or self._last["truncated"]):
            raise ValueError("cannot append after a terminal frame")
        _validate_frame(frame, self._count)
        encoded = _canonical(frame)
        # Leave room for the result and checksum footer.
        if self._stream.tell() + len(encoded) + 4096 > MAX_REPLAY_BYTES:
            raise ValueError("replay exceeds byte limit")
        self._stream.write((b"," if self._count else b"") + encoded)
        self._digest.update(encoded + b"\n")
        self._last = json.loads(encoded)
        self._count += 1

    def finish(self, result: dict[str, Any]) -> None:
        if self._stream is None or self._finished or self._last is None:
            raise RuntimeError("replay writer is not open or has no frames")
        _validate_result(result, self._last, self.header["tick_hz"])
        encoded = _canonical(result)
        self._digest.update(encoded + b"\n")
        footer = b'],"result":' + encoded + b',"payload_hash":' + _canonical(self._digest.hexdigest()) + b"}\n"
        if self._stream.tell() + len(footer) > MAX_REPLAY_BYTES:
            raise ValueError("replay exceeds byte limit")
        self._stream.write(footer)
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        self._stream = None
        os.replace(self._temporary, self.path)
        self._temporary = None
        self._finished = True

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        if self._temporary is not None:
            self._temporary.unlink(missing_ok=True)
            self._temporary = None


def validate_replay(data: Any) -> dict[str, Any]:
    """Validate the exchange format without running any simulation.

    Topological track validation remains the track builder's responsibility;
    this read-only boundary checks its structure and exact embedded hash.
    """

    if not isinstance(data, dict) or set(data) != HEADER_FIELDS | {"frames", "result", "payload_hash"}:
        raise ValueError("invalid replay document fields")
    header = {key: data[key] for key in HEADER_FIELDS}
    _validate_header(header)
    frames = data["frames"]
    if not isinstance(frames, list) or not 2 <= len(frames) <= MAX_REPLAY_FRAMES:
        raise ValueError("replay must contain an initial frame and at least one transition within frame limits")
    digest = hashlib.sha256(_canonical(header) + b"\n")
    for tick, frame in enumerate(frames):
        _validate_frame(frame, tick)
        if tick < len(frames) - 1 and (frame["terminated"] or frame["truncated"]):
            raise ValueError("terminal frame must be last")
        digest.update(_canonical(frame) + b"\n")
    _validate_result(data["result"], frames[-1], header["tick_hz"])
    digest.update(_canonical(data["result"]) + b"\n")
    _hash(data["payload_hash"], "payload hash")
    if digest.hexdigest() != data["payload_hash"]:
        raise ValueError("replay payload hash mismatch")
    return data


def load_replay(path: str | Path) -> dict[str, Any]:
    """Load a bounded local JSON replay, rejecting duplicate keys and NaN."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate replay key: {key}")
            result[key] = value
        return result

    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(MAX_REPLAY_BYTES + 1)
        if len(raw) > MAX_REPLAY_BYTES:
            raise ValueError("replay exceeds byte limit")
        data = json.loads(raw, object_pairs_hook=unique_object)
        return validate_replay(data)
    except (OSError, UnicodeError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        raise ValueError(f"cannot load replay: {exc}") from exc


def replay_result(result: Any, tick_hz: int) -> dict[str, Any]:
    return {
        "status": result.status, "steps": result.steps, "lap_time_s": result.lap_time_s,
        "elapsed_s": result.steps / tick_hz, "trajectory_hash": result.trajectory_hash,
    }
