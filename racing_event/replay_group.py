"""Bounded replay-group v1.0; single-car replay schema 1.1 stays unchanged.

Published limits: 1..4 ghosts/files, 16 KiB manifest, 16 MiB per replay,
32 MiB total INCLUDING the manifest. References are relative POSIX paths below
the manifest directory; no traversal, absolute paths, or symlinks at any level.
One shared clock starts at the identical recorded standing tick zero. A shorter
recording holds its final pose/telemetry/events, never extrapolates or restarts.
Hashes are integrity checks, not proof of provenance or tournament eligibility.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Any, Iterator

from .trajectory import MAX_REPLAY_BYTES, validate_replay


GROUP_VERSION = "1.0"
MAX_GHOSTS = 4
MAX_MANIFEST_BYTES = 16 * 1024
MAX_GROUP_BYTES = 32 * 1024 * 1024
GHOST_COLORS = ("#45f3ff", "#ffd166", "#c79bff", "#87ed8a")


def _json(raw: bytes) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value: str) -> None:
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def _reference(value: Any) -> str:
    if (not isinstance(value, str) or not value or len(value) > 512
            or any(c in value for c in ("\\", ":", "\x00"))
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError("replay reference must be a relative POSIX path without traversal")
    return value


@contextmanager
def _directory(path: Path) -> Iterator[int]:
    """Walk with directory FDs/O_NOFOLLOW, including ancestors of the manifest.

    Each opened FD pins the directory, so a swapped symlink cannot redirect a
    subsequent open. This reader is intentionally Linux/POSIX, like the project.
    """
    absolute = path.absolute()
    fd = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in absolute.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def _read(root: int, reference: str, limit: int) -> bytes:
    parts = _reference(reference).split("/")
    directory = os.dup(root)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("replay group inputs must be regular files")
            if info.st_size > limit:
                raise ValueError("replay group exceeds byte limit")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                raw = stream.read(limit + 1)
            if len(raw) > limit:
                raise ValueError("replay group exceeds byte limit")
            return raw
        finally:
            os.close(fd)
    finally:
        os.close(directory)


def _identity(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "track_id": data["track"]["meta"]["id"],
        "track_hash": data["track"]["meta"]["content_hash"],
        "engine": data["engine"], "tick_hz": data["tick_hz"], "car": data["car"],
        "tick_zero": data["frames"][0],
    }


def _standing(data: dict[str, Any]) -> None:
    frame = data["frames"][0]
    if (any(frame[key] != 0 for key in ("speed", "lateral_velocity", "yaw_rate", "steer",
                                       "steering_command", "throttle_brake")) or frame["off_track"]):
        raise ValueError("group requires a recorded standing start at tick zero")


@dataclass(frozen=True)
class ReplayParticipant:
    participant_id: str
    label: str
    color: str
    data: dict[str, Any]


@dataclass(frozen=True)
class ReplayGroup:
    manifest: dict[str, Any]
    participants: tuple[ReplayParticipant, ...]

    @property
    def tick_hz(self) -> int:
        return self.participants[0].data["tick_hz"]

    @property
    def final_tick(self) -> int:
        return max(p.data["result"]["steps"] for p in self.participants)


def _load(root: int, manifest: Any, manifest_size: int) -> ReplayGroup:
    fields = {"kind", "schema_version", "alignment", "end_behavior", "identity", "participants"}
    if not isinstance(manifest, dict) or set(manifest) != fields:
        raise ValueError("invalid replay-group fields")
    if manifest["kind"] != "racing_event_replay_group" or manifest["schema_version"] != GROUP_VERSION:
        raise ValueError("unsupported replay-group kind/version; expected 1.0")
    if manifest["alignment"] != "standing_start_tick_zero" or manifest["end_behavior"] != "hold_final_pose":
        raise ValueError("unsupported group alignment/end behavior")
    entries = manifest["participants"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_GHOSTS:
        raise ValueError("group requires 1 to 4 ghosts/files")
    participants = []
    seen_ids, seen_files, seen_colors = set(), set(), set()
    total = manifest_size
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"participant_id", "label", "color", "file", "file_sha256", "payload_hash"}:
            raise ValueError("invalid group participant fields")
        reference = _reference(entry["file"])
        label, color, identity = entry["label"], entry["color"], entry["participant_id"]
        if not isinstance(label, str) or not 1 <= len(label) <= 48 or not all(c.isprintable() for c in label):
            raise ValueError("group label must contain 1 to 48 printable characters")
        if not isinstance(identity, str) or not identity or identity in seen_ids:
            raise ValueError("group participant identities must be unique")
        if reference in seen_files or not isinstance(color, str) or color not in GHOST_COLORS or color in seen_colors:
            raise ValueError("group files and palette colors must be unique")
        raw = _read(root, reference, min(MAX_REPLAY_BYTES, MAX_GROUP_BYTES - total))
        total += len(raw)
        if hashlib.sha256(raw).hexdigest() != entry["file_sha256"]:
            raise ValueError("recorded file SHA-256 mismatch")
        data = validate_replay(_json(raw))
        if data["payload_hash"] != entry["payload_hash"]:
            raise ValueError("recorded payload hash mismatch")
        if data["source"]["participant_id"] != identity:
            raise ValueError("participant identity differs from recording")
        _standing(data)
        if _identity(data) != manifest["identity"]:
            raise ValueError("common track/config/engine/tick/car/tick-zero identity mismatch")
        seen_ids.add(identity)
        seen_files.add(reference)
        seen_colors.add(color)
        participants.append(ReplayParticipant(identity, label, color, data))
    return ReplayGroup(manifest, tuple(participants))


def load_replay_group(path: str | Path) -> ReplayGroup:
    """Validate the manifest and all referenced immutable recordings before play."""
    path = Path(path)
    try:
        with _directory(path.parent) as root:
            raw = _read(root, path.name, MAX_MANIFEST_BYTES)
            return _load(root, _json(raw), len(raw))
    except (OSError, UnicodeError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        raise ValueError(f"cannot load replay group: {exc}") from exc


def create_replay_group(output: str | Path, replays: list[str | Path], *, labels: list[str] | None = None) -> ReplayGroup:
    """Publish a validated manifest exclusively; never rewrite a replay or output.

    Replays must already be beneath output.parent. CLI paths are relative to the
    current directory; manifest references are relative to the manifest itself.
    """
    output = Path(output).absolute()
    _reference(output.name)
    if not 1 <= len(replays) <= MAX_GHOSTS or labels is not None and len(labels) != len(replays):
        raise ValueError("provide 1 to 4 replays and one label per replay")
    manifest: dict[str, Any] = {
        "kind": "racing_event_replay_group", "schema_version": GROUP_VERSION,
        "alignment": "standing_start_tick_zero", "end_behavior": "hold_final_pose",
        "identity": None, "participants": [],
    }
    with _directory(output.parent) as root:
        total = 0
        for index, path in enumerate(replays):
            try:
                reference = Path(path).absolute().relative_to(output.parent).as_posix()
            except ValueError as exc:
                raise ValueError("replays must be below the manifest directory") from exc
            if reference == output.name:
                raise ValueError("group output must be distinct from recordings")
            raw = _read(root, reference, min(MAX_REPLAY_BYTES, MAX_GROUP_BYTES - total))
            total += len(raw)
            data = validate_replay(_json(raw))
            if manifest["identity"] is None:
                manifest["identity"] = _identity(data)
            manifest["participants"].append({
                "participant_id": data["source"]["participant_id"],
                "label": labels[index] if labels is not None else data["source"]["participant_id"][:48],
                "color": GHOST_COLORS[index], "file": reference,
                "file_sha256": hashlib.sha256(raw).hexdigest(), "payload_hash": data["payload_hash"],
            })
        raw = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        if len(raw) > MAX_MANIFEST_BYTES:
            raise ValueError("manifest exceeds byte limit")
        group = _load(root, manifest, len(raw))
        # Link provides atomic, no-clobber publication (also rejects symlink
        # destinations). Open the pinned directory rather than rewalking it.
        name = None
        try:
            candidate = f".replay-group-{secrets.token_hex(16)}"
            fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=root)
            name = candidate
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(name, output.name, src_dir_fd=root, dst_dir_fd=root, follow_symlinks=False)
            os.fsync(root)
        finally:
            if name is not None:
                os.unlink(name, dir_fd=root)
    return group


__all__ = ["ReplayGroup", "ReplayParticipant", "create_replay_group", "load_replay_group"]
