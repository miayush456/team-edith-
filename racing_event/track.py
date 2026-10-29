"""Validated track geometry loading, projection, footprint checks, and rays.

The JSON format is deliberately small.  Every polyline is an implicit closed
loop: its last point is unique and the segment from the last point back to the
first point is part of the geometry.  The Python implementation is the
authoritative interpretation of that format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


_GEOMETRY_VERSION = "1.0"
_EPS = 1.0e-9


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _cross_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _point_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
    delta = end - start
    scale = max(1.0, float(np.linalg.norm(delta)), float(np.linalg.norm(point - start)))
    if abs(_cross2(delta, point - start)) > _EPS * scale * scale:
        return False
    return bool(
        np.dot(point - start, point - end) <= _EPS * scale * scale
    )


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Return true for points inside or exactly on a simple closed polygon."""

    return bool(_points_in_polygon(np.asarray(point, dtype=np.float64)[None, :], polygon)[0])


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized inclusive point-in-polygon for a simple closed polygon."""

    starts = polygon
    ends = np.roll(polygon, -1, axis=0)
    delta = ends - starts
    point_delta = points[:, None, :] - starts[None, :, :]
    cross = delta[None, :, 0] * point_delta[:, :, 1] - delta[None, :, 1] * point_delta[:, :, 0]
    scale = np.maximum(1.0, np.linalg.norm(delta, axis=1))[None, :]
    on_segment = (
        (np.abs(cross) <= _EPS * scale * scale)
        & (points[:, None, 0] >= np.minimum(starts[:, 0], ends[:, 0])[None, :] - _EPS * scale)
        & (points[:, None, 0] <= np.maximum(starts[:, 0], ends[:, 0])[None, :] + _EPS * scale)
        & (points[:, None, 1] >= np.minimum(starts[:, 1], ends[:, 1])[None, :] - _EPS * scale)
        & (points[:, None, 1] <= np.maximum(starts[:, 1], ends[:, 1])[None, :] + _EPS * scale)
    )
    above_start = starts[None, :, 1] > points[:, None, 1]
    above_end = ends[None, :, 1] > points[:, None, 1]
    crosses_ray = above_start != above_end
    with np.errstate(divide="ignore", invalid="ignore"):
        intersection_x = (delta[None, :, 0] * (points[:, None, 1] - starts[None, :, 1]) / delta[None, :, 1]) + starts[None, :, 0]
    inside = np.count_nonzero(crosses_ray & (points[:, None, 0] < intersection_x), axis=1) % 2 == 1
    return inside | np.any(on_segment, axis=1)


def _polygon_contains_all(container: np.ndarray, points: np.ndarray) -> bool:
    return bool(np.all(_points_in_polygon(points, container)))


def _edge_intersection_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return a matrix indicating which closed-polyline edges touch or cross.

    Containment is intentionally not an intersection here.  The geometry is
    made of nested rings, so containment is expected between several pairs.
    """

    first_start = first
    first_end = np.roll(first, -1, axis=0)
    second_start = second
    second_end = np.roll(second, -1, axis=0)
    first_delta = first_end - first_start
    second_delta = second_end - second_start
    first_delta_grid = first_delta[:, None, :]
    second_delta_grid = second_delta[None, :, :]
    between = second_start[None, :, :] - first_start[:, None, :]
    first_to_second_end = second_end[None, :, :] - first_start[:, None, :]
    second_to_first_end = first_end[:, None, :] - second_start[None, :, :]

    def cross_grid(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]

    orientation_1 = cross_grid(first_delta_grid, between)
    orientation_2 = cross_grid(first_delta_grid, first_to_second_end)
    orientation_3 = cross_grid(second_delta_grid, -between)
    orientation_4 = cross_grid(second_delta_grid, second_to_first_end)
    scale = np.maximum(
        1.0,
        np.maximum(
            np.linalg.norm(first_delta, axis=1)[:, None],
            np.linalg.norm(second_delta, axis=1)[None, :],
        ),
    )
    tolerance = _EPS * scale * scale
    proper = (
        (((orientation_1 > tolerance) & (orientation_2 < -tolerance)) | ((orientation_1 < -tolerance) & (orientation_2 > tolerance)))
        & (((orientation_3 > tolerance) & (orientation_4 < -tolerance)) | ((orientation_3 < -tolerance) & (orientation_4 > tolerance)))
    )

    def on_segment(points: np.ndarray, starts: np.ndarray, ends: np.ndarray, orientations: np.ndarray) -> np.ndarray:
        point_tolerance = _EPS * scale
        return (
            (np.abs(orientations) <= tolerance)
            & (points[..., 0] >= np.minimum(starts[..., 0], ends[..., 0]) - point_tolerance)
            & (points[..., 0] <= np.maximum(starts[..., 0], ends[..., 0]) + point_tolerance)
            & (points[..., 1] >= np.minimum(starts[..., 1], ends[..., 1]) - point_tolerance)
            & (points[..., 1] <= np.maximum(starts[..., 1], ends[..., 1]) + point_tolerance)
        )

    touching = (
        on_segment(second_start[None, :, :], first_start[:, None, :], first_end[:, None, :], orientation_1)
        | on_segment(second_end[None, :, :], first_start[:, None, :], first_end[:, None, :], orientation_2)
        | on_segment(first_start[:, None, :], second_start[None, :, :], second_end[None, :, :], orientation_3)
        | on_segment(first_end[:, None, :], second_start[None, :, :], second_end[None, :, :], orientation_4)
    )
    return proper | touching


def _polygons_intersect(first: np.ndarray, second: np.ndarray) -> bool:
    return bool(np.any(_edge_intersection_matrix(first, second)))


@dataclass(frozen=True)
class Projection:
    s: float
    distance: float
    lateral_offset: float
    heading: float
    segment_index: int
    point: np.ndarray


@dataclass(frozen=True)
class EdgeStatus:
    off_track: bool
    barrier_contact: bool
    maximum_left_excess: float
    maximum_right_excess: float


@dataclass
class TrackGeometry:
    track_id: str
    centerline: np.ndarray
    s: np.ndarray
    length: float
    heading: np.ndarray
    curvature: np.ndarray
    half_width_left: np.ndarray
    half_width_right: np.ndarray
    left_boundary: np.ndarray
    right_boundary: np.ndarray
    runoff_width: float
    barrier_left: np.ndarray
    barrier_right: np.ndarray
    start_line_s: float
    meta: dict[str, Any]
    content_hash: str
    _segment_vectors_cache: np.ndarray = field(init=False, repr=False)
    _segment_lengths_cache: np.ndarray = field(init=False, repr=False)
    _boundary_outer_cache: np.ndarray = field(init=False, repr=False)
    _boundary_inner_cache: np.ndarray = field(init=False, repr=False)
    _barrier_outer_cache: np.ndarray = field(init=False, repr=False)
    _barrier_inner_cache: np.ndarray = field(init=False, repr=False)
    _ray_starts_cache: np.ndarray = field(init=False, repr=False)
    _ray_ends_cache: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.centerline.ndim == 2 and self.centerline.shape[1] == 2:
            self._segment_vectors_cache = np.roll(self.centerline, -1, axis=0) - self.centerline
            self._segment_lengths_cache = np.linalg.norm(self._segment_vectors_cache, axis=1)
        else:
            self._segment_vectors_cache = np.empty((0, 2), dtype=np.float64)
            self._segment_lengths_cache = np.empty(0, dtype=np.float64)
        rings = (self.left_boundary, self.right_boundary, self.barrier_left, self.barrier_right)
        rings_ready = all(ring.ndim == 2 and ring.shape[1] == 2 and len(ring) >= 3 for ring in rings)
        if rings_ready:
            self._boundary_outer_cache, self._boundary_inner_cache = self._outer_inner(
                self.left_boundary, self.right_boundary, "boundaries"
            )
            self._barrier_outer_cache, self._barrier_inner_cache = self._outer_inner(
                self.barrier_left, self.barrier_right, "barriers"
            )
            self._ray_starts_cache = np.concatenate((self.left_boundary, self.right_boundary), axis=0)
            self._ray_ends_cache = np.concatenate(
                (np.roll(self.left_boundary, -1, axis=0), np.roll(self.right_boundary, -1, axis=0)), axis=0
            )
        else:
            empty_ring = np.empty((0, 2), dtype=np.float64)
            self._boundary_outer_cache = empty_ring
            self._boundary_inner_cache = empty_ring
            self._barrier_outer_cache = empty_ring
            self._barrier_inner_cache = empty_ring
            self._ray_starts_cache = empty_ring
            self._ray_ends_cache = empty_ring

    @classmethod
    def from_file(cls, path: str | Path) -> "TrackGeometry":
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid track file {source}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"invalid track file {source}: top-level JSON value must be an object")

        meta = data.get("meta")
        if not isinstance(meta, dict):
            raise ValueError("track meta must be an object")
        expected_hash = meta.get("content_hash")
        if not isinstance(expected_hash, str) or not expected_hash:
            raise ValueError("track meta.content_hash is mandatory")
        if len(expected_hash) != 64 or any(character not in "0123456789abcdef" for character in expected_hash):
            raise ValueError("track meta.content_hash must be a lowercase SHA-256 hex digest")
        try:
            actual_hash = cls.content_hash_for_data(data)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid track content for {source}: {exc}") from exc
        if expected_hash != actual_hash:
            raise ValueError(f"track content hash mismatch: expected {expected_hash}, got {actual_hash}")

        required = {
            "version",
            "centerline",
            "s",
            "L",
            "heading",
            "curvature",
            "half_width_left",
            "half_width_right",
            "left_boundary",
            "right_boundary",
            "runoff_width",
            "barrier_left",
            "barrier_right",
            "start_line_s",
            "meta",
        }
        unknown = set(data) - required
        missing = required - set(data)
        if missing:
            raise ValueError(f"track is missing required fields: {sorted(missing)}")
        if unknown:
            raise ValueError(f"track has unknown top-level fields: {sorted(unknown)}")
        if data["version"] != _GEOMETRY_VERSION:
            raise ValueError(f"unsupported track version: {data['version']!r}")
        if meta.get("geometry_version") != _GEOMETRY_VERSION:
            raise ValueError("meta.geometry_version must be '1.0'")

        try:
            track_id = meta["id"]
            if not isinstance(track_id, str) or not track_id:
                raise ValueError("meta.id must be a non-empty string")
            centerline = cls._array(data["centerline"], "centerline", 2)
            s = cls._array(data["s"], "s", 1)
            heading = cls._array(data["heading"], "heading", 1)
            curvature = cls._array(data["curvature"], "curvature", 1)
            half_width_left = cls._array(data["half_width_left"], "half_width_left", 1)
            half_width_right = cls._array(data["half_width_right"], "half_width_right", 1)
            left_boundary = cls._array(data["left_boundary"], "left_boundary", 2)
            right_boundary = cls._array(data["right_boundary"], "right_boundary", 2)
            barrier_left = cls._array(data["barrier_left"], "barrier_left", 2)
            barrier_right = cls._array(data["barrier_right"], "barrier_right", 2)
            length = cls._scalar(data["L"], "L")
            runoff_width = cls._scalar(data["runoff_width"], "runoff_width")
            start_line_s = cls._scalar(data["start_line_s"], "start_line_s")
            track = cls(
                track_id=track_id,
                centerline=centerline,
                s=s,
                length=length,
                heading=heading,
                curvature=curvature,
                half_width_left=half_width_left,
                half_width_right=half_width_right,
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                runoff_width=runoff_width,
                barrier_left=barrier_left,
                barrier_right=barrier_right,
                start_line_s=start_line_s,
                meta=dict(meta),
                content_hash=actual_hash,
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid track file {source}: {exc}") from exc
        track.validate()
        return track

    @staticmethod
    def _scalar(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number")
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    @staticmethod
    def _array(value: Any, name: str, columns: int) -> np.ndarray:
        try:
            result = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if result.ndim != 2 and columns == 2:
            raise ValueError(f"{name} must have shape (N, 2)")
        if result.ndim != 1 and columns == 1:
            raise ValueError(f"{name} must have shape (N,)")
        if columns == 2 and result.shape[1] != columns:
            raise ValueError(f"{name} must have shape (N, 2)")
        if result.ndim == 0 or (result.ndim == 1 and columns == 2):
            raise ValueError(f"{name} has invalid shape")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain only finite values")
        return result

    @staticmethod
    def content_hash_for_data(data: dict[str, Any]) -> str:
        """Hash canonical JSON content, excluding only ``meta.content_hash``."""

        if not isinstance(data, dict):
            raise ValueError("track data must be an object")
        normalized = json.loads(json.dumps(data, allow_nan=False))
        meta = normalized.get("meta")
        if not isinstance(meta, dict):
            raise ValueError("track meta must be an object")
        meta.pop("content_hash", None)
        encoded = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def validate(self) -> None:
        if self.meta.get("geometry_version") != _GEOMETRY_VERSION:
            raise ValueError("track geometry_version must be '1.0'")
        if not isinstance(self.track_id, str) or not self.track_id:
            raise ValueError("track_id must be a non-empty string")
        if self.centerline.ndim != 2 or self.centerline.shape[1] != 2:
            raise ValueError("centerline must have shape (N, 2)")
        n = len(self.centerline)
        if n < 8:
            raise ValueError("track needs at least eight centerline points")
        arrays = {
            "s": self.s,
            "heading": self.heading,
            "curvature": self.curvature,
            "half_width_left": self.half_width_left,
            "half_width_right": self.half_width_right,
            "left_boundary": self.left_boundary,
            "right_boundary": self.right_boundary,
            "barrier_left": self.barrier_left,
            "barrier_right": self.barrier_right,
        }
        for name, value in arrays.items():
            if not isinstance(value, np.ndarray):
                raise ValueError(f"{name} must be a numpy array")
            if len(value) != n:
                raise ValueError(f"{name} must contain {n} samples")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")
        for name in ("left_boundary", "right_boundary", "barrier_left", "barrier_right"):
            if arrays[name].ndim != 2 or arrays[name].shape != (n, 2):
                raise ValueError(f"{name} must have shape ({n}, 2)")
        for name in ("s", "heading", "curvature", "half_width_left", "half_width_right"):
            if arrays[name].ndim != 1 or arrays[name].shape != (n,):
                raise ValueError(f"{name} must have shape ({n},)")

        if not np.isfinite(self.length) or self.length <= 0:
            raise ValueError("L must be a positive finite number")
        if not np.isclose(self.s[0], 0.0, atol=1.0e-9):
            raise ValueError("s must start at zero")
        segment_vectors = np.roll(self.centerline, -1, axis=0) - self.centerline
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        if np.any(segment_lengths <= 1.0e-8):
            raise ValueError("centerline contains duplicate or zero-length segments")
        measured_length = float(np.sum(segment_lengths))
        if not np.isclose(measured_length, self.length, rtol=1.0e-6, atol=1.0e-6):
            raise ValueError("L does not match the closed centerline length")
        closed_s_deltas = np.diff(np.concatenate((self.s, np.array([self.length]))))
        if np.any(closed_s_deltas <= 0) or self.s[-1] >= self.length:
            raise ValueError("s must be strictly increasing and end before L")
        if not np.allclose(closed_s_deltas, segment_lengths, rtol=2.0e-4, atol=2.0e-4):
            raise ValueError("s increments do not match centerline arclength")
        mean_spacing = float(np.mean(segment_lengths))
        if np.any(np.abs(segment_lengths / mean_spacing - 1.0) > 0.10):
            raise ValueError("centerline samples must be approximately uniform")
        if np.any(self.half_width_left <= 0) or np.any(self.half_width_right <= 0):
            raise ValueError("track widths must be positive")
        if not np.isfinite(self.runoff_width) or self.runoff_width <= 0:
            raise ValueError("runoff_width must be a positive finite number")
        if not np.isfinite(self.start_line_s) or not (0.0 <= self.start_line_s < self.length):
            raise ValueError("start_line_s must be within the lap")

        self._validate_simple_polyline(self.centerline, closed=True, name="centerline")
        for name in ("left_boundary", "right_boundary", "barrier_left", "barrier_right"):
            self._validate_simple_polyline(getattr(self, name), closed=True, name=name)

        rings = [self.centerline, self.left_boundary, self.right_boundary, self.barrier_left, self.barrier_right]
        ring_names = ["centerline", "left_boundary", "right_boundary", "barrier_left", "barrier_right"]
        for first_index in range(len(rings)):
            for second_index in range(first_index + 1, len(rings)):
                if _polygons_intersect(rings[first_index], rings[second_index]):
                    raise ValueError(f"{ring_names[first_index]} intersects {ring_names[second_index]}")

        tangents = segment_vectors / segment_lengths[:, None]
        left_offsets = _cross_rows(tangents, self.left_boundary - self.centerline)
        right_offsets = _cross_rows(tangents, self.right_boundary - self.centerline)
        barrier_left_offsets = _cross_rows(tangents, self.barrier_left - self.centerline)
        barrier_right_offsets = _cross_rows(tangents, self.barrier_right - self.centerline)
        if np.any(left_offsets <= 0) or np.any(right_offsets >= 0):
            raise ValueError("boundaries are not consistently on the left and right sides")
        if np.any(barrier_left_offsets <= 0) or np.any(barrier_right_offsets >= 0):
            raise ValueError("barriers are not consistently on the left and right sides")
        if not np.allclose(left_offsets, self.half_width_left, rtol=0.08, atol=0.05):
            raise ValueError("left boundary is inconsistent with half_width_left")
        if not np.allclose(-right_offsets, self.half_width_right, rtol=0.08, atol=0.05):
            raise ValueError("right boundary is inconsistent with half_width_right")
        if np.any(barrier_left_offsets <= left_offsets) or np.any(barrier_right_offsets >= right_offsets):
            raise ValueError("barriers must be outside the corresponding boundaries")
        runoff_left = barrier_left_offsets - left_offsets
        runoff_right = -barrier_right_offsets - (-right_offsets)
        if not np.allclose(runoff_left, self.runoff_width, rtol=0.12, atol=0.10):
            raise ValueError("left barrier is inconsistent with runoff_width")
        if not np.allclose(runoff_right, self.runoff_width, rtol=0.12, atol=0.10):
            raise ValueError("right barrier is inconsistent with runoff_width")

        boundary_outer, boundary_inner = self._outer_inner(self.left_boundary, self.right_boundary, "boundaries")
        barrier_outer, barrier_inner = self._outer_inner(self.barrier_left, self.barrier_right, "barriers")
        if not _polygon_contains_all(boundary_outer, self.centerline):
            raise ValueError("centerline must be inside the outer boundary")
        if any(_point_in_polygon(point, boundary_inner) for point in self.centerline):
            raise ValueError("centerline must be outside the inner boundary")
        if not _polygon_contains_all(barrier_outer, boundary_outer):
            raise ValueError("outer barrier must contain the outer boundary")
        if not _polygon_contains_all(boundary_inner, barrier_inner):
            raise ValueError("inner boundary must contain the inner barrier")

    @staticmethod
    def _outer_inner(first: np.ndarray, second: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
        first_contains_second = _polygon_contains_all(first, second)
        second_contains_first = _polygon_contains_all(second, first)
        if first_contains_second == second_contains_first:
            raise ValueError(f"{name} must be strictly nested without crossing")
        if first_contains_second:
            return first, second
        return second, first

    @staticmethod
    def _validate_simple_polyline(points: np.ndarray, closed: bool, name: str) -> None:
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError(f"{name} must be a two-dimensional polyline")
        if not np.all(np.isfinite(points)):
            raise ValueError(f"{name} must contain only finite values")
        if np.linalg.norm(points[0] - points[-1]) <= 1.0e-8:
            raise ValueError(f"{name} must use an implicit closure with a unique last point")
        count = len(points) if closed else len(points) - 1
        starts = points[:count]
        ends = points[1:count + 1] if not closed else np.roll(points, -1, axis=0)
        segment_lengths = np.linalg.norm(ends - starts, axis=1)
        if np.any(segment_lengths <= 1.0e-8):
            raise ValueError(f"{name} contains a zero-length segment")
        if not closed:
            raise ValueError(f"{name} must be closed for track geometry")
        intersections = _edge_intersection_matrix(points, points)
        index_grid = np.arange(count)[:, None]
        other_grid = np.arange(count)[None, :]
        adjacent = np.abs(index_grid - other_grid) == 1
        if closed:
            adjacent |= (index_grid == 0) & (other_grid == count - 1)
            adjacent |= (other_grid == 0) & (index_grid == count - 1)
        intersections &= ~adjacent
        np.fill_diagonal(intersections, False)
        if np.any(intersections):
            raise ValueError(f"{name} self-intersects, touches, or overlaps")
        if closed:
            for index in range(count):
                other_index = (index + 1) % count
                first_vector = ends[index] - starts[index]
                second_vector = ends[other_index] - starts[other_index]
                scale = max(1.0, float(np.linalg.norm(first_vector)), float(np.linalg.norm(second_vector)))
                if abs(_cross2(first_vector, second_vector)) <= _EPS * scale * scale and np.dot(first_vector, second_vector) < 0.0:
                    raise ValueError(f"{name} has overlapping adjacent collinear segments")

    @staticmethod
    def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
        scale = max(1.0, float(np.linalg.norm(b - a)), float(np.linalg.norm(d - c)))
        tolerance = _EPS * scale * scale
        o1 = _cross2(b - a, c - a)
        o2 = _cross2(b - a, d - a)
        o3 = _cross2(d - c, a - c)
        o4 = _cross2(d - c, b - c)
        if abs(o1) <= tolerance and _point_on_segment(c, a, b):
            return True
        if abs(o2) <= tolerance and _point_on_segment(d, a, b):
            return True
        if abs(o3) <= tolerance and _point_on_segment(a, c, d):
            return True
        if abs(o4) <= tolerance and _point_on_segment(b, c, d):
            return True
        return ((o1 > tolerance and o2 < -tolerance) or (o1 < -tolerance and o2 > tolerance)) and ((o3 > tolerance and o4 < -tolerance) or (o3 < -tolerance and o4 > tolerance))

    def _segment_lengths(self) -> np.ndarray:
        return self._segment_lengths_cache

    def _segment_at_s(self, s_value: float) -> tuple[int, float]:
        try:
            value = float(s_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("s must be a finite scalar") from exc
        if not np.isfinite(value):
            raise ValueError("s must be a finite scalar")
        wrapped = value % self.length
        index = int(np.searchsorted(self.s, wrapped, side="right") - 1)
        index = max(0, min(index, len(self.centerline) - 1))
        fraction = (wrapped - float(self.s[index])) / float(self._segment_lengths()[index])
        return index, float(np.clip(fraction, 0.0, 1.0))

    def project(self, position: np.ndarray | tuple[float, float]) -> Projection:
        point = np.asarray(position, dtype=np.float64)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("position must be a finite point with shape (2,)")
        starts = self.centerline
        deltas = self._segment_vectors_cache
        squared_lengths = self._segment_lengths_cache * self._segment_lengths_cache
        fractions = np.einsum("ij,ij->i", point - starts, deltas) / squared_lengths
        fractions = np.clip(fractions, 0.0, 1.0)
        candidates = starts + fractions[:, None] * deltas
        distances_squared = np.einsum("ij,ij->i", point - candidates, point - candidates)
        index = int(np.argmin(distances_squared))
        candidate = candidates[index]
        segment_length = float(np.sqrt(squared_lengths[index]))
        tangent = deltas[index] / segment_length
        projected_s = float(self.s[index] + fractions[index] * segment_length)
        if projected_s >= self.length:
            projected_s -= self.length
        return Projection(
            s=projected_s,
            distance=float(np.sqrt(distances_squared[index])),
            lateral_offset=_cross2(tangent, point - candidate),
            heading=float(np.arctan2(tangent[1], tangent[0])),
            segment_index=index,
            point=np.array(candidate, dtype=np.float64),
        )

    def point_at_s(self, s_value: float) -> tuple[np.ndarray, float, float]:
        index, fraction = self._segment_at_s(s_value)
        next_index = (index + 1) % len(self.centerline)
        point = self.centerline[index] + fraction * (self.centerline[next_index] - self.centerline[index])
        tangent = self.centerline[next_index] - self.centerline[index]
        tangent /= np.linalg.norm(tangent)
        curvature = (1.0 - fraction) * self.curvature[index] + fraction * self.curvature[next_index]
        return np.array(point, dtype=np.float64), float(np.arctan2(tangent[1], tangent[0])), float(curvature)

    def width_at_s(self, s_value: float) -> tuple[float, float]:
        index, fraction = self._segment_at_s(s_value)
        next_index = (index + 1) % len(self.centerline)
        left = (1.0 - fraction) * self.half_width_left[index] + fraction * self.half_width_left[next_index]
        right = (1.0 - fraction) * self.half_width_right[index] + fraction * self.half_width_right[next_index]
        return float(left), float(right)

    def curvature_at_s(self, s_value: float) -> float:
        index, fraction = self._segment_at_s(s_value)
        next_index = (index + 1) % len(self.centerline)
        return float((1.0 - fraction) * self.curvature[index] + fraction * self.curvature[next_index])

    @staticmethod
    def _footprint(position: np.ndarray, heading: float, car_length: float, car_width: float) -> np.ndarray:
        forward = np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)
        side = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float64)
        corners = np.array(
            [
                position - 0.5 * car_length * forward - 0.5 * car_width * side,
                position + 0.5 * car_length * forward - 0.5 * car_width * side,
                position + 0.5 * car_length * forward + 0.5 * car_width * side,
                position - 0.5 * car_length * forward + 0.5 * car_width * side,
            ],
            dtype=np.float64,
        )
        return corners

    @staticmethod
    def _footprint_intersects_ring(footprint: np.ndarray, ring: np.ndarray) -> bool:
        if np.any(_edge_intersection_matrix(footprint, ring)):
            return True
        # A footprint wholly inside an outer ring is safe; only a ring vertex
        # inside the footprint indicates that the footprint encloses/touches
        # the ring without an edge crossing.
        return bool(np.any(_points_in_polygon(ring, footprint)))

    def edge_status(self, position: np.ndarray, heading: float, car_length: float, car_width: float) -> EdgeStatus:
        point = np.asarray(position, dtype=np.float64)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("position must be a finite point with shape (2,)")
        if not np.isfinite(heading) or not np.isfinite(car_length) or not np.isfinite(car_width) or car_length <= 0 or car_width <= 0:
            raise ValueError("heading and car dimensions must be finite and positive")
        footprint = self._footprint(point, float(heading), float(car_length), float(car_width))
        boundary_outer, boundary_inner = self._boundary_outer_cache, self._boundary_inner_cache
        barrier_outer, barrier_inner = self._barrier_outer_cache, self._barrier_inner_cache

        outside_boundary = any(not _point_in_polygon(corner, boundary_outer) for corner in footprint)
        inside_inner_boundary = any(_point_in_polygon(corner, boundary_inner) for corner in footprint)
        boundary_contact = self._footprint_intersects_ring(footprint, boundary_outer) or self._footprint_intersects_ring(footprint, boundary_inner)
        off_track = outside_boundary or inside_inner_boundary or boundary_contact

        outside_barrier = any(not _point_in_polygon(corner, barrier_outer) for corner in footprint)
        inside_inner_barrier = any(_point_in_polygon(corner, barrier_inner) for corner in footprint)
        barrier_contact = outside_barrier or inside_inner_barrier
        barrier_contact = barrier_contact or self._footprint_intersects_ring(footprint, barrier_outer) or self._footprint_intersects_ring(footprint, barrier_inner)

        maximum_left_excess = 0.0
        maximum_right_excess = 0.0
        for corner in footprint:
            projection = self.project(corner)
            left, right = self.width_at_s(projection.s)
            maximum_left_excess = max(maximum_left_excess, projection.lateral_offset - left)
            maximum_right_excess = max(maximum_right_excess, -projection.lateral_offset - right)
        return EdgeStatus(
            off_track=bool(off_track),
            barrier_contact=bool(barrier_contact),
            maximum_left_excess=float(maximum_left_excess),
            maximum_right_excess=float(maximum_right_excess),
        )

    def raycast(self, origin: np.ndarray, angle: float, max_range: float) -> float:
        point = np.asarray(origin, dtype=np.float64)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("origin must be a finite point with shape (2,)")
        if not np.isfinite(angle) or not np.isfinite(max_range) or max_range < 0:
            raise ValueError("angle must be finite and max_range must be finite and non-negative")
        if max_range == 0:
            return 0.0
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        starts = self._ray_starts_cache
        ends = self._ray_ends_cache
        segments = ends - starts
        origin_to_start = starts - point
        denominators = _cross_rows(np.broadcast_to(direction, segments.shape), segments)
        parallel = np.abs(denominators) <= 1.0e-12
        distances: list[float] = []

        nonparallel = ~parallel
        if np.any(nonparallel):
            t = _cross_rows(origin_to_start[nonparallel], segments[nonparallel]) / denominators[nonparallel]
            u = _cross_rows(origin_to_start[nonparallel], np.broadcast_to(direction, (int(np.sum(nonparallel)), 2))) / denominators[nonparallel]
            valid = (t >= -_EPS) & (u >= -_EPS) & (u <= 1.0 + _EPS)
            distances.extend(np.maximum(t[valid], 0.0).tolist())

        if np.any(parallel):
            parallel_starts = starts[parallel]
            parallel_ends = ends[parallel]
            collinear = np.abs(_cross_rows(parallel_starts - point, np.broadcast_to(direction, parallel_starts.shape))) <= 1.0e-9
            if np.any(collinear):
                start_distance = np.einsum("ij,j->i", parallel_starts[collinear] - point, direction)
                end_distance = np.einsum("ij,j->i", parallel_ends[collinear] - point, direction)
                nearest = np.maximum(np.minimum(start_distance, end_distance), 0.0)
                valid = np.maximum(start_distance, end_distance) >= -_EPS
                distances.extend(nearest[valid].tolist())
        if distances:
            return float(min(max_range, max(0.0, min(distances))))
        return float(max_range)

    def observation_heading_error(self, vehicle_heading: float, projection: Projection) -> float:
        return _wrap_angle(vehicle_heading - projection.heading)
