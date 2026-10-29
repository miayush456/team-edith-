"""Build original synthetic training-track geometry fixtures.

The curves in this module are development fixtures, not traces of real
circuits.  The builder samples smooth periodic functions by arclength and
derives every exported edge from the same centerline tangent and normal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

from racing_event.track import TrackGeometry


CAR_WIDTH_M = 2.0
TOTAL_TRACK_WIDTH_M = 10.0
HALF_TRACK_WIDTH_M = TOTAL_TRACK_WIDTH_M / 2.0
RUNOFF_WIDTH_M = 5.0
SAMPLE_COUNT = 512
SOURCE_NOTE = "Original synthetic development fixture; no real-track or held-out geometry."


def _curve_oval(t: np.ndarray) -> np.ndarray:
    return np.column_stack((112.0 * np.cos(t), 62.0 * np.sin(t)))


def _curve_flowing(t: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            104.0 * np.cos(t) + 13.0 * np.cos(2.0 * t + 0.25) - 6.0 * np.cos(3.0 * t - 0.2),
            61.0 * np.sin(t) + 10.0 * np.sin(2.0 * t + 0.4) + 4.0 * np.sin(3.0 * t),
        )
    )


def _curve_tight_loop(t: np.ndarray) -> np.ndarray:
    radius = 1.0 + 0.10 * np.cos(3.0 * t + 0.35) - 0.055 * np.sin(2.0 * t - 0.2)
    return np.column_stack((86.0 * radius * np.cos(t), 54.0 * radius * np.sin(t)))


CURVES: dict[str, tuple[str, Callable[[np.ndarray], np.ndarray]]] = {
    "synthetic_oval": ("Synthetic Oval", _curve_oval),
    "synthetic_flowing": ("Synthetic Flowing Loop", _curve_flowing),
    "synthetic_tight_loop": ("Synthetic Tighter Loop", _curve_tight_loop),
}


def _uniform_resample(curve: Callable[[np.ndarray], np.ndarray], count: int = SAMPLE_COUNT) -> np.ndarray:
    dense_count = max(20000, count * 40)
    dense_t = np.linspace(0.0, 2.0 * np.pi, dense_count + 1)
    dense = np.asarray(curve(dense_t), dtype=np.float64)
    dense_segment_lengths = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    dense_s = np.concatenate(([0.0], np.cumsum(dense_segment_lengths)))
    target_s = np.linspace(0.0, float(dense_s[-1]), count, endpoint=False)
    samples = np.column_stack(
        (
            np.interp(target_s, dense_s, dense[:, 0]),
            np.interp(target_s, dense_s, dense[:, 1]),
        )
    )
    return samples


def _geometry_data(track_id: str, display_name: str, curve: Callable[[np.ndarray], np.ndarray]) -> dict[str, object]:
    centerline = _uniform_resample(curve)
    next_points = np.roll(centerline, -1, axis=0)
    previous_points = np.roll(centerline, 1, axis=0)
    segment_vectors = next_points - centerline
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    length = float(np.sum(segment_lengths))
    s = np.concatenate(([0.0], np.cumsum(segment_lengths[:-1])))
    nominal_spacing = length / len(centerline)

    central_tangent = next_points - previous_points
    tangent_norm = np.linalg.norm(central_tangent, axis=1)
    tangent = central_tangent / tangent_norm[:, None]
    normal_left = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    left_boundary = centerline + HALF_TRACK_WIDTH_M * normal_left
    right_boundary = centerline - HALF_TRACK_WIDTH_M * normal_left
    barrier_left = centerline + (HALF_TRACK_WIDTH_M + RUNOFF_WIDTH_M) * normal_left
    barrier_right = centerline - (HALF_TRACK_WIDTH_M + RUNOFF_WIDTH_M) * normal_left

    first_derivative = (next_points - previous_points) / (2.0 * nominal_spacing)
    second_derivative = (next_points - 2.0 * centerline + previous_points) / (nominal_spacing * nominal_spacing)
    first_norm = np.linalg.norm(first_derivative, axis=1)
    curvature = (
        first_derivative[:, 0] * second_derivative[:, 1]
        - first_derivative[:, 1] * second_derivative[:, 0]
    ) / np.maximum(first_norm**3, 1.0e-12)
    heading = np.arctan2(tangent[:, 1], tangent[:, 0])
    minimum_radius = float(1.0 / np.max(np.abs(curvature)))
    if minimum_radius < 12.0:
        raise ValueError(f"{track_id} is too tight to be a physically drivable fixture: radius={minimum_radius:.2f} m")

    meta: dict[str, object] = {
        "id": track_id,
        "display_name": display_name,
        "source_note": SOURCE_NOTE,
        "geometry_version": "1.0",
        "car_width_m": CAR_WIDTH_M,
        "track_width_m": TOTAL_TRACK_WIDTH_M,
        "width_ratio": TOTAL_TRACK_WIDTH_M / CAR_WIDTH_M,
        "minimum_radius_m": minimum_radius,
    }
    data: dict[str, object] = {
        "version": "1.0",
        "centerline": centerline.tolist(),
        "s": s.tolist(),
        "L": length,
        "heading": heading.tolist(),
        "curvature": curvature.tolist(),
        "half_width_left": np.full(len(centerline), HALF_TRACK_WIDTH_M).tolist(),
        "half_width_right": np.full(len(centerline), HALF_TRACK_WIDTH_M).tolist(),
        "left_boundary": left_boundary.tolist(),
        "right_boundary": right_boundary.tolist(),
        "runoff_width": RUNOFF_WIDTH_M,
        "barrier_left": barrier_left.tolist(),
        "barrier_right": barrier_right.tolist(),
        "start_line_s": 0.0,
        "meta": meta,
    }
    meta["content_hash"] = TrackGeometry.content_hash_for_data(data)
    return data


def build_track(track_id: str) -> dict[str, object]:
    """Build and validate one named synthetic track in memory."""

    try:
        display_name, curve = CURVES[track_id]
    except KeyError as exc:
        raise ValueError(f"unknown synthetic track {track_id!r}") from exc
    data = _geometry_data(track_id, display_name, curve)
    # Run the exact loader-level validation before anything is written.
    track = _track_from_data(data)
    if track.content_hash != data["meta"]["content_hash"]:  # type: ignore[index]
        raise ValueError("builder produced an inconsistent content hash")
    return data


def _track_from_data(data: dict[str, object]) -> TrackGeometry:
    """Validate in-memory data using the same checks as ``from_file``."""

    meta = data["meta"]
    if not isinstance(meta, dict):
        raise ValueError("builder metadata must be an object")
    arrays = {
        name: np.asarray(data[name], dtype=np.float64)
        for name in (
            "centerline",
            "s",
            "heading",
            "curvature",
            "half_width_left",
            "half_width_right",
            "left_boundary",
            "right_boundary",
            "barrier_left",
            "barrier_right",
        )
    }
    track = TrackGeometry(
        track_id=str(meta["id"]),
        centerline=arrays["centerline"],
        s=arrays["s"],
        length=float(data["L"]),
        heading=arrays["heading"],
        curvature=arrays["curvature"],
        half_width_left=arrays["half_width_left"],
        half_width_right=arrays["half_width_right"],
        left_boundary=arrays["left_boundary"],
        right_boundary=arrays["right_boundary"],
        runoff_width=float(data["runoff_width"]),
        barrier_left=arrays["barrier_left"],
        barrier_right=arrays["barrier_right"],
        start_line_s=float(data["start_line_s"]),
        meta=dict(meta),
        content_hash=str(meta["content_hash"]),
    )
    track.validate()
    return track


def write_tracks(output_dir: str | Path, track_ids: list[str] | None = None) -> list[Path]:
    destination = Path(output_dir)
    if not destination.parent.exists():
        raise FileNotFoundError(f"output parent does not exist: {destination.parent}")
    destination.mkdir(exist_ok=True)
    selected = track_ids or list(CURVES)
    written: list[Path] = []
    for track_id in selected:
        data = build_track(track_id)
        path = destination / f"{track_id}.json"
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="tracks/train", help="existing or creatable output directory")
    parser.add_argument("--track", action="append", choices=sorted(CURVES), dest="track_ids")
    args = parser.parse_args()
    paths = write_tracks(args.output_dir, args.track_ids)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
