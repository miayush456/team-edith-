"""Build the public Monaco and Spa reference-track fixtures.

The source centerlines are a compact, hand-curated extraction of the public
GeoJSON paths in ``bacinger/f1-circuits``.  The project does not ship the
source GeoJSON.  These points are projected into local metres, resampled,
scaled to the published lap lengths, and surrounded with SYNAPSE-owned
uniform-width boundaries, runoff, and barriers.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing_event.track import TrackGeometry


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "tracks" / "train"
SAMPLE_COUNT = 512
CAR_WIDTH_M = 2.0
TOTAL_TRACK_WIDTH_M = 10.0
HALF_TRACK_WIDTH_M = TOTAL_TRACK_WIDTH_M / 2.0
RUNOFF_WIDTH_M = 5.0

# Public centerline reference: https://github.com/bacinger/f1-circuits
# Coordinates are longitude/latitude pairs.  The selected points retain the
# current Grand Prix shapes while keeping this generator small and auditable.
REFERENCE_TRACKS: dict[str, dict[str, object]] = {
    "monaco": {
        "display_name": "Monaco",
        "target_length_m": 3337.0,
        "smoothing_passes": 12,
        "source_name": "Circuit de Monaco, current Grand Prix layout",
        "points": [
            [7.427191, 43.739404], [7.427339, 43.739764], [7.428703, 43.741091],
            [7.429125, 43.740843], [7.429495, 43.740255], [7.429749, 43.740321],
            [7.429336, 43.740680], [7.429683, 43.740978], [7.430238, 43.741090],
            [7.430363, 43.740955], [7.430331, 43.740359], [7.430072, 43.739356],
            [7.429383, 43.738491], [7.428362, 43.737894], [7.427314, 43.737534],
            [7.425724, 43.737251], [7.425450, 43.737170], [7.425403, 43.737028],
            [7.425009, 43.737050], [7.424924, 43.737054], [7.423690, 43.736937],
            [7.422513, 43.736809], [7.422178, 43.736723], [7.421880, 43.736229],
            [7.421808, 43.735911], [7.421805, 43.735565], [7.421845, 43.735420],
            [7.422219, 43.735138], [7.422237, 43.735031], [7.422413, 43.734295],
            [7.422472, 43.734125], [7.422468, 43.734070], [7.422324, 43.733976],
            [7.422202, 43.733891], [7.422334, 43.733547], [7.422542, 43.733225],
            [7.422948, 43.732874], [7.423300, 43.732742], [7.423440, 43.732674],
            [7.423494, 43.732557], [7.423384, 43.732459], [7.422803, 43.732364],
            [7.422544, 43.732362], [7.422420, 43.732452], [7.422400, 43.732600],
            [7.422154, 43.732868], [7.422000, 43.733047], [7.421880, 43.733181],
            [7.421753, 43.733435], [7.421472, 43.734142], [7.421341, 43.734614],
            [7.421216, 43.735433], [7.421216, 43.735735], [7.421269, 43.736206],
            [7.421401, 43.736523], [7.421452, 43.736696], [7.421444, 43.736819],
            [7.421441, 43.736895], [7.421562, 43.736970], [7.421963, 43.737013],
            [7.422583, 43.737069], [7.423717, 43.737283], [7.424172, 43.737373],
            [7.424589, 43.737422], [7.424875, 43.737439], [7.425098, 43.737497],
            [7.425743, 43.737718], [7.426026, 43.737802], [7.426439, 43.737869],
            [7.426784, 43.737904], [7.427113, 43.737944], [7.427523, 43.738093],
            [7.427828, 43.738364], [7.427877, 43.738494], [7.427869, 43.738630],
            [7.427767, 43.738899], [7.427528, 43.739115], [7.427388, 43.739210],
            [7.427257, 43.739299],
        ],
    },
    "spa": {
        "display_name": "Spa-Francorchamps",
        "target_length_m": 7004.0,
        "smoothing_passes": 30,
        "source_name": "Circuit de Spa-Francorchamps, current Grand Prix layout",
        "points": [
            [5.965020, 50.444251], [5.963419, 50.446033], [5.963402, 50.446113],
            [5.963473, 50.446184], [5.963786, 50.446188], [5.964313, 50.446019],
            [5.965592, 50.445628], [5.966847, 50.445085], [5.967876, 50.444463],
            [5.970321, 50.442606], [5.971315, 50.442168], [5.971741, 50.441824],
            [5.971949, 50.441442], [5.972020, 50.441069], [5.972132, 50.440815],
            [5.973476, 50.439424], [5.974458, 50.438350], [5.974754, 50.437784],
            [5.975719, 50.435639], [5.977199, 50.432382], [5.977560, 50.431472],
            [5.977406, 50.431218], [5.977015, 50.431048], [5.976796, 50.430874],
            [5.976725, 50.430732], [5.976737, 50.430591], [5.977033, 50.429747],
            [5.977027, 50.429601], [5.976820, 50.429323], [5.976630, 50.429224],
            [5.973257, 50.427739], [5.972831, 50.427678], [5.972422, 50.427805],
            [5.972239, 50.428050], [5.972410, 50.428432], [5.974056, 50.429101],
            [5.974304, 50.429309], [5.974340, 50.429431], [5.974322, 50.429582],
            [5.973712, 50.430723], [5.973523, 50.431189], [5.973091, 50.432627],
            [5.972831, 50.433593], [5.972582, 50.433867], [5.972132, 50.434098],
            [5.971599, 50.434192], [5.970717, 50.434230], [5.970072, 50.434150],
            [5.969504, 50.433956], [5.969019, 50.433608], [5.968812, 50.433358],
            [5.967977, 50.432028], [5.967231, 50.430845], [5.966882, 50.430534],
            [5.966361, 50.430393], [5.965876, 50.430421], [5.965325, 50.430624],
            [5.964828, 50.430713], [5.964307, 50.430643], [5.963792, 50.430294],
            [5.962425, 50.428927], [5.962123, 50.428790], [5.961697, 50.428771],
            [5.960578, 50.429257], [5.959898, 50.429624], [5.959602, 50.430209],
            [5.959862, 50.430779], [5.960365, 50.431463], [5.961247, 50.432160],
            [5.962656, 50.432971], [5.963100, 50.433136], [5.965385, 50.433895],
            [5.966320, 50.434357], [5.967107, 50.435106], [5.967924, 50.436261],
            [5.968084, 50.436624], [5.968095, 50.436860], [5.967699, 50.437624],
            [5.966888, 50.438991], [5.966669, 50.439608], [5.966562, 50.440183],
            [5.966432, 50.441404], [5.966533, 50.441541], [5.966852, 50.441541],
            [5.967202, 50.441560], [5.967320, 50.441701], [5.967261, 50.441800],
            [5.966533, 50.442559],
        ],
    },
}


def _project(points: Iterable[Iterable[float]]) -> np.ndarray:
    values = np.asarray(list(points), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("reference points must have shape (N, 2)")
    lon0, lat0 = values[0]
    metres_per_degree_lat = 111_132.0
    metres_per_degree_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
    projected = np.column_stack(
        ((values[:, 0] - lon0) * metres_per_degree_lon,
         (values[:, 1] - lat0) * metres_per_degree_lat)
    )
    if np.linalg.norm(projected[0] - projected[-1]) < 1.0e-6:
        projected = projected[:-1]
    return projected


def _resample_closed(points: np.ndarray, count: int) -> np.ndarray:
    closed = np.vstack((points, points[0]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    if np.any(lengths <= 1.0e-8):
        raise ValueError("reference path contains duplicate points")
    source_s = np.concatenate(([0.0], np.cumsum(lengths)))
    targets = np.linspace(0.0, float(source_s[-1]), count, endpoint=False)
    return np.column_stack((
        np.interp(targets, source_s, closed[:, 0]),
        np.interp(targets, source_s, closed[:, 1]),
    ))


def _smooth_closed(points: np.ndarray, passes: int = 12) -> np.ndarray:
    """Smooth a sparse public path while keeping it periodic and simple."""

    result = np.asarray(points, dtype=np.float64)
    for _ in range(passes):
        result = (np.roll(result, 1, axis=0) + 2.0 * result + np.roll(result, -1, axis=0)) / 4.0
    return result


def _geometry_data(track_id: str, spec: dict[str, object]) -> dict[str, object]:
    centerline = _resample_closed(
        _smooth_closed(_project(spec["points"]), int(spec["smoothing_passes"])), SAMPLE_COUNT  # type: ignore[arg-type]
    )
    centerline *= float(spec["target_length_m"]) / float(
        np.linalg.norm(np.roll(centerline, -1, axis=0) - centerline, axis=1).sum()
    )
    next_points = np.roll(centerline, -1, axis=0)
    previous_points = np.roll(centerline, 1, axis=0)
    segment_vectors = next_points - centerline
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    length = float(segment_lengths.sum())
    s = np.concatenate(([0.0], np.cumsum(segment_lengths[:-1])))
    spacing = length / len(centerline)
    tangent_raw = next_points - previous_points
    tangent = tangent_raw / np.linalg.norm(tangent_raw, axis=1)[:, None]
    normal_left = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    first_derivative = tangent_raw / (2.0 * spacing)
    second_derivative = (next_points - 2.0 * centerline + previous_points) / (spacing * spacing)
    first_norm = np.linalg.norm(first_derivative, axis=1)
    curvature = (
        first_derivative[:, 0] * second_derivative[:, 1]
        - first_derivative[:, 1] * second_derivative[:, 0]
    ) / np.maximum(first_norm**3, 1.0e-12)
    minimum_radius = float(1.0 / np.max(np.abs(curvature)))

    meta: dict[str, object] = {
        "id": track_id,
        "display_name": spec["display_name"],
        "source_note": (
            f"Derived from the public {spec['source_name']} centerline in "
            "bacinger/f1-circuits (MIT); resampled, scaled, and bounded by SYNAPSE."
        ),
        "source_url": "https://github.com/bacinger/f1-circuits",
        "source_license": "MIT",
        "reference_length_m": float(spec["target_length_m"]),
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
        "heading": np.arctan2(tangent[:, 1], tangent[:, 0]).tolist(),
        "curvature": curvature.tolist(),
        "half_width_left": np.full(SAMPLE_COUNT, HALF_TRACK_WIDTH_M).tolist(),
        "half_width_right": np.full(SAMPLE_COUNT, HALF_TRACK_WIDTH_M).tolist(),
        "left_boundary": (centerline + HALF_TRACK_WIDTH_M * normal_left).tolist(),
        "right_boundary": (centerline - HALF_TRACK_WIDTH_M * normal_left).tolist(),
        "runoff_width": RUNOFF_WIDTH_M,
        "barrier_left": (centerline + (HALF_TRACK_WIDTH_M + RUNOFF_WIDTH_M) * normal_left).tolist(),
        "barrier_right": (centerline - (HALF_TRACK_WIDTH_M + RUNOFF_WIDTH_M) * normal_left).tolist(),
        "start_line_s": 0.0,
        "meta": meta,
    }
    meta["content_hash"] = TrackGeometry.content_hash_for_data(data)
    return data


def build_tracks(output_dir: Path = OUTPUT_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for track_id, spec in REFERENCE_TRACKS.items():
        data = _geometry_data(track_id, spec)
        track = _track_from_data(data)
        if track.content_hash != data["meta"]["content_hash"]:  # type: ignore[index]
            raise ValueError(f"{track_id} content hash is inconsistent")
        path = output_dir / f"{track_id}.json"
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        written.append(path)
    return written


def _track_from_data(data: dict[str, object]) -> TrackGeometry:
    meta = data["meta"]
    assert isinstance(meta, dict)
    arrays = {
        name: np.asarray(data[name], dtype=np.float64)
        for name in (
            "centerline", "s", "heading", "curvature", "half_width_left",
            "half_width_right", "left_boundary", "right_boundary", "barrier_left",
            "barrier_right",
        )
    }
    track = TrackGeometry(
        track_id=str(meta["id"]), centerline=arrays["centerline"], s=arrays["s"],
        length=float(data["L"]), heading=arrays["heading"], curvature=arrays["curvature"],
        half_width_left=arrays["half_width_left"], half_width_right=arrays["half_width_right"],
        left_boundary=arrays["left_boundary"], right_boundary=arrays["right_boundary"],
        runoff_width=float(data["runoff_width"]), barrier_left=arrays["barrier_left"],
        barrier_right=arrays["barrier_right"], start_line_s=float(data["start_line_s"]),
        meta=dict(meta), content_hash=str(meta["content_hash"]),
    )
    track.validate()
    return track


if __name__ == "__main__":
    for path in build_tracks():
        track = TrackGeometry.from_file(path)
        print(f"{path}: {track.length:.3f} m, min radius {track.meta['minimum_radius_m']:.2f} m")
