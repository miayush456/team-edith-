"""Focused contract tests for the public training track fixtures."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from racing_event.track import TrackGeometry


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = sorted((ROOT / "tracks" / "train").glob("*.json"))


class TrackGeometryTests(unittest.TestCase):
    def _data(self, path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_data(self, data: dict[str, object], *, preserve_hash: bool = False) -> Path:
        temporary = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        temporary.close()
        path = Path(temporary.name)
        if not preserve_hash:
            meta = data["meta"]
            assert isinstance(meta, dict)
            meta["content_hash"] = TrackGeometry.content_hash_for_data(data)
        path.write_text(json.dumps(data, allow_nan=True), encoding="utf-8")
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def test_fixture_round_trip_and_hashes(self) -> None:
        self.assertEqual(len(FIXTURES), 2)
        self.assertEqual(
            {path.stem for path in FIXTURES},
            {"monaco", "spa"},
        )
        for path in FIXTURES:
            track = TrackGeometry.from_file(path)
            self.assertEqual(track.content_hash, track.meta["content_hash"])
            self.assertEqual(track.meta["geometry_version"], "1.0")
            self.assertEqual(track.centerline.shape, (512, 2))
            self.assertAlmostEqual(track.half_width_left[0] + track.half_width_right[0], 10.0)
            self.assertEqual(track.meta["source_license"], "MIT")
            self.assertAlmostEqual(track.meta["reference_length_m"], track.length)

            point, heading, _ = track.point_at_s(track.length + 1.25)
            projection = track.project(point)
            self.assertLess(projection.distance, 1.0e-6)
            self.assertTrue(np.isfinite(heading))

    def test_hash_mutation_is_rejected(self) -> None:
        data = self._data(FIXTURES[0])
        data["centerline"][0][0] += 0.25  # type: ignore[index]
        path = self._write_data(data, preserve_hash=True)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            TrackGeometry.from_file(path)

    def test_missing_hash_is_rejected(self) -> None:
        data = self._data(FIXTURES[0])
        meta = data["meta"]
        assert isinstance(meta, dict)
        del meta["content_hash"]
        path = self._write_data(data, preserve_hash=True)
        with self.assertRaisesRegex(ValueError, "content_hash is mandatory"):
            TrackGeometry.from_file(path)

    def test_invalid_schema_finite_and_arclength_are_rejected(self) -> None:
        schema = self._data(FIXTURES[0])
        schema["version"] = "9.0"
        schema_path = self._write_data(schema)
        with self.assertRaisesRegex(ValueError, "unsupported track version"):
            TrackGeometry.from_file(schema_path)

        finite = self._data(FIXTURES[0])
        finite["centerline"][0][0] = float("nan")  # type: ignore[index]
        finite_path = self._write_data(finite, preserve_hash=True)
        with self.assertRaises(ValueError):
            TrackGeometry.from_file(finite_path)

        arclength = self._data(FIXTURES[0])
        arclength["s"][20] += 0.5  # type: ignore[index]
        arclength_path = self._write_data(arclength)
        with self.assertRaisesRegex(ValueError, "arclength|strictly increasing"):
            TrackGeometry.from_file(arclength_path)

    def test_self_crossing_and_touching_are_rejected(self) -> None:
        bow_tie = np.asarray([[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "self-intersects"):
            TrackGeometry._validate_simple_polyline(bow_tie, closed=True, name="bow_tie")

        touching = np.asarray([[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [1.0, 3.0], [1.0, 0.0], [0.0, 3.0]], dtype=np.float64)
        with self.assertRaises(ValueError):
            TrackGeometry._validate_simple_polyline(touching, closed=True, name="touching")

        collinear_overlap = np.asarray([[0.0, 0.0], [3.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]], dtype=np.float64)
        with self.assertRaises(ValueError):
            TrackGeometry._validate_simple_polyline(collinear_overlap, closed=True, name="collinear_overlap")

    def test_projection_and_raycast_use_all_segments(self) -> None:
        track = TrackGeometry.from_file(FIXTURES[0])
        point, heading, _ = track.point_at_s(0.0)
        projection = track.project(point)
        self.assertAlmostEqual(projection.distance, 0.0, places=6)
        self.assertAlmostEqual(projection.lateral_offset, 0.0, places=6)

        left_ray = track.raycast(point, heading + np.pi / 2.0, 100.0)
        right_ray = track.raycast(point, heading - np.pi / 2.0, 100.0)
        self.assertAlmostEqual(left_ray, 5.0, delta=0.2)
        self.assertAlmostEqual(right_ray, 5.0, delta=0.2)
        self.assertEqual(track.raycast(point, heading, 0.0), 0.0)

    def test_rectangle_footprint_contacts_exported_edges(self) -> None:
        track = TrackGeometry.from_file(FIXTURES[0])
        point, heading, _ = track.point_at_s(0.0)
        left_normal = np.asarray([-np.sin(heading), np.cos(heading)])

        clear = track.edge_status(point, heading, 4.8, 2.0)
        self.assertFalse(clear.off_track)
        self.assertFalse(clear.barrier_contact)

        boundary_touch = track.edge_status(point + 4.0 * left_normal, heading, 4.8, 2.0)
        self.assertTrue(boundary_touch.off_track)
        self.assertFalse(boundary_touch.barrier_contact)

        barrier_touch = track.edge_status(point + 9.0 * left_normal, heading, 4.8, 2.0)
        self.assertTrue(barrier_touch.barrier_contact)

        fully_outside_barrier = track.edge_status(point + 12.0 * left_normal, heading, 4.8, 2.0)
        self.assertTrue(fully_outside_barrier.barrier_contact)


if __name__ == "__main__":
    unittest.main()
