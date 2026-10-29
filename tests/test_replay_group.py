from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from racing_event.config import SimulationConfig
from racing_event.replay_group import (
    MAX_GROUP_BYTES, MAX_MANIFEST_BYTES, create_replay_group, load_replay_group,
)
from racing_event.track import TrackGeometry
from racing_event.trajectory import ReplayWriter, load_replay, replay_frame, replay_header


ROOT = Path(__file__).resolve().parents[1]
TRACK = TrackGeometry.from_file(ROOT / "tests/fixtures/tracks/synthetic_oval.json")
CONFIG = SimulationConfig.from_file(ROOT / "config/simulation.json")


def recording(path: Path, participant_id: str, steps: int, *, change=None, status="finished") -> Path:
    """Schema-valid boundary fixtures, not claims about physically driven laps."""
    header = replay_header(TRACK, CONFIG, participant_id=participant_id, mode="evaluation")
    if change:
        change(header)
        header["track"]["meta"]["content_hash"] = TrackGeometry.content_hash_for_data(header["track"])
    with ReplayWriter(path, header) as writer:
        for tick in range(steps + 1):
            state = {"x": float(TRACK.centerline[0][0]) + tick / 10,
                     "y": float(TRACK.centerline[0][1]), "heading": tick / 100,
                     "speed": tick / 2, "lateral_velocity": tick / 100,
                     "yaw_rate": tick / 500, "steer": tick / 300}
            writer.append(replay_frame(
                tick, state, steering_command=tick / (steps + 1), throttle_brake=-tick / (steps + 1),
                off_track=7 <= tick < 19,
                events=["off_track_start"] if tick == 7 else ["off_track_end"] if tick == 19 else
                       ["barrier_contact" if status == "dnf_crash" else "lap_complete"] if tick == steps else [],
                terminated=tick == steps,
            ))
        writer.finish({"status": status, "steps": steps, "elapsed_s": steps / header["tick_hz"],
                       "lap_time_s": steps / header["tick_hz"] if status == "finished" else None,
                       "trajectory_hash": "a" * 64})
    return path


class ReplayGroupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = [recording(self.root / "one.json", "one", 30), recording(self.root / "two.json", "two", 60)]
        self.path = self.root / "group.json"
        self.group = create_replay_group(self.path, self.paths, labels=["First", "Second"])

    def replace_manifest(self, change):
        manifest = copy.deepcopy(self.group.manifest)
        change(manifest)
        self.path.write_text(json.dumps(manifest))

    def test_round_trip_separate_schema_identities_hashes_and_unchanged_inputs(self):
        before = [p.read_bytes() for p in self.paths]
        group = load_replay_group(self.path)
        self.assertEqual(group, self.group)
        self.assertEqual(group.final_tick, 60)
        self.assertEqual(group.tick_hz, CONFIG.tick_hz)
        self.assertEqual(group.manifest["schema_version"], "1.0")
        self.assertEqual(group.manifest["end_behavior"], "hold_final_pose")
        self.assertEqual([p.participant_id for p in group.participants], ["one", "two"])
        for index, participant in enumerate(group.participants):
            self.assertEqual(participant.data, load_replay(self.paths[index]))
            self.assertEqual(participant.data["schema_version"], "1.1")
        self.assertEqual(before, [p.read_bytes() for p in self.paths])

    def test_identity_mismatches_rejected_even_for_individually_valid_recordings(self):
        changes = {
            "engine": lambda h: h["engine"].update(code_hash="b" * 64),
            "config": lambda h: h["engine"].update(config_hash="b" * 64),
            "tick rate": lambda h: h.update(tick_hz=60),
            "track id": lambda h: h["track"]["meta"].update(id="other"),
            "track content": lambda h: h["track"].update(runoff_width=10.0),
            "car": lambda h: h["car"].update(length_m=6.0),
        }
        for name, change in changes.items():
            with self.subTest(identity=name):
                other = recording(self.root / "other.json", "other", 30, change=change)
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    create_replay_group(self.root / "bad.json", [self.paths[0], other])
                self.assertFalse((self.root / "bad.json").exists())

    def test_unique_participants_files_palette_and_recorded_identity_required(self):
        duplicate = recording(self.root / "duplicate.json", "one", 60)
        with self.assertRaisesRegex(ValueError, "identities must be unique"):
            create_replay_group(self.root / "bad.json", [self.paths[0], duplicate])
        mutations = [lambda m: m["participants"][1].update(participant_id="one"),
                     lambda m: m["participants"][1].update(file="one.json"),
                     lambda m: m["participants"][1].update(color=m["participants"][0]["color"]),
                     lambda m: m["participants"][0].update(participant_id="forged"),
                     lambda m: m["participants"][0].update(label="bad\nlabel")]
        for change in mutations:
            self.replace_manifest(change)
            with self.assertRaises(ValueError):
                load_replay_group(self.path)

    def test_common_exact_standing_tick_zero_required(self):
        # Rebuild through the writer to keep all single-replay integrity checks.
        for field, value in (("x", 1.0), ("speed", 1.0), ("heading", 1.0)):
            data = load_replay(self.paths[1])
            data["frames"][0][field] = value
            with ReplayWriter(self.paths[1], {k: data[k] for k in
                              ("schema_version", "kind", "engine", "source", "tick_hz", "car", "track")}) as writer:
                for frame in data["frames"]:
                    writer.append(frame)
                writer.finish(data["result"])
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "tick-zero|standing start"):
                create_replay_group(self.root / "bad.json", self.paths)
            recording(self.paths[1], "two", 60)

    def test_content_and_payload_hash_mismatch_and_malformed_replay_rejected(self):
        self.replace_manifest(lambda m: m["participants"][0].update(file_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "file SHA-256 mismatch"):
            load_replay_group(self.path)
        self.replace_manifest(lambda m: m["participants"][0].update(payload_hash="0" * 64))
        with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
            load_replay_group(self.path)
        self.path.write_text(json.dumps(self.group.manifest))
        self.paths[0].write_bytes(self.paths[0].read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "file SHA-256 mismatch"):
            load_replay_group(self.path)
        # Even a changed file hash cannot make corrupted embedded data valid.
        import hashlib
        data = json.loads(self.paths[0].read_bytes())
        data["frames"][1]["speed"] = 987
        raw = json.dumps(data).encode()
        self.paths[0].write_bytes(raw)
        self.replace_manifest(lambda m: m["participants"][0].update(file_sha256=hashlib.sha256(raw).hexdigest()))
        with self.assertRaisesRegex(ValueError, "replay payload hash mismatch"):
            load_replay_group(self.path)

    def test_references_reject_traversal_absolute_windows_and_symlink_files_or_parents(self):
        for reference in ("../one.json", "/one.json", "sub/../one.json", "./one.json", "sub//one.json",
                          "C:/one.json", "sub\\one.json", "one.json/", ""):
            self.replace_manifest(lambda m: m["participants"][0].update(file=reference))
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                load_replay_group(self.path)
        (self.root / "link.json").symlink_to(self.paths[0])
        (self.root / "link-dir").symlink_to(self.root, target_is_directory=True)
        for reference in ("link.json", "link-dir/one.json"):
            self.replace_manifest(lambda m: m["participants"][0].update(file=reference))
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                load_replay_group(self.path)
        self.path.write_text(json.dumps(self.group.manifest))
        (self.root / "manifest-link.json").symlink_to(self.path)
        with self.assertRaises(ValueError):
            load_replay_group(self.root / "manifest-link.json")
        with self.assertRaises(ValueError):
            load_replay_group(self.root / "link-dir/group.json")
        os.mkfifo(self.root / "fifo")
        self.replace_manifest(lambda m: m["participants"][0].update(file="fifo"))
        with self.assertRaisesRegex(ValueError, "regular files"):
            load_replay_group(self.path)

    def test_published_file_manifest_and_total_byte_limits(self):
        self.assertEqual(MAX_GROUP_BYTES, 32 * 1024 * 1024)
        self.assertEqual(MAX_MANIFEST_BYTES, 16 * 1024)
        for entries in ([], self.group.manifest["participants"] * 3):
            self.replace_manifest(lambda m: m.update(participants=entries))
            with self.assertRaisesRegex(ValueError, "1 to 4"):
                load_replay_group(self.path)
        self.path.write_text(json.dumps(self.group.manifest))
        total = self.path.stat().st_size + sum(p.stat().st_size for p in self.paths)
        with mock.patch("racing_event.replay_group.MAX_GROUP_BYTES", total):
            load_replay_group(self.path)
        with mock.patch("racing_event.replay_group.MAX_GROUP_BYTES", total - 1):
            with self.assertRaisesRegex(ValueError, "byte limit"):
                load_replay_group(self.path)
        with mock.patch("racing_event.replay_group.MAX_REPLAY_BYTES", self.paths[0].stat().st_size - 1):
            with self.assertRaisesRegex(ValueError, "byte limit"):
                load_replay_group(self.path)
        self.path.write_bytes(self.path.read_bytes() + b" " * MAX_MANIFEST_BYTES)
        with self.assertRaisesRegex(ValueError, "byte limit"):
            load_replay_group(self.path)

    def test_strict_version_alignment_end_policy_fields_duplicate_keys_and_nonfinite(self):
        for field, value in (("schema_version", "1.1"), ("kind", "racing_event_replay"),
                             ("alignment", "offset"), ("end_behavior", "loop"), ("extra", True)):
            self.replace_manifest(lambda m: m.update({field: value}))
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_replay_group(self.path)
        for raw in (b'{"kind":1,"kind":2}', b'{"kind":NaN}'):
            self.path.write_bytes(raw)
            with self.assertRaises(ValueError):
                load_replay_group(self.path)

    def test_builder_never_clobbers_sources_existing_manifest_or_symlink(self):
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            create_replay_group(self.path, self.paths)
        self.assertEqual(before, self.path.read_bytes())
        source = self.paths[0].read_bytes()
        with self.assertRaisesRegex(ValueError, "distinct"):
            create_replay_group(self.paths[0], self.paths)
        self.assertEqual(source, self.paths[0].read_bytes())
        (self.root / "out-link.json").symlink_to(self.paths[0])
        with self.assertRaises(FileExistsError):
            create_replay_group(self.root / "out-link.json", self.paths)
        with self.assertRaisesRegex(ValueError, "below"):
            create_replay_group(self.root / "out.json", [ROOT / "artifacts/replay/pygame-smoke.json"])
        with self.assertRaises(ValueError):
            create_replay_group(self.root / "out.json", self.paths * 3)
        self.assertEqual(list(self.root.glob(".replay-group-*")), [])


if __name__ == "__main__":
    unittest.main()
