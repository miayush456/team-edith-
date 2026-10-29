from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from racing_event.config import SimulationConfig
from racing_event.evaluation import evaluate_policy
from racing_event.observation import TrainingContract
from racing_event.track import TrackGeometry
from racing_event.trajectory import (
    ReplayWriter, engine_code_hash, load_replay, replay_header, replay_frame, replay_result, validate_replay,
)
from racing_event.vector_env import VectorizedRacingEnv
from tools.build_replay import build
from tools.export_replay import export


ROOT = Path(__file__).resolve().parents[1]
TRACK = TrackGeometry.from_file(ROOT / "tests/fixtures/tracks/synthetic_oval.json")
CONFIG = SimulationConfig.from_file(ROOT / "config/simulation.json")


def coast(observations: np.ndarray) -> np.ndarray:
    return np.zeros((len(observations), 2), dtype=np.float64)


def make_generic_artifact(root: Path) -> None:
    contract = TrainingContract.for_config(
        CONFIG,
        track_ids=(TRACK.track_id,),
        track_hashes=(TRACK.content_hash,),
    )
    contract_path = root / "training_contract.json"
    contract.save(contract_path)
    weights_path = root / "weights.npz"
    np.savez(
        weights_path,
        dense_0_weights=np.zeros((2, 23), dtype=np.float32),
        dense_0_bias=np.zeros(2, dtype=np.float32),
    )
    manifest = {
        "format": "synapse_numpy_mlp",
        "format_version": 1,
        "environment_code_hash": engine_code_hash(),
        "observation": {"schema_version": "1.0", "size": 23, "dtype": "float32"},
        "action": {
            "schema_version": "1.0", "size": 2, "dtype": "float32",
            "bounds": [[-1.0, 1.0], [-1.0, 1.0]], "output_activation": "tanh",
        },
        "preprocessing": {"kind": "identity"},
        "layers": [{"name": "dense_0", "weights": "dense_0_weights", "bias": "dense_0_bias", "activation": "tanh"}],
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "training_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")


class ReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "replay.json"
        self.header = replay_header(TRACK, CONFIG, participant_id="test", mode="evaluation")

    def record(self, steps: int = 3):
        env = VectorizedRacingEnv(TRACK, CONFIG, n_envs=1)
        with ReplayWriter(self.path, self.header) as writer:
            result = evaluate_policy(env, coast, max_steps=steps, frame_sink=writer.append)
            writer.finish(replay_result(result, CONFIG.tick_hz))
        return result, load_replay(self.path)

    def test_round_trip_initial_pose_tick_timing_and_observer_parity(self) -> None:
        result, data = self.record()
        without_writer = evaluate_policy(VectorizedRacingEnv(TRACK, CONFIG, n_envs=1), coast, max_steps=3)
        self.assertEqual(result, without_writer)
        self.assertEqual(data["result"]["trajectory_hash"], result.trajectory_hash)
        self.assertEqual(data["schema_version"], "1.1")
        self.assertEqual(len(data["frames"]), 4)
        self.assertEqual([f["tick"] for f in data["frames"]], [0, 1, 2, 3])
        self.assertEqual(data["result"]["elapsed_s"], 3 / CONFIG.tick_hz)
        self.assertIsNone(data["result"]["lap_time_s"])
        self.assertEqual(data["track"]["meta"]["content_hash"], TRACK.content_hash)
        np.testing.assert_array_equal(data["track"]["left_boundary"], TRACK.left_boundary)
        point, heading, _ = TRACK.point_at_s(TRACK.start_line_s)
        initial = data["frames"][0]
        self.assertEqual((initial["x"], initial["y"], initial["heading"]), (*point, heading))
        self.assertEqual(initial["speed"], 0)
        self.assertEqual(initial["events"], [])

    def test_independent_snapshots_cannot_mutate_authoritative_state(self) -> None:
        env = VectorizedRacingEnv(TRACK, CONFIG, n_envs=1)

        def bad_consumer(frame):
            frame["x"] = 999999
            frame["events"].append("fake")

        actual = evaluate_policy(env, coast, max_steps=2, frame_sink=bad_consumer)
        expected = evaluate_policy(VectorizedRacingEnv(TRACK, CONFIG, n_envs=1), coast, max_steps=2)
        self.assertEqual(actual, expected)

    def test_same_run_produces_same_payload_and_source_identity(self) -> None:
        self.record()
        first = self.path.read_bytes()
        self.record()
        self.assertEqual(first, self.path.read_bytes())
        data = load_replay(self.path)
        self.assertEqual(set(data["engine"]), {"code_hash", "config_hash"})

    def test_corruption_version_finite_tick_and_terminal_rejections(self) -> None:
        _, data = self.record()
        mutations = (
            lambda d: d.update(schema_version="99.0"),
            lambda d: d["frames"][1].update(x=float("nan")),
            lambda d: d["frames"][1].update(tick=0),
            lambda d: d["frames"][1].update(tick=True),
            lambda d: d["frames"][1].update(x=42.0),
            lambda d: d["frames"][0].update(terminated=True),
            lambda d: d["frames"][1].update(terminated=True),
            lambda d: d["frames"][-1].update(terminated=True, truncated=True),
            lambda d: d["frames"][1].update(events=["unsupported"]),
            lambda d: d["track"]["centerline"][0].__setitem__(0, 42.0),
            lambda d: d["result"].update(steps=999),
            lambda d: d["result"].update(elapsed_s=999),
            lambda d: d["result"].update(lap_time_s=1.0),
            lambda d: d.update(extra="ignored?"),
        )
        for mutation in mutations:
            candidate = copy.deepcopy(data)
            mutation(candidate)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_replay(candidate)

    def test_result_consistency_and_crash_precedence(self) -> None:
        _, data = self.record()
        for status in ("finished", "dnf_crash", "dnf_stuck", "dnf_backward"):
            candidate = copy.deepcopy(data)
            candidate["result"]["status"] = status
            with self.subTest(status=status), self.assertRaises(ValueError):
                validate_replay(candidate)
        # A crash and finish on the same step cannot be presented as a win.
        last = data["frames"][-1]
        last.update(terminated=True, events=["lap_complete", "barrier_contact"])
        data["result"].update(status="finished", lap_time_s=0.1)
        with self.assertRaisesRegex(ValueError, "terminal event"):
            validate_replay(data)

    def test_abort_does_not_publish_partial_or_destroy_previous_replay(self) -> None:
        self.record()
        previous = self.path.read_bytes()
        with self.assertRaises(RuntimeError):
            with ReplayWriter(self.path, self.header):
                raise RuntimeError("simulation failed")
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])
        with ReplayWriter(self.path, self.header):
            pass  # No finish: do not publish an empty run.
        self.assertEqual(self.path.read_bytes(), previous)

    def test_writer_enforces_frame_order_and_no_post_terminal_frames(self) -> None:
        env = VectorizedRacingEnv(TRACK, CONFIG, n_envs=1)
        env.reset(evaluation=True)
        state = {k: float(v[0]) for k, v in env.state_to_dict().items()}
        with ReplayWriter(self.path, self.header) as writer:
            writer.append(replay_frame(0, state))
            with self.assertRaises(ValueError):
                writer.append(replay_frame(2, state))
            writer.append(replay_frame(1, state, truncated=True))
            with self.assertRaises(ValueError):
                writer.append(replay_frame(2, state))

    def test_reader_limits_and_duplicate_keys(self) -> None:
        from unittest.mock import patch

        self.record()
        with patch("racing_event.trajectory.MAX_REPLAY_BYTES", 100), self.assertRaisesRegex(ValueError, "byte limit"):
            load_replay(self.path)
        self.path.write_text('{"schema_version":"1.1","schema_version":"1.1"}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_replay(self.path)

    def test_build_is_offline_injection_safe_and_rejects_bad_input(self) -> None:
        self.header["source"]["participant_id"] = '</script><img src=x onerror="alert(1)">'
        self.record()
        output = self.path.with_suffix(".html")
        build(self.path, output)
        html = output.read_text()
        self.assertNotIn('</script><img', html)
        self.assertIn('\\u003c/script\\u003e', html)
        self.assertNotIn("__REPLAY_", html)
        self.assertIn('id="replay-data"', html)
        # Extract the inert data node and validate exactly what the UI consumes.
        encoded = html.split('<script type="application/json" id="replay-data">')[1].split('</script>')[0]
        self.assertEqual(validate_replay(json.loads(encoded)), load_replay(self.path))
        self.path.write_text("{}")
        before = output.read_bytes()
        with self.assertRaises(ValueError):
            build(self.path, output)
        self.assertEqual(output.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            build(self.path, self.path)

    def test_scripted_export_tool_reads_back(self) -> None:
        args = argparse.Namespace(
            track=ROOT / "tests/fixtures/tracks/synthetic_oval.json", config=ROOT / "config/simulation.json",
            weights=None, metadata=None, contract=None, participant_id="smoke",
            steps=5, output=self.path,
        )
        result = export(args)
        data = load_replay(self.path)
        self.assertEqual(data["source"]["mode"], "scripted")
        self.assertEqual(data["result"]["steps"], result["steps"])
        self.assertEqual(len(data["frames"]), 6)
        args.output = args.config
        with self.assertRaisesRegex(ValueError, "overwrite"):
            export(args)

    def test_generic_artifact_export_tool_reads_back(self) -> None:
        artifact_dir = Path(self.tmp.name) / "artifact"
        artifact_dir.mkdir()
        make_generic_artifact(artifact_dir)
        args = argparse.Namespace(
            artifact_dir=artifact_dir,
            track=ROOT / "tests/fixtures/tracks/synthetic_oval.json",
            config=ROOT / "config/simulation.json",
            weights=None,
            metadata=None,
            contract=None,
            participant_id="generic",
            steps=3,
            output=self.path,
        )
        result = export(args)
        data = load_replay(self.path)
        self.assertEqual(data["source"]["mode"], "evaluation")
        self.assertEqual(data["result"]["steps"], result["steps"])
        with self.assertRaisesRegex(ValueError, "artifact directory"):
            args.output = artifact_dir / "inside.json"
            export(args)

    def test_reference_multitrack_contract_exports_one_track_without_rewriting(self) -> None:
        from racing_event.ppo import FixedMLPActorCritic
        from racing_event.observation import RunningObservationNormalizer
        source = Path(self.tmp.name) / "reference"
        source.mkdir()
        policy = FixedMLPActorCritic(23)
        policy.save_artifact(source / "policy.pt")
        flowing = TrackGeometry.from_file(ROOT / "tests/fixtures/tracks/synthetic_flowing.json")
        contract = TrainingContract.for_config(CONFIG, track_ids=(TRACK.track_id, flowing.track_id),
                                             track_hashes=(TRACK.content_hash, flowing.content_hash))
        normalizer = RunningObservationNormalizer(23)
        normalizer.freeze()
        contract.save(source / "training_contract.json", normalizer=normalizer)
        before = (source / "training_contract.json").read_bytes()
        args = argparse.Namespace(
            track=ROOT / "tests/fixtures/tracks/synthetic_oval.json", config=ROOT / "config/simulation.json",
            weights=source / "policy.pt", metadata=source / "policy.json", contract=source / "training_contract.json",
            participant_id="multi-reference", steps=3, output=self.path,
        )
        export(args)
        self.assertEqual(load_replay(self.path)["result"]["steps"], 3)
        self.assertEqual((source / "training_contract.json").read_bytes(), before)
        payload = json.loads(args.config.read_text())
        payload["tick_hz"] = 60
        changed_path = source / "changed-config.json"
        changed_path.write_text(json.dumps(payload))
        args.config = changed_path
        with self.assertRaisesRegex(ValueError, "config hash"):
            export(args)

    def test_real_stuck_replay_preserves_engine_outcome(self) -> None:
        config = replace(CONFIG, episode=replace(CONFIG.episode, stuck_steps=2))
        header = replay_header(TRACK, config, mode="evaluation", participant_id="stuck")
        with ReplayWriter(self.path, header) as writer:
            result = evaluate_policy(VectorizedRacingEnv(TRACK, config, n_envs=1), coast, frame_sink=writer.append)
            writer.finish(replay_result(result, config.tick_hz))
        data = load_replay(self.path)
        self.assertEqual(data["result"]["status"], "dnf_stuck")
        self.assertTrue(data["frames"][-1]["truncated"])


if __name__ == "__main__":
    unittest.main()
