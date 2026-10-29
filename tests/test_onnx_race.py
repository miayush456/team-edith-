from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from racing_event.race import RaceEntrant, run_onnx_race
from racing_event.replay_group import load_replay_group
from racing_event.trajectory import load_replay
from tests.test_model_artifact import CONFIG, FLOWING, TRACK, write_artifact


HAS_ONNX = bool(importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"))


@unittest.skipUnless(HAS_ONNX, "ONNX race checks require requirements-evaluator.lock")
class OnnxRaceTests(unittest.TestCase):
    def setUp(self) -> None:
        from tools.export_onnx import export

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        source_a, source_b = root / "source-a", root / "source-b"
        write_artifact(source_a)
        write_artifact(source_b, layers=[("actor", np.full((2, 23), .02, dtype=np.float32),
                                         np.asarray([.2, .4], dtype=np.float32), "tanh")])
        self.artifact_a, self.artifact_b = root / "team-a", root / "team-b"
        export(source_a, self.artifact_a)
        export(source_b, self.artifact_b)
        self.config = CONFIG
        self.tracks = (TRACK, FLOWING)

    def test_runs_models_on_shared_tracks_and_emits_renderer_inputs(self) -> None:
        output = Path(self.temp.name) / "race"
        report = run_onnx_race(
            [RaceEntrant("team-a", self.artifact_a, "A"), RaceEntrant("team-b", self.artifact_b, "B")],
            self.tracks,
            self.config,
            output_dir=output,
            emit_replays=True,
            max_steps=4,
            repeats=2,
        )
        data = report.to_dict()
        self.assertEqual(data["kind"], "synapse_onnx_race")
        self.assertEqual(data["race_type"], "independent_ghosts")
        self.assertIsNone(data["ranking_policy"])
        self.assertEqual(len(data["entrants"]), 2)
        self.assertEqual(len(data["tracks"]), 2)
        self.assertEqual(json.loads((output / "race.json").read_text())["kind"], "synapse_onnx_race")
        for track in data["tracks"]:
            group_path = output / track["replay_group"]
            group = load_replay_group(group_path)
            self.assertEqual([item.participant_id for item in group.participants], ["team-a", "team-b"])
            for result in track["results"]:
                replay = output / result["replay"]
                self.assertEqual(load_replay(replay)["schema_version"], "1.1")
                self.assertEqual(result["result"]["steps"], 4)
                self.assertEqual(len(result["repeats"]), 2)
                self.assertEqual(result["repeats"][0], result["repeats"][1])

    def test_rejects_non_onnx_artifact_and_does_not_leave_output(self) -> None:
        from tests.test_model_artifact import write_artifact

        numpy_artifact = Path(self.temp.name) / "numpy-artifact"
        write_artifact(numpy_artifact)
        output = Path(self.temp.name) / "rejected"
        with self.assertRaisesRegex(ValueError, "only synapse_onnx_policy"):
            run_onnx_race(
                [RaceEntrant("numpy", numpy_artifact)], self.tracks[:1], self.config,
                output_dir=output, max_steps=2,
            )
        self.assertFalse(output.exists())

    def test_rejects_duplicate_artifacts_and_existing_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "artifacts must be unique"):
            run_onnx_race(
                [RaceEntrant("a", self.artifact_a), RaceEntrant("b", self.artifact_a)],
                self.tracks[:1], self.config, max_steps=2,
            )
        output = Path(self.temp.name) / "existing"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            run_onnx_race([RaceEntrant("a", self.artifact_a)], self.tracks[:1], self.config, output_dir=output)

    def test_report_only_mode_does_not_embed_track_files(self) -> None:
        output = Path(self.temp.name) / "report-only"
        report = run_onnx_race(
            [RaceEntrant("team-a", self.artifact_a)], self.tracks[:1], self.config,
            output_dir=output, max_steps=2,
        )
        self.assertIsNone(report.to_dict()["tracks"][0]["replay_group"])
        self.assertEqual([path.name for path in output.iterdir()], ["race.json"])


if __name__ == "__main__":
    unittest.main()
