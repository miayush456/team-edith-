from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from racing_event.evaluation import evaluate_participant_artifact, validate_evaluation_contract
from racing_event.model_artifact import load_participant_artifact
from racing_event.observation import RunningObservationNormalizer, TrainingContract
from racing_event.track import TrackGeometry
from racing_event.trajectory import engine_code_hash
from racing_event.config import SimulationConfig
from racing_event.vector_env import VectorizedRacingEnv


ROOT = Path(__file__).resolve().parents[1]
TRACK = TrackGeometry.from_file(ROOT / "tests/fixtures/tracks/synthetic_oval.json")
FLOWING = TrackGeometry.from_file(ROOT / "tests/fixtures/tracks/synthetic_flowing.json")
CONFIG = SimulationConfig.from_file(ROOT / "config/simulation.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_artifact(
    root: Path,
    *,
    preprocessing: dict[str, object] | None = None,
    normalizer: RunningObservationNormalizer | None = None,
    layers: list[tuple[str, np.ndarray, np.ndarray, str]] | None = None,
) -> None:
    root.mkdir(parents=True)
    contract = TrainingContract.for_config(
        CONFIG,
        track_ids=(TRACK.track_id,),
        track_hashes=(TRACK.content_hash,),
    )
    contract.save(root / "training_contract.json", normalizer=normalizer)
    layers = layers or [
        ("dense_0", np.zeros((4, 23), dtype=np.float32), np.zeros(4, dtype=np.float32), "relu"),
        ("dense_1", np.zeros((2, 4), dtype=np.float32), np.zeros(2, dtype=np.float32), "tanh"),
    ]
    arrays: dict[str, np.ndarray] = {}
    manifest_layers: list[dict[str, str]] = []
    for name, weights, bias, activation in layers:
        weights_name = f"{name}_weights"
        bias_name = f"{name}_bias"
        arrays[weights_name] = weights
        arrays[bias_name] = bias
        manifest_layers.append({
            "name": name,
            "weights": weights_name,
            "bias": bias_name,
            "activation": activation,
        })
    np.savez(root / "weights.npz", **arrays)
    preprocessing = preprocessing or {"kind": "identity"}
    manifest = {
        "format": "synapse_numpy_mlp",
        "format_version": 1,
        "environment_code_hash": engine_code_hash(),
        "observation": {"schema_version": "1.0", "size": 23, "dtype": "float32"},
        "action": {
            "schema_version": "1.0",
            "size": 2,
            "dtype": "float32",
            "bounds": [[-1.0, 1.0], [-1.0, 1.0]],
            "output_activation": "tanh",
        },
        "preprocessing": preprocessing,
        "layers": manifest_layers,
        "weights_sha256": _sha256(root / "weights.npz"),
        "training_contract_sha256": _sha256(root / "training_contract.json"),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


class ModelArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_variable_mlp_is_deterministic_and_bounded(self) -> None:
        root = Path(self.temp.name) / "artifact"
        layers = [
            ("first", np.full((4, 23), 0.01, dtype=np.float32), np.zeros(4, dtype=np.float32), "relu"),
            ("last", np.full((2, 4), 0.02, dtype=np.float32), np.zeros(2, dtype=np.float32), "tanh"),
        ]
        write_artifact(root, layers=layers)
        artifact = load_participant_artifact(root)
        observations = np.ones((2, 23), dtype=np.float32)
        first = artifact(observations)
        second = artifact(observations)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (2, 2))
        self.assertTrue(np.all(first <= 1.0))
        self.assertEqual(len(artifact.artifact_sha256), 64)

    def test_frozen_normalizer_and_affine_preprocessing(self) -> None:
        root = Path(self.temp.name) / "normalized"
        normalizer = RunningObservationNormalizer(23)
        normalizer.update(np.ones((2, 23), dtype=np.float32))
        normalizer.freeze()
        write_artifact(root, normalizer=normalizer, preprocessing={"kind": "contract_normalizer"})
        artifact = load_participant_artifact(root)
        before = artifact.normalizer.state_dict() if artifact.normalizer is not None else None
        artifact(np.zeros((1, 23), dtype=np.float32))
        self.assertEqual(before, artifact.normalizer.state_dict() if artifact.normalizer is not None else None)

        affine_root = Path(self.temp.name) / "affine"
        write_artifact(
            affine_root,
            preprocessing={
                "kind": "affine",
                "mean": [0.0] * 23,
                "scale": [1.0] * 23,
                "clip": [1.0] * 23,
            },
        )
        self.assertEqual(load_participant_artifact(affine_root).observation_size, 23)

    def test_loader_rejects_extra_files_and_hash_changes(self) -> None:
        root = Path(self.temp.name) / "bad"
        write_artifact(root)
        (root / "reward.py").write_text("raise RuntimeError('must never import')\n")
        with self.assertRaisesRegex(ValueError, "exactly"):
            load_participant_artifact(root)
        (root / "reward.py").unlink()
        np.savez(root / "weights.npz", dense_0_weights=np.zeros((4, 23), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_participant_artifact(root)

    def test_holdout_evaluation_does_not_execute_reward(self) -> None:
        root = Path(self.temp.name) / "holdout"
        write_artifact(root)
        artifact = load_participant_artifact(root)
        env = VectorizedRacingEnv(FLOWING, CONFIG, reward=None, n_envs=1)
        with self.assertRaises(ValueError):
            validate_evaluation_contract(env, artifact.contract, artifact.normalizer)
        result = evaluate_participant_artifact(env, artifact, max_steps=2)
        self.assertEqual(result.steps, 2)
        self.assertEqual(len(result.trajectory_hash), 64)


if __name__ == "__main__":
    unittest.main()
