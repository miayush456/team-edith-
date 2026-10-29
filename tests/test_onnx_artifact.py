from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from racing_event.evaluation import evaluate_participant_artifact
from racing_event.observation import RunningObservationNormalizer
from racing_event.submission import load_submission
from racing_event.vector_env import VectorizedRacingEnv
from tests.test_model_artifact import write_artifact, CONFIG, TRACK


@unittest.skipUnless(importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
                     "optional ONNX checks require requirements-evaluator.lock")
class OnnxArtifactTests(unittest.TestCase):
    def setUp(self):
        import onnx
        from tools.export_onnx import export
        self.onnx = onnx
        self.export = export
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.target = self.root / "source", self.root / "submission"
        weights = np.full((2, 23), .02, dtype=np.float32)
        write_artifact(self.source, layers=[("actor", weights, np.asarray([.1, .3], dtype=np.float32), "tanh")])
        self.export(self.source, self.target)

    def mutate(self, callback):
        model = self.onnx.load(self.target / "model.onnx")
        callback(model)
        self.onnx.save(model, self.target / "model.onnx")
        path = self.target / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["model_sha256"] = hashlib.sha256((self.target / "model.onnx").read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest))

    def test_numpy_roundtrip_preprocessing_and_deterministic_evaluation(self):
        original, artifact = load_submission(self.source), load_submission(self.target)
        observation = np.linspace(-1, 1, 23, dtype=np.float32)[None, :]
        np.testing.assert_allclose(original(observation), artifact(observation), atol=1e-7)
        np.testing.assert_array_equal(artifact(observation), artifact(observation))
        env = VectorizedRacingEnv(TRACK, CONFIG, reward=None, n_envs=1)
        first = evaluate_participant_artifact(env, artifact, max_steps=8)
        second = evaluate_participant_artifact(env, artifact, max_steps=8)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.total_reward, 0)
        with self.assertRaises(ValueError):
            artifact(np.zeros((2, 23), dtype=np.float32))
        with self.assertRaises(ValueError):
            artifact(np.full((1, 23), np.nan, dtype=np.float32))

    def test_reference_actor_and_frozen_normalizer_preserve_mapping(self):
        import torch
        from racing_event.ppo import FixedMLPActorCritic
        source = self.root / "reference"
        source.mkdir()
        torch.manual_seed(17)
        policy = FixedMLPActorCritic(23)
        policy.save_artifact(source / "policy.pt")
        normalizer = RunningObservationNormalizer(23)
        normalizer.update(np.random.default_rng(17).normal(size=(10, 23)))
        normalizer.freeze()
        load_submission(self.source).contract.save(source / "training_contract.json", normalizer=normalizer)
        self.export(source, self.root / "reference-onnx", reference=True)
        loaded = load_submission(self.root / "reference-onnx")
        for obs in np.random.default_rng(8).uniform(-1, 1, size=(10, 23)).astype(np.float32):
            expected = policy.act(normalizer.transform(obs[None, :], training=False), deterministic=True)
            np.testing.assert_allclose(expected, loaded(obs[None, :]), atol=1e-6)
        self.assertIsNone(loaded.normalizer)

    def test_affine_per_field_clipping_preserves_mapping(self):
        source, target = self.root / "affine", self.root / "affine-onnx"
        write_artifact(source, layers=[("actor", np.full((2, 23), .02, dtype=np.float32),
                                       np.asarray([.1, .3], dtype=np.float32), "tanh")],
                       preprocessing={"kind": "affine", "mean": [.2] * 23,
                                      "scale": [.5] * 23, "clip": np.linspace(.1, 1, 23).tolist()})
        self.export(source, target)
        original, exported = load_submission(source), load_submission(target)
        for observation in np.random.default_rng(14).uniform(-4, 4, (8, 23)).astype(np.float32):
            np.testing.assert_allclose(original(observation[None, :]), exported(observation[None, :]), atol=1e-7)

    def test_participant_residual_architecture_is_not_the_fixed_reference_model(self):
        from onnx import helper as h, numpy_helper as nh
        rng = np.random.default_rng(32)
        weights = rng.normal(0, .1, (23, 8)).astype(np.float32)
        output_weights = rng.normal(0, .1, (8, 2)).astype(np.float32)
        def change(model):
            graph = h.make_graph([
                h.make_node("MatMul", ["observations", "features"], ["dense"]),
                h.make_node("Relu", ["dense"], ["positive"]),
                h.make_node("Add", ["dense", "positive"], ["residual"]),
                h.make_node("MatMul", ["residual", "head"], ["logits"]),
                h.make_node("Tanh", ["logits"], ["actions"]),
            ], "custom_residual", list(model.graph.input), list(model.graph.output),
                initializer=[nh.from_array(weights, "features"), nh.from_array(output_weights, "head")])
            model.graph.CopyFrom(graph)
        self.mutate(change)
        artifact = load_submission(self.target)
        observation = rng.normal(size=(1, 23)).astype(np.float32)
        features = observation @ weights
        expected = np.tanh((features + np.maximum(features, 0)) @ output_weights)
        np.testing.assert_allclose(artifact(observation), expected, atol=1e-7)

    def test_graph_byte_node_and_parameter_limits_precede_runtime_construction(self):
        from racing_event.onnx_artifact import MAX_MODEL_BYTES, validate_model
        with self.assertRaisesRegex(ValueError, "byte limit"):
            validate_model(b"x" * (MAX_MODEL_BYTES + 1), 23)
        original = (self.target / "model.onnx").read_bytes()
        def too_many_nodes(model):
            template = model.graph.node[0]
            for _ in range(129):
                model.graph.node.add().CopyFrom(template)
        def too_many_parameters(model):
            model.graph.initializer[0].dims[:] = [1024, 1024]
        for change, error in ((too_many_nodes, "nodes"), (too_many_parameters, "parameter limit")):
            (self.target / "model.onnx").write_bytes(original)
            self.mutate(change)
            with self.assertRaisesRegex(ValueError, error):
                load_submission(self.target)

    def test_rejects_external_tensors_before_loading_external_files(self):
        def change(model):
            tensor = model.graph.initializer[0]
            tensor.data_location = self.onnx.TensorProto.EXTERNAL
            tensor.external_data.add(key="location", value="../../private.bin")
        self.mutate(change)
        with self.assertRaisesRegex(ValueError, "external"):
            load_submission(self.target)

    def test_rejects_onnx_normalizer_before_constructing_declared_arrays(self):
        path = self.target / "training_contract.json"
        contract = json.loads(path.read_text())
        contract["trainer"]["normalizer"] = {"size": 10**12}
        path.write_text(json.dumps(contract))
        manifest_path = self.target / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["training_contract_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with mock.patch("racing_event.observation.RunningObservationNormalizer.from_state_dict") as construct:
            with self.assertRaisesRegex(ValueError, "no separate normalizer"):
                load_submission(self.target)
        construct.assert_not_called()

    def test_rejects_unapproved_operators_domains_dynamic_shapes_and_nonfinite_data(self):
        original = (self.target / "model.onnx").read_bytes()
        callbacks = [
            lambda m: setattr(m.graph.node[0], "op_type", "RandomNormal"),
            lambda m: setattr(m.graph.node[0], "domain", "participant.custom"),
            lambda m: setattr(m.graph.input[0].type.tensor_type.shape.dim[0], "dim_param", "batch"),
            lambda m: setattr(m.graph.initializer[0], "raw_data", np.full((2, 23), np.nan, dtype=np.float32).tobytes()),
            lambda m: setattr(m.opset_import[0], "version", 18),
        ]
        for callback in callbacks:
            with self.subTest(callback=callback):
                (self.target / "model.onnx").write_bytes(original)
                self.mutate(callback)
                with self.assertRaises((ValueError, self.onnx.checker.ValidationError)):
                    load_submission(self.target)

    def test_rejects_hash_tampering_and_symlink_or_extra_files(self):
        original = (self.target / "model.onnx").read_bytes()
        (self.target / "model.onnx").write_bytes(original + b"bad")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_submission(self.target)
        (self.target / "model.onnx").write_bytes(original)
        (self.target / "reward.py").write_text("raise RuntimeError('must never execute')")
        with self.assertRaisesRegex(ValueError, "exactly"):
            load_submission(self.target)
        (self.target / "reward.py").unlink()
        (self.root / "linked").symlink_to(self.target, target_is_directory=True)
        with self.assertRaises(ValueError):
            load_submission(self.root / "linked")

    def test_export_never_clobbers_source_or_existing_output(self):
        before = {p.name: p.read_bytes() for p in self.target.iterdir()}
        with self.assertRaisesRegex(ValueError, "new directory"):
            self.export(self.source, self.target)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.target.iterdir()})


if __name__ == "__main__":
    unittest.main()
