"""Export a safe NumPy artifact or optional reference actor to ONNX profile v1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from racing_event.model_artifact import load_participant_artifact
from racing_event.observation import TrainingContract
from racing_event.onnx_artifact import FORMAT, load_onnx_artifact
from racing_event.trajectory import engine_code_hash
from racing_event.participant_model import build_policy_from_model, load_policy_state


def _normalizer_values(normalizer):
    if normalizer is None:
        return None
    return normalizer.mean, normalizer.std, normalizer.clip


def export_torch_policy(policy, contract, normalizer, output: Path, *, source_kind: str = "synapse") -> Path:
    """Export a participant actor using the pinned generic Torch path."""
    import torch
    from torch import nn
    import onnx
    from onnx import helper as h, numpy_helper as nh

    if output.exists() or output.is_symlink():
        raise ValueError("ONNX output must be a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = policy.actor
            values = _normalizer_values(normalizer)
            self.register_buffer("mean", None if values is None else torch.as_tensor(values[0], dtype=torch.float64))
            self.register_buffer("scale", None if values is None else torch.as_tensor(values[1], dtype=torch.float64))
            self.register_buffer("clip", None if values is None else torch.as_tensor(values[2], dtype=torch.float64))

        def forward(self, observations):
            values = observations
            if self.mean is not None:
                values = values.to(torch.float64)
                values = (values - self.mean) / self.scale
                values = torch.maximum(torch.minimum(values, self.clip), -self.clip).to(torch.float32)
            return torch.tanh(self.actor(values))

    wrapper = Wrapper().eval()
    temporary = Path(tempfile.mkdtemp(prefix=".onnx-", dir=output.parent))
    try:
        model_path = temporary / "model.onnx"
        dummy = torch.zeros((1, contract.observation.size), dtype=torch.float32)
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(model_path),
            input_names=["observations"],
            output_names=["actions"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        model = onnx.load(model_path)
        # Torch may emit Constant nodes for scalar arithmetic. Convert the
        # safe tensor form into initializers before the strict profile check.
        retained = []
        for node in model.graph.node:
            if node.op_type == "Constant" and len(node.attribute) == 1 and node.attribute[0].name == "value":
                tensor = node.attribute[0].t
                tensor.name = node.output[0]
                model.graph.initializer.append(tensor)
            else:
                retained.append(node)
        del model.graph.node[:]
        model.graph.node.extend(retained)
        model.ir_version = 10
        del model.metadata_props[:]
        onnx.save(model, model_path)
        contract.save(temporary / "training_contract.json")
        manifest = {
            "format": FORMAT,
            "format_version": 1,
            "profile": "static_vector_cpu_opset17_v1",
            "environment_code_hash": engine_code_hash(),
            "source_kind": source_kind,
        }
        for name, key in (("model.onnx", "model_sha256"), ("training_contract.json", "training_contract_sha256")):
            manifest[key] = hashlib.sha256((temporary / name).read_bytes()).hexdigest()
        (temporary / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        load_onnx_artifact(temporary)
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def export(source: Path, output: Path, *, reference: bool = False, model: Path | None = None) -> Path:
    import onnx
    from onnx import helper as h, numpy_helper as nh

    if output.exists() or output.is_symlink():
        raise ValueError("ONNX output must be a new directory")
    if reference:
        contract, normalizer = TrainingContract.load(source / "training_contract.json")
        metadata = json.loads((source / "policy.json").read_text(encoding="utf-8"))
        if model is not None or metadata.get("format") == "racing_event_actor_critic":
            model_path = model or (source / "model_source.py")
            policy, _ = build_policy_from_model(model_path, contract.observation.size)
            load_policy_state(policy, source / "policy.pt")
            return export_torch_policy(policy, contract, normalizer, output, source_kind="synapse")
        from racing_event.ppo import FixedMLPActorCritic
        policy, _ = FixedMLPActorCritic.load_artifact(source / "policy.pt", source / "policy.json")
        state = policy.state_dict()
        layers = [(state[f"{name}.weight"].numpy(), state[f"{name}.bias"].numpy(), "tanh")
                  for name in ("trunk.0", "trunk.2", "policy_mean")]
        mean, scale, clip = (None, None, None) if normalizer is None else (normalizer.mean, normalizer.std, normalizer.clip)
    else:
        artifact = load_participant_artifact(source)
        if artifact.manifest["environment_code_hash"] != engine_code_hash():
            raise ValueError("source artifact belongs to a different engine")
        contract, normalizer = artifact.contract, artifact.normalizer
        layers = [(layer.weights, layer.bias, layer.activation) for layer in artifact.layers]
        mean, scale, clip = ((normalizer.mean, normalizer.std, normalizer.clip) if normalizer is not None
                             else (artifact._affine_mean, artifact._affine_scale, artifact._affine_clip))
    nodes, initializers = [], []

    def constant(name, value):
        initializers.append(nh.from_array(np.asarray(value), name))
        return name

    value = "observations"
    if mean is not None:
        # Match the reference normalizer's float64 arithmetic, then cast once.
        dtype = np.float64 if normalizer is not None else np.float32
        if normalizer is not None:
            nodes.append(h.make_node("Cast", [value], ["obs64"], to=11))
            value = "obs64"
        nodes += [h.make_node("Sub", [value, constant("mean", np.asarray(mean, dtype=dtype))], ["centered"]),
                  h.make_node("Div", ["centered", constant("scale", np.asarray(scale, dtype=dtype))], ["scaled"])]
        bounds = np.asarray(clip, dtype=dtype)
        low, high = constant("clip_min", -bounds), constant("clip_max", bounds)
        if bounds.ndim == 0:
            nodes.append(h.make_node("Clip", ["scaled", low, high], ["clipped"]))
        else:
            # ONNX Clip accepts scalar bounds only; affine preprocessing permits
            # independently bounded fields, preserved through broadcast Min/Max.
            nodes += [h.make_node("Max", ["scaled", low], ["lower_bounded"]),
                      h.make_node("Min", ["lower_bounded", high], ["clipped"])]
        value = "clipped"
        if normalizer is not None:
            nodes.append(h.make_node("Cast", [value], ["normalized"], to=1))
            value = "normalized"
    for index, (weights, bias, activation) in enumerate(layers):
        dense = f"dense_{index}"
        final = index == len(layers) - 1
        nodes.append(h.make_node("Gemm", [value, constant(f"w{index}", weights), constant(f"b{index}", bias)], [dense], transB=1))
        value = "actions" if final else f"features_{index}"
        nodes.append(h.make_node({"tanh": "Tanh", "relu": "Relu", "identity": "Identity"}[activation], [dense], [value]))
    graph = h.make_graph(nodes, "participant_policy", [h.make_tensor_value_info("observations", 1, [1, contract.observation.size])],
                         [h.make_tensor_value_info("actions", 1, [1, 2])], initializer=initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], ir_version=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".onnx-", dir=output.parent))
    try:
        (temporary / "model.onnx").write_bytes(model.SerializeToString())
        contract.save(temporary / "training_contract.json")
        manifest = {"format": FORMAT, "format_version": 1, "profile": "static_vector_cpu_opset17_v1", "environment_code_hash": engine_code_hash(), "source_kind": "synapse"}
        for name, key in (("model.onnx", "model_sha256"), ("training_contract.json", "training_contract_sha256")):
            manifest[key] = hashlib.sha256((temporary / name).read_bytes()).hexdigest()
        (temporary / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        load_onnx_artifact(temporary)
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", action="store_true", help="source is optional Torch reference actor, not NumPy artifact")
    parser.add_argument("--model", type=Path, help="participant model.py for a pluggable reference actor")
    args = parser.parse_args()
    try:
        print(export(args.source, args.output, reference=args.reference, model=args.model))
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"ONNX export failed: {exc}\n")


if __name__ == "__main__":
    main()
