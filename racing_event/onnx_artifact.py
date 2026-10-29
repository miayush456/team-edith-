"""SYNAPSE ONNX vector-policy v1: bounded static graphs, CPU deterministic inference.

Architecture/weights belong to participants. This runtime profile supports
static dense/residual DAGs, not arbitrary ONNX programs. Use the isolated worker
for untrusted submissions: graph validation alone is not a security sandbox.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .model_artifact import _load_json, _require_hash
from .observation import TrainingContract

FORMAT = "synapse_onnx_policy"
OPSET = 17
RUNTIME_VERSION = "1.20.1"
MAX_MODEL_BYTES = 16 * 1024 * 1024
MAX_JSON_BYTES = 256 * 1024
MAX_NODES = 128
MAX_PARAMETERS = 1_000_000
MAX_TENSOR_ELEMENTS = 1_000_000
MAX_TOTAL_ELEMENTS = 4_000_000
FILES = {"manifest.json", "training_contract.json", "model.onnx"}
OPERATORS = {"Gemm", "MatMul", "Add", "Sub", "Mul", "Div", "Min", "Max", "Relu", "Tanh", "Sigmoid", "Clip", "Identity", "Cast"}


def _shape(value: Any, *, activation: bool = True) -> tuple[int, ...]:
    tensor = value.type.tensor_type
    if tensor.elem_type not in (1, 11):  # FLOAT, DOUBLE; no strings/sequences/resources.
        raise ValueError("ONNX tensors must be float32/float64")
    dims = tuple(d.dim_value if d.HasField("dim_value") else 0 for d in tensor.shape.dim)
    if len(dims) > 2 or any(d <= 0 or d > 1024 for d in dims):
        raise ValueError("ONNX requires static rank <=2 and dimensions 1..1024")
    if math.prod(dims) > MAX_TENSOR_ELEMENTS or activation and len(dims) == 2 and dims[0] != 1:
        raise ValueError("ONNX intermediate exceeds static batch-one limits")
    return dims


def validate_model(raw: bytes, observation_size: int) -> Any:
    import onnx

    if not 0 < len(raw) <= MAX_MODEL_BYTES:
        raise ValueError("ONNX model exceeds byte limit")
    model = onnx.load_model_from_string(raw)
    graph = model.graph
    if model.ir_version > 10 or model.functions or model.training_info or graph.sparse_initializer:
        raise ValueError("ONNX functions/training/sparse data or newer IR are unsupported")
    if [(op.domain, op.version) for op in model.opset_import] != [("", OPSET)]:
        raise ValueError("ONNX requires only standard opset 17")
    if len(graph.input) != 1 or len(graph.output) != 1 or not 1 <= len(graph.node) <= MAX_NODES:
        raise ValueError("ONNX requires one input/output and 1..128 nodes")
    if len(graph.initializer) > 256 or len(graph.value_info) > 512:
        raise ValueError("ONNX tensor count exceeds limits")
    if (_shape(graph.input[0]) != (1, observation_size) or graph.input[0].name != "observations"
            or graph.input[0].type.tensor_type.elem_type != 1
            or _shape(graph.output[0]) != (1, 2) or graph.output[0].name != "actions"
            or graph.output[0].type.tensor_type.elem_type != 1):
        raise ValueError("ONNX boundary must be observations float32[1,N] -> actions float32[1,2]")
    parameters = 0
    initializer_names = set()
    for tensor in graph.initializer:
        if tensor.external_data or tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.HasField("segment"):
            raise ValueError("ONNX external/segmented tensor data is forbidden")
        if tensor.name in initializer_names or tensor.name == "observations":
            raise ValueError("ONNX initializer names must be unique and not override input")
        initializer_names.add(tensor.name)
        if tensor.data_type not in (1, 11) or len(tensor.dims) > 2 or any(d <= 0 or d > 1024 for d in tensor.dims):
            raise ValueError("ONNX initializer type/shape is unsupported")
        parameters += math.prod(tensor.dims)
        if parameters > MAX_PARAMETERS:
            raise ValueError("ONNX parameter limit exceeded")
        if not np.isfinite(onnx.numpy_helper.to_array(tensor)).all():
            raise ValueError("ONNX initializers must be finite")
    for node in graph.node:
        if node.domain or node.op_type not in OPERATORS or len(node.output) != 1:
            raise ValueError(f"unsupported ONNX operator/domain: {node.domain}:{node.op_type}")
        attributes = {a.name: a for a in node.attribute}
        allowed = {"alpha", "beta", "transA", "transB"} if node.op_type == "Gemm" else {"to"} if node.op_type == "Cast" else set()
        if len(attributes) != len(node.attribute) or set(attributes) - allowed:
            raise ValueError("unsupported ONNX operator attributes/subgraphs")
        for attr in node.attribute:
            if attr.type not in (onnx.AttributeProto.FLOAT, onnx.AttributeProto.INT):
                raise ValueError("ONNX subgraphs/tensor/string attributes are forbidden")
            if attr.name in ("alpha", "beta") and (attr.type != onnx.AttributeProto.FLOAT or not math.isfinite(attr.f)):
                raise ValueError("ONNX Gemm coefficients must be finite floats")
            if attr.name in ("transA", "transB") and (attr.type != onnx.AttributeProto.INT or attr.i not in (0, 1)):
                raise ValueError("invalid ONNX transpose attribute")
            if attr.name == "to" and (attr.type != onnx.AttributeProto.INT or attr.i not in (1, 11)):
                raise ValueError("ONNX Cast supports float32/float64 only")
    if graph.node[-1].op_type != "Tanh" or list(graph.node[-1].output) != ["actions"]:
        raise ValueError("ONNX actions require a final Tanh")
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    values = {v.name: v for v in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)}
    total = parameters
    for node in inferred.graph.node:
        for name in node.output:
            if name not in values:
                raise ValueError("ONNX intermediate shape could not be proven static")
            total += math.prod(_shape(values[name]))
            if total > MAX_TOTAL_ELEMENTS:
                raise ValueError("ONNX intermediate allocation limit exceeded")
    return model


@dataclass
class OnnxPolicyArtifact:
    manifest: dict[str, Any]
    contract: TrainingContract
    artifact_sha256: str
    session: Any
    normalizer: None = None  # Preprocessing is embedded in the validated graph.

    @property
    def observation_size(self) -> int:
        return self.contract.observation.size

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        values = np.asarray(observations)
        if values.shape != (1, self.observation_size) or values.dtype.kind != "f" or not np.isfinite(values).all():
            raise ValueError("ONNX input must be a finite floating-point batch of one")
        result = self.session.run(["actions"], {"observations": values.astype(np.float32)})[0]
        if result.shape != (1, 2) or result.dtype != np.float32 or not np.isfinite(result).all() or np.any(np.abs(result) > 1):
            raise ValueError("ONNX policy produced invalid actions")
        return result.copy()


def load_onnx_artifact(path: str | Path) -> OnnxPolicyArtifact:
    """Trusted-local preview loader. Put this inside Docker for untrusted input."""
    import onnxruntime as ort

    root = Path(path)
    if root.is_symlink() or not root.is_dir() or {p.name for p in root.iterdir()} != FILES:
        raise ValueError("ONNX artifact requires exactly manifest.json, training_contract.json, model.onnx")
    contents = {}
    for name in sorted(FILES):
        file = root / name
        limit = MAX_MODEL_BYTES if name == "model.onnx" else MAX_JSON_BYTES
        if file.is_symlink() or not file.is_file() or file.stat().st_size > limit:
            raise ValueError("ONNX artifact file is not a bounded regular non-symlink file")
        with file.open("rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("ONNX artifact file exceeds limit")
        contents[name] = raw
    manifest = _load_json(root / "manifest.json")
    keys = {"format", "format_version", "profile", "environment_code_hash", "model_sha256", "training_contract_sha256"}
    optional = {"source_kind", "provenance"}
    if not keys.issubset(manifest) or set(manifest) - keys - optional or manifest["format"] != FORMAT or type(manifest["format_version"]) is not int or manifest["format_version"] != 1:
        raise ValueError("unsupported ONNX artifact manifest")
    if "source_kind" in manifest and manifest["source_kind"] not in {"synapse", "imported"}:
        raise ValueError("unsupported ONNX source kind")
    if manifest["profile"] != "static_vector_cpu_opset17_v1":
        raise ValueError("unsupported ONNX runtime profile")
    _require_hash(manifest["environment_code_hash"], "environment code hash")
    for file, field in (("model.onnx", "model_sha256"), ("training_contract.json", "training_contract_sha256")):
        if hashlib.sha256(contents[file]).hexdigest() != _require_hash(manifest[field], field):
            raise ValueError(f"ONNX artifact hash mismatch: {file}")
    # Reject ONNX-side normalizers and unbounded observation declarations before
    # TrainingContract.load can construct NumPy arrays from untrusted JSON.
    contract_data = _load_json(root / "training_contract.json")
    observation_data = contract_data.get("observation")
    fields = observation_data.get("fields") if isinstance(observation_data, dict) else None
    trainer_data = contract_data.get("trainer")
    if (not isinstance(fields, list) or not 1 <= len(fields) <= 256
            or not isinstance(trainer_data, dict) or trainer_data.get("normalizer") is not None):
        raise ValueError("ONNX contract must have a bounded observation and no separate normalizer")
    contract, normalizer = TrainingContract.load(root / "training_contract.json")
    if normalizer is not None or contract_data != contract.to_dict():
        raise ValueError("ONNX contract must be canonical with preprocessing embedded, not a separate normalizer")
    if not 1 <= contract.observation.size <= 256 or not contract.track_ids or len(contract.track_ids) != len(contract.track_hashes):
        raise ValueError("invalid ONNX observation/training-track contract")
    if len(set(contract.track_ids)) != len(contract.track_ids) or len(set(contract.track_hashes)) != len(contract.track_hashes):
        raise ValueError("ONNX training tracks must be unique")
    for value in (*contract.track_hashes, contract.config_hash):
        _require_hash(value, "contract hash")
    validate_model(contents["model.onnx"], contract.observation.size)
    if ort.__version__ != RUNTIME_VERSION:
        raise ValueError(f"ONNX runtime must be pinned to {RUNTIME_VERSION}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_cpu_mem_arena = options.enable_mem_pattern = False
    options.add_session_config_entry("session.use_deterministic_compute", "1")
    session = ort.InferenceSession(contents["model.onnx"], sess_options=options, providers=["CPUExecutionProvider"])
    digest = hashlib.sha256()
    for name, raw in contents.items():
        digest.update(name.encode() + b"\0" + raw)
    result = OnnxPolicyArtifact(manifest, contract, digest.hexdigest(), session)
    probe = np.zeros((1, contract.observation.size), dtype=np.float32)
    if not np.array_equal(result(probe), result(probe)):
        raise ValueError("ONNX deterministic inference check failed")
    return result
