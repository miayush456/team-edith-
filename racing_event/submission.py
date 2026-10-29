"""Dispatch a declared submission format without importing participant code."""
from pathlib import Path

from .model_artifact import _load_json, load_participant_artifact
from .onnx_artifact import FORMAT, MAX_JSON_BYTES, load_onnx_artifact


def load_submission(path: str | Path):
    root = Path(path)
    manifest = root / "manifest.json"
    if root.is_symlink() or manifest.is_symlink() or not manifest.is_file() or manifest.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("submission manifest must be a bounded regular non-symlink file")
    format_name = _load_json(manifest).get("format")
    if format_name == FORMAT:
        return load_onnx_artifact(root)
    if format_name == "synapse_numpy_mlp":
        return load_participant_artifact(root)
    raise ValueError("unsupported declared submission artifact format")
