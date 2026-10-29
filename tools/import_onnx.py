"""Import an externally trained ONNX policy into a SYNAPSE artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import shutil
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing_event.observation import TrainingContract
from racing_event.onnx_artifact import FORMAT, load_onnx_artifact, validate_model
from racing_event.track import TrackGeometry
from racing_event.trajectory import engine_code_hash
from racing_event.config import SimulationConfig

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_onnx(model: Path, run_dir: Path, output: Path, track_dir: Path) -> Path:
    model = Path(model)
    run_dir = Path(run_dir)
    output = Path(output)
    if model.is_symlink() or run_dir.is_symlink() or output.is_symlink():
        raise ValueError("ONNX import paths must not be symlinks")
    model = model.resolve()
    run_dir = run_dir.resolve()
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("ONNX import output must be a new directory")
    if not model.is_file():
        raise ValueError("model.onnx must be a regular file")
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise ValueError("run directory must contain run_config.json")
    config = SimulationConfig.from_file(config_path)
    tracks = [TrackGeometry.from_file(path) for path in sorted(track_dir.glob("*.json")) if path.name != "manifest.json"]
    if not tracks:
        raise ValueError("track_dir contains no public tracks")
    contract = TrainingContract.for_config(
        config,
        track_ids=[track.track_id for track in tracks],
        track_hashes=[track.content_hash for track in tracks],
    )
    raw = model.read_bytes()
    validate_model(raw, contract.observation.size)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (temporary / "model.onnx").write_bytes(raw)
        contract.save(temporary / "training_contract.json")
        manifest = {
            "format": FORMAT,
            "format_version": 1,
            "profile": "static_vector_cpu_opset17_v1",
            "environment_code_hash": engine_code_hash(),
            "source_kind": "imported",
            "provenance": {"training_claim": "participant supplied", "run_dir": str(run_dir)},
            "model_sha256": _sha256(temporary / "model.onnx"),
            "training_contract_sha256": _sha256(temporary / "training_contract.json"),
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        load_onnx_artifact(temporary)
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="finished model.onnx")
    parser.add_argument("--run-dir", type=Path, required=True, help="run/config directory describing rays and physics")
    parser.add_argument("--track-dir", type=Path, default=ROOT / "tracks/train")
    parser.add_argument("--output", type=Path, required=True, help="new exact-three-file artifact directory")
    args = parser.parse_args()
    try:
        print(import_onnx(args.model, args.run_dir, args.output, args.track_dir))
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"ONNX import failed: {exc}\n")


if __name__ == "__main__":
    main()
