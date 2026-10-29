"""Validate public training maps and build a release-facing manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing_event.track import TrackGeometry
from racing_event.config import SimulationConfig
from racing_event.observation import TrainingContract, build_observation_schema


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks-dir", type=Path, default=ROOT / "tracks" / "train")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--release-id", default="synapse-local-participant-prototype")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "simulation.json")
    return parser.parse_args()


def build_manifest(tracks_dir: Path, release_id: str, config_path: Path | None = None) -> dict:
    paths = sorted(path for path in tracks_dir.glob("*.json") if path.name != "manifest.json")
    if not paths:
        raise ValueError("no training track JSON files found")
    maps = []
    for path in paths:
        track = TrackGeometry.from_file(path)
        maps.append({
            "map_id": track.track_id,
            "display_name": track.meta["display_name"],
            "split": "train",
            "geometry_version": track.meta["geometry_version"],
            "canonical_content_hash": track.content_hash,
            "delivered_file_sha256": _sha256(path),
            "units": "metres",
            "validator": "TrackGeometry.validate",
            "source_note": track.meta.get("source_note", ""),
        })
    environment: dict[str, object] = {}
    if config_path is not None:
        config = SimulationConfig.from_file(config_path)
        schema = build_observation_schema(config)
        environment = {
            "config_hash": TrainingContract.config_hash_for(config),
            "observation_size": schema.size,
            "ray_angles_deg": list(config.sensors.ray_angles_deg),
            "ray_count": config.sensors.ray_count,
        }
    return {
        "manifest_version": "1",
        "release_id": release_id,
        "split": "train",
        "private_tracks_included": False,
        "environment": environment,
        "maps": maps,
    }


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.tracks_dir, args.release_id, args.config)
    output = args.output or args.tracks_dir.parent / "train_manifest.json"
    _atomic_write(output, manifest)
    print(f"validated {len(manifest['maps'])} public training maps")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
