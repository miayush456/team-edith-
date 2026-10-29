"""Run 1 to 4 ONNX policy artifacts as a deterministic ghost race.

The command evaluates every entrant independently on each explicit track. It
does not use the renderer and it does not simulate car-to-car interaction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing_event.config import SimulationConfig
from racing_event.race import RaceEntrant, run_onnx_race
from racing_event.track import TrackGeometry


def _labels(values: list[str] | None, ids: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("--label must use PARTICIPANT_ID=DISPLAY_LABEL")
        participant_id, label = value.split("=", 1)
        if participant_id not in ids or participant_id in result:
            raise ValueError("--label must name each participant at most once")
        result[participant_id] = label
    return result


def run(args: argparse.Namespace):
    entries = [(value[0], Path(value[1])) for value in args.entrants]
    ids = {participant_id for participant_id, _ in entries}
    labels = _labels(args.label, ids)
    entrants = tuple(
        RaceEntrant(participant_id, artifact_dir, labels.get(participant_id))
        for participant_id, artifact_dir in entries
    )
    config = SimulationConfig.from_file(args.config)
    tracks = tuple(TrackGeometry.from_file(path) for path in args.tracks)
    return run_onnx_race(
        entrants,
        tracks,
        config,
        output_dir=args.output_dir,
        emit_replays=args.emit_replays,
        max_steps=args.max_steps,
        repeats=args.repeats,
        isolate=args.isolate,
        worker_image=args.worker_image,
        worker_memory_mb=args.worker_memory_mb,
        startup_timeout=args.startup_timeout,
        action_timeout=args.action_timeout,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entrant", dest="entrants", action="append", nargs=2, required=True,
        metavar=("PARTICIPANT_ID", "ARTIFACT_DIR"),
        help="repeat for each ONNX artifact directory (1 to 4 entrants)",
    )
    parser.add_argument(
        "--label", action="append",
        help="optional display label in PARTICIPANT_ID=DISPLAY_LABEL form",
    )
    parser.add_argument("--track", dest="tracks", action="append", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for race.json and optional replays")
    parser.add_argument("--emit-replays", action="store_true", help="also emit schema-1.1 replays and replay groups")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--isolate", action="store_true", help="run each model in the model-only Docker worker")
    mode.add_argument(
        "--trusted-local-preview", action="store_true",
        help="run ONNX Runtime in this process (the default; only use trusted artifacts)",
    )
    parser.add_argument("--worker-image", default="synapse-evaluator:v1")
    parser.add_argument("--worker-memory-mb", type=int, default=512)
    parser.add_argument("--startup-timeout", type=float, default=30)
    parser.add_argument("--action-timeout", type=float, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError, TypeError, ImportError) as exc:
        raise SystemExit(f"ONNX race failed: {exc}") from exc
    print(json.dumps(report.to_dict(), sort_keys=True, indent=2, allow_nan=False))
    print(f"wrote {args.output_dir / 'race.json'}")


if __name__ == "__main__":
    main()
