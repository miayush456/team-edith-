"""Run ONNX policies live on a fixed-rate Python racing simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing_event.config import SimulationConfig
from racing_event.live_protocol import LiveStatePublisher
from racing_event.live_race import LiveRaceOptions, run_live_onnx_race
from racing_event.race import RaceEntrant
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
    track = TrackGeometry.from_file(args.track)
    publisher = LiveStatePublisher(host=args.host, port=args.port)
    options = LiveRaceOptions(
        control_hz=args.control_hz,
        policy_budget_ms=args.policy_budget_ms,
        max_steps=args.max_steps,
        late_policy=args.late_policy,
        pace=not args.no_pace,
        action_hold_ticks=args.action_hold_ticks,
        startup_timeout=args.startup_timeout,
        action_timeout=args.action_timeout,
    )
    def announce(endpoint: str) -> None:
        print(f"Live WebSocket endpoint: {endpoint}", flush=True)
        print(f"Open the renderer with ?live={endpoint}; the browser is presentation-only.", flush=True)

    def telemetry(value: dict[str, object]) -> None:
        if args.telemetry_stdout:
            print("LIVE_TELEMETRY " + json.dumps(value, sort_keys=True, separators=(",", ":")), flush=True)

    try:
        report = run_live_onnx_race(
            entrants,
            track,
            config,
            publisher=publisher,
            options=options,
            isolate=args.isolate,
            worker_image=args.worker_image,
            worker_memory_mb=args.worker_memory_mb,
            output_dir=args.output_dir,
            on_started=announce,
            on_telemetry=telemetry,
        )
        return report
    finally:
        publisher.close(grace_seconds=args.end_grace_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entrant", dest="entrants", action="append", nargs=2, required=True,
        metavar=("PARTICIPANT_ID", "ARTIFACT_DIR"),
        help="repeat for each ONNX artifact directory (1 to 4 entrants)",
    )
    parser.add_argument("--label", action="append", help="PARTICIPANT_ID=DISPLAY_LABEL")
    parser.add_argument("--track", type=Path, required=True, help="one explicit track for this live heat")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for live-race.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--control-hz", type=int, default=30)
    parser.add_argument("--policy-budget-ms", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--action-hold-ticks", type=int, default=None)
    parser.add_argument("--late-policy", choices=("strict_scored", "trusted_live"), default="strict_scored")
    parser.add_argument("--no-pace", action="store_true", help="run as fast as possible; not a live competition clock")
    parser.add_argument("--isolate", action="store_true", help="run each model in the model-only Docker worker")
    parser.add_argument("--worker-image", default="synapse-evaluator:v1")
    parser.add_argument("--worker-memory-mb", type=int, default=512)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--action-timeout", type=float, default=None)
    parser.add_argument("--end-grace-seconds", type=float, default=0.25)
    parser.add_argument("--telemetry-stdout", action="store_true", help="emit one bounded LIVE_TELEMETRY JSON line per simulation tick")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError, TypeError, ImportError) as exc:
        raise SystemExit(f"Live ONNX race failed: {exc}") from exc
    print(json.dumps(report.to_dict(), sort_keys=True, indent=2, allow_nan=False))
    print(f"wrote {args.output_dir / 'live-race.json'}")


if __name__ == "__main__":
    main()
