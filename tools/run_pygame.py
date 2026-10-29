"""Play a validated SYNAPSE replay with the read-only Pygame renderer."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing_event.pygame_renderer import MIN_WINDOW_SIZE, run_pygame


def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("size must look like WIDTHxHEIGHT") from exc
    if width < MIN_WINDOW_SIZE[0] or height < MIN_WINDOW_SIZE[1]:
        raise argparse.ArgumentTypeError("size must be at least 640x480")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--replay", type=Path, help="single-car schema-1.1 JSON (default playback behavior)")
    source.add_argument("--group", type=Path, help="replay-group v1.0 manifest; max 4 ghosts, holds final poses")
    parser.add_argument("--size", type=parse_size, default=(1280, 800))
    parser.add_argument("--speed", type=float, default=1.0)
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--start-seconds", type=float, default=0.0)
    start.add_argument("--start-tick", type=int, help="exact shared tick; clamped to recording/group range")
    parser.add_argument("--seek-event", choices=("first", "last", "previous", "next"),
                        help="seek/pause at a recorded selected-ghost event after setting the start position")
    parser.add_argument("--show-help", action="store_true", help="open visible keyboard help (also in screenshots)")
    parser.add_argument("--select-ghost", type=int, default=1, help="initial telemetry ghost number, 1..4 (default 1)")
    parser.add_argument("--headless", action="store_true", help="render one frame with SDL dummy drivers")
    parser.add_argument("--screenshot", type=Path, help="save one deterministic paused PNG at --start-seconds (implies --headless)")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after this many interactive frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0 or args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--speed and --max-frames must be positive")
    try:
        run_pygame(
            args.replay,
            group_path=args.group,
            size=args.size,
            speed=args.speed,
            headless=args.headless,
            start_seconds=args.start_seconds,
            max_frames=args.max_frames,
            screenshot=args.screenshot,
            start_tick=args.start_tick,
            seek_event=args.seek_event,
            show_help=args.show_help,
            selected_ghost=args.select_ghost,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"pygame replay failed: {exc}") from exc


if __name__ == "__main__":
    main()
