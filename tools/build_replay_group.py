"""Create replay-group v1.0: max 4 files, 16 MiB/file, 32 MiB total, 16 KiB manifest.

All replays must be beneath the manifest directory. Common recorded standing
tick zero, track/config/engine/tick identities and hashes are checked. Shorter
ghosts hold their final recorded pose; results stay in manifest order, unranked.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing_event.replay_group import create_replay_group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new manifest; existing files are never overwritten")
    parser.add_argument("--replays", type=Path, nargs="+", required=True, help="1..4 existing schema-1.1 recordings")
    parser.add_argument("--labels", nargs="+", help="one printable label per ghost (max 48 characters)")
    args = parser.parse_args()
    try:
        group = create_replay_group(args.output, args.replays, labels=args.labels)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Replay group creation failed: {exc}\n")
    print(f"wrote {args.output}: {len(group.participants)} ghosts, {group.final_tick} shared ticks; hold_final_pose")


if __name__ == "__main__":
    main()
