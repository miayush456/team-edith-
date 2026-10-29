"""Bundle a validated replay and the read-only viewer into one offline HTML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing_event.trajectory import load_replay


def build(replay_path: Path, output: Path) -> None:
    if replay_path.resolve() == output.resolve():
        raise ValueError("HTML output must not overwrite its source replay")
    data = load_replay(replay_path)
    assets = Path(__file__).resolve().parents[1] / "racing_event" / "assets"
    template = (assets / "replay.html").read_text(encoding="utf-8")
    # The JSON is inert script data, not JS. Escape HTML delimiters even in
    # participant labels so a literal </script> cannot escape the data node.
    encoded = json.dumps(data, separators=(",", ":"), allow_nan=False)
    encoded = encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    replacements = {
        "/*__REPLAY_CSS__*/": (assets / "replay.css").read_text(encoding="utf-8"),
        "/*__REPLAY_JS__*/": (assets / "replay.js").read_text(encoding="utf-8"),
        "<!--__REPLAY_DATA__-->": f'<script type="application/json" id="replay-data">{encoded}</script>',
    }
    for marker, value in replacements.items():
        if template.count(marker) != 1:
            raise RuntimeError(f"viewer template must contain exactly one {marker}")
        template = template.replace(marker, value)
    output.write_text(template, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        build(args.replay, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Replay build failed: {exc}\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
