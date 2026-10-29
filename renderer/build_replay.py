"""Bundle two SYNAPSE replay JSON files into a standalone 2.5D F1 viewer HTML file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


def bundle_replays(racer_a_path: Path, racer_b_path: Path, output_path: Path) -> None:
    renderer_dir = Path(__file__).resolve().parent

    index_html = (renderer_dir / "index.html").read_text(encoding="utf-8")
    style_css = (renderer_dir / "style.css").read_text(encoding="utf-8")
    app_js = (renderer_dir / "app.js").read_text(encoding="utf-8")

    data_a = json.loads(racer_a_path.read_text(encoding="utf-8"))
    data_b = json.loads(racer_b_path.read_text(encoding="utf-8"))

    # Validate both replays match track
    hash_a = data_a.get("track", {}).get("meta", {}).get("content_hash")
    hash_b = data_b.get("track", {}).get("meta", {}).get("content_hash")
    if hash_a != hash_b:
        print(f"Warning: Replay track hashes differ ({hash_a} vs {hash_b})", file=sys.stderr)

    # Embed CSS inline
    embedded_css = f"<style>\n{style_css}\n</style>"
    html = re.sub(r'<link rel="stylesheet" href="style\.css">', lambda _: embedded_css, index_html)

    # Embed Replay JSON Data Scripts
    embedded_data = (
        f'<script id="replay-data-a" type="application/json">\n{json.dumps(data_a)}\n</script>\n'
        f'<script id="replay-data-b" type="application/json">\n{json.dumps(data_b)}\n</script>\n'
    )
    html = html.replace('<!-- 3D canvas', f'{embedded_data}\n  <!-- 3D canvas')

    # Embed JS inline
    embedded_js = f"<script type=\"module\">\n{app_js}\n</script>"
    html = re.sub(r'<script type="module" src="app\.js"></script>', lambda _: embedded_js, html)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Bundled standalone 2.5D replay to: {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--racer-a", type=Path, required=True, help="Path to Racer A replay JSON")
    parser.add_argument("--racer-b", type=Path, required=True, help="Path to Racer B replay JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output path for standalone HTML file")
    args = parser.parse_args()

    if not args.racer_a.exists():
        parser.error(f"Racer A replay not found: {args.racer_a}")
    if not args.racer_b.exists():
        parser.error(f"Racer B replay not found: {args.racer_b}")

    bundle_replays(args.racer_a, args.racer_b, args.output)


if __name__ == "__main__":
    main()
