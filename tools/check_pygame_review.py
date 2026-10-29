"""Capture reproducible, real dummy-SDL CLI screenshots for replay presentation QA.

Uses retained full laps, plus constant-action crashes recorded by the current
authoritative evaluator. Writes only QA outputs; never changes input replays.
Run from the project root: .venv/bin/python tools/check_pygame_review.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from racing_event.config import SimulationConfig
from racing_event.evaluation import evaluate_policy
from racing_event.replay_group import load_replay_group
from racing_event.track import TrackGeometry
from racing_event.trajectory import ReplayWriter, load_replay, replay_header, replay_result
from racing_event.vector_env import VectorizedRacingEnv


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pygame-review-2026-10-19"))
    args = parser.parse_args()
    output = args.output_dir.absolute()
    output.mkdir(parents=True, exist_ok=True)
    replay_dir = ROOT / "artifacts/replay"
    retained = sorted(replay_dir.glob("pygame-imitation-*.json")) + [replay_dir / "pygame-serious.json"]
    before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in retained}
    captures = []

    def cli(*arguments: str) -> None:
        subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, check=True)

    def group(path: Path, replays: list[Path], labels: list[str]) -> None:
        if not path.exists():
            cli("tools/build_replay_group.py", "--output", path, "--replays", *replays, "--labels", *labels)
        loaded = load_replay_group(path)
        if [p.participant_id for p in loaded.participants] != [load_replay(p)["source"]["participant_id"] for p in replays]:
            raise ValueError("existing QA group has different participants; choose a fresh output directory")

    def capture(name: str, source: Path, *, is_group=False, tick=0, extra=()) -> None:
        for size in ("640x480", "1280x800"):
            target = output / f"{name}-{size}.png"
            arguments = ["tools/run_pygame.py", "--group" if is_group else "--replay", source,
                         "--size", size, "--start-tick", str(tick), *extra, "--screenshot", target]
            cli(*arguments)
            captures.append({"path": str(target.relative_to(ROOT)), "command": [str(a) for a in arguments],
                             "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})

    for path in retained[:3]:
        data = load_replay(path)
        name = data["track"]["meta"]["id"]
        for stage, tick in (("start", 0), ("mid", data["result"]["steps"] // 2), ("end", data["result"]["steps"])):
            capture(f"{name}-{stage}", path, tick=tick)
    oval = replay_dir / "pygame-imitation-synthetic_oval.json"
    capture("single-help", oval, extra=("--show-help",))
    capture("single-event-seek", oval, extra=("--seek-event", "next"))
    comparison = replay_dir / "pygame-review-oval.group.json"
    group(comparison, [oval, replay_dir / "pygame-serious.json"], ["Imitation", "PPO"])
    for stage, tick in (("start", 0), ("mid", 432), ("hold", 880), ("end", 930)):
        capture(f"group-{stage}", comparison, is_group=True, tick=tick)
    capture("group-help", comparison, is_group=True, extra=("--show-help",))
    capture("group-event-seek", comparison, is_group=True, extra=("--select-ghost", "2", "--seek-event", "next"))

    # Each crash is an actual deterministic policy evaluation, not a fabricated
    # terminal frame or a result inferred from where the sprite meets a wall.
    config = SimulationConfig.from_file(ROOT / "config/simulation.json")
    track = TrackGeometry.from_file(ROOT / "tracks/train/monaco.json")
    crashes = []
    crash_results = []
    for name, action in (("straight", (0.0, 1.0)), ("left", (0.15, 1.0)),
                         ("right", (-0.15, 1.0)), ("gentle", (0.3, 0.8))):
        path = output / f"crash-{name}.json"
        env = VectorizedRacingEnv(track, config, n_envs=1)
        header = replay_header(track, config, participant_id=f"review-{name}", mode="scripted")
        with ReplayWriter(path, header) as writer:
            result = evaluate_policy(env, lambda obs: np.tile(action, (len(obs), 1)), max_steps=1200, frame_sink=writer.append)
            writer.finish(replay_result(result, config.tick_hz))
        data = load_replay(path)
        if data["result"]["status"] != "dnf_crash":
            raise ValueError(f"constant action {action} did not produce the expected recorded crash")
        crashes.append(path)
        crash_results.append({"file": path.name, "action": action, "result": data["result"], "payload_hash": data["payload_hash"],
                              "events": [[f["tick"], f["events"]] for f in data["frames"] if f["events"]]})
    straight = load_replay(crashes[0])
    end = straight["result"]["steps"]
    capture("crash-before-contact", crashes[0], tick=end - 1)
    capture("crash-contact", crashes[0], extra=("--seek-event", "last"))
    capture("crash-event-seek", crashes[0], extra=("--seek-event", "next"))
    four = output / "crashes.group.json"
    group(four, crashes, ["Straight", "Left", "Right", "Gentle"])
    capture("four-start", four, is_group=True)
    capture("four-end-reviewed", four, is_group=True, tick=load_replay_group(four).final_tick)
    capture("four-help", four, is_group=True, extra=("--show-help",))
    capture("four-selected-hold", four, is_group=True, tick=load_replay_group(four).final_tick,
            extra=("--select-ghost", "3"))
    after = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in retained}
    if before != after:
        raise AssertionError("retained input changed")
    inventory = {"retained_sha256_before": before, "retained_sha256_after": after,
                 "crashes": crash_results, "screenshots": captures}
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print(f"Captured {len(captures)} PNGs at both sizes; retained input hashes unchanged. Inventory: {output / 'inventory.json'}")


if __name__ == "__main__":
    main()
