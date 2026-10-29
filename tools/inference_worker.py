"""Model-only JSONL worker. Run inside the evaluator image for untrusted input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from racing_event.submission import load_submission

MAX_LINE_BYTES = 16 * 1024


def send(value):
    sys.stdout.write(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        artifact = load_submission(args.artifact_dir)
        send({"ready": True, "artifact_sha256": artifact.artifact_sha256})
        sequence = 0
        while True:
            raw = sys.stdin.buffer.readline(MAX_LINE_BYTES + 1)
            if not raw:
                return
            if len(raw) > MAX_LINE_BYTES or not raw.endswith(b"\n"):
                raise ValueError("oversized or incomplete request")
            request = json.loads(raw)
            sequence += 1
            if (not isinstance(request, dict) or set(request) != {"sequence", "observations"}
                    or type(request["sequence"]) is not int or request["sequence"] != sequence):
                raise ValueError("invalid worker request sequence")
            observation = np.asarray(request["observations"], dtype=np.float32)
            if observation.shape != (1, artifact.observation_size):
                raise ValueError("worker accepts exactly one observation")
            send({"sequence": sequence, "actions": artifact(observation).tolist()})
    except Exception as exc:
        # No traceback, weights, observations, file paths or arbitrary payloads.
        send({"error": f"model worker rejected input ({type(exc).__name__})"})
        raise SystemExit(2)


if __name__ == "__main__":
    main()
