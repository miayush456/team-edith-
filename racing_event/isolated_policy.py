"""Model-only Docker worker; private tracks stay in the trusted Python evaluator.

This supplies container restrictions, not a claim that Docker is a perfect
security boundary. Dedicated, patched organizer hardware remains required.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import stat
import subprocess
import tempfile
import time
from uuid import uuid4

import numpy as np

from .model_artifact import _load_json
from .observation import TrainingContract
from .replay_group import _directory

MAX_LINE_BYTES = 16 * 1024
ROOT = Path(__file__).resolve().parents[1]


def docker_command(snapshot: Path, name: str, image: str, memory_mb: int) -> list[str]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9/_.:@-]{0,255}", image):
        raise ValueError("invalid evaluator image name")
    if "," in str(snapshot) or "\n" in str(snapshot):
        raise ValueError("model snapshot path cannot contain comma/newline")
    if type(memory_mb) is not int or not 128 <= memory_mb <= 4096:
        raise ValueError("worker memory must be 128..4096 MiB")
    return ["docker", "run", "--rm", "--pull=never", "--name", name, "-i",
            "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
            "--user=65534:65534", "--cpus=1", f"--memory={memory_mb}m", f"--memory-swap={memory_mb}m",
            "--pids-limit=64", "--ulimit=nofile=64:64", "--ulimit=core=0:0", "--ipc=none",
            "--tmpfs=/tmp:rw,noexec,nosuid,size=16m", "--env=OMP_NUM_THREADS=1", "--env=OPENBLAS_NUM_THREADS=1",
            "--mount", f"type=bind,src={snapshot},dst=/model,readonly", image,
            "python", "tools/inference_worker.py", "--artifact-dir", "/model"]


def _snapshot(source: Path, destination: Path) -> str:
    """Pin/read regular files via directory FDs; mount only a private frozen copy."""
    digest = hashlib.sha256()
    with _directory(source) as root:
        names = set(os.listdir(root))
        if names not in ({"manifest.json", "training_contract.json", "model.onnx"},
                         {"manifest.json", "training_contract.json", "weights.npz"}):
            raise ValueError("submission contains extra or missing files")
        total = 0
        for name in sorted(names):
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root)
            try:
                info = os.fstat(fd)
                limit = 256 * 1024 if name.endswith(".json") else 64 * 1024 * 1024
                if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                    raise ValueError("submission file must be bounded and regular")
                with os.fdopen(fd, "rb", closefd=False) as stream:
                    raw = stream.read(limit + 1)
            finally:
                os.close(fd)
            total += len(raw)
            if len(raw) > limit or total > 65 * 1024 * 1024:
                raise ValueError("submission snapshot exceeds limits")
            (destination / name).write_bytes(raw)
            (destination / name).chmod(0o444)
            digest.update(name.encode() + b"\0" + raw)
    destination.chmod(0o755)
    return digest.hexdigest()


def _contract_metadata(path: Path):
    # TrainingContract.load constructs a normalizer from its declared size.
    # Bound that declaration BEFORE any NumPy allocation in the trusted host.
    data = _load_json(path)
    try:
        fields = data["observation"]["fields"]
        if not isinstance(fields, list) or not 1 <= len(fields) <= 256:
            raise ValueError("unsupported worker observation size")
        normalizer = data["trainer"].get("normalizer")
        if normalizer is not None and (
            not isinstance(normalizer, dict) or type(normalizer.get("size")) is not int
            or normalizer["size"] != len(fields)
        ):
            raise ValueError("worker normalizer size must match bounded observation size")
        return TrainingContract.load(path)
    except (KeyError, TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("invalid worker contract metadata") from exc


class IsolatedPolicy:
    """Validated artifact proxy, with bounded output and per-action deadlines."""
    def __init__(self, source: str | Path, *, image: str = "synapse-evaluator:v1",
                 memory_mb: int = 512, startup_timeout: float = 30, action_timeout: float = 2):
        for value in (startup_timeout, action_timeout):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 300:
                raise ValueError("worker deadlines must be finite and within (0,300] seconds")
        parent = ROOT / "artifacts" / "isolated-inputs"
        parent.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(prefix="model-", dir=parent)
        self._process = None
        self._buffer = b""
        self._sequence = 0
        self._name = "synapse-inference-" + uuid4().hex
        self.action_timeout = action_timeout
        try:
            snapshot = Path(self._temporary.name)
            self.artifact_sha256 = _snapshot(Path(source), snapshot)
            self.manifest = _load_json(snapshot / "manifest.json")
            self.contract, self.normalizer = _contract_metadata(snapshot / "training_contract.json")
            if not 1 <= self.observation_size <= 256:
                raise ValueError("unsupported worker observation size")
            command = docker_command(snapshot, self._name, image, memory_mb)
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            ready = self._read(startup_timeout)
            if (set(ready) != {"ready", "artifact_sha256"} or ready["ready"] is not True
                    or ready["artifact_sha256"] != self.artifact_sha256):
                raise ValueError("model worker did not validate this exact snapshot")
        except BaseException:
            self.close()
            raise

    @property
    def observation_size(self):
        return self.contract.observation.size

    def _read(self, timeout):
        deadline = time.monotonic() + timeout
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self._process.stdout], [], [], remaining)[0]:
                raise TimeoutError("model worker deadline exceeded")
            raw = os.read(self._process.stdout.fileno(), 4096)
            if not raw:
                raise ValueError("model worker exited before a valid response; check Docker/image availability")
            self._buffer += raw
            if len(self._buffer) > MAX_LINE_BYTES:
                raise ValueError("model worker response exceeds byte limit")
        line, self._buffer = self._buffer.split(b"\n", 1)
        value = json.loads(line)
        if not isinstance(value, dict) or "error" in value:
            raise ValueError("model worker rejected artifact or request")
        return value

    def __call__(self, observations):
        try:
            values = np.asarray(observations)
            if values.shape != (1, self.observation_size) or values.dtype.kind != "f" or not np.isfinite(values).all():
                raise ValueError("worker input must be a finite floating batch of one")
            self._sequence += 1
            line = json.dumps({"sequence": self._sequence, "observations": values.tolist()}, allow_nan=False).encode() + b"\n"
            if len(line) > MAX_LINE_BYTES:
                raise ValueError("worker request exceeds byte limit")
            self._process.stdin.write(line)
            self._process.stdin.flush()
            result = self._read(self.action_timeout)
            if (set(result) != {"sequence", "actions"} or type(result["sequence"]) is not int
                    or result["sequence"] != self._sequence):
                raise ValueError("worker response sequence mismatch")
            action = np.asarray(result["actions"])
            if (action.shape != (1, 2) or action.dtype.kind not in "fi"
                    or not np.isfinite(action).all() or np.any((action < -1) | (action > 1))):
                raise ValueError("worker returned invalid bounded actions")
            return action.astype(np.float32)
        except BaseException:
            self.close()
            raise

    def close(self):
        process = self._process
        if process is not None:
            try:
                if process.stdin:
                    process.stdin.close()
                process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait(timeout=5)
            finally:
                if process.stdout:
                    process.stdout.close()
                try:
                    subprocess.run(["docker", "rm", "--force", self._name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                except (OSError, subprocess.SubprocessError):
                    pass
                self._process = None
        self._temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
