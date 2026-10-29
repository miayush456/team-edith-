"""Local reference-trainer checkpoints, not a participant artifact contract.

POSIX/local-filesystem only. Each generation is written once, fsynced, renamed,
then committed by one atomic JSON pointer. Root artifacts are compatibility
symlinks switched *after* that commit; recovery never reads those mirrors.
The external RunBudget is never copied or restored. A budget ahead of the
committed boundary is an interrupted update and exact continuation is refused.

Private collector/env fields are deliberately isolated here. Source/runtime
identities fail closed if those implementation details change. No pickle-based
NumPy/environment objects or participant Python are loaded.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields
try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import msvcrt
except ImportError:
    msvcrt = None
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import tempfile
from typing import Any, Iterator
import uuid

import numpy as np
import torch

from .physics import CarStateBatch
from .trajectory import engine_code_hash


VERSION = 1
POINTER = "reference_checkpoint.json"
GENERATIONS = "reference_generations"
GENERATION_RE = re.compile(r"g-[0-9a-f]{32}\Z")
MIRRORS = (
    "policy.pt", "policy.json", "training_contract.json", "training_state.json",
    "training_metrics.jsonl", "selection_report.json",
)
FILES = {*MIRRORS, "local_state.pt", "collector.json"}
ENV_ARRAYS = (
    "track_indices", "previous_s", "distance_since_spawn", "steps", "stuck_ticks",
    "sector_index", "sector_coverage_mask", "_done",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fsync_dir(path: Path) -> None:
    if os.name == "nt" or not hasattr(os, "O_DIRECTORY"):
        return
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def atomic_bytes(path: Path, data: bytes) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        fsync_dir(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_bad_constant)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"invalid reference checkpoint JSON: {path.name}") from exc


def _bad_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON: {value}")


@contextmanager
def writer_lock(output: Path) -> Iterator[None]:
    """Reject another writer for the entire invocation, including evaluation."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".reference-writer.lock").open("a+b") as stream:
        if fcntl is not None:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("another reference writer owns this output directory") from exc
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        elif msvcrt is not None:
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except (BlockingIOError, OSError) as exc:
                raise ValueError("another reference writer owns this output directory") from exc
            try:
                yield
            finally:
                try:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        else:
            yield


def runtime_identity(device: str = "cpu") -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    names = (
        "racing_event/ppo.py", "racing_event/rollout.py", "racing_event/metrics.py",
        "racing_event/reference_checkpoint.py", "racing_event/evaluation.py",
        "racing_event/evaluation_audit.py", "tools/run_reference_baseline.py",
    )
    processor: dict[str, Any] = {"device": device}
    if device.startswith("cuda"):
        processor.update({
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(torch.device(device)),
            "device_capability": list(torch.cuda.get_device_capability(torch.device(device))),
        })
    return {
        "engine": engine_code_hash(),
        "trainer_sources": {name: sha256(root / name) for name in names},
        "python": platform.python_version(), "numpy": np.__version__,
        "torch": str(torch.__version__), "machine": platform.machine(),
        "torch_threads": torch.get_num_threads(), "processor": processor,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _pack(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or not np.all(np.isfinite(value)):
            raise ValueError("unsupported/non-finite checkpoint array")
        return {"__array__": value.dtype.str, "shape": list(value.shape), "data": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"__tuple__": [_pack(item) for item in value]}
    if isinstance(value, list):
        return [_pack(item) for item in value]
    if isinstance(value, dict):
        return {key: _pack(item) for key, item in value.items()}
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError(f"unsupported checkpoint value: {type(value).__name__}")


def _unpack(value: Any) -> Any:
    if isinstance(value, dict):
        if "__array__" in value:
            if set(value) != {"__array__", "shape", "data"}:
                raise ValueError("invalid checkpoint array")
            dtype = np.dtype(value["__array__"])
            if dtype.kind not in "biuf":
                raise ValueError("unsupported checkpoint dtype")
            array = np.asarray(value["data"], dtype=dtype)
            if list(array.shape) != value["shape"] or not np.all(np.isfinite(array)):
                raise ValueError("invalid checkpoint array shape/values")
            return array
        if "__tuple__" in value:
            if set(value) != {"__tuple__"}:
                raise ValueError("invalid checkpoint tuple")
            return tuple(_unpack(item) for item in value["__tuple__"])
        return {key: _unpack(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack(item) for item in value]
    return value


def snapshot_collector(collector: Any) -> dict[str, Any]:
    if collector._invalid_reason is not None:
        raise ValueError("cannot checkpoint an invalid collector")
    env = collector.env
    return _pack({
        "version": VERSION,
        "arrays": {name: getattr(env, name).copy() for name in ENV_ARRAYS},
        "car": {field.name: getattr(env.state, field.name).copy() for field in fields(CarStateBatch)},
        "env_rng": env.rng.bit_generator.state,
        "last_info": env.last_info,
        "raw_observation": collector._raw_observation,
        "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
    })


def restore_collector(collector: Any, payload: Any) -> None:
    try:
        data = _unpack(payload)
        if set(data) != {"version", "arrays", "car", "env_rng", "last_info", "raw_observation", "numpy_rng", "python_rng"} or data["version"] != VERSION:
            raise ValueError("unsupported collector snapshot")
        env = collector.env
        if set(data["arrays"]) != set(ENV_ARRAYS):
            raise ValueError("incomplete environment snapshot")
        for name in ENV_ARRAYS:
            saved, live = data["arrays"][name], getattr(env, name)
            if saved.shape != live.shape or saved.dtype != live.dtype:
                raise ValueError(f"invalid environment array {name}")
            setattr(env, name, saved.copy())
        if np.any(env.track_indices < 0) or np.any(env.track_indices >= len(env.tracks)):
            raise ValueError("invalid track indices")
        car_names = {field.name for field in fields(CarStateBatch)}
        if set(data["car"]) != car_names:
            raise ValueError("incomplete car snapshot")
        for name in car_names:
            saved = data["car"][name]
            if saved.shape != (env.n_envs,) or saved.dtype != np.dtype("float64"):
                raise ValueError("invalid car state")
        env.state = CarStateBatch(**data["car"])
        env.rng.bit_generator.state = data["env_rng"]
        env.last_info = data["last_info"]
        raw = data["raw_observation"]
        if raw is not None and (raw.shape != (env.n_envs, env.observation_size) or raw.dtype != np.dtype("float32")):
            raise ValueError("invalid collector observation")
        collector._raw_observation = raw
        np.random.set_state(data["numpy_rng"])
        random.setstate(data["python_rng"])
    except (KeyError, TypeError, AttributeError, OverflowError, ValueError) as exc:
        raise ValueError(f"invalid collector checkpoint: {exc}") from exc


def save_local_state(path: Path, trainer: Any) -> None:
    device = next(trainer.policy.parameters()).device
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    with path.open("wb") as stream:
        torch.save({
            "optimizer": trainer.optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": cuda_rng,
        }, stream)
        stream.flush()
        os.fsync(stream.fileno())


def restore_local_state(path: Path, trainer: Any) -> None:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if set(state) not in ({"optimizer", "torch_rng"}, {"optimizer", "torch_rng", "cuda_rng"}):
            raise ValueError("missing local state")
        optimizer = state["optimizer"]
        live = trainer.optimizer.state_dict()
        if set(optimizer) != set(live) or len(optimizer["param_groups"]) != 1:
            raise ValueError("invalid optimizer groups")
        if optimizer["param_groups"] != live["param_groups"]:
            raise ValueError("optimizer configuration mismatch")
        params = list(trainer.policy.parameters())
        if optimizer["state"] and set(optimizer["state"]) != set(range(len(params))):
            raise ValueError("incomplete optimizer state")
        for index, item in optimizer["state"].items():
            if set(item) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("invalid Adam state")
            for name, tensor in item.items():
                if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
                    raise ValueError("non-finite optimizer state")
                if tensor.shape != (() if name == "step" else params[index].shape):
                    raise ValueError("optimizer tensor shape mismatch")
        trainer.optimizer.load_state_dict(optimizer)
        torch.set_rng_state(state["torch_rng"])
        device = next(trainer.policy.parameters()).device
        cuda_rng = state.get("cuda_rng")
        if device.type == "cuda":
            if not isinstance(cuda_rng, torch.Tensor):
                raise ValueError("GPU checkpoint is missing its CUDA random state")
            torch.cuda.set_rng_state(cuda_rng, device)
        elif cuda_rng is not None:
            raise ValueError("CPU training cannot resume a CUDA checkpoint")
    except Exception as exc:
        raise ValueError(f"invalid safe local Torch checkpoint: {path.name}") from exc


def new_generation() -> str:
    return "g-" + uuid.uuid4().hex


def publication_hook(stage: str, generation: str) -> None:
    """Fault-injection seam for crash tests; never an executable artifact hook."""


def publish(output: Path, generation: str, *, policy: Any, contract: Any,
            normalizer: Any, trainer: Any, collector: Any, state: dict[str, Any],
            metrics: list[dict[str, Any]], selection: Any, identity: Any,
            budget_head: str) -> Path:
    if GENERATION_RE.fullmatch(generation) is None:
        raise ValueError("invalid generation name")
    parent = output / GENERATIONS
    parent.mkdir(exist_ok=True)
    if parent.is_symlink():
        raise ValueError("reference generations directory must not be a symlink")
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))
    destination = parent / generation
    try:
        policy.save_artifact(temporary / "policy.pt", temporary / "policy.json", hyperparameters=trainer.hyperparameters)
        contract.save(temporary / "training_contract.json", normalizer=normalizer)
        atomic_json(temporary / "training_state.json", state)
        atomic_json(temporary / "selection_report.json", selection)
        atomic_json(temporary / "collector.json", snapshot_collector(collector))
        atomic_bytes(temporary / "training_metrics.jsonl", b"".join(
            (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode() for row in metrics
        ))
        save_local_state(temporary / "local_state.pt", trainer)
        atomic_json(temporary / "manifest.json", {
            "version": VERSION, "generation": generation, "identity": identity,
            "budget_head": budget_head,
            "files": {name: sha256(temporary / name) for name in sorted(FILES)},
        })
        fsync_dir(temporary)
        os.rename(temporary, destination)
        fsync_dir(parent)
        fsync_dir(output)
        publication_hook("before_pointer", generation)
        atomic_json(output / POINTER, {
            "version": VERSION, "generation": generation,
            "manifest_sha256": sha256(destination / "manifest.json"),
        })
        publication_hook("after_pointer", generation)
        repair_mirrors(output, destination, selection)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def validate_generation(output: Path, generation: str, manifest_hash: str | None = None) -> tuple[Path, dict[str, Any]]:
    if not isinstance(generation, str) or GENERATION_RE.fullmatch(generation) is None:
        raise ValueError("invalid reference generation pointer")
    directory = output / GENERATIONS / generation
    if (output / GENERATIONS).is_symlink() or directory.is_symlink() or not directory.is_dir():
        raise ValueError("missing reference generation")
    try:
        if manifest_hash is not None and sha256(directory / "manifest.json") != manifest_hash:
            raise ValueError("reference manifest hash mismatch")
        manifest = read_json(directory / "manifest.json")
        if not isinstance(manifest, dict) or set(manifest) != {"version", "generation", "identity", "budget_head", "files"} or manifest["version"] != VERSION or manifest["generation"] != generation:
            raise ValueError("reference generation manifest mismatch")
        if not isinstance(manifest["identity"], dict):
            raise ValueError("invalid reference generation identity")
        if set(manifest["files"]) != FILES:
            raise ValueError("incomplete reference generation")
        for name, digest in manifest["files"].items():
            path = directory / name
            if path.is_symlink() or sha256(path) != digest:
                raise ValueError(f"reference generation hash mismatch: {name}")
        return directory, manifest
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError("corrupt reference generation; exact recovery refused") from exc


def load_current(output: Path) -> tuple[Path, dict[str, Any]] | None:
    pointer = output / POINTER
    if not os.path.lexists(pointer):
        # Never infer a current generation from its number or compatibility files.
        if any(os.path.lexists(output / name) for name in MIRRORS) or (
            (output / GENERATIONS).exists() and any((output / GENERATIONS).iterdir())
        ):
            raise ValueError("no authoritative pointer: legacy/non-exact or interrupted checkpoint; use explicit --warm-start into a new directory")
        return None
    data = read_json(pointer)
    if not isinstance(data, dict) or set(data) != {"version", "generation", "manifest_sha256"} or data["version"] != VERSION:
        raise ValueError("invalid authoritative reference pointer")
    digest = data["manifest_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("invalid reference manifest digest in pointer")
    return validate_generation(output, data["generation"], data["manifest_sha256"])


def _atomic_symlink(path: Path, target: str) -> None:
    temporary = path.parent / (".link-" + uuid.uuid4().hex)
    try:
        temporary.symlink_to(target)
        os.replace(temporary, path)
        fsync_dir(path.parent)
        return
    except (OSError, PermissionError):
        target_path = (path.parent / target).resolve()
        if target_path.is_dir():
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            shutil.copytree(target_path, path, dirs_exist_ok=True)
        elif target_path.is_file():
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)
            shutil.copyfile(target_path, path)
    finally:
        temporary.unlink(missing_ok=True)


def repair_mirrors(output: Path, current: Path, selection: Any) -> None:
    """Only link immutable files. Never save a policy through a root symlink."""
    if selection is not None:
        best, manifest = validate_generation(output, selection["best_generation"], selection.get("best_manifest_sha256"))
        if manifest["identity"] != read_json(current / "manifest.json")["identity"]:
            raise ValueError("best generation identity mismatch")
        _atomic_symlink(output / "best", str(best.relative_to(output)))
    _atomic_symlink(output / ".reference-latest", str(current.relative_to(output)))
    for name in MIRRORS:
        src = current / name
        dst = output / name
        if src.is_file():
            try:
                if dst.is_symlink() or dst.is_file():
                    dst.unlink(missing_ok=True)
                shutil.copyfile(src, dst)
            except OSError:
                pass
