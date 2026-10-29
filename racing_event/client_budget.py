"""Persistent client-side cap for independent participant training allocations.

This is an operational local guard, not cryptographic proof of competition
compliance. A person who controls the client can delete or rewrite it. It is
intended to prevent accidental unlimited retraining through the participant
launcher and dashboard while the stronger organizer-side budget remains a
separate concern.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator

try:  # The supported prototype platform is POSIX/Linux.
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX hosts.
    fcntl = None  # type: ignore[assignment]


SCHEMA_VERSION = 1
DEFAULT_MAX_RETRAINS = 12
DEFAULT_MAX_STEPS_PER_RETRAIN = 1_000_000
_TOP_LEVEL_KEYS = {"schema_version", "max_retrains", "max_steps_per_retrain", "allocations", "head_hash"}
_ALLOCATION_KEYS = {"slot", "output_dir", "budget_steps"}


class ClientTrainingBudgetError(ValueError):
    """The local client training cap cannot satisfy a request."""


class ClientRetrainLimitExceeded(ClientTrainingBudgetError):
    """No new independent training allocation remains."""


def _write_state_locked(path: Path, state: dict[str, Any]) -> None:
    payload = {key: state[key] for key in _TOP_LEVEL_KEYS if key != "head_hash"}
    state["head_hash"] = _digest(payload)
    _atomic_write(path, state)


@dataclass(frozen=True)
class TrainingAllocation:
    slot: int
    max_retrains: int
    max_steps_per_retrain: int
    remaining_retrains: int
    reused: bool


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _path_for(path: str | os.PathLike[str]) -> Path:
    result = Path(path)
    if result.name in {"", "."}:
        raise ClientTrainingBudgetError("client budget path must name a JSON file")
    return result


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    if fcntl is None:
        raise ClientTrainingBudgetError("client training cap requires POSIX file locking")
    lock_path = Path(f"{path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _new_state(max_retrains: int, max_steps_per_retrain: int) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "max_retrains": max_retrains,
        "max_steps_per_retrain": max_steps_per_retrain,
        "allocations": [],
    }
    return {**payload, "head_hash": _digest(payload)}


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ClientTrainingBudgetError(f"client training budget is unreadable: {path}") from exc
    if not isinstance(value, dict) or set(value) != _TOP_LEVEL_KEYS:
        raise ClientTrainingBudgetError("client training budget has an invalid shape")
    payload = {key: value[key] for key in _TOP_LEVEL_KEYS if key != "head_hash"}
    if value.get("schema_version") != SCHEMA_VERSION or value.get("head_hash") != _digest(payload):
        raise ClientTrainingBudgetError("client training budget hash or schema is invalid")
    if not isinstance(value.get("max_retrains"), int) or value["max_retrains"] <= 0:
        raise ClientTrainingBudgetError("client training budget max_retrains is invalid")
    if not isinstance(value.get("max_steps_per_retrain"), int) or value["max_steps_per_retrain"] <= 0:
        raise ClientTrainingBudgetError("client training budget max_steps_per_retrain is invalid")
    allocations = value.get("allocations")
    if not isinstance(allocations, list):
        raise ClientTrainingBudgetError("client training budget allocations are invalid")
    for expected_slot, allocation in enumerate(allocations, start=1):
        if not isinstance(allocation, dict) or set(allocation) != _ALLOCATION_KEYS:
            raise ClientTrainingBudgetError("client training budget allocation is invalid")
        if allocation.get("slot") != expected_slot or not isinstance(allocation.get("output_dir"), str):
            raise ClientTrainingBudgetError("client training budget allocation identity is invalid")
        if not isinstance(allocation.get("budget_steps"), int) or allocation["budget_steps"] <= 0:
            raise ClientTrainingBudgetError("client training budget allocation steps are invalid")
        if allocation["budget_steps"] > value["max_steps_per_retrain"]:
            raise ClientTrainingBudgetError("client training budget allocation exceeds its configured cap")
    if len(allocations) > value["max_retrains"]:
        raise ClientTrainingBudgetError("client training budget contains too many allocations")
    return value


def _validate_limits(max_retrains: int, max_steps_per_retrain: int) -> None:
    if not isinstance(max_retrains, int) or isinstance(max_retrains, bool) or max_retrains <= 0:
        raise ClientTrainingBudgetError("max_retrains must be a positive integer")
    if not isinstance(max_steps_per_retrain, int) or isinstance(max_steps_per_retrain, bool) or max_steps_per_retrain <= 0:
        raise ClientTrainingBudgetError("max_steps_per_retrain must be a positive integer")


def _upgrade_step_limit(state: dict[str, Any], max_retrains: int, max_steps_per_retrain: int) -> bool:
    """Raise a persisted client cap without discarding its allocation history."""
    if state["max_retrains"] != max_retrains:
        raise ClientTrainingBudgetError("client training budget configuration changed")
    current_limit = state["max_steps_per_retrain"]
    if current_limit == max_steps_per_retrain:
        return False
    if current_limit < max_steps_per_retrain:
        state["max_steps_per_retrain"] = max_steps_per_retrain
        return True
    raise ClientTrainingBudgetError("client training budget configuration changed")


def reserve_training_allocation(
    path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    budget_steps: int,
    *,
    max_retrains: int = DEFAULT_MAX_RETRAINS,
    max_steps_per_retrain: int = DEFAULT_MAX_STEPS_PER_RETRAIN,
) -> TrainingAllocation:
    """Reserve one new output directory, or reuse its existing allocation."""

    _validate_limits(max_retrains, max_steps_per_retrain)
    if not isinstance(budget_steps, int) or isinstance(budget_steps, bool) or budget_steps <= 0:
        raise ClientTrainingBudgetError("budget_steps must be a positive integer")
    budget_path = _path_for(path)
    output_key = str(Path(output_dir).resolve())
    with _exclusive_lock(budget_path):
        state = _read(budget_path) if budget_path.exists() else _new_state(max_retrains, max_steps_per_retrain)
        if _upgrade_step_limit(state, max_retrains, max_steps_per_retrain):
            _write_state_locked(budget_path, state)
        matching = [row for row in state["allocations"] if row["output_dir"] == output_key]
        if matching:
            allocated_steps = sum(row["budget_steps"] for row in matching)
            if allocated_steps != budget_steps:
                raise ClientTrainingBudgetError(
                    "this output directory already has a different client training allocation; use its explicit extension action or choose a new directory"
                )
            return TrainingAllocation(
                matching[0]["slot"], state["max_retrains"], state["max_steps_per_retrain"],
                state["max_retrains"] - len(state["allocations"]), True,
            )
        if budget_steps > max_steps_per_retrain:
            raise ClientTrainingBudgetError(
                f"new runs are limited to {max_steps_per_retrain:,} transitions per retrain; extensions must attach to an existing output directory"
            )
        if len(state["allocations"]) >= state["max_retrains"]:
            raise ClientRetrainLimitExceeded(
                f"client retrain limit reached: {state['max_retrains']} independent training allocations"
            )
        allocation = {"slot": len(state["allocations"]) + 1, "output_dir": output_key, "budget_steps": budget_steps}
        state["allocations"].append(allocation)
        _write_state_locked(budget_path, state)
        return TrainingAllocation(
            allocation["slot"], state["max_retrains"], state["max_steps_per_retrain"],
            state["max_retrains"] - len(state["allocations"]), False,
        )


def extend_training_allocation(
    path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    additional_steps: int = DEFAULT_MAX_STEPS_PER_RETRAIN,
    *,
    max_retrains: int = DEFAULT_MAX_RETRAINS,
    max_steps_per_retrain: int = DEFAULT_MAX_STEPS_PER_RETRAIN,
) -> TrainingAllocation:
    """Spend one retraining slot to extend an existing model's transition cap."""
    _validate_limits(max_retrains, max_steps_per_retrain)
    if additional_steps != max_steps_per_retrain:
        raise ClientTrainingBudgetError("an extension must add exactly one full transition allocation")
    budget_path = _path_for(path)
    output_key = str(Path(output_dir).resolve())
    with _exclusive_lock(budget_path):
        if not budget_path.exists():
            raise ClientTrainingBudgetError("there is no existing client allocation to extend")
        state = _read(budget_path)
        if _upgrade_step_limit(state, max_retrains, max_steps_per_retrain):
            _write_state_locked(budget_path, state)
        matching = [row for row in state["allocations"] if row["output_dir"] == output_key]
        if not matching:
            raise ClientTrainingBudgetError("the selected output directory has no client training allocation")
        if len(state["allocations"]) >= state["max_retrains"]:
            raise ClientRetrainLimitExceeded("no iteration remains to extend this model")
        allocation = {"slot": len(state["allocations"]) + 1, "output_dir": output_key, "budget_steps": additional_steps}
        state["allocations"].append(allocation)
        _write_state_locked(budget_path, state)
        return TrainingAllocation(
            allocation["slot"], state["max_retrains"], state["max_steps_per_retrain"],
            state["max_retrains"] - len(state["allocations"]), False,
        )


def release_training_allocation(
    path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    max_retrains: int = DEFAULT_MAX_RETRAINS,
    max_steps_per_retrain: int = DEFAULT_MAX_STEPS_PER_RETRAIN,
) -> bool:
    """Release an allocation that failed before producing a checkpoint.

    This is deliberately narrow: callers must decide that no checkpoint was
    written.  A completed or partially published run remains allocated so a
    failed resume cannot silently create a free retrain.
    """

    _validate_limits(max_retrains, max_steps_per_retrain)
    budget_path = _path_for(path)
    output_key = str(Path(output_dir).resolve())
    with _exclusive_lock(budget_path):
        if not budget_path.exists():
            return False
        state = _read(budget_path)
        if _upgrade_step_limit(state, max_retrains, max_steps_per_retrain):
            _write_state_locked(budget_path, state)
        before = len(state["allocations"])
        state["allocations"] = [item for item in state["allocations"] if item["output_dir"] != output_key]
        if len(state["allocations"]) == before:
            return False
        for slot, allocation in enumerate(state["allocations"], start=1):
            allocation["slot"] = slot
        _write_state_locked(budget_path, state)
        return True


def client_budget_status(
    path: str | os.PathLike[str],
    *,
    max_retrains: int = DEFAULT_MAX_RETRAINS,
    max_steps_per_retrain: int = DEFAULT_MAX_STEPS_PER_RETRAIN,
) -> dict[str, Any]:
    """Return dashboard-safe status without creating the ledger."""

    _validate_limits(max_retrains, max_steps_per_retrain)
    budget_path = _path_for(path)
    with _exclusive_lock(budget_path):
        state = _read(budget_path) if budget_path.exists() else _new_state(max_retrains, max_steps_per_retrain)
        if _upgrade_step_limit(state, max_retrains, max_steps_per_retrain):
            _write_state_locked(budget_path, state)
    return {
        "max_retrains": state["max_retrains"],
        "retraining_started": len(state["allocations"]),
        "retraining_remaining": state["max_retrains"] - len(state["allocations"]),
        "max_steps_per_retrain": state["max_steps_per_retrain"],
        "path": str(budget_path),
    }


__all__ = [
    "ClientRetrainLimitExceeded",
    "ClientTrainingBudgetError",
    "DEFAULT_MAX_RETRAINS",
    "DEFAULT_MAX_STEPS_PER_RETRAIN",
    "TrainingAllocation",
    "client_budget_status",
    "extend_training_allocation",
    "release_training_allocation",
    "reserve_training_allocation",
]
