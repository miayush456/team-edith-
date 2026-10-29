"""Persistent per-run environment-step accounting.

``RunBudget`` implements a configurable ``n`` runs by ``k`` transitions per
run budget.  A call to :meth:`RunBudget.consume` reserves transitions before
the caller steps its environment.  A process crash after the reservation can
therefore conservatively count transitions that were not actually executed.
The argument is a count of environment transitions, not a count of calls to
a vectorized environment.  A vectorized caller must pass the number of
transitions represented by its batch.

The state is an append-only JSON event log.  Each record includes the hash of
the preceding record, and the whole file is replaced atomically after every
mutation.  Handles do not keep a mutable snapshot: every operation reloads
the file while holding the local lock.  This prevents two open handles from
silently losing reservations.

The current implementation requires a local POSIX filesystem with
``fcntl.flock`` (the prototype target is Linux).  It does not provide
network-filesystem or cross-host locking guarantees.  The hash chain is
tamper-evident, not an authentication mechanism: someone who controls the
machine can delete or rewrite the file and recompute the chain.  There are no
secrets, signatures, or claims that this local file proves compliance.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterator

try:  # The supported prototype platform is POSIX/Linux.
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX hosts.
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:
    msvcrt = None


SCHEMA_VERSION = 1
_ZERO_HASH = "0" * 64
_RUN_ID_RE = re.compile(r"run-[0-9]{6}\Z")
_TOP_LEVEL_KEYS = {"schema_version", "records", "head_hash"}
_GENESIS_KEYS = {"seq", "type", "config", "prev_hash", "hash"}
_EVENT_KEYS = {"seq", "type", "run_id", "steps", "prev_hash", "hash"}


class RunBudgetError(Exception):
    """Base class for errors raised by the budget API."""


class BudgetConfigurationError(RunBudgetError, ValueError):
    """The requested budget configuration is invalid or does not match."""


class BudgetCorruptionError(RunBudgetError):
    """The persisted JSON state or its hash chain is invalid."""


class RunLimitExceeded(RunBudgetError):
    """All configured logical runs have already been started."""


class StepLimitExceeded(RunBudgetError):
    """A reservation would exceed the per-run transition limit."""


class UnknownRunError(RunBudgetError, KeyError):
    """The requested logical run does not exist."""


class BudgetClosedError(RunBudgetError):
    """An operation was attempted on a closed handle."""


@dataclass
class _State:
    max_runs: int
    max_steps_per_run: int
    records: list[dict[str, Any]]
    head_hash: str
    used_steps: dict[str, int]


def _canonical_json(value: Any) -> bytes:
    """Serialize hash input without accepting non-finite JSON values."""

    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _record_hash(prev_hash: str, body: dict[str, Any]) -> str:
    hash_input = {"prev_hash": prev_hash, **body}
    return hashlib.sha256(_canonical_json(hash_input)).hexdigest()


def _make_record(prev_hash: str, body: dict[str, Any]) -> dict[str, Any]:
    record = {**body, "prev_hash": prev_hash}
    record["hash"] = _record_hash(prev_hash, body)
    return record


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_positive_int(name: str, value: Any) -> int:
    """Validate an API integer, rejecting bools, floats, and non-finite values."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be a finite positive integer")
    if not _is_strict_int(value) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_nonnegative_int(name: str, value: Any) -> int:
    if not _is_strict_int(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _resolve_config_value(
    primary_name: str,
    primary: Any,
    alias_name: str,
    alias: Any,
    *,
    required: bool,
) -> int | None:
    """Resolve the descriptive API name and the short n/k spelling."""

    if primary is not None and alias is not None:
        first = _require_positive_int(primary_name, primary)
        second = _require_positive_int(alias_name, alias)
        if first != second:
            raise BudgetConfigurationError(
                f"{primary_name} and {alias_name} specify different values"
            )
        return first
    if primary is not None:
        return _require_positive_int(primary_name, primary)
    if alias is not None:
        return _require_positive_int(alias_name, alias)
    if required:
        raise BudgetConfigurationError(
            f"{primary_name} is required and must be a positive integer"
        )
    return None


def _path_for(path: str | os.PathLike[str]) -> Path:
    try:
        result = Path(path)
    except TypeError as exc:
        raise TypeError("path must be a filesystem path") from exc
    if result.name in {"", "."}:
        raise ValueError("path must name a JSON state file")
    return result


@contextmanager
def _exclusive_lock(state_path: Path) -> Iterator[None]:
    """Lock a sidecar file so atomic state replacement does not lose updates."""

    lock_path = Path(f"{state_path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if fcntl is not None:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
    elif msvcrt is not None:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
        try:
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            yield
        finally:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            finally:
                os.close(fd)
    else:
        yield


def _read_state(path: Path) -> _State:
    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(
            text,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {value!r}")
            ),
        )
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise BudgetCorruptionError(f"cannot read budget state {path}: {exc}") from exc

    try:
        return _validate_payload(payload)
    except BudgetCorruptionError:
        raise
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise BudgetCorruptionError(f"invalid budget state {path}: {exc}") from exc


def _validate_hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a 64-character hexadecimal hash")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a 64-character hexadecimal hash") from exc
    if value != value.lower():
        raise ValueError(f"{field} must use lowercase hexadecimal")
    return value


def _validate_run_id(run_id: Any) -> str:
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError("run_id must have the form run-000001")
    return run_id


def _validate_payload(payload: Any) -> _State:
    if not isinstance(payload, dict) or set(payload) != _TOP_LEVEL_KEYS:
        raise ValueError("state must contain exactly schema_version, records, and head_hash")
    if not _is_strict_int(payload["schema_version"]) or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {payload['schema_version']!r}")
    records = payload["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list")
    head_hash = _validate_hash(payload["head_hash"], "head_hash")

    previous_hash = _ZERO_HASH
    max_runs: int | None = None
    max_steps_per_run: int | None = None
    used_steps: dict[str, int] = {}

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"record {index} must be an object")
        if index == 0:
            if set(record) != _GENESIS_KEYS:
                raise ValueError("genesis record has an invalid shape")
        elif set(record) != _EVENT_KEYS:
            raise ValueError(f"event record {index} has an invalid shape")

        if not _is_strict_int(record.get("seq")) or record["seq"] != index:
            raise ValueError(f"record {index} has an invalid sequence number")
        if record.get("prev_hash") != previous_hash:
            raise ValueError(f"record {index} has an invalid previous hash")
        _validate_hash(record.get("prev_hash"), f"record {index}.prev_hash")
        stored_hash = _validate_hash(record.get("hash"), f"record {index}.hash")

        if index == 0:
            if record.get("type") != "genesis":
                raise ValueError("first record must be genesis")
            config = record.get("config")
            if not isinstance(config, dict) or set(config) != {
                "max_runs",
                "max_steps_per_run",
            }:
                raise ValueError("genesis config has an invalid shape")
            max_runs = _require_positive_int("max_runs", config["max_runs"])
            max_steps_per_run = _require_positive_int(
                "max_steps_per_run", config["max_steps_per_run"]
            )
            body = {
                "seq": record["seq"],
                "type": record["type"],
                "config": config,
            }
        else:
            if record.get("type") not in {"run_started", "steps_reserved", "budget_extended"}:
                raise ValueError(f"record {index} has an unknown event type")
            run_id = _validate_run_id(record.get("run_id"))
            steps = record.get("steps")
            if record["type"] == "run_started":
                if steps != 0:
                    raise ValueError("run_started must have zero steps")
                expected_id = f"run-{len(used_steps) + 1:06d}"
                if run_id != expected_id:
                    raise ValueError("run IDs must be unique and sequential")
                if max_runs is None or len(used_steps) >= max_runs:
                    raise ValueError("state contains more runs than configured")
                used_steps[run_id] = 0
            elif record["type"] == "budget_extended":
                steps = _require_positive_int("steps", steps)
                if run_id not in used_steps:
                    raise ValueError("budget_extended references an unknown run")
                max_steps_per_run += steps
            else:
                steps = _require_positive_int("steps", steps)
                if run_id not in used_steps:
                    raise ValueError("steps_reserved references an unknown run")
                new_total = used_steps[run_id] + steps
                if max_steps_per_run is None or new_total > max_steps_per_run:
                    raise ValueError("state contains more steps than configured")
                used_steps[run_id] = new_total
            body = {
                "seq": record["seq"],
                "type": record["type"],
                "run_id": run_id,
                "steps": steps,
            }

        if _record_hash(previous_hash, body) != stored_hash:
            raise ValueError(f"record {index} hash does not match its contents")
        previous_hash = stored_hash

    if max_runs is None or max_steps_per_run is None:
        raise ValueError("state has no configuration")
    if head_hash != previous_hash:
        raise ValueError("head_hash does not match the final record")
    return _State(max_runs, max_steps_per_run, records, head_hash, used_steps)


def _new_state(max_runs: int, max_steps_per_run: int) -> _State:
    genesis_body = {
        "seq": 0,
        "type": "genesis",
        "config": {
            "max_runs": max_runs,
            "max_steps_per_run": max_steps_per_run,
        },
    }
    genesis = _make_record(_ZERO_HASH, genesis_body)
    return _State(max_runs, max_steps_per_run, [genesis], genesis["hash"], {})


def _append_event(state: _State, event_type: str, run_id: str, steps: int) -> None:
    body = {
        "seq": len(state.records),
        "type": event_type,
        "run_id": run_id,
        "steps": steps,
    }
    record = _make_record(state.head_hash, body)
    state.records.append(record)
    state.head_hash = record["hash"]


def _write_state(path: Path, state: _State) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "records": state.records,
        "head_hash": state.head_hash,
    }
    serialized = _canonical_json(payload) + b"\n"
    temporary_path: str | None = None
    try:
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None

        # Ensure the rename itself is durable on the local POSIX filesystem.
        if os.name != "nt" and hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


class RunBudget:
    """A persistent, per-logical-run transition budget.

    Use :meth:`create` once and :meth:`open` for later processes.  ``consume``
    is a reservation operation and returns the remaining transitions for that
    run.  It accepts partial batches, but rejects a batch larger than the
    current remainder without changing the file.
    """

    def __init__(self, path: Path, max_runs: int, max_steps_per_run: int) -> None:
        self.path = path
        self.max_runs = max_runs
        self.max_steps_per_run = max_steps_per_run
        self._closed = False

    @classmethod
    def create(
        cls,
        path: str | os.PathLike[str],
        max_runs: int | None = None,
        max_steps_per_run: int | None = None,
        *,
        n: int | None = None,
        k: int | None = None,
    ) -> "RunBudget":
        """Create a new state file with required positive ``n`` and ``k``.

        ``max_runs``/``max_steps_per_run`` are the canonical names.  ``n``
        and ``k`` are accepted as concise aliases when integrating with the
        event's terminology; specifying both forms requires equal values.
        Creation never overwrites an existing state file.
        """

        resolved_runs = _resolve_config_value(
            "max_runs", max_runs, "n", n, required=True
        )
        resolved_steps = _resolve_config_value(
            "max_steps_per_run",
            max_steps_per_run,
            "k",
            k,
            required=True,
        )
        assert resolved_runs is not None and resolved_steps is not None
        state_path = _path_for(path)
        with _exclusive_lock(state_path):
            if state_path.exists():
                raise FileExistsError(f"budget state already exists: {state_path}")
            state = _new_state(resolved_runs, resolved_steps)
            _write_state(state_path, state)
        return cls(state_path, resolved_runs, resolved_steps)

    @classmethod
    def open(
        cls,
        path: str | os.PathLike[str],
        max_runs: int | None = None,
        max_steps_per_run: int | None = None,
        *,
        n: int | None = None,
        k: int | None = None,
    ) -> "RunBudget":
        """Open and validate a state file, optionally checking its config."""

        requested_runs = _resolve_config_value(
            "max_runs", max_runs, "n", n, required=False
        )
        requested_steps = _resolve_config_value(
            "max_steps_per_run",
            max_steps_per_run,
            "k",
            k,
            required=False,
        )
        state_path = _path_for(path)
        with _exclusive_lock(state_path):
            state = _read_state(state_path)
        if requested_runs is not None and requested_runs != state.max_runs:
            raise BudgetConfigurationError(
                f"max_runs mismatch: file has {state.max_runs}, requested {requested_runs}"
            )
        if requested_steps is not None and requested_steps != state.max_steps_per_run:
            raise BudgetConfigurationError(
                "max_steps_per_run mismatch: "
                f"file has {state.max_steps_per_run}, requested {requested_steps}"
            )
        return cls(state_path, state.max_runs, state.max_steps_per_run)

    def _ensure_open(self) -> None:
        if self._closed:
            raise BudgetClosedError("RunBudget handle is closed")

    @contextmanager
    def _locked_state(self) -> Iterator[_State]:
        self._ensure_open()
        with _exclusive_lock(self.path):
            state = _read_state(self.path)
            if (
                state.max_runs != self.max_runs
                or state.max_steps_per_run < self.max_steps_per_run
            ):
                raise BudgetConfigurationError(
                    "budget configuration changed incompatibly while this handle was open"
                )
            yield state

    def _mutate(self, operation: Any) -> Any:
        with self._locked_state() as state:
            result = operation(state)
            _write_state(self.path, state)
            return result

    def start_run(self) -> str:
        """Start one new logical run and return its stable ID.

        A run remains resumable indefinitely until its per-run reservation is
        exhausted.  Reopening the budget and using the same ID resumes it; a
        second call to ``start_run`` always requests a new logical run.
        """

        def operation(state: _State) -> str:
            if len(state.used_steps) >= state.max_runs:
                raise RunLimitExceeded(
                    f"maximum of {state.max_runs} logical runs has been started"
                )
            run_id = f"run-{len(state.used_steps) + 1:06d}"
            _append_event(state, "run_started", run_id, 0)
            state.used_steps[run_id] = 0
            return run_id

        return self._mutate(operation)

    def extend_run(self, run_id: str, additional_steps: int) -> int:
        """Append an auditable increase to the active run's transition cap."""
        run_id = _validate_run_id(run_id)
        additional_steps = _require_positive_int("additional_steps", additional_steps)
        with _exclusive_lock(self.path):
            state = _read_state(self.path)
            if state.max_runs != self.max_runs or state.max_steps_per_run < self.max_steps_per_run:
                raise BudgetConfigurationError("budget configuration changed incompatibly")
            if run_id not in state.used_steps:
                raise UnknownRunError(run_id)
            _append_event(state, "budget_extended", run_id, additional_steps)
            state.max_steps_per_run += additional_steps
            _write_state(self.path, state)
            self.max_steps_per_run = state.max_steps_per_run
            return state.max_steps_per_run

    def consume(self, run_id: str, steps: int) -> int:
        """Reserve ``steps`` environment transitions and return the remainder.

        ``steps`` must be a positive integer and may be a partial batch.  The
        operation is rejected atomically if it would exceed this logical
        run's ``max_steps_per_run``.  The reservation is persisted before the
        method returns, so callers should reserve immediately before stepping
        and treat a crash afterward as conservatively charged usage.
        """

        run_id = _validate_run_id(run_id)
        steps = _require_positive_int("steps", steps)

        def operation(state: _State) -> int:
            if run_id not in state.used_steps:
                raise UnknownRunError(run_id)
            remaining = state.max_steps_per_run - state.used_steps[run_id]
            if steps > remaining:
                raise StepLimitExceeded(
                    f"reservation of {steps} exceeds {remaining} remaining steps "
                    f"for {run_id}"
                )
            _append_event(state, "steps_reserved", run_id, steps)
            state.used_steps[run_id] += steps
            return remaining - steps

        return self._mutate(operation)

    def remaining(self, run_id: str) -> int:
        """Return the current unreserved transition count for ``run_id``."""

        run_id = _validate_run_id(run_id)

        def operation(state: _State) -> int:
            if run_id not in state.used_steps:
                raise UnknownRunError(run_id)
            return state.max_steps_per_run - state.used_steps[run_id]

        with self._locked_state() as state:
            return operation(state)

    def status(self, run_id: str | None = None) -> dict[str, Any]:
        """Return a snapshot of the whole budget or one logical run."""

        if run_id is not None:
            run_id = _validate_run_id(run_id)

        with self._locked_state() as state:
            if run_id is not None and run_id not in state.used_steps:
                raise UnknownRunError(run_id)

            def run_status(identifier: str) -> dict[str, Any]:
                used = state.used_steps[identifier]
                remaining = state.max_steps_per_run - used
                return {
                    "run_id": identifier,
                    "used_steps": used,
                    "remaining_steps": remaining,
                    "status": "exhausted" if remaining == 0 else "open",
                }

            if run_id is not None:
                return run_status(run_id)
            runs = [run_status(identifier) for identifier in state.used_steps]
            return {
                "max_runs": state.max_runs,
                "max_steps_per_run": state.max_steps_per_run,
                "runs_started": len(runs),
                "runs_remaining": state.max_runs - len(runs),
                "runs": runs,
                "head_hash": state.head_hash,
            }

    def close(self) -> None:
        """Close this lightweight handle; persisted state is unaffected."""

        self._closed = True

    def __enter__(self) -> "RunBudget":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


__all__ = [
    "BudgetClosedError",
    "BudgetConfigurationError",
    "BudgetCorruptionError",
    "RunBudget",
    "RunBudgetError",
    "RunLimitExceeded",
    "StepLimitExceeded",
    "UnknownRunError",
]
