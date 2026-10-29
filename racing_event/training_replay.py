"""Background deterministic replay capture for the current best training policy."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import queue
import threading
from typing import Any
from uuid import uuid4

import torch

from .evaluation import evaluate_policy
from .observation import RunningObservationNormalizer
from .track import TrackGeometry
from .trajectory import ReplayWriter, replay_header, replay_result
from .vector_env import VectorizedRacingEnv


class BestTrainingReplay:
    """Evaluate policy snapshots off-thread and retain one best replay per run."""

    def __init__(self, output_dir: Path, track: TrackGeometry, config: Any):
        self.directory = Path(output_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        for stale in self.directory.glob(".candidate-*"):
            if stale.is_file() and not stale.is_symlink():
                stale.unlink(missing_ok=True)
        self.path = self.directory / "best_replay.json"
        self.summary_path = self.directory / "best_replay_summary.json"
        self.track, self.config = track, config
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._best_rank: tuple[float, ...] | None = None
        if self.summary_path.is_file():
            try:
                old = json.loads(self.summary_path.read_text(encoding="utf-8"))
                if old.get("track_id") == track.track_id and self.path.is_file():
                    self._best_rank = tuple(float(value) for value in old["rank"])
            except (OSError, ValueError, TypeError, KeyError):
                pass
        self._thread = threading.Thread(target=self._run, name="synapse-best-replay", daemon=True)
        self._thread.start()

    def submit(self, policy: Any, normalizer: RunningObservationNormalizer | None, update: int) -> None:
        snapshot = deepcopy(policy).to("cpu")
        snapshot.eval()
        normalizer_snapshot = None if normalizer is None else RunningObservationNormalizer.from_state_dict(normalizer.state_dict())
        item = (snapshot, normalizer_snapshot, int(update))
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            self._queue.put_nowait(item)

    def close(self) -> None:
        self._queue.join()
        self._queue.put(None)
        self._thread.join()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._evaluate(*item)
            except Exception as exc:  # Replay capture is diagnostic; training continues.
                self._write_status({"state": "error", "error": str(exc)})
            finally:
                self._queue.task_done()

    def _evaluate(self, policy: Any, normalizer: Any, update: int) -> None:
        env = VectorizedRacingEnv(self.track, self.config, reward=None, n_envs=1)
        candidate = self.directory / f".candidate-{uuid4().hex}.json"
        try:
            with ReplayWriter(candidate, replay_header(self.track, self.config, mode="evaluation", participant_id="SYNAPSE-training-best")) as writer:
                result = evaluate_policy(
                    env, policy.as_policy(deterministic=True), normalizer=normalizer,
                    max_steps=self.config.episode.max_steps, frame_sink=writer.append,
                )
                writer.finish(replay_result(result, self.config.tick_hz))
            rank = (
                float(result.status == "finished"), float(result.final_distance_m),
                float(-result.off_track_steps), float(-(result.lap_time_s or 0.0)),
            )
            with self._lock:
                if self._best_rank is None or rank > self._best_rank:
                    os.replace(candidate, self.path)
                    self._best_rank = rank
                    self._write_status({
                        "state": "ready", "update": update, "track_id": result.track_id,
                        "status": result.status, "final_distance_m": result.final_distance_m,
                        "lap_time_s": result.lap_time_s, "off_track_steps": result.off_track_steps,
                        "rank": list(rank), "replay": str(self.path),
                    })
        finally:
            candidate.unlink(missing_ok=True)

    def _write_status(self, payload: dict[str, Any]) -> None:
        merged: dict[str, Any] = {}
        try:
            merged = json.loads(self.summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        merged.update(payload)
        temporary = self.summary_path.with_name(f".{self.summary_path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(merged, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, self.summary_path)
        except OSError:
            temporary.unlink(missing_ok=True)


__all__ = ["BestTrainingReplay"]
