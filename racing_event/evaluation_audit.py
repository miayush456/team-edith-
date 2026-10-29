"""Bounded, policy-invisible physical diagnostics for one evaluation episode.

These counters are reports, not reward terms or disqualification rules. They
consume already-authoritative transition data without extra geometry queries.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .telemetry import DEFAULT_SECTOR_COUNT


class EvaluationAudit:
    def __init__(self, track_length: float, dt: float) -> None:
        self.track_length = track_length
        self.dt = dt
        self.steps = 0
        self.off_track_steps = 0
        self.off_track_run = 0
        self.max_off_track_run = 0
        self.stalled_run = 0
        self.max_stalled_run = 0
        self.stalled_steps = 0
        self.positive_progress = 0.0
        self.off_track_progress = 0.0
        self.backward_progress = 0.0
        self.max_boundary_excess = 0.0
        self.coverage_mask = 0
        self.previous_action = np.zeros(2, dtype=np.float64)
        self.action_delta_sum = np.zeros(2, dtype=np.float64)
        self.saturated_components = np.zeros(2, dtype=np.int64)
        self.split_ticks: list[int | None] = [None] * DEFAULT_SECTOR_COUNT

    def update(
        self, *, previous_distance: float, distance: float, off_track: bool,
        boundary_excess: float, coverage_mask: int, raw_action: np.ndarray,
    ) -> None:
        self.steps += 1
        delta = distance - previous_distance
        forward = max(delta, 0.0)
        self.positive_progress += forward
        self.backward_progress += max(-delta, 0.0)
        self.off_track_steps += int(off_track)
        self.off_track_progress += forward if off_track else 0.0
        self.off_track_run = self.off_track_run + 1 if off_track else 0
        self.max_off_track_run = max(self.max_off_track_run, self.off_track_run)
        stalled = abs(delta) < 0.01  # Matches the engine's negligible-progress test.
        self.stalled_steps += int(stalled)
        self.stalled_run = self.stalled_run + 1 if stalled else 0
        self.max_stalled_run = max(self.max_stalled_run, self.stalled_run)
        self.max_boundary_excess = max(self.max_boundary_excess, boundary_excess)
        self.coverage_mask = coverage_mask
        applied = np.clip(raw_action, -1.0, 1.0)
        self.action_delta_sum += np.abs(applied - self.previous_action)
        self.saturated_components += np.abs(raw_action) >= 1.0
        self.previous_action = applied.copy()
        # First forward crossing of each spawn-relative threshold, including
        # the full-lap boundary. No reset credit, backward credit, or recrossing
        # credit; all crossings on a large transition retain its integer tick.
        for index, tick in enumerate(self.split_ticks):
            boundary = self.track_length * (index + 1) / DEFAULT_SECTOR_COUNT
            if tick is None and previous_distance < boundary <= distance:
                self.split_ticks[index] = self.steps

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": "1.0",
            "elapsed_s": self.steps * self.dt,
            "off_track_duration_s": self.off_track_steps * self.dt,
            "max_contiguous_off_track_steps": self.max_off_track_run,
            "positive_progress_m": self.positive_progress,
            "positive_off_track_progress_m": self.off_track_progress,
            "positive_off_track_progress_fraction": (
                self.off_track_progress / self.positive_progress
                if self.positive_progress > 0.0 else None
            ),
            "backward_progress_m": self.backward_progress,
            "max_boundary_excess_m": self.max_boundary_excess,
            "stalled_steps": self.stalled_steps,
            "max_contiguous_stalled_steps": self.max_stalled_run,
            "sector_count": DEFAULT_SECTOR_COUNT,
            "sector_coverage_mask": self.coverage_mask,
            "sector_coverage_count": self.coverage_mask.bit_count(),
            "sector_coverage_fraction": self.coverage_mask.bit_count() / DEFAULT_SECTOR_COUNT,
            "sector_first_crossing_ticks": list(self.split_ticks),
            "action_saturation_rate": {
                name: int(self.saturated_components[index]) / self.steps if self.steps else None
                for index, name in enumerate(("steering", "longitudinal"))
            },
            "mean_abs_applied_action_delta": {
                name: float(self.action_delta_sum[index]) / self.steps if self.steps else None
                for index, name in enumerate(("steering", "longitudinal"))
            },
        }
