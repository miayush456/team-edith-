"""Example participant-owned reward function.

Copy this into your trainer workspace and replace it freely. SYNAPSE supplies
the raw signals; this function is not loaded by the organizer evaluator.
"""

from collections.abc import Mapping

import numpy as np


def reward(
    observation: np.ndarray,
    action: np.ndarray,
    signals: Mapping[str, float],
    terminated: bool,
    truncated: bool,
) -> float:
    """Prefer progress and speed while charging for crashes and abrupt steering."""

    del observation, terminated, truncated
    return float(
        12.0 * signals["progress"]
        + 0.5 * signals["speed"]
        - 1.5 * signals["off_track"]
        - 10.0 * signals["crash"]
        - 0.05 * signals["steer_smoothness"]
        + 0.02 * float(action[1])
    )
