"""Environment wrapper that makes transition budgets algorithm-independent."""

from __future__ import annotations

from typing import Any

import numpy as np

from .budget import RunBudget
from .vector_env import VectorizedRacingEnv


class BudgetedEnv:
    """Reserve active-lane transitions immediately before a vector step.

    The wrapper delegates physics, observations, rewards, and termination to
    :class:`VectorizedRacingEnv`. Reservations are conservative: if participant
    code fails after reservation, the transitions remain charged.
    """

    budget_managed = True

    def __init__(self, env: VectorizedRacingEnv, budget: RunBudget, run_id: str) -> None:
        if not isinstance(env, VectorizedRacingEnv):
            raise TypeError("env must be a VectorizedRacingEnv")
        if not isinstance(budget, RunBudget):
            raise TypeError("budget must be a RunBudget")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        self.env = env
        self.budget = budget
        self.run_id = run_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _reserve(self, active_mask: np.ndarray | None) -> np.ndarray:
        if active_mask is None:
            mask = np.ones(self.env.n_envs, dtype=bool)
        else:
            mask = np.asarray(active_mask)
            if mask.shape != (self.env.n_envs,) or mask.dtype != np.dtype(bool):
                raise ValueError(f"active_mask must be a boolean array with shape ({self.env.n_envs},)")
            mask = mask.copy()
        count = int(np.count_nonzero(mask))
        if count:
            self.budget.consume(self.run_id, count)
        return mask

    def step(self, actions: Any, *, active_mask: np.ndarray | None = None):
        mask = self._reserve(active_mask)
        return self.env.step(actions, active_mask=mask)

    def step_raw(self, actions: Any, *, active_mask: np.ndarray | None = None):
        mask = self._reserve(active_mask)
        return self.env.step_raw(actions, active_mask=mask)


__all__ = ["BudgetedEnv"]
