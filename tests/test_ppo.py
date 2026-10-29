from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from racing_event.ppo import (
    FixedMLPActorCritic,
    PPOTrainer,
    compute_masked_gae,
)
from racing_event.rollout import PPOHyperparameters, RolloutBatch


def make_batch(
    policy: FixedMLPActorCritic,
    *,
    rewards: np.ndarray | None = None,
    active_mask: np.ndarray | None = None,
    terminated: np.ndarray | None = None,
) -> RolloutBatch:
    time_steps, n_envs = (2, 2)
    observations = np.asarray(
        [
            [[0.1, 0.0, 0.2, -0.1], [0.2, 0.1, 0.0, 0.3]],
            [[0.2, 0.1, 0.1, -0.2], [0.3, 0.0, 0.2, 0.1]],
        ],
        dtype=np.float32,
    )
    with torch.no_grad():
        actions = policy.act(observations.reshape(-1, 4), deterministic=True).reshape(time_steps, n_envs, 2)
    return RolloutBatch(
        observations=observations,
        next_observations=observations + 0.01,
        actions=actions,
        rewards=np.asarray(rewards if rewards is not None else [[1.0, 0.5], [0.25, 0.0]], dtype=np.float64),
        terminated=np.asarray(
            terminated if terminated is not None else np.zeros((time_steps, n_envs), dtype=bool),
            dtype=bool,
        ),
        truncated=np.zeros((time_steps, n_envs), dtype=bool),
        active_mask=np.asarray(
            active_mask if active_mask is not None else np.ones((time_steps, n_envs), dtype=bool),
            dtype=bool,
        ),
        signals={},
        track_ids=(("track", "track"), ("track", "track")),
    )


class GaeTests(unittest.TestCase):
    def test_terminated_transition_does_not_bootstrap(self) -> None:
        advantages, returns = compute_masked_gae(
            np.asarray([[1.0]]),
            np.asarray([[2.0]]),
            np.asarray([[100.0]]),
            np.asarray([[True]]),
            np.asarray([[False]]),
            np.asarray([[True]]),
            gamma=0.99,
            gae_lambda=0.95,
        )
        np.testing.assert_allclose(advantages, [[-1.0]])
        np.testing.assert_allclose(returns, [[1.0]])

    def test_truncated_transition_bootstraps_but_does_not_carry(self) -> None:
        advantages, returns = compute_masked_gae(
            np.asarray([[1.0], [100.0]]),
            np.zeros((2, 1)),
            np.zeros((2, 1)),
            np.asarray([[False], [False]]),
            np.asarray([[True], [False]]),
            np.ones((2, 1), dtype=bool),
            gamma=0.99,
            gae_lambda=0.95,
        )
        np.testing.assert_allclose(advantages, [[1.0], [100.0]])
        np.testing.assert_allclose(returns, [[1.0], [100.0]])

    def test_inactive_lanes_are_ignored(self) -> None:
        active = np.asarray([[True, False], [True, False]])
        advantages, returns = compute_masked_gae(
            np.asarray([[1.0, 100000.0], [2.0, 100000.0]]),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.zeros((2, 2), dtype=bool),
            np.zeros((2, 2), dtype=bool),
            active,
            gamma=0.99,
            gae_lambda=0.95,
        )
        np.testing.assert_allclose(advantages[:, 0], [1.0 + 0.99 * 0.95 * 2.0, 2.0])
        np.testing.assert_allclose(advantages[:, 1], [0.0, 0.0])
        np.testing.assert_allclose(returns[:, 1], [0.0, 0.0])

    def test_both_terminal_flags_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compute_masked_gae(
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                np.asarray([[True]]),
                np.asarray([[True]]),
                np.asarray([[True]]),
                gamma=0.99,
                gae_lambda=0.95,
            )


class PpoTests(unittest.TestCase):
    def test_actions_are_bounded_and_update_changes_weights(self) -> None:
        torch.manual_seed(3)
        policy = FixedMLPActorCritic(4)
        observations = np.zeros((5, 4), dtype=np.float32)
        actions = policy.act(observations)
        self.assertEqual(actions.shape, (5, 2))
        self.assertTrue(np.all(np.isfinite(actions)))
        self.assertTrue(np.all(actions >= -1.0))
        self.assertTrue(np.all(actions <= 1.0))

        before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
        trainer = PPOTrainer(
            policy,
            PPOHyperparameters(update_epochs=2, learning_rate=1.0e-2),
        )
        result = trainer.update(make_batch(policy))
        self.assertEqual(result.transitions, 4)
        self.assertEqual(result.optimizer_steps, 2)
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in policy.state_dict().items()))
        self.assertTrue(np.isfinite(result.policy_loss))
        self.assertTrue(np.isfinite(result.value_loss))
        self.assertTrue(np.isfinite(result.approx_kl))

    def test_inactive_transition_does_not_reach_update(self) -> None:
        torch.manual_seed(11)
        first = FixedMLPActorCritic(4)
        second = FixedMLPActorCritic(4)
        second.load_state_dict(first.state_dict())
        active = np.asarray([[True, True], [True, False]])
        first_batch = make_batch(first, active_mask=active)
        second_batch = make_batch(second, active_mask=active, rewards=[[1.0, 0.5], [0.25, 999999.0]])
        # The inactive lane's action/reward can contain arbitrary collector
        # padding; PPO must use only active rows.
        second_batch = RolloutBatch(
            observations=second_batch.observations,
            next_observations=second_batch.next_observations,
            actions=second_batch.actions,
            rewards=second_batch.rewards,
            terminated=second_batch.terminated,
            truncated=second_batch.truncated,
            active_mask=second_batch.active_mask,
            signals={},
            track_ids=second_batch.track_ids,
        )
        PPOTrainer(first, PPOHyperparameters(update_epochs=1)).update(first_batch)
        PPOTrainer(second, PPOHyperparameters(update_epochs=1)).update(second_batch)
        for first_value, second_value in zip(first.state_dict().values(), second.state_dict().values()):
            torch.testing.assert_close(first_value, second_value)

    def test_constant_returns_have_no_explained_variance(self) -> None:
        torch.manual_seed(5)
        policy = FixedMLPActorCritic(4)
        result = PPOTrainer(policy, PPOHyperparameters(update_epochs=1)).update(
            make_batch(
                policy,
                rewards=np.ones((2, 2), dtype=np.float64),
                terminated=np.ones((2, 2), dtype=bool),
            )
        )
        self.assertIsNone(result.explained_variance)

    def test_safe_model_artifact_round_trip_and_metadata_rejection(self) -> None:
        torch.manual_seed(7)
        policy = FixedMLPActorCritic(4)
        observations = np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "policy.pt"
            metadata = Path(directory) / "policy.json"
            policy.save_artifact(weights, metadata)
            loaded, payload = FixedMLPActorCritic.load_artifact(weights, metadata, expected_observation_size=4)
            np.testing.assert_allclose(policy.act(observations, deterministic=True), loaded.act(observations, deterministic=True))
            self.assertEqual(payload["distribution"], "tanh_normal")
            self.assertEqual(payload["policy_mean_bias"], [0.0, 0.5])
            metadata.write_text(metadata.read_text(encoding="utf-8").replace('"action_size": 2', '"action_size": 3'), encoding="utf-8")
            with self.assertRaises(ValueError):
                FixedMLPActorCritic.load_artifact(weights, metadata)


if __name__ == "__main__":
    unittest.main()
