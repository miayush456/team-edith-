"""Small, fixed PPO reference trainer for SYNAPSE.

The environment and rollout collector remain NumPy-first.  This module is the
model/optimizer boundary only: it evaluates a fixed two-layer actor-critic,
computes masked GAE, and performs shuffled-minibatch clipped PPO updates. The model
artifact contains a state dictionary and separate JSON metadata.  Loading uses
``weights_only=True`` so a submission cannot ask PyTorch to reconstruct an
arbitrary Python object.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from numbers import Integral
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Normal

from .rollout import PPOHyperparameters, RolloutBatch


PPO_FORMAT = "racing_event_fixed_mlp"
PPO_FORMAT_VERSION = 2
PPO_HIDDEN_SIZES = (64, 64)
PPO_ACTION_SIZE = 2
PPO_POLICY_MEAN_BIAS = (0.0, 0.5)
_LOG_PROB_EPSILON = 1.0e-6


def _as_float_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _as_bool_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape or result.dtype != np.dtype(bool):
        raise ValueError(f"{name} must be a boolean array with shape {shape}")
    return result.copy()


def compute_masked_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    active_mask: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE without crossing inactive lanes or reset episode boundaries.

    ``terminated`` transitions do not bootstrap.  ``truncated`` transitions do
    bootstrap from their final observation, but neither flag carries the GAE
    recurrence into the next row because the collector may reset that lane
    before the next vector step.
    """

    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape (time, n_envs)")
    shape = rewards.shape
    values = _as_float_array(values, shape, "values")
    next_values = _as_float_array(next_values, shape, "next_values")
    rewards = _as_float_array(rewards, shape, "rewards")
    terminated = _as_bool_array(terminated, shape, "terminated")
    truncated = _as_bool_array(truncated, shape, "truncated")
    active_mask = _as_bool_array(active_mask, shape, "active_mask")

    if np.any(active_mask & terminated & truncated):
        raise ValueError("an active transition cannot be both terminated and truncated")
    try:
        gamma_value = float(gamma)
        lambda_value = float(gae_lambda)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("gamma and gae_lambda must be finite numbers") from exc
    if not np.isfinite(gamma_value) or not np.isfinite(lambda_value):
        raise ValueError("gamma and gae_lambda must be finite numbers")
    if not 0.0 <= gamma_value <= 1.0 or not 0.0 <= lambda_value <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")

    deltas = np.zeros(shape, dtype=np.float64)
    active = active_mask
    deltas[active] = (
        rewards[active]
        + gamma_value * (~terminated[active]) * next_values[active]
        - values[active]
    )

    advantages = np.zeros(shape, dtype=np.float64)
    carry = np.zeros(shape[1], dtype=np.float64)
    for step in range(shape[0] - 1, -1, -1):
        if step == shape[0] - 1:
            carry = np.zeros(shape[1], dtype=np.float64)
        else:
            same_episode = (
                active_mask[step]
                & active_mask[step + 1]
                & ~terminated[step]
                & ~truncated[step]
            )
            carry = np.where(same_episode, carry, 0.0)
        advantages[step] = np.where(
            active_mask[step],
            deltas[step] + gamma_value * lambda_value * carry,
            0.0,
        )
        carry = advantages[step]

    returns = np.where(active_mask, values + advantages, 0.0)
    return advantages, returns


class ActorCritic(nn.Module):
    """PPO policy boundary with a participant-owned actor and platform critic.

    ``actor`` must return the *pre-tanh* action mean with shape ``[B, 2]``.
    The platform owns the tanh-normal distribution and the value head so custom
    architectures cannot accidentally change the action contract.
    """

    def __init__(
        self,
        actor: nn.Module | None = None,
        observation_size: int | None = None,
        *,
        critic: nn.Module | None = None,
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        if actor is None:
            return
        if observation_size is None or isinstance(observation_size, bool) or not isinstance(observation_size, Integral) or observation_size <= 0:
            raise ValueError("observation_size must be a positive integer")
        if not isinstance(actor, nn.Module):
            raise TypeError("actor must be a torch.nn.Module")
        if not np.isfinite(float(log_std_init)):
            raise ValueError("log_std_init must be finite")
        self.actor = actor
        self.critic = critic or default_critic(int(observation_size))
        self.observation_size = int(observation_size)
        self.action_size = PPO_ACTION_SIZE
        self.hidden_sizes = ()
        self.log_std_init = float(log_std_init)
        self.log_std = nn.Parameter(torch.full((self.action_size,), self.log_std_init))

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        if not hasattr(self, "actor"):
            raise NotImplementedError("the actor-critic must implement forward")
        if observations.ndim != 2 or observations.shape[1] != self.observation_size:
            raise ValueError(f"observations must have shape (batch, {self.observation_size})")
        values = observations.float()
        mean = self.actor(values)
        if mean.ndim != 2 or mean.shape[1] != self.action_size:
            raise ValueError(f"actor must return shape (batch, {self.action_size})")
        critic_value = self.critic(values)
        if critic_value.ndim == 2 and critic_value.shape[1] == 1:
            critic_value = critic_value[:, 0]
        if critic_value.ndim != 1:
            raise ValueError("critic must return shape (batch,) or (batch, 1)")
        return mean, critic_value

    def _distribution(self, mean: Tensor) -> Normal:
        log_std = self.log_std.clamp(-5.0, 2.0).expand_as(mean)
        return Normal(mean, log_std.exp())

    def _tensor(self, value: Tensor | np.ndarray, width: int, name: str) -> Tensor:
        tensor = torch.as_tensor(value, device=self.log_std.device)
        if tensor.ndim != 2 or tensor.shape[1] != width:
            raise ValueError(f"{name} must have shape (batch, {width})")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values")
        return tensor.float()

    def evaluate_actions(
        self,
        observations: Tensor | np.ndarray,
        actions: Tensor | np.ndarray,
    ) -> tuple[Tensor, Tensor, Tensor]:
        observation_tensor = self._tensor(observations, self.observation_size, "observations")
        action_tensor = self._tensor(actions, self.action_size, "actions")
        if not torch.isfinite(action_tensor).all() or torch.any(torch.abs(action_tensor) > 1.0 + 1.0e-6):
            raise ValueError("PPO actions must be finite and inside [-1, 1]")
        mean, value = self(observation_tensor)
        distribution = self._distribution(mean)
        bounded = action_tensor.clamp(-1.0 + _LOG_PROB_EPSILON, 1.0 - _LOG_PROB_EPSILON)
        latent = torch.atanh(bounded)
        log_probability = (distribution.log_prob(latent) - torch.log(1.0 - bounded.square() + _LOG_PROB_EPSILON)).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_probability, entropy, value

    def act(self, observations: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        observation_tensor = self._tensor(observations, self.observation_size, "observations")
        with torch.no_grad():
            mean, _ = self(observation_tensor)
            latent = mean if deterministic else self._distribution(mean).sample()
            actions = torch.tanh(latent)
        return actions.cpu().numpy().astype(np.float64, copy=False)

    def value(self, observations: np.ndarray | Tensor) -> np.ndarray:
        observation_tensor = self._tensor(observations, self.observation_size, "observations")
        with torch.no_grad():
            _, values = self(observation_tensor)
        return values.cpu().numpy().astype(np.float64, copy=False)

    def as_policy(self, *, deterministic: bool = False):
        return lambda observations: self.act(observations, deterministic=deterministic)

    @property
    def parameter_count(self) -> int:
        return sum(int(value.numel()) for value in self.parameters())

    def artifact_metadata(self, hyperparameters: PPOHyperparameters | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": "racing_event_actor_critic",
            "format_version": 1,
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "distribution": "tanh_normal",
            "actor_parameters": sum(int(value.numel()) for value in self.actor.parameters()),
            "parameters": self.parameter_count,
        }
        if hyperparameters is not None:
            payload["ppo"] = hyperparameters.to_dict()
        return payload

    def save_artifact(
        self,
        weights_path: str | Path,
        metadata_path: str | Path | None = None,
        *,
        hyperparameters: PPOHyperparameters | None = None,
    ) -> tuple[Path, Path]:
        if not hasattr(self, "actor"):
            raise TypeError("generic artifact saving requires a participant-owned actor")
        weights = Path(weights_path)
        metadata = Path(metadata_path) if metadata_path is not None else weights.with_suffix(".json")
        weights.parent.mkdir(parents=True, exist_ok=True)
        state = {name: value.detach().cpu() for name, value in self.state_dict().items()}
        torch.save(state, weights)
        metadata.write_text(json.dumps(self.artifact_metadata(hyperparameters), sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return weights, metadata


def default_critic(observation_size: int) -> nn.Module:
    return nn.Sequential(nn.Linear(observation_size, 64), nn.Tanh(), nn.Linear(64, 1))


def build_actor_critic(
    actor: nn.Module,
    observation_size: int,
    *,
    critic: nn.Module | None = None,
    log_std_init: float = -0.5,
) -> ActorCritic:
    """Wrap a participant actor in the platform-owned PPO contract."""
    return ActorCritic(actor, observation_size, critic=critic, log_std_init=log_std_init)


class FixedMLPActorCritic(ActorCritic):
    """The fixed event policy: 23-ish inputs, two tanh layers, two actions."""

    def __init__(
        self,
        observation_size: int,
        *,
        action_size: int = PPO_ACTION_SIZE,
        hidden_sizes: tuple[int, ...] = PPO_HIDDEN_SIZES,
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        if isinstance(observation_size, bool) or not isinstance(observation_size, Integral) or observation_size <= 0:
            raise ValueError("observation_size must be a positive integer")
        if action_size != PPO_ACTION_SIZE:
            raise ValueError("the fixed event policy requires exactly two actions")
        if tuple(hidden_sizes) != PPO_HIDDEN_SIZES:
            raise ValueError(f"the fixed event policy requires hidden_sizes={PPO_HIDDEN_SIZES}")
        if not np.isfinite(float(log_std_init)):
            raise ValueError("log_std_init must be finite")
        self.observation_size = int(observation_size)
        self.action_size = int(action_size)
        self.hidden_sizes = PPO_HIDDEN_SIZES
        self.log_std_init = float(log_std_init)
        self.trunk = nn.Sequential(
            nn.Linear(self.observation_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.policy_mean = nn.Linear(64, self.action_size)
        self.value_head = nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.full((self.action_size,), self.log_std_init))
        with torch.no_grad():
            self.policy_mean.bias.copy_(torch.as_tensor(PPO_POLICY_MEAN_BIAS))

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        if observations.ndim != 2 or observations.shape[1] != self.observation_size:
            raise ValueError(
                f"observations must have shape (batch, {self.observation_size})"
            )
        features = self.trunk(observations.float())
        return self.policy_mean(features), self.value_head(features).squeeze(-1)

    def _distribution(self, mean: Tensor) -> Normal:
        # The clamp bounds keep an outlying optimizer update from creating a
        # non-finite Normal while retaining the learned parameter in the state.
        log_std = self.log_std.clamp(-5.0, 2.0).expand_as(mean)
        return Normal(mean, log_std.exp())

    def evaluate_actions(
        self,
        observations: Tensor | np.ndarray,
        actions: Tensor | np.ndarray,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return corrected log-probability, base entropy, and value."""

        observation_tensor = self._tensor(observations, self.observation_size, "observations")
        action_tensor = self._tensor(actions, self.action_size, "actions")
        if not torch.isfinite(action_tensor).all() or torch.any(torch.abs(action_tensor) > 1.0 + 1.0e-6):
            raise ValueError("PPO actions must be finite and inside [-1, 1]")
        mean, value = self(observation_tensor)
        distribution = self._distribution(mean)
        bounded = action_tensor.clamp(-1.0 + _LOG_PROB_EPSILON, 1.0 - _LOG_PROB_EPSILON)
        latent = torch.atanh(bounded)
        log_probability = (
            distribution.log_prob(latent)
            - torch.log(1.0 - bounded.square() + _LOG_PROB_EPSILON)
        ).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_probability, entropy, value

    def _tensor(self, value: Tensor | np.ndarray, width: int, name: str) -> Tensor:
        tensor = torch.as_tensor(value, device=self.log_std.device)
        if tensor.ndim != 2 or tensor.shape[1] != width:
            raise ValueError(f"{name} must have shape (batch, {width})")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values")
        return tensor.float()

    def act(self, observations: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        """Return a NumPy action batch for ``RolloutCollector``."""

        observation_tensor = self._tensor(observations, self.observation_size, "observations")
        with torch.no_grad():
            mean, _ = self(observation_tensor)
            distribution = self._distribution(mean)
            latent = mean if deterministic else distribution.sample()
            actions = torch.tanh(latent)
        return actions.cpu().numpy().astype(np.float64, copy=False)

    def value(self, observations: np.ndarray | Tensor) -> np.ndarray:
        observation_tensor = self._tensor(observations, self.observation_size, "observations")
        with torch.no_grad():
            _, values = self(observation_tensor)
        return values.cpu().numpy().astype(np.float64, copy=False)

    def as_policy(self, *, deterministic: bool = False):
        """Return the callable shape expected by :class:`RolloutCollector`."""

        return lambda observations: self.act(observations, deterministic=deterministic)

    def artifact_metadata(self, hyperparameters: PPOHyperparameters | None = None) -> dict[str, Any]:
        """Return JSON-safe metadata for this exact model architecture."""

        payload: dict[str, Any] = {
            "format": PPO_FORMAT,
            "format_version": PPO_FORMAT_VERSION,
            "observation_schema_version": "1.0",
            "observation_size": self.observation_size,
            "action_schema_version": "1.0",
            "action_size": self.action_size,
            "hidden_sizes": list(self.hidden_sizes),
            "activation": "tanh",
            "distribution": "tanh_normal",
            "policy_mean_bias": list(PPO_POLICY_MEAN_BIAS),
            "log_std_shape": list(self.log_std.shape),
            "log_std_init": self.log_std_init,
        }
        if hyperparameters is not None:
            payload["ppo"] = hyperparameters.to_dict()
        return payload

    def save_artifact(
        self,
        weights_path: str | Path,
        metadata_path: str | Path | None = None,
        *,
        hyperparameters: PPOHyperparameters | None = None,
    ) -> tuple[Path, Path]:
        """Save a state dictionary and its separately validated metadata."""

        weights = Path(weights_path)
        metadata = Path(metadata_path) if metadata_path is not None else weights.with_suffix(".json")
        state = {name: value.detach().cpu() for name, value in self.state_dict().items()}
        weights.parent.mkdir(parents=True, exist_ok=True)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        weight_fd, weight_name = tempfile.mkstemp(prefix=f".{weights.name}.", suffix=".tmp", dir=weights.parent)
        metadata_fd, metadata_name = tempfile.mkstemp(prefix=f".{metadata.name}.", suffix=".tmp", dir=metadata.parent)
        weight_temp = Path(weight_name)
        metadata_temp = Path(metadata_name)
        try:
            with os.fdopen(weight_fd, "wb") as stream:
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            with os.fdopen(metadata_fd, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(self.artifact_metadata(hyperparameters), sort_keys=True, indent=2) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(weight_temp, weights)
            os.replace(metadata_temp, metadata)
        finally:
            weight_temp.unlink(missing_ok=True)
            metadata_temp.unlink(missing_ok=True)
        return weights, metadata

    @classmethod
    def load_artifact(
        cls,
        weights_path: str | Path,
        metadata_path: str | Path | None = None,
        *,
        expected_observation_size: int | None = None,
    ) -> tuple["FixedMLPActorCritic", dict[str, Any]]:
        """Load a fixed model without unpickling executable user objects."""

        weights = Path(weights_path)
        metadata = Path(metadata_path) if metadata_path is not None else weights.with_suffix(".json")
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read model metadata: {metadata}") from exc
        cls._validate_metadata(payload, expected_observation_size=expected_observation_size)
        model = cls(int(payload["observation_size"]))
        try:
            state = torch.load(weights, map_location="cpu", weights_only=True)
        except Exception as exc:  # PyTorch uses several exception types here.
            raise ValueError(f"cannot load safe model weights: {weights}") from exc
        if not isinstance(state, Mapping):
            raise ValueError("model weights must be a state-dictionary mapping")
        expected_keys = set(model.state_dict())
        if set(state) != expected_keys:
            raise ValueError("model state dictionary has missing or extra keys")
        for name, value in state.items():
            if not isinstance(value, Tensor) or not torch.isfinite(value).all():
                raise ValueError(f"model state entry {name!r} is not a finite tensor")
            if value.shape != model.state_dict()[name].shape:
                raise ValueError(f"model state entry {name!r} has the wrong shape")
        try:
            model.load_state_dict(state, strict=True)
        except (RuntimeError, TypeError) as exc:
            raise ValueError("model state dictionary does not match the fixed architecture") from exc
        return model, payload

    @staticmethod
    def _validate_metadata(payload: Any, *, expected_observation_size: int | None) -> None:
        if not isinstance(payload, dict):
            raise ValueError("model metadata must be an object")
        required = {
            "format",
            "format_version",
            "observation_schema_version",
            "observation_size",
            "action_schema_version",
            "action_size",
            "hidden_sizes",
            "activation",
            "distribution",
            "policy_mean_bias",
            "log_std_shape",
            "log_std_init",
        }
        if not required.issubset(payload):
            raise ValueError("model metadata is missing required fields")
        if payload["format"] != PPO_FORMAT or payload["format_version"] != PPO_FORMAT_VERSION:
            raise ValueError("unsupported model artifact format")
        if payload["observation_schema_version"] != "1.0" or payload["action_schema_version"] != "1.0":
            raise ValueError("unsupported model schema version")
        observation_size = payload["observation_size"]
        if isinstance(observation_size, bool) or not isinstance(observation_size, Integral) or observation_size <= 0:
            raise ValueError("model observation_size is invalid")
        if expected_observation_size is not None and observation_size != expected_observation_size:
            raise ValueError("model observation_size does not match the expected environment")
        if payload["action_size"] != PPO_ACTION_SIZE:
            raise ValueError("unsupported model action_size")
        if payload["hidden_sizes"] != list(PPO_HIDDEN_SIZES):
            raise ValueError("unsupported model hidden_sizes")
        if payload["activation"] != "tanh" or payload["distribution"] != "tanh_normal":
            raise ValueError("unsupported model activation or distribution")
        if payload["policy_mean_bias"] != list(PPO_POLICY_MEAN_BIAS):
            raise ValueError("unsupported model policy mean bias")
        if payload["log_std_shape"] != [PPO_ACTION_SIZE] or not np.isfinite(float(payload["log_std_init"])):
            raise ValueError("invalid model log_std metadata")


@dataclass(frozen=True)
class PPOUpdateMetrics:
    """Diagnostics for one rollout followed by the configured PPO epochs."""

    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float | None
    transitions: int
    optimizer_steps: int
    update_epochs: int
    wall_time_s: float
    update_transitions_per_second: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "approx_kl": self.approx_kl,
            "clip_fraction": self.clip_fraction,
            "explained_variance": self.explained_variance,
            "transitions": self.transitions,
            "optimizer_steps": self.optimizer_steps,
            "update_epochs": self.update_epochs,
            "wall_time_s": self.wall_time_s,
            "update_transitions_per_second": self.update_transitions_per_second,
        }


class PPOTrainer:
    """Full-batch clipped PPO updates for the fixed actor-critic model."""

    def __init__(
        self,
        policy: ActorCritic,
        hyperparameters: PPOHyperparameters | None = None,
    ) -> None:
        if not isinstance(policy, ActorCritic):
            raise TypeError("policy must be an ActorCritic")
        self.policy = policy
        self.hyperparameters = hyperparameters or PPOHyperparameters()
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.hyperparameters.learning_rate,
            eps=1.0e-5,
        )

    def update(self, batch: RolloutBatch) -> PPOUpdateMetrics:
        """Run one rollout update and return JSON-safe PPO diagnostics."""

        started = time.perf_counter()
        observations, next_observations, actions, active = self._validate_batch(batch)
        time_steps, n_envs, observation_size = observations.shape
        if observation_size != self.policy.observation_size:
            raise ValueError("rollout observations do not match the policy")
        transitions = int(np.count_nonzero(active))
        if transitions == 0:
            raise ValueError("rollout batch contains no active transitions")

        flat_observations = observations.reshape(time_steps * n_envs, observation_size)
        flat_next_observations = next_observations.reshape(time_steps * n_envs, observation_size)
        active_flat = active.reshape(-1)
        flat_actions = actions.reshape(time_steps * n_envs, PPO_ACTION_SIZE)
        if np.any(np.abs(flat_actions[active_flat]) > 1.0 + 1.0e-6):
            raise ValueError("PPO rollout actions must be inside [-1, 1]")

        with torch.no_grad():
            observation_tensor = self.policy._tensor(
                flat_observations, observation_size, "observations"
            )
            next_observation_tensor = self.policy._tensor(
                flat_next_observations, observation_size, "next_observations"
            )
            _, old_value_tensor = self.policy(observation_tensor)
            _, next_value_tensor = self.policy(next_observation_tensor)
            old_log_probability_tensor, _, _ = self.policy.evaluate_actions(
                flat_observations[active_flat], flat_actions[active_flat]
            )

        old_values = old_value_tensor.cpu().numpy().reshape(time_steps, n_envs)
        next_values = next_value_tensor.cpu().numpy().reshape(time_steps, n_envs)
        old_log_probabilities = old_log_probability_tensor.cpu().numpy()
        advantages, returns = compute_masked_gae(
            batch.rewards,
            old_values,
            next_values,
            batch.terminated,
            batch.truncated,
            batch.active_mask,
            gamma=self.hyperparameters.gamma,
            gae_lambda=self.hyperparameters.gae_lambda,
        )
        active_advantages = advantages.reshape(-1)[active_flat]
        active_returns = returns.reshape(-1)[active_flat]
        advantage_mean = float(np.mean(active_advantages))
        advantage_std = float(np.std(active_advantages))
        if advantage_std > 1.0e-8:
            active_advantages = (active_advantages - advantage_mean) / advantage_std
        else:
            active_advantages = active_advantages - advantage_mean

        active_observations = flat_observations[active_flat]
        active_actions = flat_actions[active_flat]
        observation_tensor = self.policy._tensor(
            active_observations, observation_size, "observations"
        )
        action_tensor = self.policy._tensor(active_actions, PPO_ACTION_SIZE, "actions")
        device = observation_tensor.device
        old_log_probability_tensor = torch.as_tensor(old_log_probabilities, dtype=torch.float32, device=device)
        advantage_tensor = torch.as_tensor(active_advantages, dtype=torch.float32, device=device)
        return_tensor = torch.as_tensor(active_returns, dtype=torch.float32, device=device)

        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        optimizer_steps = 0
        minibatch_size = min(self.hyperparameters.minibatch_size, transitions)
        for _ in range(self.hyperparameters.update_epochs):
            permutation = torch.randperm(transitions, device=device)
            for start in range(0, transitions, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                new_log_probability, entropy, value = self.policy.evaluate_actions(
                    observation_tensor[indices], action_tensor[indices]
                )
                log_ratio = new_log_probability - old_log_probability_tensor[indices]
                ratio = torch.exp(log_ratio)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.hyperparameters.clip_range,
                    1.0 + self.hyperparameters.clip_range,
                )
                surrogate = torch.minimum(ratio * advantage_tensor[indices], clipped_ratio * advantage_tensor[indices])
                policy_loss = -surrogate.mean()
                unweighted_value_loss = torch.square(value - return_tensor[indices]).mean()
                value_loss = self.hyperparameters.value_loss_coefficient * unweighted_value_loss
                entropy_mean = entropy.mean()
                loss = policy_loss + value_loss - self.hyperparameters.entropy_coefficient * entropy_mean
                if not torch.isfinite(loss):
                    raise FloatingPointError("PPO loss became non-finite")

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.hyperparameters.max_grad_norm)
                if any(
                    parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    for parameter in self.policy.parameters()
                ):
                    raise FloatingPointError("PPO gradient became non-finite")
                self.optimizer.step()
                optimizer_steps += 1
                last_policy_loss = float(policy_loss.detach().cpu())
                last_value_loss = float(value_loss.detach().cpu())
                last_entropy = float(entropy_mean.detach().cpu())

        with torch.no_grad():
            final_log_probability, _, final_value = self.policy.evaluate_actions(
                observation_tensor, action_tensor
            )
            final_log_ratio = final_log_probability - old_log_probability_tensor
            final_ratio = torch.exp(final_log_ratio)
        if not torch.isfinite(final_ratio).all() or not torch.isfinite(final_log_ratio).all():
            raise FloatingPointError("PPO diagnostics became non-finite")
        ratio_array = final_ratio.cpu().numpy()
        log_ratio_array = final_log_ratio.cpu().numpy()
        value_array = final_value.cpu().numpy()
        return_variance = float(np.var(active_returns))
        explained_variance = (
            None
            if return_variance <= 1.0e-12
            else float(1.0 - np.var(active_returns - value_array) / return_variance)
        )
        wall_time_s = time.perf_counter() - started
        update_rate = float(transitions * optimizer_steps / wall_time_s) if wall_time_s > 0.0 else None
        return PPOUpdateMetrics(
            policy_loss=last_policy_loss,
            value_loss=last_value_loss,
            entropy=last_entropy,
            approx_kl=float(np.mean((ratio_array - 1.0) - log_ratio_array)),
            clip_fraction=float(
                np.mean(np.abs(ratio_array - 1.0) > self.hyperparameters.clip_range)
            ),
            explained_variance=explained_variance,
            transitions=transitions,
            optimizer_steps=optimizer_steps,
            update_epochs=self.hyperparameters.update_epochs,
            wall_time_s=float(wall_time_s),
            update_transitions_per_second=update_rate,
        )

    @staticmethod
    def _validate_batch(
        batch: RolloutBatch,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(batch, RolloutBatch):
            raise TypeError("batch must be a RolloutBatch")
        observations = np.asarray(batch.observations, dtype=np.float32)
        next_observations = np.asarray(batch.next_observations, dtype=np.float32)
        actions = np.asarray(batch.actions, dtype=np.float64)
        if observations.ndim != 3:
            raise ValueError("batch observations must have shape (time, n_envs, observation_size)")
        expected = observations.shape
        if next_observations.shape != expected:
            raise ValueError("batch next_observations must match observations")
        if actions.shape != (expected[0], expected[1], PPO_ACTION_SIZE):
            raise ValueError("batch actions must have shape (time, n_envs, 2)")
        shape = expected[:2]
        terminated = _as_bool_array(batch.terminated, shape, "batch terminated")
        truncated = _as_bool_array(batch.truncated, shape, "batch truncated")
        active = _as_bool_array(batch.active_mask, shape, "batch active_mask")
        if np.any(active & terminated & truncated):
            raise ValueError("an active rollout transition cannot be both terminated and truncated")
        for name, array in (
            ("batch observations", observations),
            ("batch next_observations", next_observations),
            ("batch actions", actions),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values")
        # GAE performs the complete reward/flag shape validation as well.  Do
        # the cheap reward check here so malformed input fails before autograd.
        _as_float_array(batch.rewards, shape, "batch rewards")
        return observations, next_observations, actions, active


__all__ = [
    "ActorCritic",
    "FixedMLPActorCritic",
    "PPOTrainer",
    "PPOUpdateMetrics",
    "build_actor_critic",
    "compute_masked_gae",
    "default_critic",
]
