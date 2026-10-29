"""Trusted-local loading of a participant actor factory for training/checking."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from torch import nn

from .ppo import ActorCritic, build_actor_critic


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_module(path: str | Path) -> ModuleType:
    source = Path(path).resolve()
    if not source.is_file() or source.suffix != ".py":
        raise ValueError(f"model file does not exist: {source}")
    module_name = f"synapse_participant_model_{sha256_file(source)[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not import model file: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_policy_from_model(path: str | Path, observation_size: int) -> tuple[ActorCritic, dict[str, Any]]:
    module = load_model_module(path)
    builder = getattr(module, "build_actor", None)
    if not callable(builder):
        raise ValueError("model file must define build_actor(obs_size)")
    actor = builder(int(observation_size))
    if not isinstance(actor, nn.Module):
        raise TypeError("build_actor(obs_size) must return torch.nn.Module")
    critic_builder = getattr(module, "build_critic", None)
    critic = critic_builder(int(observation_size)) if callable(critic_builder) else None
    if critic is not None and not isinstance(critic, nn.Module):
        raise TypeError("build_critic(obs_size) must return torch.nn.Module")
    policy = build_actor_critic(actor, observation_size, critic=critic)
    with torch.no_grad():
        probe = torch.zeros((2, int(observation_size)), dtype=torch.float32)
        mean, value = policy(probe)
    if mean.shape != (2, 2) or value.shape != (2,) or not torch.isfinite(mean).all() or not torch.isfinite(value).all():
        raise ValueError("participant actor/critic returned invalid probe shapes or values")
    return policy, {
        "model_sha256": sha256_file(path),
        "model_path": str(Path(path).resolve()),
        "actor_parameters": sum(int(value.numel()) for value in actor.parameters()),
        "critic_parameters": sum(int(value.numel()) for value in policy.critic.parameters()),
        "parameters": policy.parameter_count,
        "actor_class": f"{actor.__class__.__module__}.{actor.__class__.__qualname__}",
    }


def load_policy_state(policy: ActorCritic, weights_path: str | Path) -> None:
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"cannot load safe participant model weights: {weights_path}") from exc
    if not isinstance(state, dict):
        raise ValueError("participant model weights must be a state dictionary")
    try:
        policy.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("participant model weights do not match the supplied model.py architecture") from exc


__all__ = ["build_policy_from_model", "load_model_module", "load_policy_state", "sha256_file"]
