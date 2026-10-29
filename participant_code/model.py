"""Starter participant actor for the pluggable SYNAPSE PPO path."""

import torch
from torch import nn


def build_actor(obs_size: int) -> nn.Module:
    """Return pre-tanh means with shape [batch, 2]."""
    actor = nn.Sequential(
        nn.Linear(obs_size, 128),
        nn.Tanh(),
        nn.Linear(128, 64),
        nn.Tanh(),
        nn.Linear(64, 2),
    )
    with torch.no_grad():
        actor[-1].bias.copy_(torch.tensor([0.0, 0.5]))
    return actor
