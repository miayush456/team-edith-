"""Minimal actor construction for the platform PPO trainer."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from participant_code.model import build_actor
from racing_event.ppo import build_actor_critic, PPOTrainer
from racing_event.rollout import PPOHyperparameters


def make_policy(observation_size: int):
    actor = build_actor(observation_size)
    policy = build_actor_critic(actor, observation_size)
    return policy, PPOTrainer(policy, PPOHyperparameters())
