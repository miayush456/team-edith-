"""Python-authoritative SYNAPSE racing simulation."""

import importlib.util

from .config import SimulationConfig
from .budgeted_env import BudgetedEnv
from .env import RacingEnv
from .evaluation import EvaluationResult, evaluate_policy, validate_evaluation_contract
from .evaluation import evaluate_participant_artifact
from .model_artifact import ArtifactLimits, ParticipantModelArtifact, load_participant_artifact, save_layers
from .live_race import LiveRaceOptions, LiveRaceReport, run_live_onnx_race
from .live_protocol import LiveStatePublisher
from .observation import (
    ObservationSchema,
    RunningObservationNormalizer,
    TrainingContract,
    build_observation_schema,
)
from .reward import RewardConfig, RewardFunction, evaluate_reward_function
from .race import RaceEntrant, RaceReport, run_onnx_race
from .rollout import PPOHyperparameters, RolloutBatch, RolloutCollector, RolloutResult
from .track import TrackGeometry
from .vector_env import RacingVectorEnv, VectorizedRacingEnv

_PPO_EXPORTS = {"ActorCritic", "FixedMLPActorCritic", "PPOTrainer", "PPOUpdateMetrics", "build_actor_critic", "compute_masked_gae"}

__all__ = [
    "ObservationSchema",
    "RacingEnv",
    "BudgetedEnv",
    "RacingVectorEnv",
    "RewardConfig",
    "RewardFunction",
    "evaluate_reward_function",
    "ArtifactLimits",
    "ParticipantModelArtifact",
    "load_participant_artifact",
    "save_layers",
    "PPOHyperparameters",
    "RolloutBatch",
    "RolloutCollector",
    "RolloutResult",
    "RunningObservationNormalizer",
    "SimulationConfig",
    "TrackGeometry",
    "TrainingContract",
    "VectorizedRacingEnv",
    "build_observation_schema",
    "EvaluationResult",
    "evaluate_policy",
    "validate_evaluation_contract",
    "evaluate_participant_artifact",
    "RaceEntrant",
    "RaceReport",
    "run_onnx_race",
    "LiveRaceOptions",
    "LiveRaceReport",
    "LiveStatePublisher",
    "run_live_onnx_race",
]

if importlib.util.find_spec("torch") is not None:
    __all__.extend(sorted(_PPO_EXPORTS))


def __getattr__(name: str):
    """Load the optional PyTorch reference trainer only when requested."""

    if name in _PPO_EXPORTS:
        from . import ppo

        return getattr(ppo, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
