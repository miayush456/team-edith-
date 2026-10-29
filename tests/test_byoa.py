from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from racing_event.budget import RunBudget
from racing_event.budgeted_env import BudgetedEnv
from racing_event.client_budget import release_training_allocation, reserve_training_allocation
from racing_event.participant_model import build_policy_from_model
from racing_event.observation import TrainingContract
from racing_event.rollout import RolloutCollector
from racing_event.reward import RewardConfig, SIGNAL_NAMES
from racing_event.vector_env import VectorizedRacingEnv
from tests.test_model_artifact import CONFIG, TRACK
from tools.run_reference_baseline import run
from tools.evaluate_both import evaluate_both
from tools.export_replay import export as export_replay
import synapse

ROOT = Path(__file__).resolve().parents[1]


class BringYourOwnArchitectureTests(unittest.TestCase):
    def write_model(self, root: Path) -> Path:
        path = root / "model.py"
        path.write_text(
            "import torch\n"
            "from torch import nn\n"
            "def build_actor(obs_size):\n"
            "    return nn.Sequential(nn.Linear(obs_size, 7), nn.Tanh(), nn.Linear(7, 2))\n",
            encoding="utf-8",
        )
        return path

    def test_custom_actor_builds_platform_critic_and_trains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy, info = build_policy_from_model(self.write_model(Path(directory)), 23)
            self.assertEqual(policy.actor[-1].out_features, 2)
            self.assertGreater(info["parameters"], info["actor_parameters"])
            self.assertEqual(policy.act(np.zeros((2, 23), dtype=np.float32)).shape, (2, 2))

    def test_custom_actor_uses_reference_reward_weights_and_can_be_evaluated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = dict(RewardConfig().weights)
            weights["progress"] = 4.0
            reward_path = root / "reward.json"
            reward = RewardConfig(weights=weights)
            reward.save(reward_path)
            output = root / "run"
            run(Namespace(
                reward=reward_path,
                track_dir=ROOT / "tracks/train",
                tracks=None,
                config=ROOT / "config/simulation.json",
                output_dir=output,
                budget_steps=2,
                steps=2,
                n_envs=1,
                rollout_length=2,
                seed=17,
                model=self.write_model(root),
                hparams=None,
                init_from=None,
                warm_start=None,
                fine_tune=False,
                max_updates=None,
                learning_rate=1.0e-4,
                update_epochs=1,
                eval_every=1,
                eval_max_steps=10,
                eval_track_dir=ROOT / "tracks/train",
            ))
            contract, _ = TrainingContract.load(output / "training_contract.json")
            self.assertEqual(contract.reward_config_hash, reward.content_hash)
            report = evaluate_both(output, ROOT / "config/simulation.json", ROOT / "tracks/train", 1)
            self.assertEqual(len(report["tracks"]), 2)
            replay_path = root / "custom-actor-replay.json"
            export_replay(Namespace(
                config=ROOT / "config/simulation.json", track=ROOT / "tracks/train/monaco.json",
                weights=output / "policy.pt", metadata=output / "policy.json",
                contract=output / "training_contract.json", artifact_dir=None,
                output=replay_path, participant_id="custom-actor", steps=2,
            ))
            self.assertTrue(replay_path.is_file())
            row = json.loads((output / "training_metrics.jsonl").read_text().splitlines()[0])
            self.assertEqual(set(row["weighted_signal_sums"]), set(SIGNAL_NAMES))

    def test_budgeted_env_and_collector_charge_each_lane_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            budget = RunBudget.create(Path(directory) / "budget.json", max_runs=1, max_steps_per_run=8)
            run_id = budget.start_run()
            raw = VectorizedRacingEnv(TRACK, CONFIG, reward=None, n_envs=2)
            env = BudgetedEnv(raw, budget, run_id)
            collector = RolloutCollector(
                env, lambda observations: np.zeros((2, 2), dtype=np.float64), budget, run_id, seed=17,
            )
            result = collector.collect(3)
            self.assertEqual(result.batch.transitions, 3)
            self.assertEqual(budget.status(run_id)["used_steps"], 3)

    def test_sdk_save_accepts_the_output_directory_created_by_make_env(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            args = Namespace(
                config=ROOT / "config/simulation.json", track_dir=ROOT / "tracks/train",
                output_dir=output, steps=2, budget_steps=2, n_envs=1,
                rollout_length=2, seed=17, reward=None, hparams=None,
            )
            synapse.make_env(args)
            policy, _ = build_policy_from_model(self.write_model(root), 23)

            def fake_export(_policy, _contract, _normalizer, destination, **_kwargs):
                destination.mkdir(parents=True)
                return destination

            with patch("tools.export_onnx.export_torch_policy", side_effect=fake_export):
                submission = synapse.save(policy, args.output_dir, args)
            self.assertEqual(submission, output / "submission")
            self.assertTrue((output / "budget.json").is_file())
            self.assertTrue((output / "training_report.md").is_file())

    def test_failed_allocation_can_be_released_before_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "client.json"
            output = Path(directory) / "run"
            reserve_training_allocation(path, output, 100, max_retrains=2, max_steps_per_retrain=100)
            self.assertTrue(release_training_allocation(path, output, max_retrains=2, max_steps_per_retrain=100))
            status = reserve_training_allocation(path, output, 100, max_retrains=2, max_steps_per_retrain=100)
            self.assertFalse(status.reused)


if __name__ == "__main__":
    unittest.main()
