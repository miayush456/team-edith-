from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tools.build_training_manifest import build_manifest
from racing_event.budget import RunBudget
from racing_event.observation import TrainingContract
from tools.run_reference_baseline import DEFAULT_CONFIG, DEFAULT_REWARD, DEFAULT_TRACK_DIR, run
from tools.train_participant import build_training_command


ROOT = Path(__file__).resolve().parents[1]


class ParticipantTrainingTests(unittest.TestCase):
    def test_fresh_reference_runs_seed_initialization_before_model_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policies = []
            contracts = []
            for index in range(2):
                # Simulate arbitrary earlier Torch activity in the UI process.
                torch.manual_seed(100 + index)
                output = Path(directory) / str(index)
                run(Namespace(
                    reward=DEFAULT_REWARD, track_dir=DEFAULT_TRACK_DIR, tracks=None,
                    config=DEFAULT_CONFIG, output_dir=output, budget_steps=2,
                    steps=2, n_envs=1, rollout_length=2, seed=17,
                ))
                policies.append(torch.load(output / "policy.pt", weights_only=True))
                contracts.append((output / "training_contract.json").read_bytes())
            self.assertEqual(contracts[0], contracts[1])
            self.assertEqual(policies[0].keys(), policies[1].keys())
            for name in policies[0]:
                self.assertTrue(torch.equal(policies[0][name], policies[1][name]), name)

    def test_public_training_manifest_has_only_valid_train_maps(self) -> None:
        manifest = build_manifest(DEFAULT_TRACK_DIR, "test-release")
        self.assertEqual(manifest["release_id"], "test-release")
        self.assertFalse(manifest["private_tracks_included"])
        self.assertEqual([item["split"] for item in manifest["maps"]], ["train"] * 2)
        self.assertEqual(
            {item["map_id"] for item in manifest["maps"]},
            {"monaco", "spa"},
        )
        self.assertTrue(all(len(item["canonical_content_hash"]) == 64 for item in manifest["maps"]))
        self.assertTrue(all(len(item["delivered_file_sha256"]) == 64 for item in manifest["maps"]))

    def test_reference_baseline_resumes_persistent_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            args = Namespace(
                reward=DEFAULT_REWARD,
                track_dir=DEFAULT_TRACK_DIR,
                tracks=None,
                config=DEFAULT_CONFIG,
                output_dir=output_dir,
                budget_steps=16,
                steps=8,
                n_envs=1,
                rollout_length=8,
                seed=17,
            )
            first = run(args)
            second = run(args)
            self.assertEqual(first["budget_used"], 8)
            self.assertEqual(second["budget_used"], 16)
            self.assertTrue((output_dir / "policy.pt").exists())
            self.assertTrue((output_dir / "training_contract.json").exists())
            self.assertEqual(len((output_dir / "training_metrics.jsonl").read_text().splitlines()), 2)
            contract, _ = TrainingContract.load(output_dir / "training_contract.json")
            self.assertEqual(contract.reward_config_hash, "59dfca8e79e6b30d89dad5f3cc5a3f9883c97c649724b0dc5c3ac2bab28aa604")
            self.assertEqual(len(contract.reward_sha256 or ""), 64)

    def test_participant_launcher_canonicalizes_to_reference_trainer(self) -> None:
        args = Namespace(
            trainer=ROOT / "examples" / "participant_train.py",
            reward=DEFAULT_REWARD,
            track_dir=DEFAULT_TRACK_DIR,
            config=DEFAULT_CONFIG,
            output_dir=ROOT / "participant_run",
            steps=8,
            budget_steps=16,
            n_envs=1,
            rollout_length=8,
            seed=17,
        )
        command = build_training_command(args)
        self.assertTrue(command[1].endswith("tools/run_reference_baseline.py"))
        self.assertIn("--steps", command)
        with self.assertRaises(ValueError):
            args.trainer = ROOT.parent / "outside.py"
            build_training_command(args)

        args.trainer = ROOT / "participant_code" / "model.py"
        with self.assertRaisesRegex(ValueError, "always uses tools/run_reference_baseline.py"):
            build_training_command(args)

    def test_participant_launcher_uses_the_baseline_reward_by_default(self) -> None:
        args = Namespace(
            trainer=ROOT / "examples" / "participant_train.py",
            reward=None,
            track_dir=DEFAULT_TRACK_DIR,
            config=DEFAULT_CONFIG,
            output_dir=ROOT / "participant_run",
            steps=1,
            budget_steps=1,
            n_envs=1,
            rollout_length=1,
            seed=17,
        )
        command = build_training_command(args)
        self.assertIn("--reward", command)
        self.assertTrue(command[command.index("--reward") + 1].endswith("examples/baseline_reward.json"))
        args.config = ROOT / "artifacts" / "missing-config.json"
        with self.assertRaisesRegex(ValueError, "config does not exist"):
            build_training_command(args)

    def test_participant_launcher_rejects_a_single_public_map(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            track_dir = Path(directory)
            (track_dir / "monaco.json").write_bytes((ROOT / "tracks/train/monaco.json").read_bytes())
            args = Namespace(
                trainer=ROOT / "examples" / "participant_train.py",
                reward=DEFAULT_REWARD,
                track_dir=track_dir,
                config=DEFAULT_CONFIG,
                output_dir=ROOT / "participant_run",
                steps=1,
                budget_steps=1,
                n_envs=1,
                rollout_length=1,
                seed=17,
            )
            with self.assertRaisesRegex(ValueError, "all public training maps"):
                build_training_command(args)

    def test_reference_checkpoint_refuses_inexact_recovery_after_policy_publish_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "recover"
            args = Namespace(
                reward=DEFAULT_REWARD,
                track_dir=DEFAULT_TRACK_DIR,
                tracks=None,
                config=DEFAULT_CONFIG,
                output_dir=output_dir,
                budget_steps=8,
                steps=4,
                n_envs=1,
                rollout_length=4,
                seed=17,
            )
            from tools import run_reference_baseline

            original = run_reference_baseline.FixedMLPActorCritic.save_artifact
            calls = 0

            def flaky_save(policy, *save_args, **save_kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected checkpoint interruption")
                return original(policy, *save_args, **save_kwargs)

            with patch.object(run_reference_baseline.FixedMLPActorCritic, "save_artifact", flaky_save):
                with self.assertRaisesRegex(OSError, "injected"):
                    run_reference_baseline.run(args)
            self.assertTrue((output_dir / "training_state.json").exists())
            budget_bytes = (output_dir / "budget.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "spent steps retained, exact recovery refused"):
                run_reference_baseline.run(args)
            self.assertEqual((output_dir / "budget.json").read_bytes(), budget_bytes)
            self.assertEqual(RunBudget.open(output_dir / "budget.json").status()["runs"][0]["used_steps"], 4)
            self.assertEqual(len((output_dir / "training_metrics.jsonl").read_text().splitlines()), 0)


if __name__ == "__main__":
    unittest.main()
