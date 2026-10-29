from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from racing_event import reference_checkpoint as checkpoint
from racing_event.config import SimulationConfig
from racing_event.observation import RunningObservationNormalizer, TrainingContract
from racing_event.ppo import FixedMLPActorCritic
from racing_event.track import TrackGeometry
from racing_event.vector_env import VectorizedRacingEnv
from tools import run_reference_baseline as runner
from tests.test_reference_checkpoint import arguments, current_directory


ROOT = Path(__file__).resolve().parents[1]
OVAL = ROOT / "tracks/train/monaco.json"


class ReferenceFineTuneTests(unittest.TestCase):
    def make_source(self, root: Path, *, normalized: bool = False) -> tuple[Path, FixedMLPActorCritic]:
        source = root / "source"
        source.mkdir()
        config = SimulationConfig.from_file(runner.DEFAULT_CONFIG)
        track = TrackGeometry.from_file(OVAL)
        env = VectorizedRacingEnv(track, config, n_envs=1)
        torch.manual_seed(12)
        policy = FixedMLPActorCritic(env.observation_size)
        # This simple straight-line source is deliberately sufficient only for
        # a short public-track probe; no imitation training sweep is required.
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.zero_()
            policy.policy_mean.bias.copy_(torch.tensor([0.0, 4.0]))
        policy.save_artifact(source / "policy.pt")
        normalizer = None
        if normalized:
            normalizer = RunningObservationNormalizer(env.observation_size)
            observation, _ = env.reset(seed=7)
            normalizer.update(observation)
        env.training_contract.save(source / "training_contract.json", normalizer=normalizer)
        return source, policy

    def fine_args(self, root: Path, source: Path, **overrides):
        evaluation = root / "public-evaluation"
        evaluation.mkdir(exist_ok=True)
        link = evaluation / OVAL.name
        if not link.exists():
            link.symlink_to(OVAL)
        return arguments(root / "fine", warm_start=source, fine_tune=True, max_updates=1,
                         update_epochs=1, eval_every=1, eval_max_steps=2,
                         eval_track_dir=evaluation, **overrides)

    def test_imitation_warm_start_preserves_environment_only_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, original = self.make_source(root)
            before = {path.name: path.read_bytes() for path in source.iterdir()}
            args = self.fine_args(root, source)
            collect = runner.RolloutCollector.collect
            seen = []

            def observe(collector, steps):
                self.assertIsNone(collector.normalizer)
                rollout = collect(collector, steps)
                # With no extra normalizer, the policy sees the environment's
                # float32 observations, not newly fitted running statistics.
                self.assertTrue(np.all(np.abs(rollout.batch.observations) <= 1))
                seen.append(steps)
                return rollout

            with patch.object(runner.RolloutCollector, "collect", observe):
                state = runner.run(args)
            self.assertEqual(seen, [6])
            self.assertEqual(state["preprocessing"], "environment_only")
            _, normalizer = TrainingContract.load(args.output_dir / "training_contract.json")
            self.assertIsNone(normalizer)
            report = checkpoint.read_json(args.output_dir / "selection_report.json")
            initial = args.output_dir / checkpoint.GENERATIONS / report["history"][0]["generation"]
            loaded, _ = FixedMLPActorCritic.load_artifact(initial / "policy.pt")
            observations = np.zeros((2, original.observation_size), dtype=np.float32)
            np.testing.assert_array_equal(original.act(observations, deterministic=True), loaded.act(observations, deterministic=True))
            self.assertEqual(before, {path.name: path.read_bytes() for path in source.iterdir()})
            self.assertIn("not event scoring", report["rule"])
            self.assertNotIn("total_reward", report["history"][0]["report"]["tracks"][0])
            # Exact resume requires no continuing dependency on source files.
            args.warm_start = None
            resumed = runner.run(args)
            self.assertEqual(resumed["updates"], 2)
            self.assertIsNone(TrainingContract.load(args.output_dir / "training_contract.json")[1])

    def test_reference_warm_start_freezes_and_retains_source_normalizer_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = self.make_source(root, normalized=True)
            _, source_normalizer = TrainingContract.load(source / "training_contract.json")
            args = self.fine_args(root, source)
            state = runner.run(args)
            _, normalizer = TrainingContract.load(args.output_dir / "training_contract.json")
            expected = source_normalizer.state_dict()
            expected["training"] = False
            self.assertEqual(normalizer.state_dict(), expected)
            self.assertEqual(state["preprocessing"], "frozen_source_normalizer")

    def test_deliberately_worse_candidate_keeps_best_artifact_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = self.make_source(root)
            args = self.fine_args(root, source)
            args.eval_max_steps = 60
            original = runner.PPOTrainer.update
            retained_hashes = {}

            def worsen(trainer, batch):
                retained_hashes.update({name: checkpoint.sha256(args.output_dir / "best" / name)
                                        for name in ("policy.pt", "policy.json", "training_contract.json")})
                metrics = original(trainer, batch)
                with torch.no_grad():
                    for parameter in trainer.policy.parameters():
                        parameter.zero_()
                    trainer.policy.policy_mean.bias.copy_(torch.tensor([4.0, 4.0]))
                return metrics

            with patch.object(runner.PPOTrainer, "update", worsen):
                runner.run(args)
            report = checkpoint.read_json(args.output_dir / "selection_report.json")
            initial, worse = report["history"]
            self.assertTrue(initial["selected"])
            self.assertFalse(worse["selected"])
            self.assertGreater(worse["report"]["off_track_seconds"], initial["report"]["off_track_seconds"])
            self.assertEqual(report["best_generation"], initial["generation"])
            self.assertEqual(report["best_report"], initial["report"])
            best = (args.output_dir / "best").resolve()
            self.assertEqual(best.name, initial["generation"])
            for name in ("policy.pt", "policy.json", "training_contract.json"):
                self.assertEqual(checkpoint.sha256(best / name), retained_hashes[name])
            current = current_directory(args.output_dir)
            self.assertNotEqual((best / "policy.pt").read_bytes(), (current / "policy.pt").read_bytes())
            # The same fixed-start probe is deterministic; evaluation cannot
            # mutate the retained normalizer or any training RNG stream.
            policy, normalizer, _ = runner.load_warm_start(best, VectorizedRacingEnv(TrackGeometry.from_file(OVAL), SimulationConfig.from_file(args.config), n_envs=1))
            repeat = runner.evaluate_candidate(policy, normalizer, (TrackGeometry.from_file(OVAL),), SimulationConfig.from_file(args.config), 60)
            self.assertEqual(repeat, initial["report"])

    def test_selection_priorities_and_ties_exclude_training_reward(self) -> None:
        report = {"finish_count": 2, "off_track_seconds": 0.0, "finished_lap_seconds": 20.0}
        self.assertGreater(runner.selection_key(report), runner.selection_key({**report, "finish_count": 1, "finished_lap_seconds": 1.0}))
        self.assertGreater(runner.selection_key(report), runner.selection_key({**report, "off_track_seconds": 1.0, "finished_lap_seconds": 1.0}))
        self.assertGreater(runner.selection_key(report), runner.selection_key({**report, "finished_lap_seconds": 21.0}))
        previous = runner.select_candidate(None, report, "g-" + "a" * 32, 0)
        previous["best_manifest_sha256"] = "b" * 64
        tied = runner.select_candidate(previous, {**report, "training_reward": 1e30}, "g-" + "c" * 32, 1)
        self.assertEqual(tied["best_generation"], previous["best_generation"])
        self.assertFalse(tied["history"][-1]["selected"])

    def test_public_validation_dashboard_separates_latest_and_best(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = {
                "finish_count": 1, "off_track_seconds": 0.5, "finished_lap_seconds": 8.5,
                "tracks": [{"track_id": "monaco", "status": "finished", "lap_time_s": 8.5,
                            "mean_speed_mps": 21.0, "off_track_steps": 15}],
            }
            selection = runner.select_candidate(None, report, "g-" + "a" * 32, 3)
            worse = {**report, "finish_count": 0, "finished_lap_seconds": 0.0,
                     "tracks": [{**report["tracks"][0], "status": "dnf_crash", "lap_time_s": None}]}
            selection = runner.select_candidate(selection, worse, "g-" + "b" * 32, 5)
            view = runner._public_validation_payload(selection, 1 / 30)
            self.assertEqual(view["latest_update"], 5)
            self.assertEqual(view["latest"]["finish_count"], 0)
            self.assertEqual(view["best_update"], 3)
            self.assertEqual(view["best"]["tracks"][0]["lap_time_s"], 8.5)
            self.assertAlmostEqual(view["best"]["tracks"][0]["off_track_seconds"], 0.5)

    def test_finetune_work_caps_and_fresh_directory_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = self.make_source(root)
            args = self.fine_args(root, source)
            for name, value in (("max_updates", 33), ("steps", 65537), ("eval_max_steps", 20001), ("update_epochs", 17)):
                with self.subTest(name=name):
                    original = getattr(args, name)
                    setattr(args, name, value)
                    with self.assertRaises(ValueError):
                        runner.run(args)
                    setattr(args, name, original)
                    self.assertFalse((args.output_dir / "budget.json").exists())
            args.output_dir = source
            with self.assertRaisesRegex(ValueError, "new.*directory"):
                runner.run(args)


if __name__ == "__main__":
    unittest.main()
