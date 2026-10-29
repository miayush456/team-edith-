from __future__ import annotations

from argparse import Namespace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from racing_event import reference_checkpoint as checkpoint
from racing_event.budget import RunBudget
from tools import run_reference_baseline as runner


ROOT = Path(__file__).resolve().parents[1]


def arguments(output: Path, **overrides) -> Namespace:
    return Namespace(**{
        "reward": runner.DEFAULT_REWARD, "track_dir": runner.DEFAULT_TRACK_DIR,
        "tracks": [ROOT / "tracks/train/monaco.json"],
        "config": runner.DEFAULT_CONFIG, "output_dir": output, "budget_steps": 24,
        "steps": 6, "n_envs": 2, "rollout_length": 3, "seed": 17, **overrides,
    })


def current_directory(output: Path) -> Path:
    return checkpoint.load_current(output)[0]


class ReferenceCheckpointTests(unittest.TestCase):
    def assert_tree_equal(self, left, right, path="root") -> None:
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right), path)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys(), path)
            for key in left:
                self.assert_tree_equal(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right), path)
            for index, (a, b) in enumerate(zip(left, right)):
                self.assert_tree_equal(a, b, f"{path}[{index}]")
        else:
            self.assertEqual(left, right, path)

    def assert_same_boundary(self, left: Path, right: Path) -> None:
        left, right = current_directory(left), current_directory(right)
        for name in ("policy.pt", "local_state.pt"):
            self.assert_tree_equal(torch.load(left / name, weights_only=True), torch.load(right / name, weights_only=True), name)
        for name in ("collector.json", "training_contract.json", "training_state.json"):
            self.assertEqual(checkpoint.read_json(left / name), checkpoint.read_json(right / name), name)
        a = [json.loads(row) for row in (left / "training_metrics.jsonl").read_text().splitlines()]
        b = [json.loads(row) for row in (right / "training_metrics.jsonl").read_text().splitlines()]
        for rows in (a, b):
            for row in rows:
                row.pop("wall_time_s")
                row.pop("update_transitions_per_second")
        self.assertEqual(a, b)

    def cli(self, args: Namespace) -> subprocess.CompletedProcess:
        command = [sys.executable, str(ROOT / "tools/run_reference_baseline.py")]
        for name in ("output_dir", "config", "steps", "budget_steps", "n_envs", "rollout_length", "seed"):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        if args.tracks is not None:
            command.extend(["--tracks", *(str(path) for path in args.tracks)])
        # Direct CLI is intentionally exercised outside the project/PYTHONPATH.
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        return subprocess.run(command, cwd=args.output_dir.parent, env=environment, capture_output=True, text=True, timeout=40)

    def test_fresh_vs_split_process_resume_exact_with_random_lane_resets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = json.loads(runner.DEFAULT_CONFIG.read_text())
            config["episode"]["max_steps"] = 2
            config_path = root / "simulation.json"
            config_path.write_text(json.dumps(config))
            fresh, split = root / "fresh", root / "split"
            # Partial vector masks and done lanes at a boundary are included.
            for output, steps in ((fresh, 17), (split, 6), (split, 11)):
                result = self.cli(arguments(output, steps=steps, config=config_path, n_envs=3, rollout_length=2, tracks=None))
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assert_same_boundary(fresh, split)
            snapshot = checkpoint.read_json(current_directory(split) / "collector.json")
            env_arrays = checkpoint._unpack(snapshot)["arrays"]
            self.assertTrue(np.any(env_arrays["_done"]))
            self.assertTrue(np.all(env_arrays["steps"] <= 2))
            self.assertEqual(checkpoint.read_json(current_directory(split) / "training_state.json")["updates"], 3)

    def test_budget_extension_resumes_same_checkpoint_and_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "extended"
            args = arguments(output, budget_steps=12, steps=12)
            first = runner.run(args)
            self.assertEqual(first["budget_used"], 12)
            RunBudget.open(output / "budget.json").extend_run(first["run_id"], 12)
            continued = runner.run(arguments(output, budget_steps=24, steps=12))
            self.assertEqual(continued["budget_used"], 24)
            self.assertEqual(continued["updates"], 4)
            self.assertEqual(continued["run_id"], first["run_id"])

    def test_live_budget_extension_adds_steps_to_the_current_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "live-extended"
            args = arguments(output, budget_steps=12, steps=12)
            collect = runner.RolloutCollector.collect
            extended = False

            def collect_then_extend(collector, steps):
                nonlocal extended
                batch = collect(collector, steps)
                if not extended:
                    state = RunBudget.open(output / "budget.json").status()
                    RunBudget.open(output / "budget.json").extend_run(state["runs"][0]["run_id"], 12)
                    extended = True
                return batch

            with patch.object(runner.RolloutCollector, "collect", collect_then_extend):
                state = runner.run(args)
            self.assertEqual(state["budget_used"], 24)
            self.assertEqual(state["budget_steps"], 24)

    def test_interruption_before_pointer_keeps_previous_generation_and_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            args = arguments(output)
            old_pointer = None
            old_weights = None

            def interrupt(stage, generation):
                nonlocal old_pointer, old_weights
                updates = checkpoint.read_json(output / checkpoint.GENERATIONS / generation / "training_state.json")["updates"]
                if stage == "before_pointer" and updates == 1:
                    old_pointer = (output / checkpoint.POINTER).read_bytes()
                    old_weights = (output / "policy.pt").read_bytes()
                    raise OSError("before publication")

            with patch.object(checkpoint, "publication_hook", interrupt):
                with self.assertRaisesRegex(OSError, "before publication"):
                    runner.run(args)
            self.assertEqual((output / checkpoint.POINTER).read_bytes(), old_pointer)
            self.assertEqual((output / "policy.pt").read_bytes(), old_weights)
            self.assertEqual(checkpoint.read_json(current_directory(output) / "training_state.json")["updates"], 0)
            budget_bytes = (output / "budget.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "spent steps retained"):
                runner.run(args)
            self.assertEqual((output / "budget.json").read_bytes(), budget_bytes)
            self.assertEqual(RunBudget.open(output / "budget.json").status()["runs"][0]["used_steps"], 6)

    def test_interruption_after_pointer_repairs_mirrors_and_resumes_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, fresh = root / "split", root / "fresh"
            runner.run(arguments(fresh, steps=12))

            def interrupt(stage, generation):
                if stage == "after_pointer" and checkpoint.read_json(output / checkpoint.GENERATIONS / generation / "training_state.json")["updates"] == 1:
                    raise OSError("after publication")

            with patch.object(checkpoint, "publication_hook", interrupt):
                with self.assertRaisesRegex(OSError, "after publication"):
                    runner.run(arguments(output))
            committed = current_directory(output)
            # Compatibility view can lag after a crash, but never changes the
            # generation chosen by exact resume.
            self.assertNotEqual((output / "policy.pt").resolve().parent, committed)
            committed_hashes = {name: checkpoint.sha256(committed / name) for name in checkpoint.FILES}
            runner.run(arguments(output))
            self.assert_same_boundary(fresh, output)
            self.assertEqual((output / "policy.pt").resolve().parent, current_directory(output))
            self.assertEqual(committed_hashes, {name: checkpoint.sha256(committed / name) for name in checkpoint.FILES})

    def test_failed_optimizer_update_never_refunds_spent_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            original = runner.PPOTrainer.update

            def failed_update(trainer, batch):
                original(trainer, batch)
                raise RuntimeError("optimizer interrupted after mutation")

            with patch.object(runner.PPOTrainer, "update", failed_update):
                with self.assertRaisesRegex(RuntimeError, "after mutation"):
                    runner.run(arguments(output))
            self.assertEqual(RunBudget.open(output / "budget.json").status()["runs"][0]["used_steps"], 6)
            self.assertEqual(checkpoint.read_json(current_directory(output) / "training_state.json")["updates"], 0)
            with self.assertRaisesRegex(ValueError, "exact recovery refused"):
                runner.run(arguments(output))

    def test_same_output_writer_rejected_in_another_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with checkpoint.writer_lock(output):
                result = self.cli(arguments(output))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("another reference writer", result.stderr)
                self.assertFalse((output / "budget.json").exists())
            runner.run(arguments(output))

    def test_corrupt_generation_and_invalid_pointer_fail_closed_without_fallback(self) -> None:
        for mode in ("missing", "corrupt", "pointer", "manifest"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "run"
                runner.run(arguments(output))
                directory = current_directory(output)
                budget_bytes = (output / "budget.json").read_bytes()
                if mode == "missing":
                    (directory / "local_state.pt").unlink()
                elif mode == "corrupt":
                    (directory / "collector.json").write_text("{}")
                elif mode == "manifest":
                    (directory / "manifest.json").write_text("{}")
                else:
                    checkpoint.atomic_json(output / checkpoint.POINTER, {"version": 1, "generation": "../escape", "manifest_sha256": "0" * 64})
                with self.assertRaisesRegex(ValueError, "generation|manifest"):
                    runner.run(arguments(output))
                self.assertEqual((output / "budget.json").read_bytes(), budget_bytes)

    def test_seed_or_hyperparameter_mismatch_does_not_charge_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            runner.run(arguments(output))
            before = (output / "budget.json").read_bytes()
            for overrides in ({"seed": 18}, {"update_epochs": 1}, {"n_envs": 1}):
                with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, "identity"):
                    runner.run(arguments(output, **overrides))
            self.assertEqual((output / "budget.json").read_bytes(), before)

    def test_mixed_optimizer_generation_is_rejected_even_with_valid_file_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            runner.run(arguments(output))
            current = current_directory(output)
            initial = next(path for path in (output / checkpoint.GENERATIONS).iterdir()
                           if checkpoint.read_json(path / "training_state.json")["updates"] == 0)
            # Simulate a misassembled checkpoint, not merely a truncated file.
            (current / "local_state.pt").write_bytes((initial / "local_state.pt").read_bytes())
            manifest = checkpoint.read_json(current / "manifest.json")
            manifest["files"]["local_state.pt"] = checkpoint.sha256(current / "local_state.pt")
            checkpoint.atomic_json(current / "manifest.json", manifest)
            pointer = checkpoint.read_json(output / checkpoint.POINTER)
            pointer["manifest_sha256"] = checkpoint.sha256(current / "manifest.json")
            checkpoint.atomic_json(output / checkpoint.POINTER, pointer)
            budget_bytes = (output / "budget.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "optimizer step count"):
                runner.run(arguments(output))
            self.assertEqual((output / "budget.json").read_bytes(), budget_bytes)

    def test_local_torch_restore_rejects_pickle_objects_with_weights_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "local_state.pt"
            torch.save({"optimizer": Path("not-a-state"), "torch_rng": torch.get_rng_state()}, path)
            trainer = runner.PPOTrainer(runner.FixedMLPActorCritic(23))
            with self.assertRaisesRegex(ValueError, "safe local Torch"):
                checkpoint.restore_local_state(path, trainer)

    def test_legacy_root_files_are_never_guessed_as_exact_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "policy.pt").write_bytes(b"legacy")
            with self.assertRaisesRegex(ValueError, "legacy/non-exact.*--warm-start"):
                runner.run(arguments(output))
            self.assertEqual((output / "policy.pt").read_bytes(), b"legacy")
            self.assertFalse((output / "budget.json").exists())


if __name__ == "__main__":
    unittest.main()
