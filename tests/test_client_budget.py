from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from racing_event.client_budget import (
    ClientRetrainLimitExceeded,
    ClientTrainingBudgetError,
    DEFAULT_MAX_STEPS_PER_RETRAIN,
    client_budget_status,
    extend_training_allocation,
    reserve_training_allocation,
)


class ClientTrainingBudgetTests(unittest.TestCase):
    def test_default_cap_increases_and_existing_ledger_upgrades_in_place(self) -> None:
        self.assertEqual(DEFAULT_MAX_STEPS_PER_RETRAIN, 1_000_000)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            budget = root / "client.json"
            output = root / "old-run"
            reserve_training_allocation(budget, output, 10_000, max_retrains=2, max_steps_per_retrain=10_000)

            status = client_budget_status(budget, max_retrains=2, max_steps_per_retrain=1_000_000)
            self.assertEqual(status["max_steps_per_retrain"], 1_000_000)
            self.assertEqual(status["retraining_started"], 1)
            self.assertEqual(
                reserve_training_allocation(budget, output, 10_000, max_retrains=2, max_steps_per_retrain=1_000_000).reused,
                True,
            )
            reserve_training_allocation(budget, root / "new-run", 1_000_000, max_retrains=2, max_steps_per_retrain=1_000_000)

    def test_new_output_directories_consume_slots_and_resume_reuses_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            budget = root / "client.json"
            first = reserve_training_allocation(budget, root / "one", 10, max_retrains=2, max_steps_per_retrain=10)
            resumed = reserve_training_allocation(budget, root / "one", 10, max_retrains=2, max_steps_per_retrain=10)
            second = reserve_training_allocation(budget, root / "two", 10, max_retrains=2, max_steps_per_retrain=10)
            self.assertEqual((first.slot, first.remaining_retrains, first.reused), (1, 1, False))
            self.assertEqual((resumed.slot, resumed.remaining_retrains, resumed.reused), (1, 1, True))
            self.assertEqual((second.slot, second.remaining_retrains, second.reused), (2, 0, False))
            self.assertEqual(client_budget_status(budget, max_retrains=2, max_steps_per_retrain=10)["retraining_remaining"], 0)
            with self.assertRaises(ClientRetrainLimitExceeded):
                reserve_training_allocation(budget, root / "three", 10, max_retrains=2, max_steps_per_retrain=10)

    def test_per_retrain_step_cap_and_hash_corruption_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            budget = root / "client.json"
            with self.assertRaisesRegex(ClientTrainingBudgetError, "per retrain"):
                reserve_training_allocation(budget, root / "one", 11, max_retrains=2, max_steps_per_retrain=10)
            reserve_training_allocation(budget, root / "one", 10, max_retrains=2, max_steps_per_retrain=10)
            budget.write_text(budget.read_text(encoding="utf-8").replace('"budget_steps": 10', '"budget_steps": 9'), encoding="utf-8")
            with self.assertRaisesRegex(ClientTrainingBudgetError, "hash"):
                client_budget_status(budget, max_retrains=2, max_steps_per_retrain=10)

    def test_extension_spends_an_iteration_on_the_existing_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            budget = root / "client.json"
            output = root / "same-model"
            reserve_training_allocation(budget, output, 1_000_000, max_retrains=3)
            extended = extend_training_allocation(budget, output, max_retrains=3)
            self.assertEqual((extended.slot, extended.remaining_retrains), (2, 1))
            resumed = reserve_training_allocation(budget, output, 2_000_000, max_retrains=3)
            self.assertTrue(resumed.reused)
            self.assertEqual(resumed.remaining_retrains, 1)


if __name__ == "__main__":
    unittest.main()
