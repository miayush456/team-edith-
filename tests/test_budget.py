import json
import math
import multiprocessing
from pathlib import Path
import tempfile
import unittest

from racing_event.budget import (
    BudgetConfigurationError,
    BudgetCorruptionError,
    RunBudget,
    RunLimitExceeded,
    StepLimitExceeded,
)


def _concurrent_worker(path: str, result_queue: multiprocessing.Queue) -> None:
    try:
        budget = RunBudget.open(path)
        run_id = budget.start_run()
        remaining = budget.consume(run_id, 1)
        result_queue.put(("ok", run_id, remaining))
    except Exception as exc:  # Send the failure back to the parent test.
        result_queue.put(("error", type(exc).__name__, str(exc)))
        raise


class RunBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "budget.json"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_persistence_resume_and_run_limit(self) -> None:
        budget = RunBudget.create(self.path, max_runs=2, max_steps_per_run=5)
        first = budget.start_run()
        self.assertEqual(first, "run-000001")
        self.assertEqual(budget.consume(first, 2), 3)
        budget.close()

        resumed = RunBudget.open(self.path)
        self.assertEqual(resumed.remaining(first), 3)
        self.assertEqual(resumed.consume(first, 3), 0)
        self.assertEqual(resumed.status(first)["status"], "exhausted")

        second = resumed.start_run()
        self.assertEqual(second, "run-000002")
        self.assertEqual(resumed.remaining(second), 5)
        with self.assertRaises(RunLimitExceeded):
            resumed.start_run()

        reopened = RunBudget.open(self.path)
        self.assertEqual(reopened.status()["runs_started"], 2)
        self.assertEqual(reopened.status()["runs_remaining"], 0)

    def test_partial_batches_are_allowed_and_overconsume_is_atomic(self) -> None:
        budget = RunBudget.create(self.path, n=2, k=5)
        run_id = budget.start_run()
        self.assertEqual(budget.consume(run_id, 2), 3)
        self.assertEqual(budget.consume(run_id, 1), 2)

        before = self.path.read_bytes()
        with self.assertRaises(StepLimitExceeded):
            budget.consume(run_id, 3)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(budget.remaining(run_id), 2)
        self.assertEqual(budget.consume(run_id, 2), 0)

    def test_invalid_numbers_are_rejected_strictly(self) -> None:
        bad_values = [True, False, 1.0, 2.5, math.nan, math.inf, -1, 0]
        for bad in bad_values:
            with self.subTest(max_runs=bad):
                with self.assertRaises(ValueError):
                    RunBudget.create(self.path, max_runs=bad, max_steps_per_run=5)
            if self.path.exists():
                self.path.unlink()
            with self.subTest(max_steps_per_run=bad):
                with self.assertRaises(ValueError):
                    RunBudget.create(self.path, max_runs=2, max_steps_per_run=bad)
            if self.path.exists():
                self.path.unlink()

        budget = RunBudget.create(self.path, max_runs=2, max_steps_per_run=5)
        run_id = budget.start_run()
        for bad in (True, 1.0, math.nan):
            with self.subTest(steps=bad):
                with self.assertRaises(ValueError):
                    budget.consume(run_id, bad)
        self.assertEqual(budget.remaining(run_id), 5)

    def test_corrupt_json_and_hash_chain_are_rejected(self) -> None:
        budget = RunBudget.create(self.path, max_runs=2, max_steps_per_run=5)
        run_id = budget.start_run()
        budget.consume(run_id, 1)

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        payload["records"][1]["steps"] = 1
        # The changed record's hash is intentionally left untouched.
        payload["records"][2]["steps"] = 2
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(BudgetCorruptionError):
            RunBudget.open(self.path)

        self.path.write_text("{ definitely not json", encoding="utf-8")
        with self.assertRaises(BudgetCorruptionError):
            RunBudget.open(self.path)

    def test_config_mismatch_is_rejected(self) -> None:
        RunBudget.create(self.path, max_runs=2, max_steps_per_run=5)
        with self.assertRaises(BudgetConfigurationError):
            RunBudget.open(self.path, max_runs=3, max_steps_per_run=5)
        with self.assertRaises(BudgetConfigurationError):
            RunBudget.open(self.path, n=2, k=6)

    def test_stale_handles_reload_before_writing(self) -> None:
        first_handle = RunBudget.create(self.path, max_runs=2, max_steps_per_run=5)
        second_handle = RunBudget.open(self.path)
        run_id = first_handle.start_run()

        self.assertEqual(second_handle.consume(run_id, 2), 3)
        # This handle was opened before the other reservation, but it reloads
        # under the lock and cannot overwrite those two reserved steps.
        self.assertEqual(first_handle.consume(run_id, 3), 0)
        self.assertEqual(second_handle.remaining(run_id), 0)

    def test_extension_is_hashed_and_visible_to_an_already_open_handle(self) -> None:
        budget = RunBudget.create(self.path, max_runs=1, max_steps_per_run=5)
        run_id = budget.start_run()
        budget.consume(run_id, 5)
        stale_handle = RunBudget.open(self.path)
        RunBudget.open(self.path).extend_run(run_id, 7)
        self.assertEqual(stale_handle.remaining(run_id), 7)
        self.assertEqual(stale_handle.consume(run_id, 7), 0)
        reopened = RunBudget.open(self.path, max_runs=1, max_steps_per_run=12)
        self.assertEqual(reopened.status(run_id)["used_steps"], 12)

    def test_concurrent_processes_get_distinct_runs_without_lost_updates(self) -> None:
        RunBudget.create(self.path, max_runs=2, max_steps_per_run=5)
        context = multiprocessing.get_context("fork")
        result_queue = context.Queue()
        processes = [
            context.Process(target=_concurrent_worker, args=(str(self.path), result_queue))
            for _ in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=10)
            self.assertFalse(process.is_alive())
            self.assertEqual(process.exitcode, 0)

        results = [result_queue.get(timeout=2) for _ in processes]
        self.assertTrue(all(result[0] == "ok" for result in results), results)
        self.assertEqual({result[1] for result in results}, {"run-000001", "run-000002"})

        final = RunBudget.open(self.path)
        self.assertEqual(final.status()["runs_started"], 2)
        self.assertEqual(
            {run["used_steps"] for run in final.status()["runs"]}, {1}
        )


if __name__ == "__main__":
    unittest.main()
