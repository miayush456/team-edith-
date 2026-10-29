from __future__ import annotations

import math
import tempfile
from pathlib import Path
import unittest

from racing_event.reward import RewardConfig


class RewardConfigTests(unittest.TestCase):
    def test_missing_signals_default_to_zero(self) -> None:
        config = RewardConfig(weights={"progress": 2})
        self.assertEqual(config.weights["progress"], 2.0)
        self.assertEqual(config.weights["speed"], 0.0)

    def test_non_finite_weights_are_rejected(self) -> None:
        for value in (math.nan, math.inf, -math.inf, "not-a-number"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    RewardConfig(weights={"progress": value})

    def test_weight_bounds_and_canonical_identity(self) -> None:
        with self.assertRaises(ValueError):
            RewardConfig(weights={"progress": 100.01})
        with self.assertRaises(ValueError):
            RewardConfig(weights={"progress": True})
        first = RewardConfig(weights={"progress": 2})
        second = RewardConfig(weights={"speed": 0, "progress": 2})
        self.assertEqual(first.to_dict()["schema_version"], 1)
        self.assertEqual(first.content_hash, second.content_hash)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reward.json"
            first.save(path)
            self.assertEqual(RewardConfig.from_file(path).content_hash, first.content_hash)


if __name__ == "__main__":
    unittest.main()
