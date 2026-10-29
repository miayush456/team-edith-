from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from racing_event.config import SimulationConfig
from racing_event.observation import RunningObservationNormalizer, TrainingContract, build_observation_schema


ROOT = Path(__file__).resolve().parents[1]
CONFIG = SimulationConfig.from_file(ROOT / "config" / "simulation.json")


class ObservationContractTests(unittest.TestCase):
    def test_schema_has_fixed_names_bounds_and_dtype(self) -> None:
        schema = build_observation_schema(CONFIG)
        self.assertEqual(schema.version, "1.0")
        self.assertEqual(schema.size, 23)
        self.assertEqual(schema.names[:4], ("speed", "longitudinal_velocity", "lateral_velocity", "yaw_rate"))
        self.assertEqual(schema.names[-1], "local_width")
        self.assertTrue(np.all(schema.lows <= schema.highs))
        self.assertEqual(schema.ray_angles_deg, CONFIG.sensors.ray_angles_deg)

    def test_ray_layout_changes_size_and_invalid_layouts_are_rejected(self) -> None:
        from dataclasses import replace
        from racing_event.config import SensorConfig

        smaller = replace(CONFIG, sensors=SensorConfig(ray_angles_deg=(0.0, -30.0, 30.0)))
        self.assertEqual(build_observation_schema(smaller).size, 17)
        for angles in ((-1.0, -1.0, 0.0), (1.0,), (0.0, 91.0), (float("nan"), 0.0)):
            with self.subTest(angles=angles):
                with self.assertRaises(ValueError):
                    SensorConfig(ray_angles_deg=angles)

    def test_normalizer_and_contract_round_trip(self) -> None:
        schema = build_observation_schema(CONFIG)
        normalizer = RunningObservationNormalizer(schema.size)
        values = np.linspace(-0.5, 0.5, schema.size, dtype=np.float32)
        normalizer.update(np.stack([values, values * 0.5]))
        normalizer.freeze()
        contract = TrainingContract.for_config(CONFIG, track_ids=("oval",), track_hashes=("a" * 64,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            contract.save(path, normalizer=normalizer)
            loaded_contract, loaded_normalizer = TrainingContract.load(path)
        self.assertEqual(loaded_contract.to_dict()["action"]["shape"], [2])
        self.assertEqual(loaded_contract.observation.names, schema.names)
        self.assertIsNotNone(loaded_normalizer)
        assert loaded_normalizer is not None
        np.testing.assert_array_equal(normalizer.transform(values, training=False), loaded_normalizer.transform(values, training=False))

    def test_training_contract_carries_reward_provenance(self) -> None:
        contract = TrainingContract(
            observation=build_observation_schema(CONFIG),
            track_ids=("oval",),
            track_hashes=("a" * 64,),
            config_hash=TrainingContract.config_hash_for(CONFIG),
            reward_config_hash="b" * 64,
            reward_sha256="c" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            contract.save(path)
            loaded, normalizer = TrainingContract.load(path)
        self.assertIsNone(normalizer)
        self.assertEqual(loaded.reward_config_hash, "b" * 64)
        self.assertEqual(loaded.reward_sha256, "c" * 64)
        self.assertEqual(
            loaded.to_dict()["training_provenance"],
            {"reward_config_hash": "b" * 64, "reward_sha256": "c" * 64},
        )


if __name__ == "__main__":
    unittest.main()
