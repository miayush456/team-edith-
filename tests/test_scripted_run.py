from __future__ import annotations

from pathlib import Path
import unittest

from tools.run_scripted import run


ROOT = Path(__file__).resolve().parents[1]


class ScriptedRunTests(unittest.TestCase):
    def test_recorded_run_has_versioned_exchange_shape(self) -> None:
        result = run(
            ROOT / "tests" / "fixtures" / "tracks" / "synthetic_oval.json",
            ROOT / "config" / "simulation.json",
            steps=8,
            participant_id="test",
        )
        self.assertEqual(result["schema_version"], "1.0")
        self.assertEqual(result["track_id"], "synthetic_oval")
        self.assertEqual(result["tick_hz"], 30)
        self.assertEqual(len(result["frames"]), 8)
        self.assertEqual(set(result["frames"][0]), {"tick", "x", "y", "heading", "speed", "steer", "throttle_brake", "yaw_rate", "off_track", "events"})


if __name__ == "__main__":
    unittest.main()
