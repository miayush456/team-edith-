from __future__ import annotations

import unittest

from racing_event.telemetry import sector_bit, sector_coverage_count, sector_index


class TelemetryTests(unittest.TestCase):
    def test_sector_index_is_spawn_relative_and_wrap_aware(self) -> None:
        self.assertEqual(sector_index(0.0, 100.0, 4), 0)
        self.assertEqual(sector_index(24.99, 100.0, 4), 0)
        self.assertEqual(sector_index(25.0, 100.0, 4), 1)
        self.assertEqual(sector_index(99.99, 100.0, 4), 3)
        self.assertEqual(sector_index(100.0, 100.0, 4), 0)
        self.assertEqual(sector_index(-1.0, 100.0, 4), 3)

    def test_sector_masks_are_bounded_and_countable(self) -> None:
        mask = sector_bit(0, 4) | sector_bit(3, 4)
        self.assertEqual(mask, 0b1001)
        self.assertEqual(sector_coverage_count(mask), 2)

        with self.assertRaises(ValueError):
            sector_bit(4, 4)
        with self.assertRaises(ValueError):
            sector_index(0.0, 100.0, 65)


if __name__ == "__main__":
    unittest.main()
