"""Policy-invisible progress zones for checkpoint rewards and coverage metrics."""

from __future__ import annotations

import math
from numbers import Integral


DEFAULT_SECTOR_COUNT = 16
MAX_SECTOR_COUNT = 64


def _validate_sector_count(sector_count: int) -> int:
    if isinstance(sector_count, bool) or not isinstance(sector_count, Integral):
        raise ValueError("sector_count must be an integer from 1 through 64")
    value = int(sector_count)
    if value < 1 or value > MAX_SECTOR_COUNT:
        raise ValueError("sector_count must be an integer from 1 through 64")
    return value


def sector_index(
    distance_since_spawn: float,
    track_length: float,
    sector_count: int = DEFAULT_SECTOR_COUNT,
) -> int:
    """Return a wrap-aware sector relative to the current episode spawn."""

    count = _validate_sector_count(sector_count)
    try:
        distance = float(distance_since_spawn)
        length = float(track_length)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("distance_since_spawn and track_length must be finite") from exc
    if not math.isfinite(distance) or not math.isfinite(length) or length <= 0.0:
        raise ValueError("distance_since_spawn must be finite and track_length must be positive")
    wrapped_distance = distance % length
    return min(count - 1, int(math.floor((wrapped_distance / length) * count)))


def sector_bit(index: int, sector_count: int = DEFAULT_SECTOR_COUNT) -> int:
    """Return the bit for a valid sector index."""

    count = _validate_sector_count(sector_count)
    if isinstance(index, bool) or not isinstance(index, Integral):
        raise ValueError("sector index must be an integer")
    value = int(index)
    if value < 0 or value >= count:
        raise ValueError(f"sector index must be between 0 and {count - 1}")
    return 1 << value


def sector_coverage_count(mask: int) -> int:
    """Return the number of sectors represented by a non-negative bit mask."""

    if isinstance(mask, bool) or not isinstance(mask, Integral) or int(mask) < 0:
        raise ValueError("sector coverage mask must be a non-negative integer")
    return int(mask).bit_count()
