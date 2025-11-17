"""Region helpers."""

from __future__ import annotations

from typing import Iterable, Optional

from .exceptions import RegionError
from .models import Region


def normalize_region(region: Optional[Region | Iterable[int]]) -> Optional[Region]:
    """Convert tuples/lists/None into a Region instance."""

    if region is None:
        return None
    if isinstance(region, Region):
        return region
    try:
        values = list(int(x) for x in region)  # type: ignore[arg-type]
    except TypeError as exc:  # pragma: no cover - defensive branch
        raise RegionError("Region must be a Region or iterable of four integers") from exc
    if len(values) != 4:
        raise RegionError("Region iterable must have exactly four values")
    return Region.from_sequence(values)


def to_pyautogui_region(region: Optional[Region | Iterable[int]]) -> Optional[tuple[int, int, int, int]]:
    normalized = normalize_region(region)
    return normalized.as_tuple() if normalized else None

