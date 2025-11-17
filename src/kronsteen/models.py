"""Dataclasses shared across Kronsteen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Region:
    """Rectangle describing an area on screen."""

    left: int
    top: int
    width: int
    height: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.width, self.height)

    def to_box(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.left + self.width, self.top + self.height)

    def center(self) -> tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)

    def area(self) -> int:
        return self.width * self.height

    def expand(self, padding: int) -> "Region":
        return Region(
            left=self.left - padding,
            top=self.top - padding,
            width=self.width + padding * 2,
            height=self.height + padding * 2,
        )

    @classmethod
    def from_corners(cls, top_left: tuple[int, int], bottom_right: tuple[int, int]) -> "Region":
        left, top = top_left
        right, bottom = bottom_right
        return cls(left=left, top=top, width=right - left, height=bottom - top)

    @classmethod
    def from_sequence(cls, values: Sequence[int]) -> "Region":
        if len(values) != 4:
            raise ValueError("Region sequence must contain four integers")
        return cls(left=int(values[0]), top=int(values[1]), width=int(values[2]), height=int(values[3]))


@dataclass(frozen=True)
class TextMatch:
    text: str
    confidence: float
    region: Region


@dataclass(frozen=True)
class ImageMatch:
    region: Region
    confidence: float
    template_name: str | None = None


@dataclass(frozen=True)
class ColorMatch:
    color_hex: str
    region: Region

