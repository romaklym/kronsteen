"""Utility helpers."""

from __future__ import annotations

import base64
import io
import time
from typing import Callable, Generator, Iterable, Iterator, Optional

from PIL import Image

from .exceptions import MatchNotFoundError


def encode_image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def normalize_hex_color(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized.startswith("#"):
        normalized = f"#{normalized}"
    if len(normalized) not in {4, 7}:
        raise ValueError("Hex color must be #RGB or #RRGGBB")
    if len(normalized) == 4:
        normalized = "#" + "".join(ch * 2 for ch in normalized[1:])
    int(normalized[1:], 16)  # validate
    return normalized


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    normalized = normalize_hex_color(value)
    return tuple(int(normalized[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]


def rgb_to_hex(rgb: Iterable[int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def retry_loop(
    timeout: float,
    interval: float,
    task: Callable[[], Optional[object]],
    on_retry: Optional[Callable[[float], None]] = None,
) -> object:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while True:
        try:
            result = task()
            if result is not None:
                return result
        except Exception as exc:  # pragma: no cover - general retry support
            last_error = exc
        if time.monotonic() >= deadline:
            if last_error:
                raise last_error
            raise MatchNotFoundError("Timed out while waiting for condition")
        if on_retry:
            on_retry(max(0.0, deadline - time.monotonic()))
        time.sleep(interval)

