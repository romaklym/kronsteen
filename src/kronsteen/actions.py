"""PyAutoGUI wrappers used by Kronsteen."""

from __future__ import annotations

from typing import Iterable, Optional

import pyautogui

from .models import Region
from .regions import to_pyautogui_region


def move_to(x: int, y: int, duration: float = 0.0) -> None:
    pyautogui.moveTo(x, y, duration=duration)


def click(
    x: Optional[int] = None,
    y: Optional[int] = None,
    *,
    clicks: int = 1,
    interval: float = 0.0,
    button: str = "left",
    duration: float = 0.0,
) -> None:
    pyautogui.click(x=x, y=y, clicks=clicks, interval=interval, button=button, duration=duration)


def double_click(x: Optional[int] = None, y: Optional[int] = None, *, button: str = "left") -> None:
    click(x=x, y=y, clicks=2, button=button, interval=0.1)


def right_click(x: Optional[int] = None, y: Optional[int] = None) -> None:
    click(x=x, y=y, button="right")


def click_and_drag(
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    duration: float = 0.3,
    button: str = "left",
) -> None:
    pyautogui.moveTo(*start)
    pyautogui.dragTo(*end, duration=duration, button=button)


def scroll(clicks: int) -> None:
    pyautogui.scroll(clicks)


def type_text(text: str, *, interval: float = 0.0, press_enter: bool = False) -> None:
    pyautogui.typewrite(text, interval=interval)
    if press_enter:
        pyautogui.press("enter")


def hotkey(*keys: str) -> None:
    pyautogui.hotkey(*keys)


def press(key: str) -> None:
    pyautogui.press(key)


def screenshot(region: Optional[Region | tuple[int, int, int, int]] = None) -> "Image.Image":
    from PIL import Image  # imported lazily to avoid heavy dependency during startup

    region_tuple = to_pyautogui_region(region)
    return pyautogui.screenshot(region=region_tuple)


def save_screenshot(path: str, region: Optional[Region | tuple[int, int, int, int]] = None) -> None:
    image = screenshot(region=region)
    image.save(path)


def get_screen_size() -> tuple[int, int]:
    return pyautogui.size()


def current_position() -> tuple[int, int]:
    return pyautogui.position()

