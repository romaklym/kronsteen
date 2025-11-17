"""High-level find helpers (text, image, color)."""

from __future__ import annotations

import re
import time
from typing import Iterable, Literal, Optional

import numpy as np
import pyautogui

from .actions import click as click_action
from .actions import screenshot as take_screenshot
from .exceptions import MatchNotFoundError
from .models import ColorMatch, ImageMatch, Region, TextMatch
from .ocr import DeepSeekOCRClient
from .regions import normalize_region, to_pyautogui_region
from .utils import hex_to_rgb, retry_loop

MatchMode = Literal["contains", "equals", "starts-with", "regex"]


def _text_matches(
    query: str,
    candidate: str,
    *,
    match_mode: MatchMode,
    case_sensitive: bool,
) -> bool:
    left = candidate if case_sensitive else candidate.lower()
    right = query if case_sensitive else query.lower()
    if match_mode == "equals":
        return left == right
    if match_mode == "contains":
        return right in left
    if match_mode == "starts-with":
        return left.startswith(right)
    if match_mode == "regex":
        flags = 0 if case_sensitive else re.IGNORECASE
        return re.search(query, candidate, flags=flags) is not None
    raise ValueError(f"Unsupported match mode: {match_mode}")


def find_text(
    query: str,
    *,
    ocr_client: DeepSeekOCRClient,
    region: Optional[Region | Iterable[int]] = None,
    match_mode: MatchMode = "contains",
    case_sensitive: bool = False,
    timeout: float = 0.0,
    retry_interval: float = 0.5,
    min_confidence: float = 0.4,
) -> TextMatch:
    normalized_region = normalize_region(region)

    def task() -> Optional[TextMatch]:
        screenshot = take_screenshot(region=normalized_region)
        matches = ocr_client.extract_text(screenshot, region=normalized_region)
        for match in matches:
            if match.confidence < min_confidence:
                continue
            if _text_matches(query, match.text, match_mode=match_mode, case_sensitive=case_sensitive):
                return match
        return None

    if timeout > 0:
        return retry_loop(timeout, retry_interval, task)
    result = task()
    if not result:
        raise MatchNotFoundError(f"Text '{query}' not found on screen")
    return result


def find_all_text(
    query: Optional[str],
    *,
    ocr_client: DeepSeekOCRClient,
    region: Optional[Region | Iterable[int]] = None,
    match_mode: MatchMode = "contains",
    case_sensitive: bool = False,
    min_confidence: float = 0.4,
) -> list[TextMatch]:
    normalized_region = normalize_region(region)
    screenshot = take_screenshot(region=normalized_region)
    matches = ocr_client.extract_text(screenshot, region=normalized_region)
    filtered: list[TextMatch] = []
    for match in matches:
        if match.confidence < min_confidence:
            continue
        if query is None or _text_matches(query, match.text, match_mode=match_mode, case_sensitive=case_sensitive):
            filtered.append(match)
    return filtered


def click_on_text(
    query: str,
    *,
    ocr_client: DeepSeekOCRClient,
    region: Optional[Region | Iterable[int]] = None,
    timeout: float = 5.0,
    retry_interval: float = 0.5,
    match_mode: MatchMode = "contains",
    case_sensitive: bool = False,
    min_confidence: float = 0.4,
) -> TextMatch:
    match = find_text(
        query,
        ocr_client=ocr_client,
        region=region,
        timeout=timeout,
        retry_interval=retry_interval,
        match_mode=match_mode,
        case_sensitive=case_sensitive,
        min_confidence=min_confidence,
    )
    click_action(*match.region.center())
    return match


def find_text_and_click(*args, **kwargs) -> TextMatch:
    return click_on_text(*args, **kwargs)


def wait_for_text(
    query: str,
    *,
    ocr_client: DeepSeekOCRClient,
    timeout: float,
    retry_interval: float = 0.5,
    region: Optional[Region | Iterable[int]] = None,
    match_mode: MatchMode = "contains",
    case_sensitive: bool = False,
    min_confidence: float = 0.4,
) -> TextMatch:
    return find_text(
        query,
        ocr_client=ocr_client,
        timeout=timeout,
        retry_interval=retry_interval,
        region=region,
        match_mode=match_mode,
        case_sensitive=case_sensitive,
        min_confidence=min_confidence,
    )


def wait_for_text_to_disappear(
    query: str,
    *,
    ocr_client: DeepSeekOCRClient,
    timeout: float,
    retry_interval: float = 0.5,
    region: Optional[Region | Iterable[int]] = None,
    match_mode: MatchMode = "contains",
    case_sensitive: bool = False,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            find_text(
                query,
                ocr_client=ocr_client,
                region=region,
                timeout=0,
                match_mode=match_mode,
                case_sensitive=case_sensitive,
            )
        except MatchNotFoundError:
            return
        time.sleep(retry_interval)
    raise MatchNotFoundError(f"Text '{query}' still visible after {timeout} seconds")


def find_image(
    image_path: str,
    *,
    region: Optional[Region | Iterable[int]] = None,
    confidence: float = 0.9,
    grayscale: bool = False,
) -> ImageMatch:
    loc = pyautogui.locateOnScreen(
        image_path,
        region=to_pyautogui_region(region),
        confidence=confidence,
        grayscale=grayscale,
    )
    if not loc:
        raise MatchNotFoundError(f"Image '{image_path}' not found on screen")
    region_obj = Region(left=loc.left, top=loc.top, width=loc.width, height=loc.height)
    return ImageMatch(region=region_obj, confidence=confidence, template_name=image_path)


def wait_for_image(
    image_path: str,
    *,
    region: Optional[Region | Iterable[int]] = None,
    confidence: float = 0.9,
    grayscale: bool = False,
    timeout: float = 10.0,
    retry_interval: float = 0.5,
) -> ImageMatch:
    return retry_loop(
        timeout,
        retry_interval,
        lambda: find_image(
            image_path,
            region=region,
            confidence=confidence,
            grayscale=grayscale,
        ),
    )


def find_color(
    hex_color: str,
    *,
    region: Optional[Region | Iterable[int]] = None,
) -> ColorMatch:
    normalized_region = normalize_region(region)
    screenshot = take_screenshot(region=normalized_region)
    rgb = np.array(screenshot)
    target = hex_to_rgb(hex_color)
    matches = np.where(
        (rgb[:, :, 0] == target[0]) & (rgb[:, :, 1] == target[1]) & (rgb[:, :, 2] == target[2])
    )
    if matches[0].size == 0:
        raise MatchNotFoundError(f"Color {hex_color} not found")
    top, left = matches[0][0], matches[1][0]
    base_left = normalized_region.left if normalized_region else 0
    base_top = normalized_region.top if normalized_region else 0
    absolute_region = Region(left=base_left + int(left), top=base_top + int(top), width=1, height=1)
    return ColorMatch(color_hex=hex_color, region=absolute_region)


def find_template(
    template_path: str,
    *,
    confidence: float = 0.8,
    region: Optional[Region | Iterable[int]] = None,
    grayscale: bool = True,
) -> ImageMatch:
    """
    Find a template image on screen using template matching.
    
    Args:
        template_path: Path to the template image file
        confidence: Minimum confidence level (0.0-1.0, default: 0.8)
        region: Optional region to search in
        grayscale: Whether to use grayscale matching (default: True, faster)
    
    Returns:
        ImageMatch with the location of the template
    
    Raises:
        MatchNotFoundError: If template not found
    """
    # Use pyautogui's locateOnScreen which uses OpenCV template matching
    location = pyautogui.locateOnScreen(
        template_path,
        confidence=confidence,
        region=to_pyautogui_region(region),
        grayscale=grayscale
    )
    
    if location is None:
        raise MatchNotFoundError(f"Template image '{template_path}' not found on screen")
    
    # Convert to our Region format
    match_region = Region(
        left=location.left,
        top=location.top,
        width=location.width,
        height=location.height
    )
    
    return ImageMatch(image_path=template_path, region=match_region, confidence=confidence)


def wait_for_template(
    template_path: str,
    *,
    timeout: float = 10.0,
    confidence: float = 0.8,
    region: Optional[Region | Iterable[int]] = None,
    grayscale: bool = True,
    retry_interval: float = 0.5,
) -> ImageMatch:
    """
    Wait for a template image to appear on screen.
    
    Args:
        template_path: Path to the template image file
        timeout: Maximum seconds to wait (default: 10.0)
        confidence: Minimum confidence level (0.0-1.0, default: 0.8)
        region: Optional region to search in
        grayscale: Whether to use grayscale matching (default: True)
        retry_interval: Seconds between retries (default: 0.5)
    
    Returns:
        ImageMatch when found
    
    Raises:
        MatchNotFoundError: If template not found within timeout
    """
    return retry_loop(
        timeout,
        retry_interval,
        lambda: find_template(
            template_path,
            confidence=confidence,
            region=region,
            grayscale=grayscale,
        ),
    )


def click_on_template(
    template_path: str,
    *,
    timeout: float = 5.0,
    confidence: float = 0.8,
    region: Optional[Region | Iterable[int]] = None,
    grayscale: bool = True,
) -> ImageMatch:
    """
    Find a template image on screen and click on its center.
    
    Args:
        template_path: Path to the template image file
        timeout: Maximum seconds to wait (default: 5.0)
        confidence: Minimum confidence level (0.0-1.0, default: 0.8)
        region: Optional region to search in
        grayscale: Whether to use grayscale matching (default: True)
    
    Returns:
        ImageMatch that was clicked
    
    Raises:
        MatchNotFoundError: If template not found within timeout
    """
    match = wait_for_template(
        template_path,
        timeout=timeout,
        confidence=confidence,
        region=region,
        grayscale=grayscale,
    )
    
    # Click on the center of the matched region
    center_x, center_y = match.region.center()
    click_action(center_x, center_y)
    
    return match
