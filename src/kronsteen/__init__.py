"""Public Kronsteen API."""

from __future__ import annotations

from typing import Iterable, Optional

from ._version import __version__
from .client import Kronsteen
from .config import KronsteenSettings
from .launcher import find_application, launch_application
from .logging_config import get_logger, setup_logging
from .models import ColorMatch, ImageMatch, Region, TextMatch
from .ocr import DeepSeekOCRClient
from .ocr_tesseract import TesseractOCRClient
from .window_monitor import (
    WindowFocusMonitor,
    check_window_focus,
    get_active_window_title,
    get_window_monitor,
    is_window_active,
    start_window_monitoring,
    stop_window_monitoring,
)
from .vision import (
    VisionClient,
    Detection,
    Segmentation,
    Classification,
    detect_objects,
    segment_objects,
    classify_image,
)

__all__ = [
    "__version__",
    "Kronsteen",
    "Region",
    "TextMatch",
    "ImageMatch",
    "ColorMatch",
    "TesseractOCRClient",
    "DeepSeekOCRClient",
    "setup_logging",
    "get_logger",
    "configure",
    "set_default_client",
    "use_ocr_engine",
    "click",
    "double_click",
    "right_click",
    "click_on_text",
    "find_text",
    "find_all_text",
    "find_text_and_click",
    "wait_for_text",
    "wait_for_text_to_disappear",
    "find_image",
    "wait_for_image",
    "find_template",
    "wait_for_template",
    "click_on_template",
    "find_color",
    "find_shape",
    "screenshot",
    "save_screenshot",
    "type_text",
    "type",
    "hotkey",
    "press",
    "move_to",
    "click_and_drag",
    "scroll",
    "sleep",
    "countdown",
    "record_region_interactively",
    "launch",
    "close_app",
    "find_application",
    "launch_application",
    "WindowFocusMonitor",
    "start_window_monitoring",
    "stop_window_monitoring",
    "check_window_focus",
    "get_window_monitor",
    "is_window_active",
    "get_active_window_title",
    "VisionClient",
    "Detection",
    "Segmentation",
    "Classification",
    "detect_objects",
    "segment_objects",
    "classify_image",
]

_default_client = Kronsteen()


def _client() -> Kronsteen:
    return _default_client


def set_default_client(client: Kronsteen) -> None:
    global _default_client
    _default_client = client


def use_ocr_engine(engine: str = "tesseract") -> None:
    """
    Switch the OCR engine used for text recognition.
    
    Args:
        engine: OCR engine to use - "tesseract" (default) or "deepseek"
    
    Example:
        >>> kronsteen.use_ocr_engine("tesseract")  # Fast, lightweight (default)
        >>> kronsteen.use_ocr_engine("deepseek")   # More accurate, requires GPU
    """
    global _default_client
    if engine.lower() == "deepseek":
        _default_client = Kronsteen(ocr_engine="deepseek")
    else:
        _default_client = Kronsteen(ocr_engine="tesseract")


def configure(**kwargs) -> Kronsteen:
    _client().configure(**kwargs)
    return _client()


# Wrappers ----------------------------------------------------------------------------------------
def click(x: Optional[int] = None, y: Optional[int] = None, *, clicks: int = 1, interval: float = 0.0, button: str = "left", duration: float = 0.0) -> None:
    """
    Click at the specified coordinates.
    
    Args:
        x: X coordinate (None for current position)
        y: Y coordinate (None for current position)
        clicks: Number of clicks (default: 1)
        interval: Seconds between clicks (default: 0.0)
        button: Mouse button - "left", "right", or "middle" (default: "left")
        duration: Seconds to move to position (default: 0.0)
    
    Example:
        >>> kronsteen.click(100, 200)  # Click at (100, 200)
        >>> kronsteen.click(button="right")  # Right-click at current position
        >>> kronsteen.click(100, 200, clicks=2)  # Double-click
    """
    _client().click(x, y, clicks=clicks, interval=interval, button=button, duration=duration)


def double_click(x: Optional[int] = None, y: Optional[int] = None, *, button: str = "left") -> None:
    """
    Double-click at the specified coordinates.
    
    Args:
        x: X coordinate (None for current position)
        y: Y coordinate (None for current position)
        button: Mouse button - "left", "right", or "middle" (default: "left")
    
    Example:
        >>> kronsteen.double_click(100, 200)
        >>> kronsteen.double_click()  # Double-click at current position
    """
    _client().double_click(x, y, button=button)


def right_click(x: Optional[int] = None, y: Optional[int] = None) -> None:
    """
    Right-click at the specified coordinates.
    
    Args:
        x: X coordinate (None for current position)
        y: Y coordinate (None for current position)
    
    Example:
        >>> kronsteen.right_click(100, 200)
        >>> kronsteen.right_click()  # Right-click at current position
    """
    _client().right_click(x, y)


def click_and_drag(start: tuple[int, int], end: tuple[int, int], *, duration: float = 0.3, button: str = "left") -> None:
    """
    Click and drag from start to end coordinates.
    
    Args:
        start: Starting (x, y) coordinates
        end: Ending (x, y) coordinates
        duration: Seconds to take for the drag (default: 0.3)
        button: Mouse button - "left", "right", or "middle" (default: "left")
    
    Example:
        >>> kronsteen.click_and_drag((100, 100), (200, 200))  # Drag from (100,100) to (200,200)
        >>> kronsteen.click_and_drag((0, 0), (500, 500), duration=1.0)  # Slow drag
    """
    _client().click_and_drag(start, end, duration=duration, button=button)


def move_to(x: int, y: int, duration: float = 0.0) -> None:
    """
    Move the mouse cursor to the specified coordinates.
    
    Args:
        x: X coordinate
        y: Y coordinate
        duration: Seconds to take for the movement (default: 0.0 = instant)
    
    Example:
        >>> kronsteen.move_to(100, 200)  # Move instantly
        >>> kronsteen.move_to(100, 200, duration=1.0)  # Move over 1 second
    """
    _client().move_to(x, y, duration)


def type_text(text: str, *, interval: float = 0.0, press_enter: bool = False) -> None:
    """
    Type text using the keyboard.
    
    Args:
        text: Text to type
        interval: Seconds between keystrokes (default: 0.0)
        press_enter: Press Enter after typing (default: False)
    
    Example:
        >>> kronsteen.type_text("Hello World")
        >>> kronsteen.type_text("Search query", press_enter=True)
        >>> kronsteen.type_text("Slow typing", interval=0.1)
    """
    _client().type_text(text, interval=interval, press_enter=press_enter)


def type(*args, **kwargs) -> None:
    _client().type(*args, **kwargs)


def hotkey(*keys: str) -> None:
    """
    Press multiple keys simultaneously (keyboard shortcut).
    
    Args:
        *keys: Keys to press together
    
    Common shortcuts:
        - Copy: hotkey("command", "c") on macOS or hotkey("ctrl", "c") on Windows/Linux
        - Paste: hotkey("command", "v") or hotkey("ctrl", "v")
        - Select All: hotkey("command", "a") or hotkey("ctrl", "a")
        - Save: hotkey("command", "s") or hotkey("ctrl", "s")
    
    Example:
        >>> kronsteen.hotkey("command", "c")  # Copy on macOS
        >>> kronsteen.hotkey("ctrl", "c")  # Copy on Windows/Linux
        >>> kronsteen.hotkey("ctrl", "shift", "esc")  # Task Manager
    """
    _client().hotkey(*keys)


def press(key: str) -> None:
    """
    Press a single key.
    
    Args:
        key: Key to press (single letter, number, or special key name)
    
    Special keys:
        - "enter", "return" - Enter key
        - "tab" - Tab key
        - "esc", "escape" - Escape key
        - "space" - Space bar
        - "backspace" - Backspace
        - "delete" - Delete key
        - "up", "down", "left", "right" - Arrow keys
        - "home", "end" - Home/End keys
        - "pageup", "pagedown" - Page Up/Down
        - "f1" through "f12" - Function keys
        - "shift", "ctrl", "alt", "command" - Modifier keys
    
    Example:
        >>> kronsteen.press("enter")  # Press Enter
        >>> kronsteen.press("a")  # Press 'a' key
        >>> kronsteen.press("esc")  # Press Escape
        >>> kronsteen.press("tab")  # Press Tab
    """
    _client().press(key)


def scroll(clicks: int) -> None:
    """
    Scroll the mouse wheel.
    
    Args:
        clicks: Number of "clicks" to scroll (positive = up, negative = down)
    
    Example:
        >>> kronsteen.scroll(5)  # Scroll up 5 clicks
        >>> kronsteen.scroll(-3)  # Scroll down 3 clicks
    """
    _client().scroll(clicks)


def screenshot(region: Optional[Region | tuple[int, int, int, int]] = None):
    """
    Take a screenshot of the entire screen or a specific region.
    
    Args:
        region: Optional region to capture (Region object or (x, y, width, height) tuple)
    
    Returns:
        PIL Image object
    
    Example:
        >>> img = kronsteen.screenshot()  # Full screen
        >>> img = kronsteen.screenshot(region=(0, 0, 500, 500))  # Top-left 500x500
        >>> img.save("screenshot.png")
    """
    return _client().screenshot(region)


def save_screenshot(path: str, region: Optional[Region | tuple[int, int, int, int]] = None):
    """
    Take a screenshot and save it to a file.
    
    Args:
        path: File path to save the screenshot
        region: Optional region to capture (Region object or (x, y, width, height) tuple)
    
    Example:
        >>> kronsteen.save_screenshot("screenshot.png")
        >>> kronsteen.save_screenshot("region.png", region=(0, 0, 500, 500))
    """
    return _client().save_screenshot(path, region)


def find_text(query: str, *, timeout: float = 5.0, match_mode: str = "contains", min_confidence: float = 0.5, region: Optional[Region] = None):
    """
    Find text on screen using OCR.
    
    Args:
        query: Text to search for
        timeout: Maximum seconds to wait (default: 5.0)
        match_mode: How to match text - "contains", "equals", "starts-with", or "regex" (default: "contains")
        min_confidence: Minimum OCR confidence (0.0-1.0, default: 0.5)
        region: Optional region to search in
    
    Returns:
        TextMatch object with text, confidence, and region
    
    Example:
        >>> match = kronsteen.find_text("Login")
        >>> print(f"Found at: {match.region.center()}")
        >>> match = kronsteen.find_text("Submit", match_mode="equals")
    """
    return _client().find_text(query, timeout=timeout, match_mode=match_mode, min_confidence=min_confidence, region=region)


def find_all_text(query: Optional[str] = None, *, min_confidence: float = 0.5, region: Optional[Region] = None):
    """
    Find all text on screen using OCR.
    
    Args:
        query: Optional text to filter by (None = return all text)
        min_confidence: Minimum OCR confidence (0.0-1.0, default: 0.5)
        region: Optional region to search in
    
    Returns:
        List of TextMatch objects
    
    Example:
        >>> matches = kronsteen.find_all_text()  # Get all text
        >>> for match in matches:
        ...     print(f"{match.text} at {match.region.center()}")
        >>> buttons = kronsteen.find_all_text("Button")  # Find all "Button" text
    """
    return _client().find_all_text(query, min_confidence=min_confidence, region=region)


def click_on_text(query: str, *, timeout: float = 5.0, match_mode: str = "contains", min_confidence: float = 0.5, region: Optional[Region] = None):
    """
    Find text on screen and click on it.
    
    Args:
        query: Text to search for and click
        timeout: Maximum seconds to wait (default: 5.0)
        match_mode: How to match text - "contains", "equals", "starts-with", or "regex" (default: "contains")
        min_confidence: Minimum OCR confidence (0.0-1.0, default: 0.5)
        region: Optional region to search in
    
    Returns:
        TextMatch object that was clicked
    
    Example:
        >>> kronsteen.click_on_text("Login")  # Find and click "Login" button
        >>> kronsteen.click_on_text("Submit", match_mode="equals")  # Exact match
        >>> kronsteen.click_on_text("OK", timeout=10)  # Wait up to 10 seconds
    """
    return _client().click_on_text(query, timeout=timeout, match_mode=match_mode, min_confidence=min_confidence, region=region)


def find_text_and_click(*args, **kwargs):
    return _client().find_text_and_click(*args, **kwargs)


def wait_for_text(query: str, *, timeout: float = 10.0, match_mode: str = "contains", min_confidence: float = 0.5, region: Optional[Region] = None):
    """
    Wait for text to appear on screen.
    
    Args:
        query: Text to wait for
        timeout: Maximum seconds to wait (default: 10.0)
        match_mode: How to match text - "contains", "equals", "starts-with", or "regex" (default: "contains")
        min_confidence: Minimum OCR confidence (0.0-1.0, default: 0.5)
        region: Optional region to search in
    
    Returns:
        TextMatch object when found
    
    Raises:
        MatchNotFoundError: If text not found within timeout
    
    Example:
        >>> kronsteen.wait_for_text("Loading", timeout=30)  # Wait up to 30 seconds
        >>> match = kronsteen.wait_for_text("Complete")
        >>> print(f"Found at: {match.region.center()}")
    """
    return _client().wait_for_text(query, timeout=timeout, match_mode=match_mode, min_confidence=min_confidence, region=region)


def wait_for_text_to_disappear(*args, **kwargs):
    return _client().wait_for_text_to_disappear(*args, **kwargs)


def find_image(*args, **kwargs):
    return _client().find_image(*args, **kwargs)


def wait_for_image(*args, **kwargs):
    return _client().wait_for_image(*args, **kwargs)


def find_template(template_path: str, *, confidence: float = 0.8, region: Optional[Region] = None, grayscale: bool = True):
    """
    Find a template image on screen using template matching.
    
    Args:
        template_path: Path to the template image file (screenshot you took)
        confidence: Minimum confidence level (0.0-1.0, default: 0.8)
        region: Optional region to search in
        grayscale: Whether to use grayscale matching (default: True, faster)
    
    Returns:
        ImageMatch with the location of the template
    
    Example:
        >>> match = kronsteen.find_template("button.png")
        >>> print(f"Found at: {match.region.center()}")
        >>> match = kronsteen.find_template("icon.png", confidence=0.9)
    """
    return _client().find_template(template_path, confidence=confidence, region=region, grayscale=grayscale)


def wait_for_template(template_path: str, *, timeout: float = 10.0, confidence: float = 0.8, region: Optional[Region] = None, grayscale: bool = True):
    """
    Wait for a template image to appear on screen.
    
    Args:
        template_path: Path to the template image file (screenshot you took)
        timeout: Maximum seconds to wait (default: 10.0)
        confidence: Minimum confidence level (0.0-1.0, default: 0.8)
        region: Optional region to search in
        grayscale: Whether to use grayscale matching (default: True)
    
    Returns:
        ImageMatch when found
    
    Example:
        >>> match = kronsteen.wait_for_template("loading.png", timeout=30)
        >>> print(f"Appeared at: {match.region.center()}")
    """
    return _client().wait_for_template(template_path, timeout=timeout, confidence=confidence, region=region, grayscale=grayscale)


def click_on_template(template_path: str, *, timeout: float = 5.0, confidence: float = 0.8, region: Optional[Region] = None, grayscale: bool = True):
    """
    Find a template image on screen and click on its center.
    
    Args:
        template_path: Path to the template image file (screenshot you took)
        timeout: Maximum seconds to wait (default: 5.0)
        confidence: Minimum confidence level (0.0-1.0, default: 0.8)
        region: Optional region to search in
        grayscale: Whether to use grayscale matching (default: True)
    
    Returns:
        ImageMatch that was clicked
    
    Example:
        >>> kronsteen.click_on_template("login_button.png")  # Finds and clicks
        >>> kronsteen.click_on_template("icon.png", confidence=0.9)
    """
    return _client().click_on_template(template_path, timeout=timeout, confidence=confidence, region=region, grayscale=grayscale)


def find_color(*args, **kwargs):
    return _client().find_color(*args, **kwargs)


def find_shape(*args, **kwargs):
    return _client().find_shape(*args, **kwargs)


def sleep(seconds: float):
    """
    Pause execution for the specified number of seconds.
    
    Args:
        seconds: Number of seconds to sleep
    
    Example:
        >>> kronsteen.sleep(2)  # Wait 2 seconds
        >>> kronsteen.sleep(0.5)  # Wait half a second
    """
    return _client().sleep(seconds)


def countdown(*args, **kwargs):
    return _client().countdown(*args, **kwargs)


def record_region_interactively(*args, **kwargs):
    return _client().record_region_interactively(*args, **kwargs)


def launch(app_name: str, *, args: Optional[list[str]] = None):
    """
    Launch an application by name (cross-platform).
    
    Args:
        app_name: Application name (e.g., "Chrome", "Safari", "Firefox")
        args: Optional command-line arguments to pass to the application
    
    Supported applications:
        - "Chrome", "Google Chrome" - Google Chrome browser
        - "Safari" - Safari browser (macOS)
        - "Firefox" - Mozilla Firefox
        - "Edge" - Microsoft Edge
        - "Brave" - Brave browser
        - And many more...
    
    Example:
        >>> kronsteen.launch("Chrome")  # Launch Chrome
        >>> kronsteen.launch("Safari")  # Launch Safari (macOS)
        >>> kronsteen.launch("Chrome", args=["--incognito"])  # Chrome in incognito mode
    """
    return _client().launch(app_name, args=args)


def close_app(app_name: str) -> None:
    """
    Close an application by name (cross-platform).
    
    Args:
        app_name: Application name (e.g., "Chrome", "Safari", "Firefox")
    
    Example:
        >>> kronsteen.launch("Chrome")
        >>> # ... do automation ...
        >>> kronsteen.close_app("Chrome")  # Close Chrome when done
    """
    return _client().close_app(app_name)

