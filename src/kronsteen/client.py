"""High-level Kronsteen client."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Iterable, Optional

import pyautogui

from . import actions, finders
from .config import KronsteenSettings
from .launcher import find_application, launch_application
from .logging_config import log_action, log_ocr_result
from .models import ColorMatch, ImageMatch, Region, TextMatch
from .ocr import DeepSeekOCRClient
from .ocr_tesseract import TesseractOCRClient
from .window_monitor import check_window_focus

class Kronsteen:
    """Coordinates OCR + automation helpers."""

    def __init__(
        self,
        *,
        settings: Optional[KronsteenSettings] = None,
        ocr_client: Optional[TesseractOCRClient | DeepSeekOCRClient] = None,
        ocr_engine: str = "tesseract",
    ) -> None:
        self.settings = settings or KronsteenSettings.from_env()
        
        # Initialize OCR client based on engine choice
        if ocr_client:
            self.ocr_client = ocr_client
        elif ocr_engine.lower() == "deepseek":
            self.ocr_client = DeepSeekOCRClient()
        else:  # Default to tesseract
            self.ocr_client = TesseractOCRClient()
        
        self._apply_pyautogui_settings()

    # region configuration helpers -------------------------------------------------
    def _apply_pyautogui_settings(self) -> None:
        pyautogui.FAILSAFE = self.settings.fail_safe
        pyautogui.PAUSE = self.settings.default_pause

    def configure(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            if not hasattr(self.settings, key):
                raise AttributeError(f"Unknown Kronsteen setting '{key}'")
            setattr(self.settings, key, value)
        self._apply_pyautogui_settings()

    # region simple input wrappers -------------------------------------------------
    @log_action("Click")
    def click(self, *args, **kwargs) -> None:
        actions.click(*args, **kwargs)

    @log_action("Double Click")
    def double_click(self, *args, **kwargs) -> None:
        actions.double_click(*args, **kwargs)

    @log_action("Right Click")
    def right_click(self, *args, **kwargs) -> None:
        actions.right_click(*args, **kwargs)

    @log_action("Click and Drag")
    def click_and_drag(self, start: tuple[int, int], end: tuple[int, int], **kwargs) -> None:
        actions.click_and_drag(start, end, **kwargs)

    @log_action("Move Mouse")
    def move_to(self, x: int, y: int, duration: float = 0.0) -> None:
        actions.move_to(x, y, duration)

    @log_action("Type Text")
    def type_text(self, text: str, *, interval: float = 0.0, press_enter: bool = False) -> None:
        actions.type_text(text, interval=interval, press_enter=press_enter)

    def type(self, text: str, **kwargs) -> None:
        self.type_text(text, **kwargs)

    @log_action("Hotkey")
    def hotkey(self, *keys: str) -> None:
        actions.hotkey(*keys)

    @log_action("Press Key")
    def press(self, key: str) -> None:
        actions.press(key)

    @log_action("Scroll")
    def scroll(self, clicks: int) -> None:
        actions.scroll(clicks)

    # region OCR powered helpers ---------------------------------------------------
    @log_action("Find Text", log_result=True)
    def find_text(self, *args, timeout: Optional[float] = None, **kwargs) -> TextMatch:
        resolved_timeout = timeout if timeout is not None else self.settings.default_timeout
        result = finders.find_text(*args, ocr_client=self.ocr_client, timeout=resolved_timeout, **kwargs)
        log_ocr_result("Find Text", [result], args[0] if args else None)
        return result

    @log_action("Find All Text", log_result=True)
    def find_all_text(self, *args, **kwargs) -> list[TextMatch]:
        result = finders.find_all_text(*args, ocr_client=self.ocr_client, **kwargs)
        log_ocr_result("Find All Text", result, args[0] if args else None)
        return result

    @log_action("Click on Text", log_result=True)
    def click_on_text(self, *args, timeout: Optional[float] = None, **kwargs) -> TextMatch:
        resolved_timeout = timeout if timeout is not None else self.settings.default_timeout
        result = finders.click_on_text(*args, ocr_client=self.ocr_client, timeout=resolved_timeout, **kwargs)
        log_ocr_result("Click on Text", [result], args[0] if args else None)
        return result

    def find_text_and_click(self, *args, **kwargs) -> TextMatch:
        return self.click_on_text(*args, **kwargs)

    @log_action("Wait for Text", log_result=True)
    def wait_for_text(self, *args, timeout: Optional[float] = None, **kwargs) -> TextMatch:
        resolved_timeout = timeout if timeout is not None else self.settings.default_timeout
        result = finders.wait_for_text(*args, ocr_client=self.ocr_client, timeout=resolved_timeout, **kwargs)
        log_ocr_result("Wait for Text", [result], args[0] if args else None)
        return result

    def wait_for_text_to_disappear(self, *args, timeout: Optional[float] = None, **kwargs) -> None:
        resolved_timeout = timeout if timeout is not None else self.settings.default_timeout
        return finders.wait_for_text_to_disappear(
            *args,
            ocr_client=self.ocr_client,
            timeout=resolved_timeout,
            **kwargs,
        )

    def find_image(self, *args, **kwargs) -> ImageMatch:
        return finders.find_image(*args, **kwargs)

    def wait_for_image(self, *args, timeout: Optional[float] = None, **kwargs) -> ImageMatch:
        resolved_timeout = timeout if timeout is not None else self.settings.default_timeout
        return finders.wait_for_image(*args, timeout=resolved_timeout, **kwargs)
    
    # region template matching helpers ---------------------------------------------
    @log_action("Find Template", log_result=True)
    def find_template(self, template_path: str, *, confidence: float = 0.8, region: Optional[Region] = None, grayscale: bool = True) -> ImageMatch:
        """
        Find a template image on screen.
        
        Args:
            template_path: Path to the template image file
            confidence: Minimum confidence level (0.0-1.0, default: 0.8)
            region: Optional region to search in
            grayscale: Whether to use grayscale matching (default: True)
        
        Returns:
            ImageMatch with the location of the template
        """
        return finders.find_template(template_path, confidence=confidence, region=region, grayscale=grayscale)
    
    @log_action("Wait for Template", log_result=True)
    def wait_for_template(self, template_path: str, *, timeout: Optional[float] = None, confidence: float = 0.8, region: Optional[Region] = None, grayscale: bool = True) -> ImageMatch:
        """
        Wait for a template image to appear on screen.
        
        Args:
            template_path: Path to the template image file
            timeout: Maximum seconds to wait
            confidence: Minimum confidence level (0.0-1.0, default: 0.8)
            region: Optional region to search in
            grayscale: Whether to use grayscale matching (default: True)
        
        Returns:
            ImageMatch when found
        """
        resolved_timeout = timeout if timeout is not None else self.settings.default_timeout
        return finders.wait_for_template(template_path, timeout=resolved_timeout, confidence=confidence, region=region, grayscale=grayscale)
    
    @log_action("Click on Template", log_result=True)
    def click_on_template(self, template_path: str, *, timeout: Optional[float] = None, confidence: float = 0.8, region: Optional[Region] = None, grayscale: bool = True) -> ImageMatch:
        """
        Find a template image on screen and click on its center.
        
        Args:
            template_path: Path to the template image file
            timeout: Maximum seconds to wait
            confidence: Minimum confidence level (0.0-1.0, default: 0.8)
            region: Optional region to search in
            grayscale: Whether to use grayscale matching (default: True)
        
        Returns:
            ImageMatch that was clicked
        """
        resolved_timeout = timeout if timeout is not None else self.settings.default_timeout
        return finders.click_on_template(template_path, timeout=resolved_timeout, confidence=confidence, region=region, grayscale=grayscale)

    def find_color(self, *args, **kwargs) -> ColorMatch:
        return finders.find_color(*args, **kwargs)

    def find_shape(self, template_path: str, **kwargs) -> ImageMatch:
        """Shape detection implemented via template matching helper."""
        return self.find_image(template_path, **kwargs)

    # region screenshot + utility --------------------------------------------------
    def screenshot(self, region: Optional[Region | Iterable[int]] = None):
        return actions.screenshot(region=region)

    def save_screenshot(self, path: str, region: Optional[Region | Iterable[int]] = None) -> Path:
        actions.save_screenshot(path, region=region)
        return Path(path)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def countdown(self, seconds: float) -> None:
        for remaining in range(int(seconds), 0, -1):
            print(f"{remaining}...")
            time.sleep(1)

    def record_region_interactively(self) -> Region:
        print("Move mouse to TOP-LEFT corner and press Enter in the terminal...")
        input()
        start = actions.current_position()
        print("Now move mouse to BOTTOM-RIGHT corner and press Enter...")
        input()
        end = actions.current_position()
        region = Region.from_corners(start, end)
        return region

    # region launch helpers --------------------------------------------------------
    @log_action("Launch Application")
    def launch(
        self,
        executable: str | Path | list[str],
        *,
        cwd: Optional[str | Path] = None,
        wait: bool = False,
        args: Optional[list[str]] = None,
    ):
        """
        Launch an application.
        
        Args:
            executable: Can be:
                - Full path to executable (e.g., "/Applications/Safari.app/Contents/MacOS/Safari")
                - Application name (e.g., "Chrome", "Safari", "Brave") - will auto-detect path
                - List of command parts (e.g., ["open", "-a", "Safari"])
            cwd: Working directory for the process
            wait: Whether to wait for the process to complete
            args: Additional command-line arguments (only used with app name or path)
        
        Returns:
            subprocess.Popen object
        
        Examples:
            >>> kronsteen.launch("Chrome")  # Auto-finds Chrome
            >>> kronsteen.launch("Safari", args=["https://google.com"])
            >>> kronsteen.launch("/usr/bin/firefox")  # Full path
        """
        if isinstance(executable, list):
            cmd = [str(part) for part in executable]
            process = subprocess.Popen(cmd, cwd=cwd and str(cwd))
        else:
            path = Path(executable)
            
            # If it's a full path that exists, use it directly
            if path.exists():
                cmd = [str(path)]
                if args:
                    cmd.extend(args)
                process = subprocess.Popen(cmd, cwd=cwd and str(cwd))
            else:
                # Try to find the application by name
                try:
                    process = launch_application(
                        str(executable),
                        args=args,
                        wait=False,
                    )
                except FileNotFoundError:
                    # Fallback to original behavior (might be a command in PATH)
                    cmd = [str(executable)]
                    if args:
                        cmd.extend(args)
                    process = subprocess.Popen(cmd, cwd=cwd and str(cwd))
        
        if wait:
            process.wait()
        return process
    
    @log_action("Close Application")
    def close_app(self, app_name: str) -> None:
        """
        Close an application by name.
        
        Args:
            app_name: Application name (e.g., "Chrome", "Safari", "Firefox")
        
        Examples:
            >>> kronsteen.close_app("Chrome")
            >>> kronsteen.close_app("Safari")
        """
        import platform
        
        system = platform.system()
        
        if system == "Darwin":  # macOS
            # Use AppleScript to quit the application
            script = f'tell application "{app_name}" to quit'
            subprocess.run(["osascript", "-e", script], check=False)
        elif system == "Windows":
            # Use taskkill on Windows
            subprocess.run(["taskkill", "/F", "/IM", f"{app_name}.exe"], check=False)
        else:  # Linux
            # Use pkill on Linux
            subprocess.run(["pkill", "-f", app_name], check=False)
