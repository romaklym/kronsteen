"""Window focus monitoring for pausing automation when window loses focus."""

from __future__ import annotations

import platform
import subprocess
import time
from typing import Optional
import logging


def get_active_window_title() -> Optional[str]:
    """
    Get the title of the currently active window.
    
    Returns:
        Window title string, or None if unable to determine
    """
    system = platform.system()
    
    try:
        if system == "Darwin":  # macOS
            script = '''
                tell application "System Events"
                    set frontApp to name of first application process whose frontmost is true
                    return frontApp
                end tell
            '''
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=False
            )
            return result.stdout.strip() if result.returncode == 0 else None
            
        elif system == "Windows":
            # Use PowerShell to get active window
            script = '''
                Add-Type @"
                    using System;
                    using System.Runtime.InteropServices;
                    public class Window {
                        [DllImport("user32.dll")]
                        public static extern IntPtr GetForegroundWindow();
                        [DllImport("user32.dll")]
                        public static extern int GetWindowText(IntPtr hWnd, System.Text.StringBuilder text, int count);
                    }
"@
                $handle = [Window]::GetForegroundWindow()
                $title = New-Object System.Text.StringBuilder 256
                [void][Window]::GetWindowText($handle, $title, 256)
                $title.ToString()
            '''
            result = subprocess.run(
                ["powershell", "-Command", script],
                capture_output=True,
                text=True,
                check=False
            )
            return result.stdout.strip() if result.returncode == 0 else None
            
        else:  # Linux
            # Try xdotool first
            try:
                result = subprocess.run(
                    ["xdotool", "getactivewindow", "getwindowname"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                return result.stdout.strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback to wmctrl
                try:
                    result = subprocess.run(
                        ["wmctrl", "-a", ":ACTIVE:"],
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    return result.stdout.strip() if result.returncode == 0 else None
                except FileNotFoundError:
                    return None
                    
    except Exception:
        return None


def is_window_active(window_name: str, partial_match: bool = True) -> bool:
    """
    Check if a window with the given name is currently active.
    
    Args:
        window_name: Name of the window to check (e.g., "Chrome", "Safari")
        partial_match: If True, checks if window_name is in active window title (default: True)
    
    Returns:
        True if window is active, False otherwise
    """
    active_window = get_active_window_title()
    
    if active_window is None:
        return False
    
    if partial_match:
        return window_name.lower() in active_window.lower()
    else:
        return window_name.lower() == active_window.lower()


class WindowFocusMonitor:
    """
    Monitor window focus and pause automation when target window loses focus.
    
    Example:
        >>> monitor = WindowFocusMonitor("Chrome")
        >>> monitor.start()
        >>> # Automation will pause if Chrome loses focus
        >>> monitor.stop()
    """
    
    def __init__(
        self,
        window_name: str,
        check_interval: float = 0.5,
        partial_match: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize window focus monitor.
        
        Args:
            window_name: Name of window to monitor (e.g., "Chrome", "Safari")
            check_interval: How often to check focus in seconds (default: 0.5)
            partial_match: If True, matches partial window names (default: True)
            logger: Optional logger for status messages
        """
        self.window_name = window_name
        self.check_interval = check_interval
        self.partial_match = partial_match
        self.logger = logger
        self._monitoring = False
        self._paused = False
    
    def start(self) -> None:
        """Start monitoring window focus."""
        self._monitoring = True
        if self.logger:
            self.logger.info(f"Started monitoring window focus: '{self.window_name}'")
    
    def stop(self) -> None:
        """Stop monitoring window focus."""
        self._monitoring = False
        self._paused = False
        if self.logger:
            self.logger.info(f"Stopped monitoring window focus: '{self.window_name}'")
    
    def check_and_wait(self) -> None:
        """
        Check if target window is active. If not, pause and wait until it becomes active.
        Call this method before each major automation action.
        """
        if not self._monitoring:
            return
        
        while self._monitoring:
            if is_window_active(self.window_name, self.partial_match):
                # Window is active
                if self._paused:
                    self._paused = False
                    if self.logger:
                        self.logger.info(f"✓ Window '{self.window_name}' is active again - resuming automation")
                return
            else:
                # Window is not active - pause
                if not self._paused:
                    self._paused = True
                    if self.logger:
                        self.logger.warning(f"⏸ Window '{self.window_name}' lost focus - pausing automation...")
                
                # Wait and check again
                time.sleep(self.check_interval)
    
    def is_paused(self) -> bool:
        """Check if automation is currently paused."""
        return self._paused
    
    def is_monitoring(self) -> bool:
        """Check if monitoring is active."""
        return self._monitoring


# Global monitor instance
_global_monitor: Optional[WindowFocusMonitor] = None


def start_window_monitoring(
    window_name: str,
    check_interval: float = 0.5,
    partial_match: bool = True,
    logger: Optional[logging.Logger] = None
) -> WindowFocusMonitor:
    """
    Start monitoring a window's focus globally.
    
    Args:
        window_name: Name of window to monitor (e.g., "Chrome", "Safari")
        check_interval: How often to check focus in seconds (default: 0.5)
        partial_match: If True, matches partial window names (default: True)
        logger: Optional logger for status messages
    
    Returns:
        WindowFocusMonitor instance
    
    Example:
        >>> monitor = kronsteen.start_window_monitoring("Chrome")
        >>> # Automation will pause if Chrome loses focus
        >>> monitor.stop()
    """
    global _global_monitor
    
    _global_monitor = WindowFocusMonitor(
        window_name=window_name,
        check_interval=check_interval,
        partial_match=partial_match,
        logger=logger
    )
    _global_monitor.start()
    
    return _global_monitor


def stop_window_monitoring() -> None:
    """Stop global window monitoring."""
    global _global_monitor
    
    if _global_monitor:
        _global_monitor.stop()
        _global_monitor = None


def get_window_monitor() -> Optional[WindowFocusMonitor]:
    """Get the global window monitor instance."""
    return _global_monitor


def check_window_focus() -> None:
    """
    Check if monitored window has focus. Pauses if not.
    Call this before major automation actions.
    """
    global _global_monitor
    
    if _global_monitor:
        _global_monitor.check_and_wait()
