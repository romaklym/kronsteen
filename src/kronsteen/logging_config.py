"""Logging configuration and decorators for Kronsteen."""

import functools
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import pyautogui


# Global logger instance
_logger: Optional[logging.Logger] = None
_screenshots_dir: Optional[Path] = None
_screenshot_counter = 0


def setup_logging(
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    enable_screenshots: bool = False,
) -> logging.Logger:
    """
    Set up logging for Kronsteen.
    
    Args:
        log_dir: Directory for logs and screenshots (default: ./logs)
        level: Logging level (default: logging.INFO)
        enable_screenshots: Whether to capture screenshots (default: False)
    
    Returns:
        Configured logger instance
    """
    global _logger, _screenshots_dir, _screenshot_counter
    
    # Create logger
    logger = logging.getLogger("kronsteen")
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Setup directory structure
    if log_dir is None:
        log_dir = "logs"
    
    base_dir = Path(log_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logs subdirectory
    logs_subdir = base_dir / "logs"
    logs_subdir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_subdir / f"automation_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Setup screenshots directory
    if enable_screenshots:
        _screenshots_dir = base_dir / "screenshots"
        _screenshots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Screenshots enabled: {_screenshots_dir.absolute()}")
    else:
        _screenshots_dir = None
        logger.info("Screenshots disabled")
    
    _logger = logger
    _screenshot_counter = 0
    
    logger.info("=" * 80)
    logger.info("Kronsteen automation started")
    logger.info(f"Log file: {log_file.absolute()}")
    logger.info("=" * 80)
    
    return logger


def get_logger() -> logging.Logger:
    """Get the Kronsteen logger instance."""
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger


def capture_screenshot(action_name: str) -> Optional[Path]:
    """
    Capture a screenshot for the current action.
    
    Args:
        action_name: Name of the action being performed
    
    Returns:
        Path to the saved screenshot, or None if screenshots disabled
    """
    global _screenshot_counter, _screenshots_dir
    
    if _screenshots_dir is None:
        return None
    
    try:
        _screenshot_counter += 1
        
        # Clean action name for filename
        clean_action = action_name.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
        
        # Simple naming: action_name.png (or action_name_2.png if duplicate)
        filename = f"{clean_action}.png"
        screenshot_path = _screenshots_dir / filename
        
        # If file exists, add counter
        if screenshot_path.exists():
            filename = f"{clean_action}_{_screenshot_counter}.png"
            screenshot_path = _screenshots_dir / filename
        
        # Capture screenshot
        screenshot = pyautogui.screenshot()
        screenshot.save(screenshot_path)
        
        return screenshot_path
    except Exception as e:
        get_logger().warning(f"Failed to capture screenshot: {e}")
        return None


def log_action(
    action_name: Optional[str] = None,
    log_args: bool = True,
    log_result: bool = False,
    check_focus: bool = True,
):
    """
    Decorator to log function calls and optionally capture screenshots.
    
    Args:
        action_name: Custom action name (default: function name)
        log_args: Log function arguments (default: True)
        log_result: Log function result (default: False)
        check_focus: Check window focus before action (default: True)
    
    Example:
        @log_action("Click button")
        def click_button():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Check window focus before action (if monitoring is active)
            if check_focus:
                from .window_monitor import check_window_focus
                check_window_focus()
            
            logger = get_logger()
            name = action_name or func.__name__.replace("_", " ").title()
            
            # Check if screenshot is requested via kwargs
            take_screenshot = kwargs.pop('screenshot', False)
            
            # Log function call
            if log_args and (args or kwargs):
                # Filter out 'self' from args for cleaner logs
                filtered_args = [repr(a) for a in args[1:]] if args and hasattr(args[0], '__class__') else [repr(a) for a in args]
                kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
                params = ", ".join(filter(None, [", ".join(filtered_args), kwargs_str]))
                if params:
                    logger.info(f"→ {name}({params})")
                else:
                    logger.info(f"→ {name}")
            else:
                logger.info(f"→ {name}")
            
            # Execute function
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                # Log success
                if log_result and result is not None:
                    logger.info(f"✓ {name} completed in {elapsed:.2f}s - Result: {result}")
                else:
                    logger.info(f"✓ {name} completed in {elapsed:.2f}s")
                
                # Capture screenshot after successful completion
                if take_screenshot:
                    screenshot_path = capture_screenshot(name)
                    if screenshot_path:
                        logger.info(f"  Screenshot: {screenshot_path.name}")
                
                return result
                
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"✗ {name} failed after {elapsed:.2f}s - Error: {e}")
                
                # Capture error screenshot if requested
                if take_screenshot:
                    screenshot_path = capture_screenshot(f"{name}_error")
                    if screenshot_path:
                        logger.info(f"  Screenshot (error): {screenshot_path.name}")
                
                raise
        
        return wrapper
    return decorator


def log_ocr_result(action_name: str, matches: list, query: Optional[str] = None):
    """
    Log OCR results.
    
    Args:
        action_name: Name of the OCR action
        matches: List of TextMatch objects
        query: Optional search query
    """
    logger = get_logger()
    
    if query:
        logger.info(f"  OCR: Searching for '{query}' - Found {len(matches)} matches")
    else:
        logger.info(f"  OCR: Found {len(matches)} text elements")
    
    # Log first few matches at debug level
    for i, match in enumerate(matches[:5], 1):
        center = match.region.center()
        logger.debug(f"    {i}. '{match.text[:50]}' at ({center[0]}, {center[1]}) [{match.confidence:.0%}]")
    
    if len(matches) > 5:
        logger.debug(f"    ... and {len(matches) - 5} more")
