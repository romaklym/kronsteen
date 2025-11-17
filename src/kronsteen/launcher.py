"""Cross-platform application launcher utilities."""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def find_application(app_name: str) -> Optional[str]:
    """
    Find an application by name across different platforms.
    
    Args:
        app_name: Application name (e.g., "Chrome", "Safari", "Brave Browser")
    
    Returns:
        Full path to the application executable, or None if not found
    
    Examples:
        >>> find_application("Chrome")
        '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
        >>> find_application("Safari")
        '/Applications/Safari.app/Contents/MacOS/Safari'
    """
    system = platform.system()
    
    if system == "Darwin":
        return _find_macos_app(app_name)
    elif system == "Windows":
        return _find_windows_app(app_name)
    else:  # Linux and others
        return _find_linux_app(app_name)


def _find_macos_app(app_name: str) -> Optional[str]:
    """Find application on macOS."""
    # Common application directories
    search_paths = [
        Path("/Applications"),
        Path("/System/Applications"),
        Path.home() / "Applications",
    ]
    
    # Normalize app name - add .app if not present
    if not app_name.endswith(".app"):
        app_name_with_ext = f"{app_name}.app"
    else:
        app_name_with_ext = app_name
    
    # Search for the app
    for base_path in search_paths:
        if not base_path.exists():
            continue
            
        # Direct match
        app_path = base_path / app_name_with_ext
        if app_path.exists():
            # Find the executable inside the app bundle
            executable = app_path / "Contents" / "MacOS" / app_name.replace(".app", "")
            if executable.exists():
                return str(executable)
            
            # Some apps have different executable names
            macos_dir = app_path / "Contents" / "MacOS"
            if macos_dir.exists():
                executables = list(macos_dir.iterdir())
                if executables:
                    return str(executables[0])
        
        # Search for partial matches (case-insensitive)
        try:
            for item in base_path.iterdir():
                if item.is_dir() and item.suffix == ".app":
                    item_name_lower = item.stem.lower()
                    search_name_lower = app_name.replace(".app", "").lower()
                    
                    if search_name_lower in item_name_lower or item_name_lower in search_name_lower:
                        macos_dir = item / "Contents" / "MacOS"
                        if macos_dir.exists():
                            executables = list(macos_dir.iterdir())
                            if executables:
                                return str(executables[0])
        except PermissionError:
            continue
    
    # Try using 'open' command as fallback
    try:
        result = subprocess.run(
            ["mdfind", f"kMDItemKind == 'Application' && kMDItemDisplayName == '*{app_name}*'"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            app_path = Path(result.stdout.strip().split("\n")[0])
            if app_path.exists():
                macos_dir = app_path / "Contents" / "MacOS"
                if macos_dir.exists():
                    executables = list(macos_dir.iterdir())
                    if executables:
                        return str(executables[0])
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return None


def _find_windows_app(app_name: str) -> Optional[str]:
    """Find application on Windows."""
    # Add .exe if not present
    if not app_name.lower().endswith(".exe"):
        exe_name = f"{app_name}.exe"
    else:
        exe_name = app_name
    
    # Check if it's in PATH
    path_result = shutil.which(exe_name)
    if path_result:
        return path_result
    
    # Common installation directories
    program_files = [
        Path(r"C:\Program Files"),
        Path(r"C:\Program Files (x86)"),
    ]
    
    # Common browser locations
    browser_paths = {
        "chrome": [
            "Google/Chrome/Application/chrome.exe",
            "Google Chrome/Application/chrome.exe",
        ],
        "firefox": [
            "Mozilla Firefox/firefox.exe",
        ],
        "edge": [
            "Microsoft/Edge/Application/msedge.exe",
        ],
        "brave": [
            "BraveSoftware/Brave-Browser/Application/brave.exe",
        ],
        "opera": [
            "Opera/launcher.exe",
        ],
    }
    
    # Try known browser paths
    app_lower = app_name.lower().replace(".exe", "")
    if app_lower in browser_paths:
        for base in program_files:
            for rel_path in browser_paths[app_lower]:
                full_path = base / rel_path
                if full_path.exists():
                    return str(full_path)
    
    # Search in Program Files
    for base in program_files:
        if not base.exists():
            continue
        
        try:
            # Direct search
            for item in base.rglob(exe_name):
                if item.is_file():
                    return str(item)
            
            # Partial match search
            search_name = app_name.replace(".exe", "").lower()
            for item in base.iterdir():
                if item.is_dir() and search_name in item.name.lower():
                    for exe in item.rglob("*.exe"):
                        if search_name in exe.stem.lower():
                            return str(exe)
        except PermissionError:
            continue
    
    return None


def _find_linux_app(app_name: str) -> Optional[str]:
    """Find application on Linux."""
    # Check if it's in PATH
    path_result = shutil.which(app_name)
    if path_result:
        return path_result
    
    # Common binary locations
    search_paths = [
        Path("/usr/bin"),
        Path("/usr/local/bin"),
        Path("/snap/bin"),
        Path("/opt"),
        Path.home() / ".local" / "bin",
    ]
    
    # Common browser names
    browser_names = {
        "chrome": ["google-chrome", "google-chrome-stable", "chrome"],
        "firefox": ["firefox", "firefox-esr"],
        "brave": ["brave", "brave-browser"],
        "opera": ["opera"],
        "edge": ["microsoft-edge", "microsoft-edge-stable"],
    }
    
    app_lower = app_name.lower()
    
    # Try known browser names
    if app_lower in browser_names:
        for name in browser_names[app_lower]:
            result = shutil.which(name)
            if result:
                return result
    
    # Search in common paths
    for base_path in search_paths:
        if not base_path.exists():
            continue
        
        try:
            # Direct match
            app_path = base_path / app_name
            if app_path.exists() and app_path.is_file():
                return str(app_path)
            
            # Partial match
            for item in base_path.iterdir():
                if item.is_file() and app_lower in item.name.lower():
                    return str(item)
        except PermissionError:
            continue
    
    return None


def launch_application(
    app_name: str,
    *,
    args: Optional[list[str]] = None,
    wait: bool = False,
) -> subprocess.Popen:
    """
    Launch an application by name across different platforms.
    
    Args:
        app_name: Application name (e.g., "Chrome", "Safari", "Brave")
        args: Additional command-line arguments
        wait: Whether to wait for the process to complete
    
    Returns:
        The subprocess.Popen object
    
    Raises:
        FileNotFoundError: If the application cannot be found
    
    Examples:
        >>> launch_application("Chrome")
        >>> launch_application("Safari", args=["https://google.com"])
    """
    system = platform.system()
    
    # Try to find the application
    app_path = find_application(app_name)
    
    if not app_path:
        raise FileNotFoundError(
            f"Could not find application '{app_name}' on {system}. "
            f"Please provide the full path or ensure the application is installed."
        )
    
    # Build command
    cmd = [app_path]
    if args:
        cmd.extend(args)
    
    # Launch the application
    if system == "Darwin":
        # On macOS, use 'open' command for better integration
        # But only if no specific args are provided
        if not args:
            cmd = ["open", "-a", app_path]
    
    process = subprocess.Popen(cmd)
    
    if wait:
        process.wait()
    
    return process
