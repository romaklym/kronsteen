```
██╗  ██╗██████╗  ██████╗ ███╗   ██╗███████╗████████╗███████╗███████╗███╗   ██╗
██║ ██╔╝██╔══██╗██╔═══██╗████╗  ██║██╔════╝╚══██╔══╝██╔════╝██╔════╝████╗  ██║
█████╔╝ ██████╔╝██║   ██║██╔██╗ ██║███████╗   ██║   █████╗  █████╗  ██╔██╗ ██║
██╔═██╗ ██╔══██╗██║   ██║██║╚██╗██║╚════██║   ██║   ██╔══╝  ██╔══╝  ██║╚██╗██║
██║  ██╗██║  ██║╚██████╔╝██║ ╚████║███████║   ██║   ███████╗███████╗██║ ╚████║
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚══════╝╚══════╝╚═╝  ╚═══╝
```

<div align="center">

### *"The perfect automation is invisible"*

**Vision-Powered Desktop Automation Framework**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Cross-Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey)](https://github.com/yourusername/kronsteen)

</div>

---

## 🎯 Mission Brief

Inspired by **SPECTRE's #5**, the master planner from Ian Fleming's *From Russia with Love*, **Kronsteen** is your strategic automation framework. Like its namesake, it operates with precision, intelligence, and flawless execution.

Kronsteen combines **computer vision (OCR)** with **human-like automation** to interact with any desktop application—no API required. It sees what you see, clicks what you click, and types what you type.

### Why Kronsteen?

- 🎯 **Vision-First** - Uses OCR to find and interact with UI elements
- 🚀 **Universal** - Works with any application, any platform
- 🧠 **Intelligent** - Template matching, window focus monitoring, smart retries
- ⚡ **Fast** - Tesseract OCR processes screens in ~100ms
- 🛡️ **Reliable** - Built-in error handling and logging
- 🌍 **Cross-Platform** - macOS, Windows, Linux support

---

## ✨ Key Features

### 🎭 Core Capabilities

| Feature | Description |
|---------|-------------|
| **🔍 OCR Text Finding** | Find and click text anywhere on screen using Tesseract OCR |
| **🖼️ Template Matching** | Match images and click on them with confidence thresholds |
| **🚀 Universal Launcher** | Launch apps by name on any platform (no paths needed) |
| **🖱️ Mouse & Keyboard** | Full control with human-like timing and movements |
| **🪟 Window Monitoring** | Pause automation when target window loses focus |
| **📸 Smart Screenshots** | Capture screens with automatic Retina display scaling |
| **🎨 Color Detection** | Find UI elements by color patterns |
| **📊 Logging System** | Automatic logging with optional screenshot capture |
| **⚙️ Configurable** | Timeouts, retries, confidence levels, and more |

### 🧩 Framework Architecture

```
kronsteen/
├── 🎯 client.py              # Main orchestrator
├── 🔍 ocr_tesseract.py       # Tesseract OCR engine (Retina support)
├── 🖼️ ocr.py                 # DeepSeek OCR engine (GPU/CPU)
├── 🎪 finders.py             # Text/image/template finding
├── 🎬 actions.py             # Mouse/keyboard automation
├── 🚀 launcher.py            # Cross-platform app launcher
├── 🪟 window_monitor.py      # Window focus tracking
├── 📝 logging_config.py      # Logging & screenshots
├── 🎨 models.py              # Data structures
└── ⚙️ config.py              # Configuration management
```

---

## 🚀 Installation

**Two simple steps:**

### 1. Install Tesseract OCR

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt update && sudo apt install tesseract-ocr

# Windows
# Download installer: https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. Install Kronsteen

```bash
pip install kronsteen
```

This installs all Python dependencies:
- `pyautogui` - Mouse and keyboard automation
- `pytesseract` - Python wrapper for Tesseract
- `opencv-python` - Computer vision and template matching
- `Pillow` - Image processing
- `numpy` - Numerical operations

✅ **Done!** Start automating in seconds.

---

## ⚡ Quick Start

### Your First Mission

```python
import kronsteen

# Setup logging (optional)
kronsteen.setup_logging(enable_screenshots=False)

# Launch Chrome - works on all platforms!
kronsteen.launch("Chrome")

# Wait for page to load using OCR
kronsteen.wait_for_text("Google", timeout=10)

# Click on text found by OCR
kronsteen.click_on_text("Search", match_mode="contains")

# Type like a human
kronsteen.type_text("Hello World", press_enter=True)

# Mission accomplished! 🎯
```

### 30-Second Demo

```python
import kronsteen

# Configure for your mission
kronsteen.configure(default_timeout=20)

# Launch target application
kronsteen.launch("Chrome")
kronsteen.sleep(2)

# Use OCR to find and interact
match = kronsteen.find_text("Sign In")
print(f"Found at: {match.region.center()}")

# Click on it
kronsteen.click_on_text("Sign In")

# Type credentials
kronsteen.type_text("agent007@mi6.gov.uk")
kronsteen.press("tab")
kronsteen.type_text("martini_shaken")
kronsteen.press("enter")
```

---

## 📚 Complete Guide

### 🔍 OCR Text Finding

Kronsteen's vision system can find any text on screen:

```python
# Find text with OCR
match = kronsteen.find_text("Login")
print(f"Found '{match.text}' at {match.region.center()}")
print(f"Confidence: {match.confidence}")

# Find all text on screen
all_matches = kronsteen.find_all_text(None)
for match in all_matches:
    print(f"- {match.text}")

# Click on text
kronsteen.click_on_text("Submit", match_mode="contains")

# Wait for text to appear
kronsteen.wait_for_text("Welcome", timeout=30)

# Wait for text to disappear
kronsteen.wait_for_text_to_disappear("Loading...", timeout=10)

# Search in specific region only
match = kronsteen.find_text(
    "Button",
    region=(0, 0, 500, 500),  # Top-left quadrant
    min_confidence=0.8
)
```

**Match Modes:**
- `"contains"` - Text contains the query (default)
- `"equals"` - Exact match
- `"starts-with"` - Text starts with query  
- `"regex"` - Regular expression match

### 🖼️ Template Matching

Find and click images on screen:

```python
# Find template image
match = kronsteen.find_template(
    "button.png",
    confidence=0.8,
    grayscale=True
)

# Wait for template to appear
match = kronsteen.wait_for_template(
    "loading_icon.png",
    timeout=10
)

# Find and click in one step
match = kronsteen.click_on_template(
    "submit_button.png",
    confidence=0.9
)
```

### 🚀 Universal Launcher

Launch apps by name—no paths needed:

```python
# Launch by name (cross-platform)
kronsteen.launch("Chrome")    # Works everywhere
kronsteen.launch("Safari")    # macOS
kronsteen.launch("Firefox")   # All platforms
kronsteen.launch("Terminal")  # macOS/Linux

# Launch with arguments
kronsteen.launch("Chrome", args=["--incognito"])

# Find app path
path = kronsteen.find_application("Chrome")
print(f"Chrome is at: {path}")

# Close app when done
kronsteen.close_app("Chrome")
```

### 🪟 Window Focus Monitoring

Pause automation when target window loses focus:

```python
# Start monitoring Chrome window
monitor = kronsteen.start_window_monitoring(
    window_name="Chrome",
    check_interval=0.5  # Check every 0.5s
)

# Automation pauses if Chrome loses focus
kronsteen.click_on_text("Button")  # Pauses if Chrome not active
kronsteen.type_text("Hello")       # Resumes when Chrome regains focus

# Stop monitoring
kronsteen.stop_window_monitoring()
```

### 🖱️ Mouse Control

```python
# Click
kronsteen.click(x=100, y=200)
kronsteen.double_click(x=100, y=200)
kronsteen.right_click(x=100, y=200)

# Move mouse
kronsteen.move_to(x=500, y=300, duration=0.5)

# Drag
kronsteen.click_and_drag(
    start_x=100, start_y=100,
    end_x=500, end_y=500,
    duration=1.0
)

# Scroll
kronsteen.scroll(clicks=5)   # Scroll down
kronsteen.scroll(clicks=-5)  # Scroll up
```

### ⌨️ Keyboard Control

```python
# Type text
kronsteen.type_text("Hello World")
kronsteen.type_text("Search query", press_enter=True)

# Press keys
kronsteen.press("enter")
kronsteen.press("tab")
kronsteen.press("escape")

# Hotkeys (keyboard shortcuts)
kronsteen.hotkey("command", "c")  # Copy on macOS
kronsteen.hotkey("ctrl", "c")     # Copy on Windows/Linux
kronsteen.hotkey("command", "l")  # Focus address bar
```

### 📸 Screenshots & Colors

```python
# Capture full screen
img = kronsteen.screenshot()

# Capture region
img = kronsteen.screenshot(region=(0, 0, 500, 500))

# Save screenshot
kronsteen.save_screenshot("screenshot.png")

# Find color on screen
match = kronsteen.find_color(
    color=(255, 0, 0),  # RGB red
    tolerance=10
)
```

### 📝 Logging & Configuration

```python
# Setup logging with screenshots
kronsteen.setup_logging(
    log_dir="logs",
    enable_screenshots=True
)

# Get logger
logger = kronsteen.get_logger()
logger.info("Starting automation")

# Configure global settings
kronsteen.configure(
    default_timeout=20,
    retry_interval=0.5,
    fail_safe=True,
    default_pause=0.1
)

# Switch OCR engines
kronsteen.use_ocr_engine("tesseract")  # Fast (default)
kronsteen.use_ocr_engine("deepseek")   # Accurate (GPU)
```

---

## 🎬 Real-World Examples

### Example 1: Web Automation

```python
"""Automate Google search."""
import kronsteen

# Setup
kronsteen.setup_logging()
kronsteen.configure(default_timeout=25)

# Launch Chrome
kronsteen.launch("Chrome")
kronsteen.sleep(3)

# Wait for Google to load
kronsteen.wait_for_text("Google", timeout=30)

# Focus address bar and search
kronsteen.hotkey("command", "l")  # Cmd+L on macOS
kronsteen.sleep(0.5)
kronsteen.type_text("Kronsteen automation", press_enter=True)

# Wait for results
kronsteen.sleep(3)
print("✓ Search completed!")
```

### Example 2: Form Filling

```python
"""Fill out a web form."""
import kronsteen

# Find and fill form fields
kronsteen.click_on_text("Email")
kronsteen.type_text("agent@mi6.gov.uk")

kronsteen.press("tab")  # Move to next field
kronsteen.type_text("SecretPassword123")

kronsteen.press("tab")
kronsteen.type_text("James Bond")

# Submit
kronsteen.click_on_text("Submit")
kronsteen.wait_for_text("Success", timeout=10)
print("✓ Form submitted!")
```

### Example 3: Multi-Step Workflow

```python
"""Complete multi-step automation workflow."""
import kronsteen

def automate_workflow():
    # Setup
    kronsteen.setup_logging(enable_screenshots=True)
    logger = kronsteen.get_logger()
    
    try:
        # Step 1: Launch application
        logger.info("Step 1: Launching application")
        kronsteen.launch("Chrome")
        kronsteen.sleep(2)
        
        # Step 2: Navigate
        logger.info("Step 2: Navigating to site")
        kronsteen.hotkey("command", "l")
        kronsteen.type_text("https://example.com", press_enter=True)
        
        # Step 3: Wait for page load
        logger.info("Step 3: Waiting for page load")
        kronsteen.wait_for_text("Welcome", timeout=30)
        
        # Step 4: Interact with UI
        logger.info("Step 4: Clicking login")
        kronsteen.click_on_text("Login")
        
        # Step 5: Fill credentials
        logger.info("Step 5: Entering credentials")
        kronsteen.type_text("username")
        kronsteen.press("tab")
        kronsteen.type_text("password")
        kronsteen.press("enter")
        
        # Step 6: Verify success
        logger.info("Step 6: Verifying login")
        kronsteen.wait_for_text("Dashboard", timeout=20)
        
        logger.info("✓ Workflow completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Workflow failed: {e}")
        return False
    
    finally:
        # Cleanup
        kronsteen.close_app("Chrome")

if __name__ == "__main__":
    success = automate_workflow()
    exit(0 if success else 1)
```

---

## 🌍 Platform Support

### macOS
- ✅ **Retina Display Support** - Automatic coordinate scaling
- ✅ **Universal Launcher** - `.app` bundle detection
- ✅ **Spotlight Integration** - Fallback app search
- ✅ **AppleScript Support** - Window management

### Windows  
- ✅ **Program Files Search** - Auto-detect installed apps
- ✅ **System PATH** - Command-line app support
- ✅ **Registry Integration** - Browser detection
- ✅ **PowerShell Support** - Window management

### Linux
- ✅ **Standard Directories** - `/usr/bin`, `/usr/local/bin`
- ✅ **Snap/Flatpak** - Modern package format support
- ✅ **Desktop Files** - `.desktop` file integration
- ✅ **xdotool/wmctrl** - Window management

---

## ⚡ Performance

| Feature | Speed | Notes |
|---------|-------|-------|
| **Tesseract OCR** | ~100ms | Fast, CPU-based |
| **DeepSeek OCR** | ~500ms (GPU) / ~5s (CPU) | Accurate, GPU recommended |
| **Screenshot** | ~10ms | Instant capture |
| **Template Match** | ~50-200ms | Depends on image size |
| **Mouse/Keyboard** | Instant | PyAutoGUI |
| **App Launch** | ~1-3s | Platform dependent |

### Optimization Tips

- ✅ Use **Tesseract** for speed (default)
- ✅ Use **DeepSeek** for accuracy (GPU required)
- ✅ Specify **regions** to limit search area
- ✅ Use **template matching** for repeated UI elements
- ✅ Enable **window monitoring** to prevent errors
- ✅ Cache **app paths** for faster launches

---

## 🔧 Troubleshooting

### Tesseract Not Found

The `tesseract` package should bundle the binary automatically. If you still get errors:

**Option 1: Reinstall**
```bash
pip uninstall kronsteen tesseract pytesseract
pip install kronsteen
```

**Option 2: System Installation (fallback)**
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

**Verify:**
```python
import pytesseract
print(pytesseract.get_tesseract_version())
```

### Text Not Found

```python
# Lower confidence threshold
match = kronsteen.find_text("text", min_confidence=0.5)

# Use different match mode
match = kronsteen.find_text("text", match_mode="contains")

# Search in specific region
match = kronsteen.find_text("text", region=(0, 0, 500, 500))

# Try different OCR engine
kronsteen.use_ocr_engine("deepseek")  # More accurate
```

### Retina Display Issues

Kronsteen automatically handles Retina scaling. To verify:

```python
from kronsteen.ocr_tesseract import TesseractOCRClient
ocr = TesseractOCRClient()
print(f"Scale factor: {ocr.scale_factor}")  # Should be 2.0 on Retina
```

### Window Focus Not Working

```python
# Check if window name is correct
active = kronsteen.get_active_window_title()
print(f"Active window: {active}")

# Use partial match
kronsteen.start_window_monitoring("Chrome", partial_match=True)
```

---

## 📁 Examples

Check out the `examples/` directory for complete working examples:
- **`example.py`** - Google search automation with window monitoring and OCR

---

## 🎓 Why Kronsteen?

### The SPECTRE Connection

Named after **Kronsteen**, SPECTRE's #5 and master strategist from Ian Fleming's *From Russia with Love*. Like the chess grandmaster who planned the perfect operation, this framework executes automation with precision and intelligence.

> *"The plan is perfect. The execution will be flawless."* - Kronsteen

### Why This Framework?

- 🎯 **No API Required** - Works with any application
- 🧠 **Vision-Based** - Sees the UI like a human
- 🚀 **Fast Development** - Write automation in minutes
- 🛡️ **Reliable** - Built-in error handling and retries
- 🌍 **Universal** - One codebase, all platforms
- 📚 **Well-Documented** - Clear examples and guides

### Why Tesseract OCR?

- ✅ **Fast** - ~100ms per screenshot
- ✅ **Accurate** - Industry-standard since 1985
- ✅ **Portable** - Bundle binary with your app
- ✅ **Multi-language** - Supports 100+ languages
- ✅ **Lightweight** - ~10MB binary + language data
- ✅ **Free** - Open source, Apache License 2.0
- ✅ **Battle-tested** - Used by Google, Microsoft, and more

---

## 🤝 Contributing

Contributions are welcome! Whether it's:
- 🐛 Bug reports
- 💡 Feature requests
- 📝 Documentation improvements
- 🔧 Code contributions

Please feel free to open issues and pull requests.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Credits

**Built With:**
- [PyAutoGUI](https://pyautogui.readthedocs.io/) - Mouse and keyboard automation
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - Text recognition engine
- [OpenCV](https://opencv.org/) - Computer vision and template matching
- [Pillow](https://python-pillow.org/) - Image processing

**Inspired By:**
- Ian Fleming's *From Russia with Love*
- SPECTRE's master planner, Kronsteen
- The need for intelligent, vision-based automation

---

<div align="center">

### *"The perfect automation is invisible"*

**Made with ❤️ by automation enthusiasts**

**Star ⭐ this repo if you find it useful!**

[Report Bug](https://github.com/yourusername/kronsteen/issues) · [Request Feature](https://github.com/yourusername/kronsteen/issues) · [Documentation](https://github.com/yourusername/kronsteen)

</div>
