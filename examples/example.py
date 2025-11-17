"""Example automation script using Kronsteen with logging and window monitoring."""

import kronsteen


def main() -> None:
    """
    Example: Launch Chrome and perform a Google search using OCR.
    
    Features:
    - Universal launcher: works on macOS, Windows, and Linux
    - OCR: Choose between Tesseract (fast) or DeepSeek (accurate)
    - Retina display support: automatically handles scaling
    - Automatic logging and screenshots
    - Window focus monitoring: pauses if Chrome loses focus
    """
    # Setup logging - creates logs/ folder with logs/ and screenshots/ subfolders
    kronsteen.setup_logging(
        log_dir="logs",  # Creates logs/logs/ and logs/screenshots/
        enable_screenshots=False  # Disable screenshots for this example
    )
    
    logger = kronsteen.get_logger()
    logger.info("Starting Google search automation")
    logger.info("=" * 80)
    
    # ========== OCR ENGINE SELECTION ==========
    # Choose your OCR engine:
    
    # Option 1: Tesseract OCR (Default - Fast, lightweight, CPU-based)
    # RECOMMENDED FOR MACOS - DeepSeek has issues on CPU-only systems
    kronsteen.use_ocr_engine("tesseract")
    logger.info("Using Tesseract OCR (fast, CPU-based)")
    
    # Option 2: DeepSeek OCR (More accurate, auto-detects GPU/CPU)
    # NOTE: Currently has compatibility issues on macOS (CPU-only)
    # Works best on Windows/Linux with NVIDIA GPU
    # kronsteen.use_ocr_engine("deepseek")
    # logger.info("Using DeepSeek OCR (accurate, auto-detects GPU/CPU)")
    # logger.info("Note: First run will download the model (~2GB)")
    
    # ==========================================
    
    kronsteen.configure(default_timeout=25)
    
    # Launch Chrome (works on all platforms!)
    logger.info("Launching Chrome...")
    kronsteen.launch("Chrome")
    kronsteen.sleep(3)
    
    # Start monitoring Chrome window focus
    # If Chrome loses focus, automation will pause automatically
    logger.info("Starting window focus monitoring for Chrome")
    logger.info("TIP: Try switching to another app - automation will pause!")
    monitor = kronsteen.start_window_monitoring(
        window_name="Chrome",
        check_interval=0.5,  # Check every 0.5 seconds
        logger=logger
    )
    
    # Wait for Chrome to load
    kronsteen.wait_for_text("Google", timeout=30, match_mode="contains")
    
    logger.info("Chrome loaded successfully!")
    
    # Type search query in address bar
    kronsteen.hotkey("command", "l")  # Focus address bar (Cmd+L on macOS)
    kronsteen.sleep(0.5)
    kronsteen.type_text("Tesseract OCR", press_enter=True)
    
    # Wait for search results
    kronsteen.sleep(3)
    
    logger.info("Google search automation completed successfully!")
    
    # Stop window monitoring
    logger.info("Stopping window focus monitoring")
    kronsteen.stop_window_monitoring()
    
    # Close Chrome when done
    kronsteen.sleep(2)  # Wait a moment to see results
    kronsteen.close_app("Chrome")
    
    logger.info("=" * 80)
    logger.info("✓ Automation completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    """
    Quick Start Guide:
    
    1. To use Tesseract OCR (RECOMMENDED - Current setup):
       - Keep lines 32-33 uncommented
       - Works on all platforms (macOS, Windows, Linux)
       - Fast and reliable
    
    2. To use DeepSeek OCR (Advanced - GPU recommended):
       - Comment out lines 32-33
       - Uncomment lines 38-40
       - Requires: pip install torch transformers
       - ⚠️  WARNING: Currently has issues on macOS (CPU-only)
       - Works best on Windows/Linux with NVIDIA GPU
       - First run downloads ~2GB model
    
    3. Platform Compatibility:
       - Tesseract: ✅ macOS, ✅ Windows, ✅ Linux (all CPU)
       - DeepSeek: ⚠️ macOS (issues), ✅ Windows (GPU), ✅ Linux (GPU)
    
    4. Performance:
       - Tesseract: ~100ms per OCR operation (CPU)
       - DeepSeek: ~500ms (GPU) or very slow (CPU)
    """
    main()
