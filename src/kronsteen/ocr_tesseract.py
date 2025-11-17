"""Tesseract OCR client - lightweight alternative to DeepSeek."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Optional

import pyautogui
from PIL import Image

from .models import Region, TextMatch


class TesseractOCRClient:
    """Lightweight OCR using Tesseract (much faster than DeepSeek)."""
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """
        Initialize Tesseract OCR client.
        
        Args:
            tesseract_path: Optional path to tesseract binary. If not provided,
                           will look for bundled binary in project folder.
        """
        try:
            import pytesseract
            self.pytesseract = pytesseract
        except ImportError:
            raise ImportError(
                "pytesseract not installed. Install with: pip install pytesseract\n"
                "Download Tesseract binary from: https://github.com/tesseract-ocr/tesseract/releases"
            )
        
        # Set tesseract command path
        self._set_tesseract_path(tesseract_path)
        
        # Detect Retina display scaling on macOS
        self.scale_factor = self._detect_scale_factor()
    
    def _set_tesseract_path(self, custom_path: Optional[str] = None) -> None:
        """
        Set the path to tesseract binary.
        
        Priority:
        1. Custom path provided by user
        2. Bundled binary in project folder (./tesseract/)
        3. System installation (brew, apt, etc.)
        """
        if custom_path:
            self.pytesseract.pytesseract.tesseract_cmd = custom_path
            return
        
        # Look for bundled tesseract in project folder
        bundled_paths = [
            # Relative to current working directory
            Path.cwd() / "tesseract" / "tesseract",
            Path.cwd() / "tesseract" / "tesseract.exe",
            # Relative to this file
            Path(__file__).parent.parent.parent / "tesseract" / "tesseract",
            Path(__file__).parent.parent.parent / "tesseract" / "tesseract.exe",
        ]
        
        for path in bundled_paths:
            if path.exists():
                self.pytesseract.pytesseract.tesseract_cmd = str(path)
                print(f"✓ Using bundled Tesseract: {path}")
                return
        
        # Fall back to system installation
        # pytesseract will use default system path
        print("ℹ️  Using system Tesseract installation")
    
    def _detect_scale_factor(self) -> float:
        """
        Detect display scale factor (for Retina displays).
        
        On macOS Retina displays, screenshots are 2x the logical screen size.
        """
        if platform.system() != "Darwin":
            return 1.0
        
        # Take a small screenshot to check actual vs logical size
        screen_width, screen_height = pyautogui.size()
        screenshot = pyautogui.screenshot()
        
        # Compare screenshot size to logical screen size
        scale_x = screenshot.width / screen_width
        scale_y = screenshot.height / screen_height
        
        # Use the average (should be 2.0 on Retina, 1.0 on non-Retina)
        scale = (scale_x + scale_y) / 2.0
        
        return scale
    
    def extract_text(
        self,
        image: Image.Image,
        *,
        region: Optional[Region] = None,
        prompt: Optional[str] = None,
    ) -> list[TextMatch]:
        """
        Extract text from image using Tesseract OCR.
        
        Args:
            image: PIL Image to extract text from
            region: Optional region offset (for coordinate adjustment)
            prompt: Ignored (for compatibility with DeepSeek interface)
        
        Returns:
            List of TextMatch objects with text and coordinates
        """
        offset_x = region.left if region else 0
        offset_y = region.top if region else 0
        
        # Get OCR data with bounding boxes
        data = self.pytesseract.image_to_data(
            image,
            output_type=self.pytesseract.Output.DICT
        )
        
        matches: list[TextMatch] = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if not text:  # Skip empty text
                continue
            
            conf = float(data['conf'][i])
            if conf < 0:  # Skip invalid confidence
                continue
            
            # Get bounding box (in screenshot coordinates)
            x = int(data['left'][i])
            y = int(data['top'][i])
            w = int(data['width'][i])
            h = int(data['height'][i])
            
            # Scale coordinates to logical screen coordinates (for Retina displays)
            x_logical = int(x / self.scale_factor)
            y_logical = int(y / self.scale_factor)
            w_logical = int(w / self.scale_factor)
            h_logical = int(h / self.scale_factor)
            
            # Create region with offset (in logical coordinates)
            text_region = Region(
                left=offset_x + x_logical,
                top=offset_y + y_logical,
                width=w_logical,
                height=h_logical
            )
            
            # Normalize confidence to 0-1 range (Tesseract gives 0-100)
            confidence = conf / 100.0
            
            matches.append(TextMatch(
                text=text,
                confidence=confidence,
                region=text_region
            ))
        
        return matches
    
    def close(self) -> None:
        """Close client (no-op for Tesseract)."""
        pass
    
    def __enter__(self) -> "TesseractOCRClient":
        return self
    
    def __exit__(self, *exc_info: object) -> None:
        self.close()
