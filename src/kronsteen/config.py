"""Global settings for Kronsteen."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import Optional


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class KronsteenSettings:
    """Runtime knobs for automation behavior."""

    default_timeout: float = 10.0
    retry_interval: float = 0.5
    screenshot_dir: Optional[Path] = None
    fail_safe: bool = True
    default_pause: float = 0.1
    confidence: float = 0.8

    @classmethod
    def from_env(cls) -> "KronsteenSettings":
        screenshot_dir = os.getenv("KRONSTEEN_SCREENSHOT_DIR")
        return cls(
            default_timeout=float(os.getenv("KRONSTEEN_DEFAULT_TIMEOUT", "10")),
            retry_interval=float(os.getenv("KRONSTEEN_RETRY_INTERVAL", "0.5")),
            screenshot_dir=Path(screenshot_dir).expanduser() if screenshot_dir else None,
            fail_safe=_bool_env("KRONSTEEN_FAILSAFE", True),
            default_pause=float(os.getenv("KRONSTEEN_DEFAULT_PAUSE", "0.1")),
            confidence=float(os.getenv("KRONSTEEN_DEFAULT_CONFIDENCE", "0.8")),
        )


@dataclass
class OCRSettings:
    """Configuration for the DeepSeek local OCR client."""

    model_name: str = "deepseek-ai/DeepSeek-OCR"
    prompt: str = "<image>\\nFree OCR."
    device: str = "auto"  # Auto-detect: use CUDA if available, otherwise CPU
    torch_dtype: Optional[str] = None  # Auto-detect based on device
    base_size: int = 1024
    image_size: int = 640
    crop_mode: bool = True
    save_results: bool = True
    test_compress: bool = True
    attn_implementation: Optional[str] = None  # Don't use flash_attention by default
    trust_remote_code: bool = True
    output_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None

    @classmethod
    def from_env(cls) -> "OCRSettings":
        output_dir = os.getenv("DEEPSEEK_OUTPUT_DIR")
        cache_dir = os.getenv("DEEPSEEK_CACHE_DIR")
        torch_dtype = os.getenv("DEEPSEEK_TORCH_DTYPE")
        if torch_dtype and torch_dtype.lower() == "none":
            torch_dtype = None
        attn_impl = os.getenv("DEEPSEEK_ATTN_IMPL")
        if attn_impl and attn_impl.lower() == "none":
            attn_impl = None
        # Auto-detect device if set to "auto"
        device_env = os.getenv("DEEPSEEK_DEVICE", "auto")
        if device_env == "auto":
            # Try to import torch to check CUDA availability
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        else:
            device = device_env
        
        return cls(
            model_name=os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-ai/DeepSeek-OCR"),
            prompt=os.getenv("DEEPSEEK_PROMPT", "<image>\\nFree OCR."),
            device=device,  # Auto-detect or use specified device
            torch_dtype=torch_dtype,  # None by default (auto-detect)
            base_size=int(os.getenv("DEEPSEEK_BASE_SIZE", "1024")),
            image_size=int(os.getenv("DEEPSEEK_IMAGE_SIZE", "640")),
            crop_mode=_bool_env("DEEPSEEK_CROP_MODE", True),
            save_results=_bool_env("DEEPSEEK_SAVE_RESULTS", True),
            test_compress=_bool_env("DEEPSEEK_TEST_COMPRESS", True),
            attn_implementation=attn_impl,  # None by default (no flash attention)
            trust_remote_code=_bool_env("DEEPSEEK_TRUST_REMOTE_CODE", True),
            output_dir=Path(output_dir).expanduser() if output_dir else None,
            cache_dir=Path(cache_dir).expanduser() if cache_dir else None,
        )
