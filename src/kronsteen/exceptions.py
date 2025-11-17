"""Kronsteen specific exception hierarchy."""

from __future__ import annotations


class KronsteenError(Exception):
    """Base error for the automation framework."""


class OCRConfigurationError(KronsteenError):
    """Raised when OCR is misconfigured (e.g., missing dependencies or invalid settings)."""


class OCRServiceError(KronsteenError):
    """Raised when the OCR provider fails to process the request."""


class MatchNotFoundError(KronsteenError):
    """Raised when text, image, or color cannot be located on screen."""


class RegionError(KronsteenError):
    """Raised when an invalid region is supplied."""
