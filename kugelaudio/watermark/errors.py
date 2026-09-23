"""Typed errors for watermark detection."""

from __future__ import annotations


class WatermarkError(Exception):
    """Base class for every watermark-detection failure."""


class WatermarkDependencyError(WatermarkError):
    """The optional detection dependencies are not installed."""


class WatermarkAudioError(WatermarkError):
    """The supplied audio cannot be scored as given.

    Raised instead of silently coercing the input (downmixing channels,
    zero-padding a too-short clip, guessing an integer scale), because each
    of those changes the answer the detector returns.
    """


class WatermarkModelError(WatermarkError):
    """The detector model could not be loaded."""


__all__ = [
    "WatermarkAudioError",
    "WatermarkDependencyError",
    "WatermarkError",
    "WatermarkModelError",
]
