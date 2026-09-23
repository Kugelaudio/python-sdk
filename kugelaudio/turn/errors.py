"""Typed failures raised by the local turn-detection runtime."""

from __future__ import annotations

from kugelaudio.exceptions import KugelAudioError


class TurnDetectionError(KugelAudioError):
    """Base failure for local semantic turn detection."""


class TurnDependencyError(TurnDetectionError):
    """The optional CPU runtime dependencies are not installed."""


class TurnModelDownloadError(TurnDetectionError):
    """The requested model revision could not be acquired from Hugging Face."""


class TurnBundleError(TurnDetectionError):
    """A downloaded model bundle is corrupt or incompatible."""


class TurnAudioError(TurnDetectionError):
    """Audio supplied to the detector violates its explicit input contract."""


class UnsupportedTurnLanguageError(TurnDetectionError):
    """No validated endpoint policy exists for the requested language."""


class TurnStateError(TurnDetectionError):
    """Turn-session methods were called in an invalid lifecycle order."""
