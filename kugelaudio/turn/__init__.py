"""Local, CPU-native semantic turn detection."""

from ._bundle import DEFAULT_REPO_ID, DEFAULT_REVISION, download_bundle, verify_bundle
from ._detector import (
    TurnDecision,
    TurnDecisionReason,
    TurnDetector,
    TurnLanguage,
    TurnOutcomeKind,
    TurnSession,
)
from ._runtime import TurnMessage, TurnPredictor, TurnProbabilities
from .errors import (
    TurnAudioError,
    TurnBundleError,
    TurnDependencyError,
    TurnDetectionError,
    TurnModelDownloadError,
    TurnStateError,
    UnsupportedTurnLanguageError,
)

__all__ = [
    "DEFAULT_REPO_ID",
    "DEFAULT_REVISION",
    "TurnAudioError",
    "TurnBundleError",
    "TurnDecision",
    "TurnDecisionReason",
    "TurnDependencyError",
    "TurnDetectionError",
    "TurnDetector",
    "TurnLanguage",
    "TurnMessage",
    "TurnModelDownloadError",
    "TurnOutcomeKind",
    "TurnPredictor",
    "TurnProbabilities",
    "TurnSession",
    "TurnStateError",
    "UnsupportedTurnLanguageError",
    "download_bundle",
    "verify_bundle",
]
