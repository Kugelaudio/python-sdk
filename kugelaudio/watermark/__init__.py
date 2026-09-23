"""Local verification of the KugelAudio AI-generated-audio watermark.

Optional: ``pip install "kugelaudio[watermark]"``.

    from kugelaudio.watermark import WatermarkDetector

    detector = WatermarkDetector()
    result = detector.detect_file("speech.wav")
    if result.ai_generated:
        print(result.confidence, result.customer_id)
"""

from .errors import (
    WatermarkAudioError,
    WatermarkDependencyError,
    WatermarkError,
    WatermarkModelError,
)
from ._detector import (
    DETECTOR_RATE,
    MIN_SECONDS,
    PRESENCE_THRESHOLD,
    WatermarkDetector,
    WatermarkResult,
)

__all__ = [
    "DETECTOR_RATE",
    "MIN_SECONDS",
    "PRESENCE_THRESHOLD",
    "WatermarkAudioError",
    "WatermarkDependencyError",
    "WatermarkDetector",
    "WatermarkError",
    "WatermarkModelError",
    "WatermarkResult",
]
