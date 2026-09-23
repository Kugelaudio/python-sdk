"""Detect the KugelAudio AI-generated-audio watermark in a clip.

Audio produced by the API carries an in-band watermark (EU AI Act Art. 50),
added inside the acoustic decode. The detector is bundled with this package
(~152 KB) and runs on numpy, so ``pip install "kugelaudio[watermark]"`` stays
small and works offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence, Union

from ._lite import LiteDetector
from ._payload import PAYLOAD_MIN_MARGIN, PayloadCodec, decode_with_sync
from ._resample import resample
from .errors import (
    WatermarkAudioError,
    WatermarkDependencyError,
    WatermarkModelError,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

DETECTOR_RATE = 24_000
"""The detector runs at the engine's native rate. The mark is produced inside
the acoustic decode at 24 kHz, so — unlike the previous AudioSeal-based
detector — nothing is resampled to 16 kHz first."""

MIN_SECONDS = 1.0
"""Shortest clip that yields a trustworthy verdict. Accuracy rises with
duration because more windows are averaged, and very short runs are unreliable
for both marks. Scoring less raises rather than returning a confident-looking
guess."""

PRESENCE_THRESHOLD = 0.5

PAYLOAD_OFFSET_STEP = 64
"""Sample-offset search granularity used by the payload research harness.

At 24 kHz this evaluates at most 50 phases across one 3,200-sample detector
window. Only offsets backed by a complete 30-window codeword are considered.
"""

# Known limitation: the detector's negative class during training was clean
# speech, so the low false-positive rate (~0.04 measured on real decoded
# speech) holds for speech-like audio. Broadband signals such as white noise
# are out of distribution and can cross the threshold; treat a positive on
# non-speech material as unreliable.

AudioLike = Union["np.ndarray", Sequence[float]]


@dataclass(frozen=True)
class WatermarkResult:
    """Outcome of scoring one clip.

    Attributes:
        ai_generated: Whether the watermark is present above the threshold.
        confidence: Mean presence probability in [0, 1] behind that decision.
        customer_id: Source id from the payload (0-4095) — ``0`` is the
            KugelAudio cloud platform. ``None`` when no watermark was found, or
            when the clip is shorter than the ~4 s a codeword spans.
        duration_seconds: Duration actually scored.
        windows: Number of 133 ms detector windows averaged.
    """

    ai_generated: bool
    confidence: float
    customer_id: Optional[int]
    duration_seconds: float
    windows: Optional[int] = None


def validate_audio(
    audio: AudioLike, sample_rate: int, min_seconds: float = MIN_SECONDS
) -> "tuple[np.ndarray, float]":
    """Check a clip is scoreable and return ``(samples, duration_seconds)``.

    Split out from :meth:`WatermarkDetector.detect` so the contract can be
    exercised without any model. Every branch raises rather than coercing: an
    implicit downmix, pad, or integer rescale would quietly change the verdict
    the caller is about to act on.

    Raises:
        WatermarkAudioError: Audio is multi-channel, non-float, empty, or
            shorter than ``min_seconds``.
    """
    import numpy as np

    samples = np.asarray(audio)
    if samples.ndim != 1:
        raise WatermarkAudioError(
            f"expected mono 1-D audio, got shape {samples.shape}. Select or "
            "mix down a channel explicitly — an implicit downmix would change "
            "the detector's verdict."
        )
    if not np.issubdtype(samples.dtype, np.floating):
        raise WatermarkAudioError(
            f"expected floating-point samples in [-1, 1], got dtype "
            f"{samples.dtype}. Divide integer PCM by its full-scale value "
            "first (e.g. int16 / 32768.0)."
        )
    if samples.size == 0:
        raise WatermarkAudioError("audio is empty")
    if not np.all(np.isfinite(samples)):
        raise WatermarkAudioError("audio samples must all be finite")
    peak = float(np.max(np.abs(samples)))
    if peak > 1.0:
        raise WatermarkAudioError(
            f"audio samples must be in [-1, 1], got peak magnitude {peak:.6g}"
        )
    if sample_rate <= 0:
        raise WatermarkAudioError(f"sample_rate must be positive, got {sample_rate}")

    duration = samples.size / float(sample_rate)
    if duration < min_seconds:
        raise WatermarkAudioError(
            f"clip is {duration:.3f}s; at least {min_seconds:.3f}s is needed for "
            "a reliable verdict. Pass min_seconds= to override."
        )
    return samples, duration


class WatermarkDetector:
    """Scores clips for the KugelAudio watermark.

    Hold on to one instance rather than constructing it per file.
    """

    def __init__(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError as exc:
            raise WatermarkDependencyError(
                "Watermark detection needs optional dependencies; install "
                '"kugelaudio[watermark]".'
            ) from exc
        self._lite = LiteDetector()

    def detect(
        self,
        audio: AudioLike,
        sample_rate: int,
        *,
        min_seconds: float = MIN_SECONDS,
    ) -> WatermarkResult:
        """Score one mono clip.

        Args:
            audio: 1-D float samples in [-1, 1]. Integer PCM must be scaled by
                the caller — guessing the scale here would silently change the
                verdict.
            sample_rate: Rate of ``audio`` in Hz. Resampled internally when it
                differs from the detector's rate.
            min_seconds: Reject clips shorter than this. Lower it only if you
                accept a less reliable answer.

        Raises:
            WatermarkAudioError: The audio is multi-channel, empty, too short,
                or not floating point.
        """
        samples, duration = validate_audio(audio, sample_rate, min_seconds)
        return self._detect_lite(resample(samples, sample_rate, DETECTOR_RATE), duration)

    def _detect_lite(
        self, native_samples: "np.ndarray", duration: float
    ) -> WatermarkResult:
        assert self._lite is not None
        out = self._lite.score(native_samples)
        if not 0.0 <= out.presence <= 1.0:
            raise WatermarkModelError(
                f"watermark model returned invalid presence score {out.presence}"
            )
        present = out.presence > PRESENCE_THRESHOLD
        customer_id: Optional[int] = None
        if present:
            # The id is a codeword spread over CODEWORD_CHUNKS chunks, not two
            # raw bits. Shorter clips report presence but no id rather than
            # naming the wrong customer.
            codec = PayloadCodec()
            if out.bit_logits is not None and out.windows >= codec.chunks:
                needed_samples = codec.chunks * self._lite.window_samples
                max_offset = min(
                    self._lite.window_samples - 1,
                    native_samples.size - needed_samples,
                )
                logits_by_offset = {
                    0: out.bit_logits[: codec.chunks].reshape(-1),
                }
                for offset in range(
                    PAYLOAD_OFFSET_STEP,
                    max_offset + 1,
                    PAYLOAD_OFFSET_STEP,
                ):
                    segment = native_samples[offset : offset + needed_samples]
                    offset_out = self._lite.score(segment)
                    if offset_out.bit_logits is None:
                        raise WatermarkModelError(
                            "watermark model omitted payload logits"
                        )
                    logits_by_offset[offset] = offset_out.bit_logits[
                        : codec.chunks
                    ].reshape(-1)
                decoded, _ = decode_with_sync(
                    codec,
                    logits_by_offset,
                    min_margin=PAYLOAD_MIN_MARGIN,
                )
                customer_id = decoded.customer_id
        return WatermarkResult(
            ai_generated=present,
            confidence=out.presence,
            customer_id=customer_id,
            duration_seconds=duration,
            windows=out.windows,
        )


    def detect_file(
        self, path: str, *, min_seconds: float = MIN_SECONDS
    ) -> WatermarkResult:
        """Score an audio file (any format libsndfile reads).

        Raises:
            WatermarkDependencyError: ``soundfile`` is not installed.
            WatermarkAudioError: The file is multi-channel or too short.
        """
        try:
            import soundfile
        except ImportError as exc:
            raise WatermarkDependencyError(
                'Reading audio files needs "kugelaudio[watermark]" (soundfile).'
            ) from exc

        samples, rate = soundfile.read(path, dtype="float32", always_2d=False)
        if samples.ndim != 1:
            raise WatermarkAudioError(
                f"{path} has {samples.shape[-1]} channels; the detector scores "
                "mono. Select a channel explicitly before scoring."
            )
        return self.detect(samples, rate, min_seconds=min_seconds)


__all__ = [
    "DETECTOR_RATE",
    "MIN_SECONDS",
    "PAYLOAD_OFFSET_STEP",
    "PRESENCE_THRESHOLD",
    "WatermarkDetector",
    "WatermarkResult",
    "validate_audio",
]
