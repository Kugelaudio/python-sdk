"""Contract tests for the optional watermark extra.

The point of the extra is that verifying a clip costs a numpy install rather
than a torch install, so the first tests here guard exactly that. Everything
below runs on numpy alone — the bundled lite detector needs nothing else.

True-positive detection (marked audio in, right payload out) needs the
generator and lives in `models/tts/scripts/debug/watermark_lite_numpy_parity.py`,
which also asserts parity against the torch reference on a GPU box.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from kugelaudio.watermark import (
    DETECTOR_RATE,
    WatermarkAudioError,
    WatermarkDetector,
    WatermarkModelError,
)
from kugelaudio.watermark._detector import validate_audio
from kugelaudio.watermark._lite import (
    LiteDetector,
    LiteDetectorOutput,
    _asset_path,
)
from kugelaudio.watermark._payload import PayloadCodec
from kugelaudio.watermark._resample import resample


def _run(code: str) -> str:
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def test_importing_the_base_sdk_stays_light() -> None:
    """The whole reason the extra exists: `import kugelaudio` must stay thin."""
    heavy = _run(
        "import sys; import kugelaudio; "
        "print(','.join(m for m in ('torch','audioseal','numpy') if m in sys.modules))"
    )
    assert heavy == "", f"base import pulled: {heavy}"


def test_detecting_never_imports_torch() -> None:
    """Constructing and running the default detector must not pull a tensor
    runtime — that is the promise the bundled numpy model exists to keep."""
    out = _run(
        "import sys, numpy as np; from kugelaudio.watermark import WatermarkDetector; "
        "d = WatermarkDetector(); "
        "d.detect(np.zeros(32000, dtype=np.float32), 16000); "
        "print('torch' in sys.modules or 'audioseal' in sys.modules)"
    )
    assert out == "False"


class TestBundledLiteDetector:


    def test_a_tampered_asset_fails_loudly(self, tmp_path) -> None:
        """A swapped model would silently change every verdict."""
        corrupt = tmp_path / "corrupt.npz"
        corrupt.write_bytes(_asset_path().read_bytes() + b"\x00")
        with pytest.raises(WatermarkModelError, match="SHA-256"):
            LiteDetector(asset_path=corrupt)

    def test_a_missing_asset_fails_loudly(self, tmp_path) -> None:
        with pytest.raises(WatermarkModelError, match="missing"):
            LiteDetector(asset_path=tmp_path / "absent.npz")

    def test_ambiguous_ood_positive_has_no_customer_id(self) -> None:
        """Presence on OOD audio must not become false source attribution."""
        rng = np.random.default_rng(1)
        noise = (rng.standard_normal(DETECTOR_RATE * 4) * 0.2).astype(np.float32)
        result = WatermarkDetector().detect(noise, DETECTOR_RATE)
        assert result.ai_generated
        assert result.customer_id is None


    def test_clip_shorter_than_one_window_is_rejected(self) -> None:
        det = LiteDetector()
        with pytest.raises(WatermarkModelError, match="shorter than one detector"):
            det.score(np.zeros(2000, dtype=np.float32))


class TestValidateAudio:

    def test_rejects_multichannel_instead_of_downmixing(self) -> None:
        with pytest.raises(WatermarkAudioError, match="mono"):
            validate_audio(np.zeros((24_000, 2), dtype=np.float32), 24_000)

    def test_rejects_integer_pcm_instead_of_guessing_the_scale(self) -> None:
        with pytest.raises(WatermarkAudioError, match="int16"):
            validate_audio(np.zeros(24_000, dtype=np.int16), 24_000)


    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_nonfinite_samples(self, value: float) -> None:
        samples = np.zeros(24_000, dtype=np.float32)
        samples[100] = value
        with pytest.raises(WatermarkAudioError, match="finite"):
            validate_audio(samples, 24_000)

    @pytest.mark.parametrize("value", [1.0001, -1.0001])
    def test_rejects_float_samples_outside_pcm_range(self, value: float) -> None:
        samples = np.zeros(24_000, dtype=np.float32)
        samples[100] = value
        with pytest.raises(WatermarkAudioError, match=r"\[-1, 1\]"):
            validate_audio(samples, 24_000)

    def test_rejects_a_clip_too_short_to_score_reliably(self) -> None:
        with pytest.raises(WatermarkAudioError, match="at least"):
            validate_audio(np.zeros(2_400, dtype=np.float32), 24_000)

    def test_min_seconds_override_is_honoured(self) -> None:
        _, duration = validate_audio(
            np.zeros(2_400, dtype=np.float32), 24_000, min_seconds=0.05
        )
        assert duration == pytest.approx(0.1)


class TestResample:


    def test_preserves_a_tone_through_24k_to_16k(self) -> None:
        """A 1 kHz tone must survive at the right frequency and amplitude —
        the property detection actually depends on."""
        freq = 1000.0
        t = np.arange(24_000) / 24_000
        tone = np.sin(2 * np.pi * freq * t).astype(np.float32)

        trimmed = resample(tone, 24_000, 16_000)[500:-500]

        spectrum = np.abs(np.fft.rfft(trimmed))
        peak_hz = np.fft.rfftfreq(trimmed.size, 1 / 16_000)[int(np.argmax(spectrum))]
        assert peak_hz == pytest.approx(freq, abs=15.0)
        rms = float(np.sqrt(np.mean(trimmed.astype(np.float64) ** 2)))
        assert rms == pytest.approx(np.sqrt(0.5), rel=0.05)


    def test_44100_to_24k_does_not_allocate_an_explicit_upsample(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A minute-long common ratio must keep intermediates bounded."""
        samples = np.zeros(44_100 * 60, dtype=np.float32)
        original_zeros = np.zeros

        def bounded_zeros(
            shape: int | tuple[int, ...], *args: object, **kwargs: object
        ) -> np.ndarray:
            size = int(np.prod(shape))
            assert size < 500_000, f"unbounded zero allocation of {size} elements"
            return original_zeros(shape, *args, **kwargs)

        monkeypatch.setattr(np, "zeros", bounded_zeros)
        output = resample(samples, 44_100, 24_000)
        assert output.shape == (24_000 * 60,)
        assert output.dtype == np.float32
        assert np.count_nonzero(output) == 0


class TestPayloadSynchronization:
    def test_fractional_window_offset_recovers_the_payload(self) -> None:
        codec = PayloadCodec()
        expected_customer_id = 731
        symbols = codec._bipolar[expected_customer_id].reshape(codec.chunks, 2)
        window_samples = 3_200
        offset = 64
        samples = np.zeros(codec.chunks * window_samples + offset, dtype=np.float32)
        for chunk, symbol in enumerate(symbols):
            start = offset + chunk * window_samples
            samples[start : start + 2] = symbol

        class FakeLiteDetector:
            window_samples = 3_200

            def score(self, clip: np.ndarray) -> LiteDetectorOutput:
                windows = clip.size // self.window_samples
                framed = clip[: windows * self.window_samples].reshape(
                    windows, self.window_samples
                )
                logits = framed[:, :2]
                return LiteDetectorOutput(
                    presence=0.9,
                    bits=(0, 0),
                    windows=windows,
                    bit_logits=logits,
                )

        detector = WatermarkDetector.__new__(WatermarkDetector)
        detector._lite = FakeLiteDetector()
        result = detector._detect_lite(samples, samples.size / DETECTOR_RATE)

        assert result.customer_id == expected_customer_id
