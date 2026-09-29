"""Load WAV files for speech enhancement.

``load_audio`` loads a whole recording for ``client.enhance.generate``;
``load_audio_stream`` loads one as PCM16 chunks for ``client.enhance.stream``.
"""

from __future__ import annotations

import io
import os
import sys
import wave
from array import array
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from kugelaudio.exceptions import ValidationError

AudioSource = Union[str, os.PathLike, bytes]
"""A WAV file path or the WAV file's bytes."""

_FORMAT_HINT = (
    "Use a 16-bit PCM WAV, or client.enhance.generate(load_audio(...)), "
    "which accepts any supported WAV."
)


def _read_source(source: AudioSource) -> tuple[str, bytes]:
    """Return ``(filename, bytes)`` for a WAV path or WAV bytes."""
    if isinstance(source, (bytes, bytearray)):
        filename, data = "audio.wav", bytes(source)
    elif isinstance(source, (str, os.PathLike)):
        path = Path(source)
        try:
            data = path.read_bytes()
        except OSError as e:
            raise ValidationError(f"Cannot read audio file {path}: {e}") from e
        filename = path.name
    else:
        raise TypeError(
            f"Audio source must be a WAV path or WAV bytes, got {type(source).__name__}."
        )
    if not data:
        raise ValidationError("Audio must not be empty.")
    return filename, data


@dataclass(frozen=True)
class Audio:
    """A WAV recording loaded with :func:`load_audio`."""

    data: bytes
    """The WAV file bytes, sent as-is."""
    filename: str
    """File name sent with the upload."""
    duration: float | None = None
    """Duration in seconds when the header is readable here, else ``None``."""

    def __repr__(self) -> str:
        return (
            f"Audio(filename={self.filename!r}, bytes={len(self.data)}, "
            f"duration={self.duration!r})"
        )


def load_audio(source: AudioSource) -> Audio:
    """Load a WAV file for ``client.enhance.generate``.

    Any WAV the API supports is accepted (PCM 16/24/32-bit or float, mono or
    stereo, 8-48 kHz, at most 300 s); the server decodes it.

    Args:
        source: WAV file path or WAV file bytes.

    Example:
        audio = load_audio("meeting.wav")
        result = await client.enhance.generate(audio, model="clarity-1")
    """
    filename, data = _read_source(source)
    duration: float | None = None
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            if reader.getframerate() > 0:
                duration = reader.getnframes() / reader.getframerate()
    except (wave.Error, EOFError):
        pass  # e.g. float WAVs: the server decodes them
    return Audio(data=data, filename=filename, duration=duration)


def _downmix(frames: bytes, channels: int) -> bytes:
    """Average interleaved PCM16 channels into mono."""
    samples = array("h", frames)
    if sys.byteorder == "big":
        samples.byteswap()
    lanes = [samples[c::channels] for c in range(channels)]
    mono = array("h", (sum(frame) // channels for frame in zip(*lanes)))
    if sys.byteorder == "big":
        mono.byteswap()
    return mono.tobytes()


class AudioStream:
    """Mono PCM16 audio served in fixed-length chunks.

    Iterate it for ``bytes`` chunks; iterating again starts from the beginning.
    Create one with :func:`load_audio_stream`.
    """

    def __init__(self, pcm: bytes, sample_rate: int, chunk_seconds: float = 0.1):
        self._pcm = pcm
        self.sample_rate = sample_rate
        """Sample rate of the audio in Hz."""
        self._chunk_bytes = max(1, int(sample_rate * chunk_seconds)) * 2

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self._pcm) // 2 / self.sample_rate

    def __iter__(self) -> Iterator[bytes]:
        step = self._chunk_bytes
        for start in range(0, len(self._pcm), step):
            yield self._pcm[start : start + step]


def load_audio_stream(
    source: AudioSource, *, chunk_seconds: float = 0.1
) -> AudioStream:
    """Load a 16-bit PCM WAV file for ``client.enhance.stream``.

    Stereo (or multi-channel) audio is downmixed to mono. The sample rate is
    read from the file.

    Args:
        source: WAV file path or WAV file bytes.
        chunk_seconds: Length of each chunk in seconds, greater than 0 and at
            most 1 (default 0.1). Chunks are sent as fast as the connection
            allows; small ones let enhanced audio start coming back sooner.

    Example:
        audio = load_audio_stream("meeting.wav")
        async for chunk in client.enhance.stream(audio, model="clarity-1"):
            ...
    """
    if not 0 < chunk_seconds <= 1:
        raise ValidationError("chunk_seconds must be greater than 0 and at most 1.")
    _, data = _read_source(source)
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            channels, width = reader.getnchannels(), reader.getsampwidth()
            rate = reader.getframerate()
            frames = reader.readframes(reader.getnframes())
    except (wave.Error, EOFError) as e:
        raise ValidationError(f"Cannot read the WAV file ({e}). {_FORMAT_HINT}") from e
    if width != 2:
        raise ValidationError(f"The WAV file is {8 * width}-bit. {_FORMAT_HINT}")
    if not frames:
        raise ValidationError("Audio must not be empty.")
    if channels > 1:
        frames = _downmix(frames, channels)
    return AudioStream(frames, rate, chunk_seconds)


__all__ = ["Audio", "AudioSource", "AudioStream", "load_audio", "load_audio_stream"]
