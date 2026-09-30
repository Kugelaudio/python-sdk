"""KugelAudio speech enhancement as a Pipecat input audio filter.

Example:
    from pipecat.transports.base_transport import TransportParams
    from kugelaudio.pipecat import KugelAudioEnhanceFilter

    params = TransportParams(
        audio_in_enabled=True,
        audio_in_filter=KugelAudioEnhanceFilter(),  # reads KUGELAUDIO_API_KEY
    )
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

from kugelaudio._diagnostics import INTEGRATION_PIPECAT
from kugelaudio._enhance_live import (
    DEFAULT_ENHANCE_MODEL,
    LiveEnhancer,
    PlayoutBuffer,
    client_from_options,
)

if TYPE_CHECKING:
    from kugelaudio.audio import Audio
    from kugelaudio.client import KugelAudio

# How far the output may trail the input beyond max_backlog_s before the
# oldest audio is dropped (enhanced audio arrives in bursts).
_PLAYOUT_HEADROOM_S = 0.5


def _stream_resampler() -> BaseAudioResampler:
    from pipecat.audio import utils

    # create_stream_resampler keeps state across chunks (pipecat >= 0.0.101);
    # older releases only have the per-chunk default resampler.
    factory = getattr(utils, "create_stream_resampler", None) or getattr(
        utils, "create_default_resampler"
    )
    resampler: BaseAudioResampler = factory()
    return resampler


class KugelAudioEnhanceFilter(BaseAudioFilter):
    """Removes background noise (or every voice but ``speaker``) from the
    user's audio before VAD and STT, using KugelAudio Clarity over the network.

    Pass it as ``TransportParams(audio_in_filter=...)``. The filter never
    waits on the network: each frame comes back at the same size, delayed by
    the network round trip and the server's processing. While
    the stream connects, reconnects, or after a fatal error (bad key, no
    credits) it returns the original audio and logs why.
    ``FilterEnableFrame(False)`` switches it to pass-through and closes the
    stream; ``FilterEnableFrame(True)`` reopens it.

    Args:
        api_key: KugelAudio API key; defaults to ``KUGELAUDIO_API_KEY``.
        client: An existing :class:`~kugelaudio.KugelAudio` client, instead of
            ``api_key``/``base_url``.
        model: Enhancement model.
        speaker: A clean 2-8 s sample of the one voice to keep, from
            :func:`~kugelaudio.load_audio`. Without it, noise is removed and
            every voice is kept.
        base_url: API base URL override.
        max_backlog_s: How far the server may fall behind before the filter
            falls back to the original audio and reconnects.
        on_error: Called once with the error that disabled enhancement.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: KugelAudio | None = None,
        model: str = DEFAULT_ENHANCE_MODEL,
        speaker: Audio | None = None,
        base_url: str | None = None,
        max_backlog_s: float = 2.0,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self._client = client_from_options(
            api_key=api_key,
            client=client,
            base_url=base_url,
            integration=INTEGRATION_PIPECAT,
        )
        self._model = model
        self._speaker = speaker
        self._max_backlog_s = max_backlog_s
        self._on_error = on_error
        self._enabled = True
        self._drain_pending = False
        self._sample_rate: int | None = None
        self._enhancer: LiveEnhancer | None = None
        self._playout: PlayoutBuffer | None = None
        self._resampler: BaseAudioResampler | None = None

    @property
    def enhancer(self) -> LiveEnhancer | None:
        """The running enhancer (state, failure count), or ``None`` before
        :meth:`start`."""
        return self._enhancer

    async def start(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self._enhancer = LiveEnhancer(
            self._client,
            model=self._model,
            speaker=self._speaker,
            max_backlog_s=self._max_backlog_s,
            on_error=self._on_error,
        )
        self._playout = PlayoutBuffer(
            sample_rate, max_delay_s=self._max_backlog_s + _PLAYOUT_HEADROOM_S
        )
        self._resampler = _stream_resampler()
        self._enhancer.prewarm()

    async def stop(self) -> None:
        if self._enhancer is not None:
            await self._enhancer.aclose()

    async def process_frame(self, frame: FilterControlFrame) -> None:
        if isinstance(frame, FilterEnableFrame):
            if not frame.enable and self._enabled and self._enhancer is not None:
                # Hand back what is still delayed on the next frame, then run
                # without delay; unanswered input returns as original audio.
                self._enhancer.pause()
                self._drain_pending = True
            elif frame.enable:
                self._drain_pending = False
            self._enabled = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        enhancer, playout = self._enhancer, self._playout
        if enhancer is None or playout is None or self._sample_rate is None:
            raise RuntimeError("KugelAudioEnhanceFilter.filter() called before start().")
        if not self._enabled:
            if not self._drain_pending:
                return audio
            self._drain_pending = False
            await self._write(enhancer, playout)
            return playout.drain() + audio
        enhancer.push(audio, self._sample_rate)
        await self._write(enhancer, playout)
        return playout.read(len(audio))

    async def _write(self, enhancer: LiveEnhancer, playout: PlayoutBuffer) -> None:
        for segment in enhancer.pull():
            pcm = segment.pcm
            if segment.sample_rate != playout.sample_rate:
                assert self._resampler is not None  # set in start()
                pcm = await self._resampler.resample(
                    pcm, segment.sample_rate, playout.sample_rate
                )
            playout.write(pcm)
