"""KugelAudio speech enhancement as a LiveKit Agents noise-cancellation processor.

Example:
    from livekit.agents import room_io
    from kugelaudio.livekit import enhancement

    await session.start(
        room=ctx.room,
        agent=agent,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(noise_cancellation=enhancement()),
        ),
    )
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from livekit import rtc

from kugelaudio._diagnostics import INTEGRATION_LIVEKIT
from kugelaudio._enhance_live import (
    DEFAULT_ENHANCE_MODEL,
    LiveEnhancer,
    LiveEnhancerState,
    LiveSegment,
    PlayoutBuffer,
    client_from_options,
)
from kugelaudio.enhance import ENHANCED_SAMPLE_RATE
from kugelaudio.exceptions import ValidationError

if TYPE_CHECKING:
    from kugelaudio.audio import Audio
    from kugelaudio.client import KugelAudio

if not hasattr(rtc, "FrameProcessor"):
    raise ImportError(
        "KugelAudio enhancement for LiveKit needs livekit-agents>=1.3.10 "
        "(rtc.FrameProcessor). Upgrade with: pip install -U 'livekit-agents>=1.3.10'"
    )

# How far the output may trail the input beyond max_backlog_s before the
# oldest audio is dropped (enhanced audio arrives in bursts).
_PLAYOUT_HEADROOM_S = 0.5


class EnhanceFrameProcessor(rtc.FrameProcessor[rtc.AudioFrame]):
    """Removes background noise (or every voice but ``speaker``) from the
    participant's audio, using KugelAudio Clarity over the network.

    Build it with :func:`enhancement`. LiveKit calls :meth:`_process` on the
    event loop for every input frame; it never waits on the network and
    returns a frame of the same size, delayed by the network round trip and
    the server's processing. While the stream connects,
    reconnects, or after a fatal error (bad key, no credits) it returns the
    original audio and logs why.
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
        self._enhancer = LiveEnhancer(
            client_from_options(
                api_key=api_key,
                client=client,
                base_url=base_url,
                integration=INTEGRATION_LIVEKIT,
            ),
            model=model,
            speaker=speaker,
            max_backlog_s=max_backlog_s,
            on_error=on_error,
        )
        self._max_delay_s = max_backlog_s + _PLAYOUT_HEADROOM_S
        self._enabled = True
        self._playout: PlayoutBuffer | None = None
        self._resampler: rtc.AudioResampler | None = None
        # Built inside the agent's entrypoint the loop is running; otherwise
        # this is skipped and attaching to the track warms instead.
        self._enhancer.prewarm()

    @property
    def enhancer(self) -> LiveEnhancer:
        """The underlying enhancer (state, failure count)."""
        return self._enhancer

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        if not value and self._enabled:
            # LiveKit stops calling _process, so audio still delayed in the
            # buffer cannot be handed back; drop it with the stream.
            self._reset()
        self._enabled = value

    def _on_stream_info_updated(
        self, *, room_name: str, participant_identity: str, publication_sid: str
    ) -> None:
        # Attached to a participant's track: the first frames are close, and a
        # connection warmed at construction may have gone idle since.
        self._enhancer.prewarm()

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        if frame.num_channels != 1:
            if self._enhancer.state is not LiveEnhancerState.DISABLED:
                self._enhancer.disable(
                    ValidationError(
                        f"Enhancement needs mono input, got {frame.num_channels} "
                        "channels; set AudioInputOptions(num_channels=1)."
                    )
                )
            return frame
        rate = frame.sample_rate
        playout = self._playout
        if playout is None or playout.sample_rate != rate:
            playout = self._playout = PlayoutBuffer(rate, max_delay_s=self._max_delay_s)
            self._resampler = None
        pcm = bytes(frame.data.cast("B"))
        self._enhancer.push(pcm, rate)
        for segment in self._enhancer.pull():
            playout.write(self._at_rate(segment, rate))
        out = playout.read(len(pcm))
        return rtc.AudioFrame(out, rate, 1, len(out) // 2)

    def _at_rate(self, segment: LiveSegment, rate: int) -> bytes:
        if segment.sample_rate == rate:
            return segment.pcm
        if self._resampler is None:
            self._resampler = rtc.AudioResampler(ENHANCED_SAMPLE_RATE, rate, num_channels=1)
        return b"".join(
            bytes(f.data.cast("B")) for f in self._resampler.push(bytearray(segment.pcm))
        )

    def _close(self) -> None:
        # LiveKit may reuse the processor for the participant's next track,
        # so this ends the stream but keeps the processor usable.
        self._reset()

    def _reset(self) -> None:
        self._enhancer.close()
        self._playout = None
        self._resampler = None


def enhancement(
    *,
    api_key: str | None = None,
    client: KugelAudio | None = None,
    model: str = DEFAULT_ENHANCE_MODEL,
    speaker: Audio | None = None,
    base_url: str | None = None,
    max_backlog_s: float = 2.0,
    on_error: Callable[[Exception], None] | None = None,
) -> EnhanceFrameProcessor:
    """KugelAudio speech enhancement for ``AudioInputOptions(noise_cancellation=...)``.

    Args:
        api_key: KugelAudio API key; defaults to ``KUGELAUDIO_API_KEY``.
        client: An existing :class:`~kugelaudio.KugelAudio` client, instead of
            ``api_key``/``base_url``.
        model: Enhancement model.
        speaker: A clean 2-8 s sample of the one voice to keep, from
            :func:`~kugelaudio.load_audio`. Without it, noise is removed and
            every voice is kept.
        base_url: API base URL override.
        max_backlog_s: How far the server may fall behind before the
            processor falls back to the original audio and reconnects.
        on_error: Called once with the error that disabled enhancement.

    Example:
        room_io.AudioInputOptions(noise_cancellation=enhancement())
    """
    return EnhanceFrameProcessor(
        api_key=api_key,
        client=client,
        model=model,
        speaker=speaker,
        base_url=base_url,
        max_backlog_s=max_backlog_s,
        on_error=on_error,
    )
