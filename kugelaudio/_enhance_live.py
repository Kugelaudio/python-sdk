"""Real-time speech enhancement for frame-based audio pipelines.

Framework-free core behind the LiveKit and Pipecat input filters. Both
frameworks call their filter on the event loop once per audio frame and
expect an answer at once, so nothing here ever waits on the network:

* :class:`LiveEnhancer` feeds pushed audio to a stream on one warm
  ``client.enhance.session()`` from a background task and hands back, on
  :meth:`LiveEnhancer.pull`, whatever has come back so far. The session's
  socket outlives each stream: a new sample rate, a rotation or the end of a
  pause sends a new config on it, and it is reopened transparently when the
  server closed it. Whenever the server is not delivering (connecting,
  reconnecting after a failure, disabled after a fatal one) the ORIGINAL
  audio is handed back instead, in order, so the caller's timeline never
  has a hole or a repeat.
* :class:`PlayoutBuffer` turns that bursty output (enhanced audio arrives in
  bursts) into frames of exactly the caller's size: a delay line.
  LiveKit's input path needs this, because its automatic gain control
  rejects an empty frame and aborts the process on one that is not a
  multiple of 10 ms.
"""

from __future__ import annotations

import array
import asyncio
import enum
import logging
import os
import sys
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING

from websockets.exceptions import WebSocketException

from kugelaudio.enhance import ENHANCED_SAMPLE_RATE
from kugelaudio.exceptions import (
    AuthenticationError,
    InsufficientCreditsError,
    KugelAudioError,
    ValidationError,
)
from kugelaudio.exceptions import ConnectionError as KugelAudioConnectionError

if TYPE_CHECKING:
    from kugelaudio.audio import Audio
    from kugelaudio.client import KugelAudio
    from kugelaudio.enhance import EnhanceSession

logger = logging.getLogger("kugelaudio.enhance")

DEFAULT_ENHANCE_MODEL = "clarity-1"

# Retrying cannot fix these: a bad key, no credits, or a request the server
# rejects as invalid. Enhancement stays off for the rest of the session.
_FATAL_ERRORS = (AuthenticationError, InsufficientCreditsError, ValidationError)
# Rate limits, network trouble, restarts, timeouts: reconnect with backoff.
_TRANSIENT_ERRORS = (
    KugelAudioError,
    WebSocketException,
    OSError,
    TimeoutError,
    asyncio.TimeoutError,
)
# The server caps a stream, and a session connection, at 3600 s; rotate a
# minute before that (the session replaces its socket at the same age).
_MAX_STREAM_S = 3540.0
# Delay-line trimming: look at the lowest buffer level over this much output,
# keep this much slack for jitter, and only drop 10 ms windows this quiet.
_TRIM_WINDOW_S = 5.0
_TRIM_SLACK_S = 0.1
_TRIM_WINDOW_MS = 10
_SILENCE_PEAK = 300  # about -40 dBFS


class LiveEnhancerState(str, enum.Enum):
    """Where a :class:`LiveEnhancer` is. Every state but ``LIVE`` passes the
    original audio through."""

    IDLE = "idle"
    """No stream open; the next push opens one."""
    WARMING = "warming"
    """Stream connecting; audio passes through unchanged until it is ready."""
    LIVE = "live"
    """Audio goes to the server and enhanced audio comes back."""
    RECONNECTING = "reconnecting"
    """A transient failure; audio passes through until the backoff ends."""
    DISABLED = "disabled"
    """A fatal failure; audio passes through for the rest of the session."""


@dataclass(frozen=True, slots=True)
class LiveSegment:
    """One piece of output, in timeline order: enhanced audio (always at
    24 kHz) or original audio (at the rate it was pushed at)."""

    pcm: bytes
    sample_rate: int
    enhanced: bool


def client_from_options(
    *,
    api_key: str | None,
    client: KugelAudio | None,
    base_url: str | None,
    integration: str,
) -> KugelAudio:
    """The client a filter uses: the one passed in, or one built from the key.

    Raises:
        ValueError: both a client and connection options were given, or no API
            key was given and ``KUGELAUDIO_API_KEY`` is not set.
    """
    if client is not None:
        if api_key is not None or base_url is not None:
            raise ValueError("Pass either client= or api_key=/base_url=, not both.")
        return client
    key = api_key or os.environ.get("KUGELAUDIO_API_KEY")
    if not key:
        raise ValueError("KUGELAUDIO_API_KEY must be set or api_key must be provided")
    from kugelaudio.client import KugelAudio

    return KugelAudio(api_key=key, api_url=base_url, _integration=integration)


class _Stream:
    """One stream on the session and the input it has not answered."""

    def __init__(self, sample_rate: int, started: float) -> None:
        self.sample_rate = sample_rate
        self.started = started
        self.last_push = started
        # ``None`` ends the input: the SDK sends ``end`` and the stream
        # finishes with ``done``, leaving the session socket open.
        self.queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.finishing = False
        # Pushed input the server has not answered yet. On failure it goes
        # back to the caller as original audio, so no speech is lost.
        self.unanswered = bytearray()
        self.answered_samples = 0
        self.enhanced_samples = 0
        self.task: asyncio.Task[None] | None = None
        self.idle_timer: asyncio.TimerHandle | None = None

    async def audio(
        self, on_ready: Callable[[_Stream], None]
    ) -> AsyncIterator[bytes]:
        # The SDK starts reading the input only once the server said ready,
        # so the first read is the moment this stream goes live.
        on_ready(self)
        while (chunk := await self.queue.get()) is not None:
            yield chunk

    def finish(self) -> None:
        """End the input; the server answers what it has and ``done``."""
        self.finishing = True
        self.queue.put_nowait(None)


class LiveEnhancer:
    """Non-blocking speech enhancement over one reconnecting stream.

    Call :meth:`push` with each input frame and :meth:`pull` for the output,
    both from the event loop. The first push opens the stream, on the socket
    :meth:`prewarm` opened when it ran.

    Failure policy: :class:`~kugelaudio.AuthenticationError`,
    :class:`~kugelaudio.InsufficientCreditsError` and
    :class:`~kugelaudio.ValidationError` disable enhancement for the rest of
    the session (logged once at ERROR, ``on_error`` called once). Anything
    else (rate limit, network, not ready within ``ready_timeout_s``, more
    than ``max_backlog_s`` unanswered) passes audio through while it
    reconnects with capped exponential backoff, honouring ``retry_after``;
    each is logged at WARNING and counted in :attr:`failures`.

    ``ready_timeout_s`` (default 30 s) leaves room for a cold start: after a
    quiet period the server can take several seconds to ready enhancement,
    and audio passes through unchanged until it does.
    """

    def __init__(
        self,
        client: KugelAudio,
        *,
        model: str = DEFAULT_ENHANCE_MODEL,
        speaker: Audio | None = None,
        max_backlog_s: float = 2.0,
        ready_timeout_s: float = 30.0,
        idle_close_s: float = 10.0,
        backoff_initial_s: float = 0.5,
        backoff_max_s: float = 30.0,
        on_error: Callable[[Exception], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        for name, value in (
            ("max_backlog_s", max_backlog_s),
            ("ready_timeout_s", ready_timeout_s),
            ("idle_close_s", idle_close_s),
            ("backoff_initial_s", backoff_initial_s),
            ("backoff_max_s", backoff_max_s),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        self._client = client
        self._model = model
        self._speaker = speaker
        self._max_backlog_s = max_backlog_s
        self._ready_timeout_s = ready_timeout_s
        self._idle_close_s = idle_close_s
        self._backoff_initial_s = backoff_initial_s
        self._backoff_max_s = backoff_max_s
        self._on_error = on_error
        self._clock = clock
        self._state = LiveEnhancerState.IDLE
        self._stream: _Stream | None = None
        self._output: list[LiveSegment] = []
        self._attempt = 0
        self._retry_at = 0.0
        self._failures = 0
        self._fatal_error: Exception | None = None
        # Every stream task still running, including ones _end_stream already
        # cancelled that are closing their socket, so aclose() can wait for all.
        self._tasks: set[asyncio.Task[None]] = set()
        self._prewarm_task: asyncio.Task[None] | None = None
        self._session: EnhanceSession | None = None

    @property
    def state(self) -> LiveEnhancerState:
        return self._state

    @property
    def failures(self) -> int:
        """Transient failures so far (each one fell back to pass-through)."""
        return self._failures

    @property
    def fatal_error(self) -> Exception | None:
        """The error that disabled enhancement, or ``None`` while it is not
        disabled."""
        return self._fatal_error

    def push(self, pcm: bytes, sample_rate: int) -> None:
        """Queue one frame of mono PCM16; never waits.

        Raises:
            ValueError: ``pcm`` is not whole 16-bit samples.
            RuntimeError: called outside a running event loop.
        """
        if len(pcm) % 2:
            raise ValueError("PCM16 audio must be a whole number of 2-byte samples.")
        if not pcm:
            return
        now = self._clock()
        stream = self._stream
        if stream is not None and stream.sample_rate != sample_rate:
            logger.info(
                "Input sample rate changed from %d to %d Hz; starting a new "
                "enhancement stream.",
                stream.sample_rate,
                sample_rate,
            )
            self._end_stream(stream, LiveEnhancerState.IDLE, graceful=True)
            stream = None
        if stream is not None and now - stream.started > _MAX_STREAM_S:
            logger.info("Rotating the enhancement stream before its 3600 s limit.")
            self._end_stream(stream, LiveEnhancerState.IDLE, graceful=True)
            stream = None
        if stream is None:
            if self._state is LiveEnhancerState.DISABLED or (
                self._state is LiveEnhancerState.RECONNECTING and now < self._retry_at
            ):
                self._pass_through(pcm, sample_rate)
                return
            stream = self._open(sample_rate, now)
        if self._state is LiveEnhancerState.WARMING:
            if now - stream.started > self._ready_timeout_s:
                self._fail(
                    stream,
                    KugelAudioConnectionError(
                        "The enhancement stream was not ready within "
                        f"{self._ready_timeout_s:g}s."
                    ),
                )
            self._pass_through(pcm, sample_rate)
            return
        stream.last_push = now
        stream.queue.put_nowait(pcm)
        stream.unanswered += pcm
        backlog_s = len(stream.unanswered) / 2 / sample_rate
        if backlog_s > self._max_backlog_s:
            self._fail(
                stream,
                KugelAudioConnectionError(
                    f"The enhancement stream fell {backlog_s:.1f}s behind "
                    f"(limit {self._max_backlog_s:g}s)."
                ),
            )

    def prewarm(self) -> None:
        """Ready enhancement in the background, before the first audio: send
        the warm-up request (``client.enhance.prewarm``) and open the
        session's socket, both at once.

        Never waits and never raises: audio pushed meanwhile passes through as
        before. A no-op while a prewarm is still running or the socket is
        open, and outside a running event loop (the adapters call it again
        once the pipeline runs).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("Enhancement prewarm skipped: no running event loop yet.")
            return
        if self._prewarm_task is not None and not self._prewarm_task.done():
            return
        task = loop.create_task(self._prewarm())
        self._prewarm_task = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def pull(self) -> list[LiveSegment]:
        """Every output segment since the last pull, in timeline order."""
        output, self._output = self._output, []
        return output

    def disable(self, error: Exception) -> None:
        """Turn enhancement off for the rest of the session (the adapters use
        this for input the server cannot take, e.g. stereo)."""
        self._disable(self._stream, error)

    def pause(self) -> None:
        """End the stream; the next push starts a new one on the same socket.
        Input the server had not answered comes back as original audio on the
        next pull."""
        if self._stream is not None:
            self._end_stream(self._stream, LiveEnhancerState.IDLE, graceful=True)

    def close(self) -> None:
        """End the stream, stop a running prewarm, and drop any output not
        pulled yet. The session's socket stays open for the next stream (a
        LiveKit processor is reused for the participant's next track) until
        the server closes it after 60 idle seconds, or :meth:`aclose`."""
        self._stop(graceful=True)

    async def aclose(self) -> None:
        """End the stream and close the session's socket, then wait until
        every connection this enhancer opened is closed, including streams it
        already replaced (reconnect, rotation, sample-rate change)."""
        self._stop(graceful=False)
        session, self._session = self._session, None
        if session is not None:
            self._spawn(session.aclose())
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    # ------------------------------------------------------------ internals

    def _stop(self, *, graceful: bool) -> None:
        if self._stream is not None:
            self._end_stream(self._stream, LiveEnhancerState.IDLE, graceful=graceful)
        self._output = []
        if self._prewarm_task is not None and not self._prewarm_task.done():
            self._prewarm_task.cancel()

    def _session_for_streams(self) -> EnhanceSession:
        if self._session is None:
            self._session = self._client.enhance.session()
        return self._session

    def _spawn(self, work: Coroutine[object, object, None]) -> None:
        try:
            task = asyncio.get_running_loop().create_task(work)
        except RuntimeError:
            work.close()  # no loop: nothing was opened that needs closing
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _prewarm(self) -> None:
        await asyncio.gather(self._warm_up(), self._connect_session())

    async def _warm_up(self) -> None:
        try:
            # Logs a network error or a refused warm-up itself; never raises those.
            await self._client.enhance.prewarm()
        except Exception:
            # KEEP-JUSTIFIED: a bug, logged with its traceback; it must not
            # reach the audio path or stop the socket from opening.
            logger.warning("Warming up enhancement failed.", exc_info=True)

    async def _connect_session(self) -> None:
        try:
            await self._session_for_streams().connect()
        except _TRANSIENT_ERRORS as e:
            # KEEP-JUSTIFIED: warming is an optimisation; the first stream
            # connects on its own and handles the same failure by its policy.
            logger.warning(
                "Could not prewarm the enhancement connection (%s: %s); the first "
                "stream will connect instead.",
                type(e).__name__,
                e,
            )
        except Exception:
            # KEEP-JUSTIFIED: a bug, logged with its traceback; it must not
            # reach the audio path.
            logger.warning("Prewarming the enhancement connection failed.", exc_info=True)

    def _pass_through(self, pcm: bytes, sample_rate: int) -> None:
        self._output.append(LiveSegment(pcm, sample_rate, enhanced=False))

    def _open(self, sample_rate: int, now: float) -> _Stream:
        stream = _Stream(sample_rate, now)
        self._stream = stream
        self._state = LiveEnhancerState.WARMING
        # RISK: requires a running loop; both frameworks call filters from
        # coroutines on the loop, and push() raises RuntimeError otherwise.
        task = asyncio.get_running_loop().create_task(self._run(stream))
        stream.task = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return stream

    async def _run(self, stream: _Stream) -> None:
        source: AsyncIterator[bytes] | None = None
        try:
            source = self._session_for_streams().stream(
                stream.audio(self._on_ready),
                model=self._model,
                sample_rate=stream.sample_rate,
                speaker=self._speaker,
            )
            async for chunk in source:
                self._on_enhanced(stream, chunk)
        except _FATAL_ERRORS as e:
            self._disable(stream, e)
        except _TRANSIENT_ERRORS as e:
            self._fail(stream, e)
        except Exception as e:  # noqa: BLE001 - a bug: disable loudly, never retry-loop
            self._disable(stream, e, unexpected=True)
        else:
            if not stream.finishing:
                self._fail(
                    stream,
                    KugelAudioConnectionError(
                        "The enhancement stream ended unexpectedly."
                    ),
                )
        finally:
            close = getattr(source, "aclose", None)
            if close is not None:
                await close()

    def _on_ready(self, stream: _Stream) -> None:
        if stream is not self._stream:
            return
        self._state = LiveEnhancerState.LIVE
        stream.last_push = self._clock()
        self._arm_idle_timer(stream, self._idle_close_s)

    def _arm_idle_timer(self, stream: _Stream, delay: float) -> None:
        loop = asyncio.get_running_loop()
        stream.idle_timer = loop.call_later(delay, self._check_idle, stream)

    def _check_idle(self, stream: _Stream) -> None:
        if stream is not self._stream:
            return
        idle = self._clock() - stream.last_push
        if idle >= self._idle_close_s:
            # No frames (a muted track): end the stream before the server's
            # 30 s no-audio rule turns it into a failure. The socket stays
            # open for the next push until the server's 60 s idle close.
            logger.info("Ending the enhancement stream after %.0fs without audio.", idle)
            self._end_stream(stream, LiveEnhancerState.IDLE, graceful=True)
            return
        self._arm_idle_timer(stream, self._idle_close_s - idle)

    def _on_enhanced(self, stream: _Stream, chunk: bytes) -> None:
        if stream is not self._stream or not chunk:
            return
        self._attempt = 0
        self._output.append(LiveSegment(chunk, ENHANCED_SAMPLE_RATE, enhanced=True))
        # Output has the input's duration, so it answers input by time.
        stream.enhanced_samples += len(chunk) // 2
        answered = stream.enhanced_samples * stream.sample_rate // ENHANCED_SAMPLE_RATE
        drop = min(2 * (answered - stream.answered_samples), len(stream.unanswered))
        if drop > 0:
            del stream.unanswered[:drop]
            stream.answered_samples += drop // 2

    def _end_stream(
        self, stream: _Stream, next_state: LiveEnhancerState, *, graceful: bool = False
    ) -> None:
        """Stop feeding ``stream``. ``graceful`` ends its input so it finishes
        with ``done`` and the session socket stays reusable; otherwise it is
        cancelled and the session drops the socket."""
        self._stream = None
        self._state = next_state
        if stream.unanswered:
            self._pass_through(bytes(stream.unanswered), stream.sample_rate)
            stream.unanswered.clear()
        if stream.idle_timer is not None:
            stream.idle_timer.cancel()
        task = stream.task
        if task is None or task.done():
            return
        if graceful:
            stream.finish()
            # The next stream waits for this one's done on the session; a done
            # that never comes (a dead network) must not hold it past the
            # ready timeout, so the stream is cancelled then.
            asyncio.get_running_loop().call_later(
                self._ready_timeout_s, _cancel_unless_done, task
            )
        # _run ends on its own when it is the caller; cancelling it there
        # would interrupt its own connection cleanup.
        elif task is not asyncio.current_task():
            task.cancel()

    def _fail(self, stream: _Stream, error: Exception) -> None:
        if stream is not self._stream:
            return
        self._end_stream(stream, LiveEnhancerState.RECONNECTING)
        self._failures += 1
        delay = min(self._backoff_max_s, self._backoff_initial_s * 2**self._attempt)
        self._attempt += 1
        retry_after = getattr(error, "retry_after", None)
        if isinstance(retry_after, (int, float)) and retry_after > delay:
            delay = float(retry_after)
        self._retry_at = self._clock() + delay
        logger.warning(
            "Speech enhancement unavailable (%s: %s); passing the original audio "
            "through and retrying in %.1fs (failure %d).",
            type(error).__name__,
            error,
            delay,
            self._failures,
        )

    def _disable(
        self, stream: _Stream | None, error: Exception, *, unexpected: bool = False
    ) -> None:
        if self._state is LiveEnhancerState.DISABLED or stream is not self._stream:
            return
        if stream is not None:
            self._end_stream(stream, LiveEnhancerState.DISABLED)
        self._state = LiveEnhancerState.DISABLED
        self._fatal_error = error
        logger.error(
            "Speech enhancement disabled for this session (%s: %s); passing the "
            "original audio through.",
            type(error).__name__,
            error,
            exc_info=error if unexpected else None,
        )
        if self._on_error is not None:
            try:
                self._on_error(error)
            except Exception:  # noqa: BLE001
                # KEEP-JUSTIFIED: a broken user callback must not take the
                # audio path down with it; it is logged with its traceback.
                logger.exception("The enhancement on_error callback raised.")


def _cancel_unless_done(task: asyncio.Task[None]) -> None:
    if not task.done():
        task.cancel()


class PlayoutBuffer:
    """A delay line that returns output frames of exactly the input's size.

    Writes are bursty; :meth:`read` always returns the requested number of
    bytes. When too little has arrived it pads with silence (the delay grows
    by that much and :attr:`underrun_s` counts it); when the buffer stays
    above what jitter needs it trims quiet 10 ms windows to shrink the delay
    again (:attr:`trimmed_s`). Past ``max_delay_s`` the oldest audio is
    dropped, logged at WARNING and counted in :attr:`dropped_s`.
    """

    def __init__(self, sample_rate: int, *, max_delay_s: float) -> None:
        if max_delay_s <= 0:
            raise ValueError(f"max_delay_s must be positive, got {max_delay_s}.")
        self.sample_rate = sample_rate
        self._buffer = bytearray()
        self._max_bytes = 2 * int(max_delay_s * sample_rate)
        self._window_bytes = 2 * int(_TRIM_WINDOW_S * sample_rate)
        self._slack_bytes = 2 * int(_TRIM_SLACK_S * sample_rate)
        self._trim_bytes = 2 * (sample_rate * _TRIM_WINDOW_MS // 1000)
        self._window_read = 0
        self._window_min: int | None = None
        self.underrun_s = 0.0
        self.trimmed_s = 0.0
        self.dropped_s = 0.0

    def __len__(self) -> int:
        return len(self._buffer)

    def write(self, pcm: bytes) -> None:
        self._buffer += pcm
        excess = len(self._buffer) - self._max_bytes
        if excess > 0:
            excess += excess % 2
            del self._buffer[:excess]
            dropped = excess / 2 / self.sample_rate
            self.dropped_s += dropped
            logger.warning(
                "Enhancement output fell %.2fs behind the input; dropped it to "
                "bound the delay (%.2fs dropped so far).",
                dropped,
                self.dropped_s,
            )

    def read(self, n_bytes: int) -> bytes:
        """Exactly ``n_bytes`` of the delayed stream (silence-padded)."""
        available = min(n_bytes, len(self._buffer))
        out = bytes(self._buffer[:available])
        del self._buffer[:available]
        if available < n_bytes:
            self.underrun_s += (n_bytes - available) / 2 / self.sample_rate
            out += bytes(n_bytes - available)
        self._track_level(n_bytes)
        return out

    def drain(self) -> bytes:
        """Everything buffered; the buffer is empty afterwards."""
        out = bytes(self._buffer)
        self._buffer.clear()
        return out

    def _track_level(self, n_bytes: int) -> None:
        level = len(self._buffer)
        self._window_min = level if self._window_min is None else min(self._window_min, level)
        self._window_read += n_bytes
        if self._window_read < self._window_bytes:
            return
        excess = self._window_min - self._slack_bytes
        self._window_read = 0
        self._window_min = None
        if excess >= self._trim_bytes:
            self._trim_quiet(excess)

    def _trim_quiet(self, excess: int) -> None:
        step = self._trim_bytes
        kept = bytearray()
        trimmed = 0
        for start in range(0, len(self._buffer) - len(self._buffer) % step, step):
            window = self._buffer[start : start + step]
            if trimmed + step <= excess and _is_quiet(window):
                trimmed += step
            else:
                kept += window
        kept += self._buffer[len(self._buffer) - len(self._buffer) % step :]
        self._buffer = kept
        self.trimmed_s += trimmed / 2 / self.sample_rate


def _is_quiet(pcm: bytearray) -> bool:
    samples = array.array("h", bytes(pcm))
    if sys.byteorder == "big":
        samples.byteswap()  # PCM16 on the wire is little-endian
    return max(samples) < _SILENCE_PEAK and -min(samples) < _SILENCE_PEAK
