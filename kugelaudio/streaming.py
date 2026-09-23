"""Streaming TTS session for text-in/audio-out streaming."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    NoReturn,
    Optional,
    Set,
    Union,
)

from kugelaudio.exceptions import (
    AuthenticationError,
    ConnectionError as KugelAudioConnectionError,
    InsufficientCreditsError,
    KugelAudioError,
    ServerRestartingError,
    WS_CLOSE_INSUFFICIENT_CREDITS,
    WS_CLOSE_MODEL_UNAVAILABLE,
    WS_CLOSE_RATE_LIMITED,
    WS_CLOSE_SERVICE_RESTART,
    WS_CLOSE_TRY_AGAIN_LATER,
    WS_CLOSE_UNAUTHORIZED,
    classify_ws_close,
    classify_ws_frame,
    classify_ws_handshake_error,
    ws_handshake_error_types,
)
from kugelaudio._diagnostics import (
    DISABLED as _DISABLED_DIAGNOSTICS,
    OP_MULTI_CONTEXT,
    OP_STREAM_SESSION,
    STAGE_FINALIZING,
    STAGE_HANDSHAKE,
    STAGE_SENDING_REQUEST,
    TRANSPORT_WEBSOCKET,
    Diagnostics,
    Operation,
)
from kugelaudio._sdk_metadata import sdk_query_string

# Server-initiated WS close codes that indicate an error the caller must see.
# Other codes (e.g. 1000/1001 normal close) end the stream cleanly.
_WS_ERROR_CLOSE_CODES = frozenset({
    WS_CLOSE_UNAUTHORIZED,
    WS_CLOSE_INSUFFICIENT_CREDITS,
    WS_CLOSE_RATE_LIMITED,
    WS_CLOSE_MODEL_UNAVAILABLE,
    WS_CLOSE_SERVICE_RESTART,
    WS_CLOSE_TRY_AGAIN_LATER,
})
# The replica is going away (rolling deploy): a typed, retryable error
# rather than a silent end-of-stream, so callers resend instead of
# treating a cut turn as complete.
_WS_RESTART_CLOSE_CODES = frozenset({
    WS_CLOSE_SERVICE_RESTART,
    WS_CLOSE_TRY_AGAIN_LATER,
})
from kugelaudio.models import (
    AudioChunk,
    SessionUsage,
    StreamConfig,
    WordTimestamp,
    clamp_cfg_scale,
)

logger = logging.getLogger(__name__)


def _append_sdk_query(url: str) -> str:
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{sdk_query_string()}"

# Default timeout for waiting on audio generation responses
_DEFAULT_RECV_TIMEOUT_S = 30.0
# Short poll interval for checking if more messages are available
_POLL_TIMEOUT_S = 0.05
# After a chunk_complete arrives, wait this long for another chunk to
# start before deciding the server is idle. The server splits multi-
# sentence text into one chunk per sentence, so a single send(flush=True)
# can produce N chunk_complete frames. Exiting on the first one truncates
# audio (KUG-421/422). 2 s covers inter-chunk scheduling latency.
_INTER_CHUNK_IDLE_S = 2.0

# How long to wait before reconnecting after a rolling-deploy close (1012 /
# 1013). The WebSocket close frame has no Retry-After header, so this is the
# portable default used by both public SDKs.
_RESTART_RECONNECT_DELAY_S = 1.0


def _restart_delay_s(err: ServerRestartingError) -> float:
    """Seconds to wait before reconnecting after a rolling-deploy close.

    Honours the server's ``retry_after`` hint, falling back to
    ``_RESTART_RECONNECT_DELAY_S`` when the close carried none.
    """
    retry_after = err.retry_after
    if retry_after is None:
        return _RESTART_RECONNECT_DELAY_S
    return max(0.0, float(retry_after))


# Generation parameters changeable mid-connection via ``update_settings``
# (KUG-1166). Identity / audio-format fields (voice_id, model_id, sample_rate,
# output_format, project_id, dictionary_ids) are NOT in this set — they are
# fixed for the connection's lifetime and the server rejects them in an update_settings body.
_UPDATABLE_SETTINGS = (
    "cfg_scale",
    "temperature",
    "max_new_tokens",
    "language",
    "normalize",
    "speed",
)


def _settings_update_body(
    *,
    cfg_scale: Optional[float] = None,
    temperature: Optional[float] = None,
    speed: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    language: Optional[str] = None,
    normalize: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build an ``update_settings`` body from the provided fields (KUG-1166).

    Drops fields left as ``None`` so a message updates only what it carries.
    Raises ``ValueError`` when nothing was provided — an empty update is a
    caller mistake, not a no-op to send.
    """
    body: Dict[str, Any] = {
        "cfg_scale": cfg_scale,
        "temperature": temperature,
        "speed": speed,
        "max_new_tokens": max_new_tokens,
        "language": language,
        "normalize": normalize,
    }
    body = {name: value for name, value in body.items() if value is not None}
    if not body:
        raise ValueError(
            "update_settings requires at least one parameter to change "
            f"(one of {', '.join(_UPDATABLE_SETTINGS)})"
        )
    return body


class _DiagnosedSession:
    """Diagnostics and connection seam shared by the two WebSocket sessions.

    Operation scope follows the contract: establishing the socket is its own
    operation, and every turn is its own operation (keyed per context for
    multi-context, under ``None`` for the single-stream session). A turn
    starts with the first text sent after the previous turn ended and ends at
    its final/done frame (success), an error (failure) or a cancellation, so
    retries, elapsed time and audio counters never leak from one turn into
    the next, and cancelling one context never touches another's turn.
    """

    #: ``kugel.operation`` value for this session class.
    _operation_name = OP_STREAM_SESSION

    _diagnostics: Diagnostics
    _turn_ops: Dict[Optional[str], Operation]
    _connect_task: Optional["asyncio.Task[None]"]

    def _turn(self, key: Optional[str] = None) -> Operation:
        """The live turn for *key*, minting one when the last turn settled."""
        op = self._turn_ops.get(key)
        if op is None or op.settled:
            op = self._diagnostics.operation(
                self._operation_name, TRANSPORT_WEBSOCKET, STAGE_SENDING_REQUEST
            )
            self._turn_ops[key] = op
        return op

    def _live_turn(self, key: Optional[str] = None) -> Optional[Operation]:
        op = self._turn_ops.get(key)
        return op if op is not None and not op.settled else None

    def _finish_turn(self, key: Optional[str] = None) -> None:
        """The server's final/done frame for *key*: the turn succeeded."""
        op = self._turn_ops.pop(key, None)
        if op is not None:
            op.succeed()

    def _cancel_turn(self, key: Optional[str] = None) -> None:
        """Caller-initiated cancellation of *key*'s turn only."""
        op = self._turn_ops.pop(key, None)
        if op is not None:
            op.cancel()

    def _raise_frame_error(
        self, data: Dict[str, Any], key: Optional[str] = None
    ) -> NoReturn:
        """Raise the typed error for a server error frame.

        The live turn for *key* is marked as failed by the server, so the
        event is ``request_failed`` (``retry_exhausted`` after a retry) with
        the frame's ``error_code`` and request id, never the
        ``stream_interrupted`` its receive stage would suggest.
        """
        err = classify_ws_frame(data)
        op = self._live_turn(key)
        if op is not None:
            op.mark_server_error(err)
        raise err

    def _cancel_all_turns(self) -> None:
        for key in list(self._turn_ops):
            self._cancel_turn(key)

    @staticmethod
    async def _within_turn(
        op: Optional[Operation], stage: str, frames: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[AudioChunk]:
        """Yield *frames* as part of *op*, or plainly when no turn is active.

        Flushing or closing after the turn already ended is not a new unit
        of work: minting an operation for it would count a second success
        for one turn (the per-turn close the Pipecat wrapper sends).
        """
        if op is None:
            async for chunk in frames:
                yield chunk
            return
        async with op.step(stage):
            async for chunk in frames:
                op.record_chunk(len(chunk.audio))
                yield chunk

    async def connect(self) -> None:
        """Open the socket with establishment owned by this session's close()."""
        await self._establish(None)

    async def _establish(self, turn: Optional[Operation]) -> None:
        """Open the socket, as its own operation or as part of *turn*.

        A reconnect while recovering a turn passes that turn, so a failed
        reconnect is the turn's failure (and ``retry_exhausted``), never a
        second event.
        """
        if self._connect_task is not None:
            await asyncio.shield(self._connect_task)
            return
        task = asyncio.create_task(self._open_connection(turn))
        self._connect_task = task
        try:
            await task
        finally:
            if self._connect_task is task:
                self._connect_task = None

    async def _open_connection(self, turn: Optional[Operation]) -> None:
        op = turn
        if op is None:
            op = self._diagnostics.operation(self._operation_name, TRANSPORT_WEBSOCKET)
        op.set_stage(STAGE_HANDSHAKE)
        try:
            await self._connect_socket()
        except BaseException as exc:
            op.fail(exc)
            raise
        if turn is None:
            op.succeed()

    async def _connect_socket(self) -> None:
        raise NotImplementedError


class StreamingSession(_DiagnosedSession):
    """WebSocket session for streaming text input and audio output.

    This allows streaming text (e.g., from an LLM) and receiving audio
    as it's generated. Text is buffered and processed when sentence
    boundaries are detected or when explicitly flushed.

    Example:
        async with client.tts.streaming_session(voice_id=123) as session:
            async for token in llm_stream:
                async for chunk in session.send(token):
                    play_audio(chunk)

            # Flush remaining text
            async for chunk in session.flush():
                play_audio(chunk)
    """

    def __init__(
        self,
        api_key: str,
        tts_url: str,
        config: Optional[StreamConfig] = None,
        on_word_timestamps: Optional[Callable[[List[WordTimestamp]], None]] = None,
        diagnostics: Optional[Diagnostics] = None,
    ):
        self._diagnostics = diagnostics or _DISABLED_DIAGNOSTICS
        self._turn_ops: Dict[Optional[str], Operation] = {}
        self._api_key = api_key
        self._tts_url = tts_url
        self._config = config or StreamConfig()
        self._ws: Optional[Any] = None
        self._connect_task: asyncio.Task[None] | None = None
        self._session_id: Optional[str] = None
        self._is_started = False
        self._config_sent = False
        self._pending_messages: List[Dict[str, Any]] = []
        self._on_word_timestamps = on_word_timestamps
        self._last_word_timestamps: List[WordTimestamp] = []
        # drain() sets these so a subsequent end_session()/close() can return
        # the same stats without re-running the close handshake.
        self._session_ended = False
        self._session_stats: Dict[str, Any] = {}
        # Typed usage from the most recently closed session (KUG-1192).
        self._last_usage: Optional[SessionUsage] = None
        # Raw ``final`` end-of-audio frame from the most recently completed
        # turn (KUG-1238) — see :attr:`last_final`.
        self._last_final: Optional[Dict[str, Any]] = None
        # Replay state for rolling-deploy closes (1012 / 1013): the frames
        # of the current turn, whether any audio for it has arrived, and
        # whether the turn was already replayed once.
        self._turn_frames: List[Dict[str, Any]] = []
        self._turn_audio = False
        self._turn_replayed = False

    def _reset_turn(self) -> None:
        self._turn_frames = []
        self._turn_audio = False
        self._turn_replayed = False

    def _forget_socket(self) -> None:
        """Drop a dead socket so the next :meth:`send` reconnects."""
        self._ws = None
        self._is_started = False
        self._config_sent = False

    @staticmethod
    def _restart_code(exc: BaseException) -> Optional[int]:
        """Return the close code when ``exc`` is a rolling-deploy close."""
        if "ConnectionClosed" not in str(type(exc)):
            return None
        code = getattr(exc, "code", None)
        return code if code in _WS_RESTART_CLOSE_CODES else None

    async def _send_frame(self, frame: Dict[str, Any]) -> None:
        """Send one turn frame, remembering it for a possible replay."""
        self._turn_frames.append(frame)
        msg = dict(frame)
        if not self._config_sent:
            from kugelaudio.client import _warn_if_no_language

            _warn_if_no_language(self._config.language, self._config.normalize)
            msg.update(self._config.to_dict())
            self._config_sent = True
        try:
            await self._ws.send(json.dumps(msg))
        except Exception as e:
            if self._restart_code(e) is None:
                raise
            await self._recover_from_restart(e)

    async def _recover_from_restart(self, exc: BaseException) -> bool:
        """Absorb a rolling-deploy close (1012 / 1013).

        When no audio for the current turn has been delivered yet, wait
        the close's ``retry_after`` hint, reconnect and resend the turn's
        frames once (config included), so the caller never notices the
        deploy. Returns ``True`` when frames were replayed and the caller
        should keep receiving on the new socket, ``False`` when there was
        nothing to replay (the next :meth:`send` reconnects lazily).

        Raises :class:`ServerRestartingError` when audio already went out
        (a replay would repeat the start of the sentence) or when this turn
        was replayed once already.
        """
        err = classify_ws_close(exc.code, getattr(exc, "reason", None))
        self._forget_socket()
        if self._turn_audio or self._turn_replayed:
            self._reset_turn()
            raise err from exc
        frames = self._turn_frames
        self._turn_frames = []
        if not frames:
            return False
        delay = _restart_delay_s(err)
        logger.info(
            "Server restarting (%s); reconnecting in %.1fs and replaying "
            "%d frame(s)",
            exc.code,
            delay,
            len(frames),
        )
        turn = self._live_turn()
        if turn is not None:
            turn.record_retry()
        await asyncio.sleep(delay)
        await self._establish(turn)
        self._turn_replayed = True
        for frame in frames:
            await self._send_frame(frame)
        if turn is not None:
            # The reconnect left the turn at ``handshake``; it is now waiting
            # for the replayed turn's audio again.
            turn.expect_audio()
        return True

    @property
    def last_word_timestamps(self) -> List[WordTimestamp]:
        """Return the most recently received word timestamps.

        Updated each time the server sends a ``word_timestamps`` message.
        """
        return self._last_word_timestamps

    @property
    def last_usage(self) -> Optional[SessionUsage]:
        """Per-session usage from the most recently closed session.

        Populated when the server sends ``session_closed`` (after
        :meth:`drain`, :meth:`end_session`, or a fully-drained :meth:`send`).
        ``None`` before the first session closes. Use this to bill your own
        customers per conversation — see :class:`~kugelaudio.SessionUsage`.
        """
        return self._last_usage

    @property
    def last_final(self) -> Optional[Dict[str, Any]]:
        """End-of-audio stats from the most recently completed turn.

        The server marks the end of every gracefully completed turn with a
        ``{"final": true, ...}`` frame (the ElevenLabs ``isFinal``
        equivalent, KUG-1238) carrying ``total_audio_seconds`` /
        ``total_text_chunks`` / ``total_audio_chunks``. ``None`` before the
        first turn completes, and not updated on a barge-in cancel.
        """
        return self._last_final

    def _capture_session_stats(self, data: Dict[str, Any]) -> None:
        """Record the ``session_closed`` payload and parse typed usage.

        ``session_closed`` is the terminal frame of the turn: it succeeded.
        """
        self._finish_turn()
        self._session_stats = data
        usage = SessionUsage.from_session_payload(data)
        if usage is not None:
            self._last_usage = usage

    async def __aenter__(self) -> StreamingSession:
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def _connect_socket(self) -> None:
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets required. Install with: pip install websockets"
            )

        ws_url = self._tts_url.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = _append_sdk_query(f"{ws_url}/ws/tts/stream?api_key={self._api_key}")

        handshake_errors = ws_handshake_error_types(websockets)
        try:
            # See MultiContextSession.connect for why we override the websockets
            # ping defaults — keep the WS alive across ~20-30s NAT/proxy idle
            # timeouts that otherwise drop it without a close handshake.
            self._ws = await websockets.connect(
                ws_url,
                compression=None,
                ping_interval=15.0,
                ping_timeout=15.0,
                close_timeout=5.0,
            )
            self._is_started = True
            logger.debug("Streaming session connected")
        except handshake_errors as e:
            typed = classify_ws_handshake_error(e)
            if typed is not None:
                raise typed from e
            raise KugelAudioConnectionError(
                f"KugelAudio streaming WebSocket handshake failed: {e}."
            ) from e

    async def _ensure_config_sent(self) -> None:
        """Send session config on first text message.

        The server's /ws/tts/stream endpoint does NOT send a session_started
        response. Instead, it reads config from the first messages in the
        main loop. We send config fields alongside the first text message.
        """
        if self._config_sent:
            return
        self._config_sent = True

    async def send(
        self,
        text: str,
        flush: bool = False,
    ) -> AsyncIterator[AudioChunk]:
        """Send text and yield any generated audio chunks.

        Args:
            text: Text to add to buffer
            flush: Force flush the buffer

        Yields:
            AudioChunk as audio is generated
        """
        # Establishing the socket is its own operation; the turn starts
        # with the text.
        if not self._is_started:
            await self.connect()

        op = self._turn()
        async with op.step(STAGE_SENDING_REQUEST):
            await self._send_frame({"text": text, "flush": flush})

            # When flush=True the server is committed to producing audio, so
            # the iterator must wait the full receive timeout for the first
            # frame. Otherwise a partial token may not trigger synthesis at
            # all and we return immediately after the short poll.
            op.expect_audio()
            async for chunk in self._receive_until_idle(wait_for_first=flush):
                op.record_chunk(len(chunk.audio))
                yield chunk

    async def flush(self) -> AsyncIterator[AudioChunk]:
        """Flush the text buffer and yield remaining audio.

        Yields:
            AudioChunk as remaining audio is generated
        """
        if not self._is_started or not self._ws:
            return

        op = self._turn()
        async with op.step(STAGE_SENDING_REQUEST):
            await self._send_frame({"flush": True})

            op.expect_audio()
            async for chunk in self._receive_until_idle(wait_for_first=True):
                op.record_chunk(len(chunk.audio))
                yield chunk

    async def _receive_until_idle(
        self, *, wait_for_first: bool = False
    ) -> AsyncIterator[AudioChunk]:
        """Receive messages until we get a chunk_complete or session_closed.

        The server sends:
          - {"generation_started": true, ...} when synthesis begins
          - {"audio": "<base64>", ...}       for each audio chunk
          - {"word_timestamps": [...], ...}  per-chunk word timestamps
          - {"chunk_complete": true, ...}    when a text chunk is done
          - {"session_closed": true, ...}    on session end
          - {"error": "..."}                 on error

        Timeout strategy:
          - _DEFAULT_RECV_TIMEOUT_S (30s) when ``wait_for_first`` is set
            (caller knows the server is committed to replying — e.g. a
            ``flush=True`` send or an explicit ``flush()``), or once we
            have received the first message from the server.
          - _POLL_TIMEOUT_S (50ms) on the very first read of an open-ended
            ``send(text)`` whose buffered text may not cross a sentence
            boundary; if the server stays silent we return immediately and
            let the caller continue feeding text.

        Once any message has arrived, we always wait the full receive
        timeout for the authoritative ``chunk_complete`` / ``session_closed``
        signal so audio chunks separated by inter-frame latency are not
        truncated (KUG-421, KUG-422).

        ``chunk_complete`` marks the end of ONE server-side chunk (one
        sentence). Multi-sentence ``send(flush=True)`` / ``flush()`` calls
        produce N chunk_complete frames, so we must not exit on the first
        one — we wait ``_INTER_CHUNK_IDLE_S`` for a follow-up
        ``generation_started`` / audio frame and only exit if the server
        stays quiet.
        """
        generation_active = False
        last_chunk_complete = False

        while True:
            if last_chunk_complete:
                timeout = _INTER_CHUNK_IDLE_S
            elif generation_active or wait_for_first:
                timeout = _DEFAULT_RECV_TIMEOUT_S
            else:
                timeout = _POLL_TIMEOUT_S

            try:
                msg = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
                data = json.loads(msg)

                if data.get("error"):
                    self._raise_frame_error(data)

                if data.get("generation_started"):
                    generation_active = True
                    last_chunk_complete = False
                    continue

                if data.get("audio"):
                    generation_active = True
                    last_chunk_complete = False
                    self._turn_audio = True
                    yield AudioChunk.from_dict(data)

                if "word_timestamps" in data:
                    stamps = [
                        WordTimestamp.from_dict(w) for w in data["word_timestamps"]
                    ]
                    self._last_word_timestamps = stamps
                    if self._on_word_timestamps:
                        self._on_word_timestamps(stamps)
                    continue

                if data.get("final"):
                    # End-of-audio marker for the turn (KUG-1238): no more
                    # audio frames are coming. ``session_closed`` follows
                    # immediately and remains the terminal signal here (it
                    # carries the usage payload).
                    self._last_final = data
                    self._finish_turn()
                    continue

                if data.get("session_closed"):
                    self._capture_session_stats(data)
                    self._reset_turn()
                    break

                if data.get("chunk_complete"):
                    last_chunk_complete = True
                    continue

            except asyncio.TimeoutError:
                if last_chunk_complete:
                    # Server idle after last chunk_complete — caller's
                    # text is fully rendered.
                    break
                if generation_active:
                    logger.warning(
                        "Timed out waiting for chunk_complete after %.0fs "
                        "(generation was active — possible server issue)",
                        _DEFAULT_RECV_TIMEOUT_S,
                    )
                break
            except Exception as e:
                if "ConnectionClosed" in str(type(e)):
                    code = getattr(e, "code", None)
                    if code in _WS_RESTART_CLOSE_CODES:
                        # Rolling deploy: replay the turn on a fresh socket
                        # when nothing audible was lost, else raise.
                        if not await self._recover_from_restart(e):
                            break
                        generation_active = False
                        last_chunk_complete = False
                        continue
                    if code in _WS_ERROR_CLOSE_CODES:
                        # The socket is gone either way; forget it so the
                        # next send() reconnects instead of writing to it.
                        self._forget_socket()
                        raise classify_ws_close(
                            code, getattr(e, "reason", None)
                        ) from e
                    break
                raise

    async def drain(self) -> AsyncIterator[AudioChunk]:
        """Signal end-of-input and yield any remaining audio chunks.

        Sends ``{"close": true}`` once, then yields every audio chunk the
        server emits until it responds with ``session_closed``. Use this
        before exiting the session if you need the tail audio that the
        server still has in flight — otherwise :meth:`close` /
        :meth:`end_session` will silently discard those chunks (KUG-421).

        After ``drain()`` completes the server-side session has ended and
        the next :meth:`send` will start a fresh session on the same
        WebSocket. Stats are captured and returned by the subsequent
        :meth:`end_session` / :meth:`close` call.

        Example:
            async with client.tts.streaming_session(voice_id=123) as session:
                async for token in llm_stream:
                    async for chunk in session.send(token):
                        play(chunk)
                async for chunk in session.drain():
                    play(chunk)
        """
        if not self._ws or self._session_ended:
            return
        async for chunk in self._within_turn(
            self._live_turn(), STAGE_FINALIZING, self._drain_frames()
        ):
            yield chunk

    async def _drain_frames(self) -> AsyncIterator[AudioChunk]:
        """The close-handshake drain behind :meth:`drain`."""
        if not self._ws or self._session_ended:
            return

        self._session_ended = True
        replay = False
        try:
            await self._ws.send(json.dumps({"close": True}))

            while True:
                msg = await asyncio.wait_for(
                    self._ws.recv(), timeout=_DEFAULT_RECV_TIMEOUT_S
                )
                data = json.loads(msg)

                if data.get("error"):
                    self._raise_frame_error(data)

                if data.get("audio"):
                    self._turn_audio = True
                    yield AudioChunk.from_dict(data)

                if "word_timestamps" in data:
                    stamps = [
                        WordTimestamp.from_dict(w) for w in data["word_timestamps"]
                    ]
                    self._last_word_timestamps = stamps
                    if self._on_word_timestamps:
                        self._on_word_timestamps(stamps)

                if data.get("session_closed"):
                    self._capture_session_stats(data)
                    self._reset_turn()
                    break
        except Exception as e:
            if "ConnectionClosed" in str(type(e)) or isinstance(
                e, (OSError, asyncio.TimeoutError)
            ):
                self._forget_socket()
                if self._restart_code(e) is not None:
                    # Rolling deploy: the close handshake never completed.
                    # Replay the unspoken turn on a fresh socket and drain
                    # again; raises when audio already went out.
                    self._session_ended = False
                    replay = await self._recover_from_restart(e)
            else:
                raise

        if replay:
            async for chunk in self._drain_frames():
                yield chunk

    async def end_session(self) -> Dict[str, Any]:
        """End the current session but keep the WebSocket connection open.

        This allows starting a new session on the same connection, avoiding
        the overhead of a new WebSocket handshake (~200-300ms).

        After calling this, use :meth:`update_config` to change voice/model
        settings, then call :meth:`send` to start the next session.

        Audio chunks the server still has in flight at close time are
        silently discarded — see :meth:`drain` for the opt-in tail-drain
        path (KUG-421). If :meth:`drain` has already been called, this
        method returns the stats captured there without re-running the
        close handshake.

        Returns:
            Session statistics from the ended session
        """
        if self._session_ended:
            stats = self._session_stats
            self._session_ended = False
            self._session_stats = {}
            self._config_sent = False
            self._last_word_timestamps = []
            return stats

        stats: Dict[str, Any] = {}
        if not self._ws:
            return stats

        try:
            await self._ws.send(json.dumps({"close": True}))

            while True:
                msg = await asyncio.wait_for(
                    self._ws.recv(), timeout=_DEFAULT_RECV_TIMEOUT_S
                )
                data = json.loads(msg)
                if data.get("error"):
                    self._raise_frame_error(data)
                if data.get("session_closed"):
                    self._capture_session_stats(data)
                    stats = data
                    break
        except Exception as e:
            # If the underlying websocket died (proxy/LB/server restart),
            # tear down state so the next send() reconnects cleanly rather
            # than calling .send() on a dead socket.
            # A rolling-deploy close (1012 / 1013) lands here too: the
            # server discarded the in-flight audio, which is what
            # end_session() does anyway, so it is not surfaced.
            if "ConnectionClosed" in str(type(e)) or isinstance(e, (OSError, asyncio.TimeoutError)):
                self._ws = None
                self._is_started = False
            else:
                raise

        # Reset session state so next send() starts a fresh session
        self._config_sent = False
        self._last_word_timestamps = []
        self._reset_turn()
        return stats

    async def cancel_current(self) -> None:
        """Interrupt (barge-in) the current generation, keeping the socket open.

        Use this when the end user starts speaking over the agent: it tells
        the server to **stop generating audio for the current turn
        immediately** and drop any text that was buffered or queued but not
        yet spoken. Unlike :meth:`end_session` / :meth:`drain`, no remaining
        text is flushed — the turn is abandoned.

        The WebSocket stays open and a fresh session is ready, so the next
        :meth:`send` starts the next user turn immediately (config is re-sent
        automatically on that first send).

        Returns once the server acknowledges with an ``interrupted`` frame, or
        after a quiet timeout if the server stays silent. Any audio chunks the
        server still has in flight at cancel time are received and discarded so
        they don't leak into the next turn.

        Example:
            # VAD detected the user speaking over the agent:
            await session.cancel_current()
            # Socket is still open — start the next turn immediately:
            async for chunk in session.send(next_text, flush=True):
                play(chunk)
        """
        if not self._ws:
            return

        # Barge-in is a caller-initiated cancellation of the current turn,
        # not a failure: no event, only the ``sdk_stats`` cancellation count.
        self._cancel_turn()

        # A pending drain()/end_session() handshake is moot once we barge in.
        self._session_ended = False
        self._session_stats = {}
        self._reset_turn()

        try:
            await self._ws.send(json.dumps({"cancel": True}))

            # Drain until the server acks ``interrupted``. The per-recv timeout
            # resets on every frame, so late audio chunks from the cancelled
            # turn (which we discard) keep the wait alive; only a genuinely
            # silent server trips the quiet fuse.
            while True:
                msg = await asyncio.wait_for(
                    self._ws.recv(), timeout=_DEFAULT_RECV_TIMEOUT_S
                )
                data = json.loads(msg)
                if data.get("error"):
                    self._raise_frame_error(data)
                if data.get("interrupted"):
                    break
                # Discard any stale audio / timestamps from the cancelled turn.
        except asyncio.TimeoutError:
            # Server stayed silent — treat the barge-in as acknowledged.
            pass
        except Exception as e:
            # Underlying socket died — tear down so the next send() reconnects
            # cleanly instead of writing to a dead socket.
            if "ConnectionClosed" in str(type(e)) or isinstance(
                e, (OSError, asyncio.TimeoutError)
            ):
                self._ws = None
                self._is_started = False
            else:
                raise

        # The server starts a fresh session after a cancel — reset so the
        # next send() re-sends config.
        self._config_sent = False
        self._last_word_timestamps = []

    def update_config(self, config: Optional[StreamConfig] = None, **kwargs) -> None:
        """Update session configuration for the next session.

        Call this after :meth:`end_session` and before the next :meth:`send`
        to change voice, model, language, or other settings.

        Args:
            config: Full replacement config. If None, update individual fields.
            **kwargs: Individual fields to update (voice_id, model_id, language, etc.)
        """
        if config is not None:
            self._config = config
        else:
            for key, value in kwargs.items():
                if hasattr(self._config, key):
                    setattr(self._config, key, value)
        self._config_sent = False
        self._reset_turn()

    async def update_settings(
        self,
        *,
        cfg_scale: Optional[float] = None,
        temperature: Optional[float] = None,
        speed: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        language: Optional[str] = None,
        normalize: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Change generation parameters mid-connection without reconnecting (KUG-1166).

        Sends an ``update_settings`` message and waits for the server's
        ``settings_updated`` acknowledgement, returning the generation
        parameters now in effect. Only the six parameters in this signature
        are updatable; identity / audio-format fields (``voice_id``,
        ``model_id``, ``sample_rate``, ``output_format``, ``project_id``,
        ``dictionary_ids``) are fixed for the connection — change those with :meth:`update_config`
        after :meth:`end_session` instead.

        The change applies to the **next turn**: a turn already streaming
        keeps the settings it started with, so call this between turns.

        Args:
            cfg_scale: Classifier-free guidance scale (0.0–10.0).
            temperature: Sampling variance (0.0–1.0).
            speed: Playback speed multiplier (0.8–1.2).
            max_new_tokens: Maximum tokens per generation (1–2048).
            language: Language code for normalization (e.g. ``"de"``).
            normalize: Enable text normalization.

        Returns:
            The generation parameters now in effect (the server's echo).

        Raises:
            ValueError: if no parameter was provided.
            KugelAudioError: if the server rejects the update (e.g. a value
                out of range) or stays silent past the receive timeout.
        """
        body = _settings_update_body(
            cfg_scale=cfg_scale,
            temperature=temperature,
            speed=speed,
            max_new_tokens=max_new_tokens,
            language=language,
            normalize=normalize,
        )
        if not self._is_started or not self._ws:
            await self.connect()

        await self._ws.send(json.dumps({"update_settings": body}))
        effective = await self._await_settings_ack()

        # Keep the local config in sync only after the server accepts the
        # change; rejected updates must not poison later config re-sends.
        for name, value in body.items():
            if hasattr(self._config, name):
                setattr(self._config, name, value)
        return effective

    async def _await_settings_ack(self) -> Dict[str, Any]:
        """Wait for the ``settings_updated`` frame; return its ``settings``.

        Frames unrelated to the ack (e.g. stray audio left over from a turn
        that was still draining) are discarded — ``update_settings`` is
        documented as a between-turns call.
        """
        while True:
            try:
                msg = await asyncio.wait_for(
                    self._ws.recv(), timeout=_DEFAULT_RECV_TIMEOUT_S
                )
            except asyncio.TimeoutError as exc:
                raise KugelAudioError(
                    "Timed out waiting for settings_updated acknowledgement"
                ) from exc
            data = json.loads(msg)
            if data.get("error"):
                self._raise_frame_error(data)
            if data.get("settings_updated"):
                settings = data.get("settings")
                return settings if isinstance(settings, dict) else {}

    async def close(self) -> Dict[str, Any]:
        """Close the session and the WebSocket connection.

        Sends a close command, drains messages until the server responds
        with ``session_closed``, then closes the WebSocket. Audio chunks
        the server emits during this final drain are silently discarded
        — call :meth:`drain` first if you need them (KUG-421).

        For session reuse without closing the connection, use
        :meth:`end_session` instead.

        Returns:
            Session statistics
        """
        # A pending handshake is cancelled by the caller's close: its connect
        # operation settles as a cancellation (counted, never an event).
        if self._connect_task is not None:
            self._connect_task.cancel()
            # The connect caller observes cancellation; wait for transport cleanup.
            await asyncio.gather(self._connect_task, return_exceptions=True)

        # Closing finalizes the turn in progress, if any; ``session_closed``
        # settles it as a success.
        op = self._live_turn()
        if op is not None:
            op.set_stage(STAGE_FINALIZING)
        try:
            stats = await self.end_session()
        except BaseException as exc:
            if op is not None:
                op.fail(exc)
            raise

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
            self._is_started = False

        # A turn the server never closed was abandoned by this close.
        self._cancel_all_turns()
        return stats


class StreamingSessionSync:
    """Synchronous wrapper for StreamingSession."""

    def __init__(self, session: StreamingSession):
        self._session = session
        self._loop = asyncio.new_event_loop()

    def __enter__(self) -> StreamingSessionSync:
        self._loop.run_until_complete(self._session.connect())
        return self

    def __exit__(self, *args) -> None:
        self._loop.run_until_complete(self._session.close())
        self._loop.close()

    def send(self, text: str, flush: bool = False) -> List[AudioChunk]:
        """Send text and return generated audio chunks."""

        async def collect() -> List[AudioChunk]:
            chunks: List[AudioChunk] = []
            async for chunk in self._session.send(text, flush=flush):
                chunks.append(chunk)
            return chunks

        return self._loop.run_until_complete(collect())

    def flush(self) -> List[AudioChunk]:
        """Flush buffer and return remaining audio chunks."""

        async def collect() -> List[AudioChunk]:
            chunks: List[AudioChunk] = []
            async for chunk in self._session.flush():
                chunks.append(chunk)
            return chunks

        return self._loop.run_until_complete(collect())

    def drain(self) -> List[AudioChunk]:
        """End the session and return any remaining audio chunks.

        See :meth:`StreamingSession.drain` for the rationale.
        """

        async def collect() -> List[AudioChunk]:
            chunks: List[AudioChunk] = []
            async for chunk in self._session.drain():
                chunks.append(chunk)
            return chunks

        return self._loop.run_until_complete(collect())

    def cancel_current(self) -> None:
        """Interrupt (barge-in) the current generation, keeping the socket open.

        See :meth:`StreamingSession.cancel_current` for the rationale.
        """
        return self._loop.run_until_complete(self._session.cancel_current())

    def update_settings(
        self,
        *,
        cfg_scale: Optional[float] = None,
        temperature: Optional[float] = None,
        speed: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        language: Optional[str] = None,
        normalize: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Change generation parameters mid-connection (KUG-1166).

        See :meth:`StreamingSession.update_settings` for the semantics.
        """
        return self._loop.run_until_complete(
            self._session.update_settings(
                cfg_scale=cfg_scale,
                temperature=temperature,
                speed=speed,
                max_new_tokens=max_new_tokens,
                language=language,
                normalize=normalize,
            )
        )

    @property
    def last_word_timestamps(self) -> List[WordTimestamp]:
        """Return the most recently received word timestamps."""
        return self._session.last_word_timestamps

    @property
    def last_usage(self) -> Optional[SessionUsage]:
        """Per-session usage from the most recently closed session."""
        return self._session.last_usage

    @property
    def last_final(self) -> Optional[Dict[str, Any]]:
        """End-of-audio stats from the most recently completed turn."""
        return self._session.last_final

    def close(self) -> Dict[str, Any]:
        """Close the session."""
        return self._loop.run_until_complete(self._session.close())


class MultiContextSession(_DiagnosedSession):
    """WebSocket session for multi-context TTS streaming.

    Allows managing up to 20 independent audio generation contexts over
    a single WebSocket connection. Each context has its own text buffer,
    voice settings, and generation queue.

    Use cases:
    - Multi-speaker conversations with different voices
    - Pre-buffering audio while another stream plays
    - Interleaved audio generation for dynamic conversations

    Example:
        async with client.tts.multi_context_session() as session:
            # Create contexts with different voices
            await session.create_context("narrator", voice_id=123)
            await session.create_context("character", voice_id=456)

            # Send text to different speakers
            async for chunk in session.send("narrator", "The story begins."):
                play_audio("narrator", chunk)

            async for chunk in session.send("character", "Hello!"):
                play_audio("character", chunk)

            # Close specific context
            await session.close_context("narrator")
    """

    _operation_name = OP_MULTI_CONTEXT

    def __init__(
        self,
        api_key: str,
        tts_url: str,
        default_voice_id: Optional[int] = None,
        model_id: Optional[str] = None,
        sample_rate: int = 24000,
        output_format: Optional[str] = None,
        cfg_scale: float = 2.0,
        temperature: Optional[float] = None,
        max_new_tokens: int = 2048,
        normalize: bool = True,
        language: Optional[str] = None,
        dictionary_ids: Optional[List[int]] = None,
        inactivity_timeout: float = 20.0,
        word_timestamps: bool = False,
        on_word_timestamps: Optional[Callable[[str, List[WordTimestamp]], None]] = None,
        diagnostics: Optional[Diagnostics] = None,
        project_id: Optional[int] = None,
    ):
        cfg_scale = clamp_cfg_scale(cfg_scale)
        self._diagnostics = diagnostics or _DISABLED_DIAGNOSTICS
        self._turn_ops: Dict[Optional[str], Operation] = {}
        self._api_key = api_key
        self._tts_url = tts_url
        self._default_voice_id = default_voice_id
        self._model_id = model_id
        self._sample_rate = sample_rate
        # Combined codec+rate token (e.g. 'ulaw_8000'); opt-in, set-once per session.
        self._output_format = output_format
        self._cfg_scale = cfg_scale
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._normalize = normalize
        self._language = language
        # Per-request dictionary selection (None = project default,
        # [] = opt-out, list = exactly those dictionaries).
        self._dictionary_ids = dictionary_ids
        # Project whose dictionaries apply; required for any dictionary to
        # apply and for a non-empty dictionary_ids.
        self._project_id = project_id
        self._inactivity_timeout = inactivity_timeout
        self._word_timestamps = word_timestamps
        self._ws: Optional[Any] = None
        self._connect_task: asyncio.Task[None] | None = None
        self._session_id: Optional[str] = None
        self._is_started = False
        self._contexts: Set[str] = set()
        self._pending_messages: List[Dict[str, Any]] = []
        self._on_word_timestamps = on_word_timestamps
        self._last_word_timestamps: Dict[str, List[WordTimestamp]] = {}
        # Per-context usage (KUG-1192): each context is its own conversation,
        # so usage arrives on that context's ``context_closed`` frame. Keyed
        # by context_id.
        self._context_usage: Dict[str, SessionUsage] = {}
        # Replay state for rolling-deploy closes (1012 / 1013): every
        # context created on this socket with its voice override (insertion
        # order), the frames of each context's current turn, the contexts
        # that already received audio for that turn, and the contexts whose
        # turn was replayed once already.
        self._context_voices: Dict[str, Optional[int]] = {}
        self._turn_frames: Dict[str, List[Dict[str, Any]]] = {}
        self._turn_audio: Set[str] = set()
        self._turn_replayed: Set[str] = set()
        self._recovering = False

    def _end_turn(self, context_id: Optional[str]) -> None:
        if context_id is None:
            return
        self._turn_frames.pop(context_id, None)
        self._turn_audio.discard(context_id)
        self._turn_replayed.discard(context_id)

    def _forget_context(self, context_id: str) -> None:
        self._contexts.discard(context_id)
        self._context_voices.pop(context_id, None)
        self._end_turn(context_id)

    def _note_turn_event(self, context_id: Optional[str], data: Dict[str, Any]) -> None:
        """Track turn boundaries from a server frame, for any context."""
        if context_id is None:
            return
        if data.get("audio"):
            self._turn_audio.add(context_id)
        if data.get("final") or data.get("context_closed"):
            self._end_turn(context_id)
            # The context's final/done frame: its turn succeeded.
            self._finish_turn(context_id)

    async def _recover_from_restart(
        self,
        context_id: Optional[str],
        code: Optional[int],
        reason: Optional[str],
        cause: Optional[BaseException] = None,
    ) -> bool:
        """Absorb a rolling-deploy close (1012 / 1013).

        Reconnects after the close's ``retry_after`` hint, re-creates every
        context that existed on the old socket (same voice override) and
        resends the unspoken turn of each context that had not received
        audio yet, so the caller never notices the deploy. A context whose
        turn was already partly spoken loses the rest of that turn (logged
        as a warning) because replaying it would repeat audio.

        Returns ``True`` when frames of *context_id* were replayed and the
        caller should keep receiving on the new socket, ``False`` otherwise.

        Raises :class:`ServerRestartingError` for *context_id* when its
        turn already produced audio, was replayed once already, or the
        close happened while a recovery was in progress.
        """
        err = classify_ws_close(code, reason)
        if (
            self._recovering
            or context_id in self._turn_audio
            or context_id in self._turn_replayed
        ):
            self._reset_dead_session()
            raise err from cause
        contexts = list(self._context_voices.items())
        frames = self._turn_frames
        spoken = self._turn_audio
        self._reset_dead_session()
        self._recovering = True
        # The reconnect is a retry of the turn that hit the close; a failed
        # reconnect is that turn's failure, not a separate connect event.
        trigger_turn = (
            self._live_turn(context_id) if context_id is not None else None
        )
        if trigger_turn is not None:
            trigger_turn.record_retry()
        try:
            delay = _restart_delay_s(err)
            logger.info(
                "Server restarting (%s); reconnecting in %.1fs and restoring "
                "%d context(s)",
                code,
                delay,
                len(contexts),
            )
            await asyncio.sleep(delay)
            await self._establish(trigger_turn)
            # Setup reads consume context_created acknowledgements. Finish
            # every setup before replay can produce audio on the shared socket.
            for cid, voice_id in contexts:
                await self.create_context(cid, voice_id)
            replayed = False
            for cid, _voice_id in contexts:
                if cid in spoken:
                    logger.warning(
                        "Context %s lost the rest of its in-flight turn to a "
                        "server restart; resend it on the new socket",
                        cid,
                    )
                    continue
                turn = frames.get(cid) or []
                if not turn:
                    continue
                self._turn_replayed.add(cid)
                replayed_op = self._live_turn(cid)
                if cid != context_id and replayed_op is not None:
                    replayed_op.record_retry()
                for frame in turn:
                    await self._ws_send(frame, replayable=True)
                if replayed_op is not None:
                    replayed_op.expect_audio()
                if cid == context_id:
                    replayed = True
            if trigger_turn is not None:
                # Left at ``handshake`` by the reconnect; waiting for audio again.
                trigger_turn.expect_audio()
            return replayed
        finally:
            self._recovering = False

    def _capture_context_usage(self, data: Dict[str, Any]) -> None:
        """Parse a ``context_closed`` frame's per-context usage, if present."""
        context_id = data.get("context_id")
        usage = SessionUsage.from_session_payload(data)
        if context_id is not None and usage is not None:
            self._context_usage[context_id] = usage

    def usage_for(self, context_id: str) -> Optional[SessionUsage]:
        """Per-context usage (audio time + charge) for a closed context.

        Populated when the server sends ``context_closed`` for *context_id*.
        ``None`` if that context hasn't closed yet. Use this to bill your own
        customers per conversation — see :class:`~kugelaudio.SessionUsage`.
        """
        return self._context_usage.get(context_id)

    @property
    def context_usage(self) -> Dict[str, SessionUsage]:
        """Map of context_id → per-context usage for all closed contexts."""
        return dict(self._context_usage)

    def get_word_timestamps(self, context_id: str) -> List[WordTimestamp]:
        """Return the most recently received word timestamps for *context_id*."""
        return self._last_word_timestamps.get(context_id, [])

    @property
    def is_alive(self) -> bool:
        """True iff the underlying WebSocket is still usable for send().

        False after the WS has been closed (clean or otherwise) or has been
        reset by ``_reset_dead_session`` after a mid-stream drop. Callers that
        cache the session across turns (e.g. the Pipecat wrapper) should check
        this before reuse and reconnect when False.
        """
        ws = self._ws
        if ws is None:
            return False
        # websockets exposes a ``state`` enum; State.OPEN == 1. Avoid importing
        # the enum to stay compatible across websockets minor versions.
        state = getattr(ws, "state", None)
        if state is not None:
            name = getattr(state, "name", "")
            return name == "OPEN"
        # Fallback: assume open if we have a ws and no state attribute.
        return True

    def _reset_dead_session(self) -> None:
        """Drop all references to a dead WebSocket so the next operation
        can either reconnect (via ``connect``) or raise cleanly. Idempotent.
        """
        self._ws = None
        self._is_started = False
        self._contexts.clear()
        self._pending_messages.clear()
        self._context_voices = {}
        self._turn_frames = {}
        self._turn_audio = set()
        self._turn_replayed = set()

    async def _ws_send(
        self, payload: Dict[str, Any], *, replayable: bool = False
    ) -> bool:
        """Write a JSON payload to the WS, classifying a mid-send drop.

        A half-open connection (TCP gone, no close handshake) leaves
        ``ws.state == OPEN``, so the ``is_alive`` guard in :meth:`send`
        cannot pre-empt it — the drop only surfaces when ``ws.send`` itself
        fails with the raw ``ConnectionClosedError("no close frame received
        or sent")``. Mirror the teardown + classification that
        :meth:`_receive_audio` already performs so every WS write path
        raises a typed :class:`KugelAudioConnectionError` (or a server-close
        classification) instead of leaking a raw websockets exception.

        A rolling-deploy close (1012 / 1013) is absorbed by
        :meth:`_recover_from_restart`. Returns ``True`` when that happened:
        the socket is new, every context was re-created, and *payload* was
        resent only if it was a *replayable* turn frame (already recorded in
        ``_turn_frames``). Other payloads must be resent by the caller.
        """
        try:
            await self._ws.send(json.dumps(payload))
            return False
        except Exception as e:
            if "ConnectionClosed" in str(type(e)):
                code = getattr(e, "code", None)
                if code in _WS_RESTART_CLOSE_CODES:
                    await self._recover_from_restart(
                        payload.get("context_id"), code, getattr(e, "reason", None), e
                    )
                    return True
                # Tear down so the next send() reconnects on a fresh socket
                # rather than writing to the dead one again.
                self._reset_dead_session()
                if code in _WS_ERROR_CLOSE_CODES:
                    raise classify_ws_close(
                        code, getattr(e, "reason", None)
                    ) from e
                raise KugelAudioConnectionError(
                    "KugelAudio WebSocket dropped while sending "
                    f"(close code={code}). Caller should retry on a "
                    "fresh session."
                ) from e
            raise

    async def __aenter__(self) -> "MultiContextSession":
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def _connect_socket(self) -> None:
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets required. Install with: pip install websockets"
            )

        ws_url = self._tts_url.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = _append_sdk_query(f"{ws_url}/ws/tts/multi?api_key={self._api_key}")

        handshake_errors = ws_handshake_error_types(websockets)
        try:
            # Tighter than the websockets default of 20s/20s. A 15s ping cadence
            # keeps the WS alive across NAT/proxy idle timeouts that show up in
            # the wild around 30s (some macOS NAT, corporate firewalls, mobile
            # carriers). Without these, an idle conversation pause of ~20-30s
            # can drop the connection without a close handshake, and the next
            # send() raises ConnectionClosedError("no close frame received").
            self._ws = await websockets.connect(
                ws_url, ping_interval=15.0, ping_timeout=15.0, close_timeout=5.0
            )
        except handshake_errors as e:
            typed = classify_ws_handshake_error(e)
            if typed is not None:
                raise typed from e
            raise KugelAudioConnectionError(
                f"KugelAudio multi-context WebSocket handshake failed: {e}."
            ) from e

    async def _start_session(
        self, context_id: str, voice_id: Optional[int] = None
    ) -> None:
        """Start the session with first context creation.

        The server's /ws/tts/multi endpoint does NOT send a ``session_started``
        message.  Instead, the first message that references a ``context_id``
        triggers an implicit ``create_context`` on the server side, which
        responds with ``{"context_created": true, "context_id": "..."}``.

        Per-context voice / generation settings must be nested inside a
        ``voice_settings`` dict (see ``create_context`` helper in the server).
        """
        if self._is_started:
            return

        if not self._ws:
            await self.connect()

        self._context_voices.setdefault(context_id, voice_id)

        from kugelaudio.client import _warn_if_no_language

        _warn_if_no_language(self._language, self._normalize)

        effective_voice_id = voice_id or self._default_voice_id
        init_msg: Dict[str, Any] = {
            "text": " ",
            "context_id": context_id,
            "sample_rate": self._sample_rate,
            "normalize": self._normalize,
            "word_timestamps": self._word_timestamps,
        }
        # Top-level output_format (mirrors sample_rate); resolved server-side
        # in SessionConfig.apply_update. Opt-in: absent ⇒ legacy PCM16.
        if self._output_format is not None:
            init_msg["output_format"] = self._output_format
        if self._model_id is not None:
            init_msg["model_id"] = self._model_id
        if self._language is not None:
            init_msg["language"] = self._language
        if self._temperature is not None:
            init_msg["temperature"] = self._temperature
        if self._project_id is not None:
            init_msg["project_id"] = self._project_id
        # [] is meaningful (explicit opt-out) and must be sent; only None
        # (use the project default) is omitted.
        if self._dictionary_ids is not None:
            init_msg["dictionary_ids"] = self._dictionary_ids
        if effective_voice_id is not None:
            init_msg["voice_settings"] = {
                "voice_id": effective_voice_id,
                "cfg_scale": self._cfg_scale,
                "max_new_tokens": self._max_new_tokens,
            }

        if await self._ws_send(init_msg):
            return  # the restart recovery re-created every context

        # Wait for context_created confirmation (no session_started event)
        while True:
            msg = await asyncio.wait_for(
                self._ws.recv(), timeout=_DEFAULT_RECV_TIMEOUT_S
            )
            data = json.loads(msg)

            if data.get("error"):
                self._raise_frame_error(data, context_id)

            if data.get("context_created"):
                self._contexts.add(data["context_id"])
                self._is_started = True
                logger.debug(
                    "Multi-context session started, first context: %s",
                    data["context_id"],
                )
                break

    async def create_context(
        self,
        context_id: str,
        voice_id: Optional[int] = None,
    ) -> None:
        """Create a new context with optional voice override.

        Args:
            context_id: Unique identifier for this context
            voice_id: Optional voice ID (uses default if not specified)
        """
        if not self._is_started:
            await self._start_session(context_id, voice_id)
            return

        if context_id in self._contexts:
            return  # Already exists

        self._context_voices.setdefault(context_id, voice_id)

        # Send context initialization — voice_id must be in voice_settings
        effective_voice_id = voice_id or self._default_voice_id
        msg: Dict[str, Any] = {"text": " ", "context_id": context_id}
        if self._model_id is not None:
            msg["model_id"] = self._model_id
        if effective_voice_id is not None:
            msg["voice_settings"] = {
                "voice_id": effective_voice_id,
                "cfg_scale": self._cfg_scale,
                "max_new_tokens": self._max_new_tokens,
            }

        if await self._ws_send(msg):
            return  # the restart recovery re-created every context

        # Wait for confirmation
        while True:
            response = await self._ws.recv()
            data = json.loads(response)

            if data.get("error"):
                self._raise_frame_error(data, context_id)

            if data.get("context_created") and data.get("context_id") == context_id:
                self._contexts.add(context_id)
                break

    async def update_settings(
        self,
        *,
        cfg_scale: Optional[float] = None,
        temperature: Optional[float] = None,
        speed: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        language: Optional[str] = None,
        normalize: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Change the session's generation parameters mid-connection (KUG-1166).

        Session-scoped (there is no ``context_id``): the update applies to
        contexts started **after** it — a context already streaming keeps the
        settings it began with, since generation parameters are bound when a
        context's engine session opens. With the common one-context-per-turn
        pattern that means the change takes effect on the next turn.

        Only the six parameters in this signature are updatable; identity /
        audio-format fields are fixed for the connection. Per-context
        ``cfg_scale`` / ``max_new_tokens`` passed to :meth:`create_context`
        still win for that context.

        Returns:
            The generation parameters now in effect (the server's echo).

        Raises:
            ValueError: if no parameter was provided.
            KugelAudioError: if the server rejects the update.
        """
        body = _settings_update_body(
            cfg_scale=cfg_scale,
            temperature=temperature,
            speed=speed,
            max_new_tokens=max_new_tokens,
            language=language,
            normalize=normalize,
        )
        if not self._ws:
            await self.connect()

        if await self._ws_send({"update_settings": body}):
            await self._ws_send({"update_settings": body})
        while True:
            msg = await asyncio.wait_for(
                self._ws.recv(), timeout=_DEFAULT_RECV_TIMEOUT_S
            )
            data = json.loads(msg)
            if data.get("error"):
                self._raise_frame_error(data)
            if data.get("settings_updated"):
                settings = data.get("settings")
                # Mirror onto local session fields only after the server
                # accepts the update, so rejected values do not leak into
                # future context creation.
                for name, value in body.items():
                    attr = f"_{name}"
                    if hasattr(self, attr):
                        setattr(self, attr, value)
                return settings if isinstance(settings, dict) else {}

    async def send(
        self,
        context_id: str,
        text: str,
        flush: bool = False,
        chunk_complete_idle_timeout: Optional[float] = None,
    ) -> AsyncIterator[AudioChunk]:
        """Send text to a specific context and yield audio chunks.

        Args:
            context_id: Context to send text to
            text: Text to synthesize
            flush: Force flush the buffer
            chunk_complete_idle_timeout: Override how long to wait after
                ``chunk_complete`` for follow-up chunks. ``0`` returns
                immediately after the first completion signal.

        Yields:
            AudioChunk as audio is generated
        """
        op = self._turn(context_id)
        async with op.step(STAGE_SENDING_REQUEST):
            async for chunk in self._send_frames(
                context_id,
                text,
                flush=flush,
                chunk_complete_idle_timeout=chunk_complete_idle_timeout,
                op=op,
            ):
                op.record_chunk(len(chunk.audio))
                yield chunk

    async def _send_frames(
        self,
        context_id: str,
        text: str,
        *,
        flush: bool,
        chunk_complete_idle_timeout: Optional[float],
        op: Operation,
    ) -> AsyncIterator[AudioChunk]:
        """The send/receive engine behind :meth:`send`."""
        if not self.is_alive:
            close_code = getattr(self._ws, "close_code", None)
            if close_code in _WS_RESTART_CLOSE_CODES:
                # The server closed an idle socket for a rolling deploy;
                # nobody was receiving, so it surfaces here. Reconnect and
                # restore the contexts before sending.
                await self._recover_from_restart(context_id, close_code, None)
            else:
                # WS was dropped (proxy idle, network blip) and state was
                # already torn down by ``_reset_dead_session``. Surface a
                # typed error rather than crashing inside ``await
                # self._ws.send`` with the stack-trace-only
                # ConnectionClosedError("no close frame received or sent")
                # that callers can't easily catch.
                self._reset_dead_session()
                raise KugelAudioConnectionError(
                    "KugelAudio WebSocket is not connected. Reconnect with "
                    "session.connect() (or recreate the session) before send()."
                )

        if not self._is_started:
            await self._start_session(context_id)

        if context_id not in self._contexts:
            await self.create_context(context_id)

        frame = {"text": text, "context_id": context_id, "flush": flush}
        self._turn_frames.setdefault(context_id, []).append(frame)
        op.set_stage(STAGE_SENDING_REQUEST)
        await self._ws_send(frame, replayable=True)
        op.expect_audio()

        # Collect audio for this context. The server marks each text chunk's
        # audio delivery with `chunk_complete`; the sole terminal signal is
        # `context_closed`, which only arrives once `close_context` is called.
        async for chunk in self._receive_audio(
            context_id,
            wait_for_chunk_complete=flush,
            chunk_complete_idle_timeout=chunk_complete_idle_timeout,
        ):
            yield chunk

    async def flush(self, context_id: str) -> AsyncIterator[AudioChunk]:
        """Flush a specific context's buffer.

        Args:
            context_id: Context to flush

        Yields:
            AudioChunk as remaining audio is generated
        """
        if not self._is_started or context_id not in self._contexts:
            return

        op = self._live_turn(context_id)
        async for chunk in self._within_turn(
            op, STAGE_SENDING_REQUEST, self._flush_frames(context_id, op)
        ):
            yield chunk

    async def _flush_frames(
        self, context_id: str, op: Optional[Operation]
    ) -> AsyncIterator[AudioChunk]:
        """The flush exchange behind :meth:`flush`."""
        frame = {"flush": True, "context_id": context_id}
        self._turn_frames.setdefault(context_id, []).append(frame)
        await self._ws_send(frame, replayable=True)

        # flush yields every audio chunk produced by the buffer drain,
        # ending once chunk_complete arrives (or we idle out). Use
        # close_context for terminal drains.
        if op is not None:
            op.expect_audio()
        async for chunk in self._receive_audio(
            context_id,
            wait_for_chunk_complete=True,
        ):
            yield chunk

    async def close_context(
        self, context_id: str, immediate: bool = False
    ) -> AsyncIterator[AudioChunk]:
        """Close a specific context and get remaining audio.

        Args:
            context_id: Context to close
            immediate: When ``True``, **barge-in**: the server cancels the
                context's in-flight generation immediately and discards any
                buffered or queued text instead of draining it. Use when the
                end user speaks over the agent — the iterator then yields
                little or no tail audio. When ``False`` (default), queued
                sentences finish first.

        Yields:
            AudioChunk for any remaining buffered text
        """
        if not self._is_started or context_id not in self._contexts:
            return

        if immediate:
            # Barge-in: a caller-initiated cancellation of THIS context's
            # turn only, counted and never reported as an event. Other
            # contexts' turns are untouched.
            self._cancel_turn(context_id)
        async for chunk in self._within_turn(
            None if immediate else self._live_turn(context_id),
            STAGE_FINALIZING,
            self._close_context_frames(context_id, immediate),
        ):
            yield chunk

    async def _close_context_frames(
        self, context_id: str, immediate: bool
    ) -> AsyncIterator[AudioChunk]:
        """The close-context exchange behind :meth:`close_context`."""
        msg: Dict[str, Any] = {
            "close_context": True,
            "context_id": context_id,
        }
        if immediate:
            msg["immediate"] = True
        if await self._ws_send(msg):
            await self._ws_send(msg)

        async for chunk in self._receive_audio(context_id, wait_for_close=True):
            yield chunk

        self._forget_context(context_id)

    async def keep_alive(self, context_id: str) -> None:
        """Reset inactivity timeout for a context.

        Args:
            context_id: Context to keep alive
        """
        if self._is_started and context_id in self._contexts:
            await self._ws.send(
                json.dumps(
                    {
                        "text": "",
                        "context_id": context_id,
                    }
                )
            )

    async def _receive_audio(
        self,
        context_id: str,
        wait_for_close: bool = False,
        wait_for_chunk_complete: bool = False,
        chunk_complete_idle_timeout: Optional[float] = None,
    ) -> AsyncIterator[AudioChunk]:
        """Receive audio messages for a specific context.

        The server sends (per context):
          - {"generation_started": true, "context_id": "...", ...}
          - {"audio": "<b64>", "context_id": "...", ...}
          - {"word_timestamps": [...], "context_id": "...", ...}
          - {"chunk_complete": true, "context_id": "...", ...}
          - {"context_closed": true, "context_id": "..."}

        Multi-sentence ``send(flush=True)`` / ``flush()`` calls produce N
        ``chunk_complete`` frames (one per server-side sentence). We wait
        ``_INTER_CHUNK_IDLE_S`` after each ``chunk_complete`` before
        returning, so the caller receives every chunk.
        """
        received_any_audio = False
        last_chunk_complete = False
        idle_after_chunk_complete = (
            _INTER_CHUNK_IDLE_S
            if chunk_complete_idle_timeout is None
            else chunk_complete_idle_timeout
        )

        while True:
            # Use longer timeout when waiting for a definitive server signal
            if wait_for_close:
                timeout = _DEFAULT_RECV_TIMEOUT_S
            elif last_chunk_complete:
                timeout = idle_after_chunk_complete
            elif wait_for_chunk_complete:
                timeout = _DEFAULT_RECV_TIMEOUT_S
            else:
                timeout = (
                    _POLL_TIMEOUT_S if received_any_audio else _DEFAULT_RECV_TIMEOUT_S
                )

            try:
                msg = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
                data = json.loads(msg)

                if data.get("error"):
                    self._raise_frame_error(data, context_id)

                msg_ctx = data.get("context_id")
                self._note_turn_event(msg_ctx, data)

                # Skip messages for other contexts — stash if needed later
                if msg_ctx != context_id:
                    self._pending_messages.append(data)
                    continue

                if data.get("generation_started"):
                    received_any_audio = False
                    last_chunk_complete = False
                    continue

                # Only yield audio for matching context
                if data.get("audio"):
                    received_any_audio = True
                    last_chunk_complete = False
                    yield AudioChunk.from_dict(data)

                if "word_timestamps" in data:
                    stamps = [
                        WordTimestamp.from_dict(w) for w in data["word_timestamps"]
                    ]
                    self._last_word_timestamps[context_id] = stamps
                    if self._on_word_timestamps:
                        self._on_word_timestamps(context_id, stamps)
                    continue

                # chunk_complete = one text-chunk done (there may be more)
                if data.get("chunk_complete"):
                    if not wait_for_close and idle_after_chunk_complete <= 0:
                        break
                    last_chunk_complete = True
                    continue

                # final = all audio admitted before the client's flush has
                # been delivered for this context (KUG-1238; the ElevenLabs
                # ``is_final`` equivalent). Deterministic flush-complete
                # signal — exit immediately instead of waiting out the
                # inter-chunk idle heuristic. On a close drain the terminal
                # signal stays ``context_closed`` (final precedes it).
                if data.get("final"):
                    if not wait_for_close:
                        break
                    continue

                # context_closed = context removed (terminal). Carries this
                # context's per-conversation usage (audio time + charge).
                if data.get("context_closed"):
                    self._capture_context_usage(data)
                    if wait_for_close:
                        break

            except asyncio.TimeoutError:
                # Idle after chunk_complete → caller's text is fully
                # rendered; idle on the initial poll → nothing more yet.
                break
            except Exception as e:
                if "ConnectionClosed" in str(type(e)):
                    code = getattr(e, "code", None)
                    if code in _WS_RESTART_CLOSE_CODES:
                        # Rolling deploy: restore the contexts on a fresh
                        # socket and replay this context's unspoken turn.
                        # Raises when audio for it already went out.
                        if last_chunk_complete and not wait_for_close:
                            # Every admitted chunk was delivered; nothing
                            # of this turn is left to replay.
                            self._end_turn(context_id)
                        if not await self._recover_from_restart(
                            context_id, code, getattr(e, "reason", None), e
                        ):
                            break
                        received_any_audio = False
                        last_chunk_complete = False
                        continue
                    # Tear down session state so the next send() reconnects
                    # cleanly instead of calling .send() on a dead socket and
                    # raising ConnectionClosedError("no close frame received").
                    self._reset_dead_session()
                    if last_chunk_complete and not wait_for_close:
                        # The server confirmed every admitted chunk; the
                        # socket died only during the idle linger for a
                        # possible next sentence. The caller received the
                        # whole turn, so this is a reconnect condition for
                        # the next send, not a delivery failure.
                        logger.warning(
                            "KugelAudio WebSocket closed after chunk_complete "
                            "(close code=%s); next turn will reconnect",
                            code,
                        )
                        break
                    if code in _WS_ERROR_CLOSE_CODES:
                        raise classify_ws_close(
                            code, getattr(e, "reason", None)
                        ) from e
                    raise KugelAudioConnectionError(
                        "KugelAudio WebSocket dropped mid-stream "
                        f"(close code={code}). Caller should retry on a "
                        "fresh session."
                    ) from e
                raise

    async def close(self) -> Dict[str, Any]:
        """Close the session and return stats.

        Sends a close command and drains messages until the server responds
        with ``session_closed``.  Intermediate messages (context_closed,
        audio, chunk_complete) are silently consumed.

        Returns:
            Session statistics including total audio generated
        """
        if self._connect_task is not None:
            self._connect_task.cancel()
            # The connect caller observes cancellation; wait for transport cleanup.
            await asyncio.gather(self._connect_task, return_exceptions=True)

        stats: Dict[str, Any] = {}

        if self._ws:
            try:
                await self._ws.send(json.dumps({"close_socket": True}))

                # Drain messages until session_closed or timeout. Capture
                # per-context usage from any context_closed frames seen here.
                while True:
                    msg = await asyncio.wait_for(
                        self._ws.recv(), timeout=_DEFAULT_RECV_TIMEOUT_S
                    )
                    data = json.loads(msg)

                    if data.get("context_closed"):
                        self._capture_context_usage(data)
                        self._note_turn_event(data.get("context_id"), data)
                    if data.get("session_closed"):
                        stats = data
                        break

            except Exception:
                pass
            finally:
                await self._ws.close()
                self._reset_dead_session()

        # Turns the server never finished were abandoned by this close.
        self._cancel_all_turns()
        return stats

    @property
    def active_contexts(self) -> Set[str]:
        """Get the set of active context IDs."""
        return self._contexts.copy()

    @property
    def session_id(self) -> Optional[str]:
        """Get the session ID."""
        return self._session_id
