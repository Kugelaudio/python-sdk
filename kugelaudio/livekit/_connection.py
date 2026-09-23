# Copyright 2024 KugelAudio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transport layer for the KugelAudio LiveKit plugin.

Holds everything that talks to ``/ws/tts/multi``: the wire options
(:class:`_TTSOptions` and the validators that mirror the server's accepted
ranges), the outbound frame dataclasses, the per-context bookkeeping, and
:class:`_Connection` itself — the single persistent WebSocket that
multiplexes every context.

This module must never import from ``kugelaudio.livekit.tts``; the
dependency runs one way only (``tts`` -> ``_connection``).
"""

from __future__ import annotations

import asyncio
import base64
import collections
import json
import logging
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import aiohttp
from kugelaudio._diagnostics import (
    DISABLED as _DISABLED_DIAGNOSTICS,
    OP_MULTI_CONTEXT,
    TRANSPORT_WEBSOCKET,
    Operation,
)
from kugelaudio._sdk_metadata import sdk_query_string
from kugelaudio.exceptions import (
    ConnectionError as KugelAudioConnectionError,
    KugelAudioError,
    classify_handshake_rejection,
    classify_ws_frame,
)
from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.voice.io import TimedString

from .models import TTSModels

if TYPE_CHECKING:
    # Type-only: never imported at runtime, so ``tts.py`` -> ``_connection.py``
    # stays a one-way dependency.
    from .tts import SynthesizeStream

logger = logging.getLogger("kugelaudio.livekit")


def _append_sdk_query(url: str) -> str:
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{sdk_query_string()}"


def _word_timestamps_to_timed(
    timestamps: list[dict[str, Any]],
) -> list[TimedString]:
    """Convert server word_timestamps payload to LiveKit TimedString list.

    The server sends timestamps with ``start_ms`` / ``end_ms`` (integers),
    while LiveKit expects ``start_time`` / ``end_time`` in seconds (floats).
    """
    return [
        TimedString(
            ts["word"],
            start_time=ts["start_ms"] / 1000.0,
            end_time=ts["end_ms"] / 1000.0,
        )
        for ts in timestamps
    ]


SUPPORTED_LANGUAGES: set[str] = {
    # Germanic
    "de",
    "en",
    "nl",
    "sv",
    "da",
    "no",
    # Romance
    "fr",
    "es",
    "it",
    "pt",
    "ro",
    # Slavic
    "pl",
    "cs",
    "uk",
    "bg",
    "sk",
    "sl",
    "hr",
    "sr",
    # Uralic
    "fi",
    "hu",
    # Other European
    "el",
    "tr",
    "ru",
    # CJK + SEA
    "zh",
    "ja",
    "ko",
    "vi",
    "yue",
    "th",
    "id",
    "ms",
    # Semitic + Indic + Other
    "ar",
    "hi",
    "he",
    "fa",
    "ur",
    "bn",
    "ta",
}


def _validate_language(language: str | None) -> str | None:
    """Validate that language is an ISO 639-1 code supported by the API.

    Raises:
        ValueError: If *language* is not a supported ISO 639-1 code.
            Common mistake: passing BCP 47 locale tags such as ``"de-DE"``
            instead of ``"de"``.
    """
    if language is None:
        return None
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"language must be a supported ISO 639-1 code "
            f"({', '.join(sorted(SUPPORTED_LANGUAGES))}), got {language!r}. "
            f"Note: BCP 47 tags like 'de-DE' are not accepted — use 'de' instead."
        )
    return language


MIN_SPEED = 0.8
MAX_SPEED = 1.2


def _validate_speed(speed: float | None) -> float | None:
    """Validate the playback-speed multiplier against the server's range.

    ``None`` means "not set": the option is omitted from the wire so the
    server applies its own default of 1.0.

    Raises:
        ValueError: If *speed* is outside ``[0.8, 1.2]``.  The ingress
            **rejects** out-of-range values instead of clamping them, so
            clamping here would silently change what the caller asked for
            (and still not match what the server would have done).
    """
    if speed is None:
        return None
    if not MIN_SPEED <= speed <= MAX_SPEED:
        raise ValueError(
            f"speed must be within [{MIN_SPEED}, {MAX_SPEED}], got {speed!r}. "
            f"The API rejects out-of-range speed values rather than clamping "
            f"them; pass a value inside the range or leave speed unset."
        )
    return speed


# Public sampling-variance scale, matching ``TEMPERATURE_MIN``/``MAX`` in
# ``services/ingress/src/ingress/routes/models.py``. The server rejects
# out-of-range values (422) rather than clamping them.
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 1.0


def _validate_temperature(temperature: float | None) -> float | None:
    """Validate the sampling temperature against the server's range.

    ``None`` means "not set": the option is omitted from the wire so the
    engine applies its own default.

    Raises:
        ValueError: If *temperature* is outside [0.0, 1.0].
    """
    if temperature is None:
        return None
    if not MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE:
        raise ValueError(
            f"temperature must be within [{MIN_TEMPERATURE}, {MAX_TEMPERATURE}], "
            f"got {temperature!r}. The API rejects out-of-range values rather "
            f"than clamping them; pass a value inside the range or leave "
            f"temperature unset."
        )
    return temperature


def _validate_dictionary_selection(
    project_id: int | None, dictionary_ids: list[int] | None
) -> None:
    """Refuse a dictionary selection that names no project.

    The server rejects a non-empty ``dictionary_ids`` without ``project_id``
    on every synthesis, so fail at configuration time instead of on the
    first turn. ``[]`` (explicit opt-out
    from the project's defaults) needs no project.

    Raises:
        ValueError: If *dictionary_ids* is non-empty and *project_id* is None.
    """
    if dictionary_ids and project_id is None:
        raise ValueError(
            "dictionary_ids requires project_id: the server rejects a "
            "dictionary selection without its project."
        )


@dataclass
class _TTSOptions:
    model: TTSModels | str
    voice_id: int | None
    sample_rate: int
    cfg_scale: float
    max_new_tokens: int
    api_key: str
    base_url: str
    word_timestamps: bool = False
    normalize: bool = True
    language: str | None = None
    speed: float | None = None
    temperature: float | None = None
    project_id: int | None = None
    # None = the project's default dictionaries; [] = explicit opt-out.
    dictionary_ids: list[int] | None = None

    def __post_init__(self) -> None:
        self.language = _validate_language(self.language)
        self.speed = _validate_speed(self.speed)
        self.temperature = _validate_temperature(self.temperature)
        _validate_dictionary_selection(self.project_id, self.dictionary_ids)


@dataclass
class _SynthesizeContent:
    """Text message to send to a specific context."""

    context_id: str
    text: str
    flush: bool = False


@dataclass
class _CloseContext:
    """Signal to close a specific context on the server.

    ``immediate=True`` is sent on barge-in (caller cancelled _run).  The
    server cancels in-flight generation instead of draining — stops
    wasted GPU on audio the client will discard anyway.  End-of-stream
    closes from ``_input_task`` use the default (graceful drain).
    """

    context_id: str
    immediate: bool = False


def _disabled_turn() -> Operation:
    return _DISABLED_DIAGNOSTICS.operation(OP_MULTI_CONTEXT, TRANSPORT_WEBSOCKET)


@dataclass
class _ContextData:
    """Per-context state tracked inside _Connection.

    ``last_activity_at`` is refreshed every time the recv loop routes a
    server message to this context.  The per-run idle timeout uses it to
    distinguish "server is silent" from "server is still working".

    ``op`` is the owning stream's diagnostics operation, assigned by the
    stream right after registering (one operation per stream across LiveKit's
    ``_run`` retries). The connection only records onto it (audio, server
    error, close code); the stream alone settles it, because only the stream
    knows whether LiveKit will retry the attempt.
    """

    emitter: tts.AudioEmitter
    waiter: asyncio.Future[None]
    stream: "SynthesizeStream | None" = None
    last_activity_at: float = field(default_factory=time.monotonic)
    op: Operation = field(default_factory=_disabled_turn)


async def _wait_for_context_idle(
    ctx: _ContextData,
    waiter: asyncio.Future[None],
    idle_threshold: float,
) -> None:
    """Block until *waiter* resolves, failing only if the server has been
    silent for this context for longer than *idle_threshold* seconds.

    Unlike ``asyncio.wait_for(waiter, idle_threshold)``, this helper polls
    ``ctx.last_activity_at`` on every tick so long-running generations do
    not time out as long as frames keep arriving.
    """
    # Poll cadence is small so the effective idle-deadline resolution is
    # sub-second; callers only care that we do not fail early when frames
    # keep arriving.
    tick = min(1.0, max(0.1, idle_threshold / 4))
    while True:
        try:
            await asyncio.wait_for(asyncio.shield(waiter), timeout=tick)
            return
        except asyncio.TimeoutError:
            if time.monotonic() - ctx.last_activity_at >= idle_threshold:
                raise


def _api_status_error_from_ingress_payload(
    data: dict[str, Any], *, request_id: str = ""
) -> APIStatusError:
    """Convert an ingress WS error frame into LiveKit's status exception."""
    err = classify_ws_frame(data)
    status_code = err.status_code
    if status_code is None:
        code = data.get("code")
        status_code = code if isinstance(code, int) else 500
    return APIStatusError(
        message=err.message,
        status_code=status_code,
        request_id=request_id,
        body=data,
    )


def _handshake_rejection(exc: aiohttp.WSServerHandshakeError) -> KugelAudioError:
    """The typed SDK error for a rejected WS upgrade, carrying the rejection
    response's ``x-request-id`` exactly like the core SDK's websockets path."""
    status_code = exc.status if isinstance(exc.status, int) else 500
    typed = classify_handshake_rejection(
        status_code, exc.headers, exc.message or str(exc)
    )
    typed.__cause__ = exc
    return typed


def _api_status_error_from_handshake(
    exc: aiohttp.WSServerHandshakeError,
) -> APIStatusError:
    """Convert a rejected WS upgrade into LiveKit's status exception."""
    status_code = exc.status if isinstance(exc.status, int) else 500
    message = exc.message or str(exc)
    return APIStatusError(
        message=message,
        status_code=status_code,
        request_id=_handshake_rejection(exc).request_id or "",
        body={"error": message, "code": status_code},
    )


def _dropped_error() -> APIConnectionError:
    """LiveKit's error for a dropped socket, caused by the SDK's typed one."""
    error = APIConnectionError("WebSocket connection closed unexpectedly")
    error.__cause__ = KugelAudioConnectionError(
        "WebSocket connection closed unexpectedly"
    )
    return error


def _diagnostic_error(exc: BaseException) -> BaseException:
    """The error to report: the typed SDK error a LiveKit error wraps, if any.

    It carries ``error_code`` and the server request id; the LiveKit wrapper
    only carries the HTTP status.
    """
    cause = exc.__cause__
    return cause if isinstance(cause, KugelAudioError) else exc


class _Connection:
    """Single persistent WebSocket to /ws/tts/multi with background loops.

    Each synthesis (chunked or streaming) registers a unique context_id.
    The recv loop routes server messages by context_id; unknown IDs are
    silently dropped — this is what makes barge-in safe.
    """

    def __init__(self, opts: _TTSOptions, session: aiohttp.ClientSession) -> None:
        self._opts = replace(opts)
        self._session = session
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._is_current = True
        self._closed = False
        self._active_contexts: set[str] = set()
        self._input_queue: asyncio.Queue[_SynthesizeContent | _CloseContext | None] = (
            asyncio.Queue()
        )
        self._context_data: dict[str, _ContextData] = {}
        # ctx_ids we have already logged a "dropped late frame" for —
        # bounded so a long-lived connection doesn't grow unboundedly.
        self._logged_late_drops: collections.deque[str] = collections.deque(maxlen=64)
        self._send_task: asyncio.Task | None = None
        self._recv_task: asyncio.Task | None = None

    @property
    def is_current(self) -> bool:
        return self._is_current

    def mark_non_current(self) -> None:
        self._is_current = False

    async def connect(self, timeout: float = 10.0) -> None:
        ws_url = self._opts.base_url.replace("https://", "wss://").replace(
            "http://", "ws://"
        )
        ws_url = _append_sdk_query(f"{ws_url}/ws/tts/multi?api_key={self._opts.api_key}")
        try:
            self._ws = await self._session.ws_connect(
                ws_url,
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=timeout),
            )
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.WSServerHandshakeError as e:
            raise _api_status_error_from_handshake(e) from _handshake_rejection(e)
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Failed to connect to /ws/tts/multi: {e}") from e
        self._send_task = asyncio.create_task(self._send_loop())
        self._recv_task = asyncio.create_task(self._recv_loop())
        logger.debug("Connection established to /ws/tts/multi")

    def register_context(
        self,
        context_id: str,
        emitter: tts.AudioEmitter,
        waiter: asyncio.Future[None],
        stream: "SynthesizeStream | None" = None,
    ) -> _ContextData | None:
        # If the connection closed between _ensure_connection returning and
        # this call (e.g., update_options raced with an in-flight _run),
        # reject immediately instead of inserting a waiter nobody will resolve.
        if self._closed:
            if not waiter.done():
                waiter.set_exception(
                    APIConnectionError("Connection closed before request started")
                )
            return None
        ctx = _ContextData(
            emitter=emitter,
            waiter=waiter,
            stream=stream,
        )
        self._context_data[context_id] = ctx
        return ctx

    def send_content(self, content: _SynthesizeContent) -> None:
        self._input_queue.put_nowait(content)

    def request_close_context(
        self, context_id: str, immediate: bool = False,
    ) -> None:
        self._input_queue.put_nowait(_CloseContext(context_id, immediate=immediate))

    def _cleanup_context(self, context_id: str) -> None:
        self._context_data.pop(context_id, None)
        self._active_contexts.discard(context_id)

    async def _send_loop(self) -> None:
        assert self._ws is not None
        try:
            if not self._closed and self._opts.voice_id is not None:
                # Config-only preparation runs before any text and creates no
                # synthesis context. Older servers ignore this frame; the
                # first text of every context still carries the full settings.
                await self._ws.send_str(json.dumps({
                    "text": "",
                    "voice_id": self._opts.voice_id,
                    "model_id": self._opts.model,
                }))
            while not self._closed:
                item = await self._input_queue.get()
                if item is None:
                    break

                if isinstance(item, _CloseContext):
                    payload: dict[str, Any] = {
                        "close_context": True,
                        "context_id": item.context_id,
                    }
                    if item.immediate:
                        payload["immediate"] = True
                    await self._ws.send_str(json.dumps(payload))
                    continue

                # _SynthesizeContent
                msg: dict[str, Any] = {
                    "text": item.text,
                    "context_id": item.context_id,
                }
                if item.flush:
                    msg["flush"] = True

                # First message for a new context: attach session + voice config
                if item.context_id not in self._active_contexts:
                    self._active_contexts.add(item.context_id)
                    msg["model_id"] = self._opts.model
                    msg["sample_rate"] = self._opts.sample_rate
                    msg["word_timestamps"] = self._opts.word_timestamps
                    msg["normalize"] = self._opts.normalize
                    if self._opts.language is not None:
                        msg["language"] = self._opts.language
                    # ``speed`` is a TOP-LEVEL StreamUpdate field on
                    # /ws/tts/multi.  Nesting it inside ``voice_settings``
                    # makes the server discard it with only a server-side
                    # log line — the client would never learn the rate was
                    # ignored.  Omitted when unset so the server default
                    # (1.0) applies.
                    if self._opts.speed is not None:
                        msg["speed"] = self._opts.speed
                    # Same rule as ``speed``: a session-wide StreamUpdate
                    # field, dropped with only a server log if nested.
                    if self._opts.temperature is not None:
                        msg["temperature"] = self._opts.temperature
                    # Top-level StreamUpdate fields too. Dictionaries apply
                    # only when the frame names their project; [] is an
                    # explicit opt-out and is sent, only None is omitted.
                    if self._opts.project_id is not None:
                        msg["project_id"] = self._opts.project_id
                    if self._opts.dictionary_ids is not None:
                        msg["dictionary_ids"] = self._opts.dictionary_ids
                    voice_settings: dict[str, Any] = {}
                    if self._opts.voice_id is not None:
                        voice_settings["voice_id"] = self._opts.voice_id
                    if self._opts.cfg_scale != 2.0:
                        voice_settings["cfg_scale"] = self._opts.cfg_scale
                    if self._opts.max_new_tokens != 2048:
                        voice_settings["max_new_tokens"] = self._opts.max_new_tokens
                    if voice_settings:
                        msg["voice_settings"] = voice_settings

                await self._ws.send_str(json.dumps(msg))
                sent_ctx = self._context_data.get(item.context_id)
                if sent_ctx is not None:
                    sent_ctx.op.expect_audio()
        except Exception:
            # Mark dead so _ensure_connection reconnects next call, and
            # close the WS so _recv_loop exits and rejects pending waiters.
            self._is_current = False
            if self._ws and not self._ws.closed:
                await self._ws.close()
            raise

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        try:
            await self._recv_loop_inner()
        finally:
            # Connection dropped or closed — mark dead so _ensure_connection
            # reconnects on the next call instead of handing out this zombie.
            self._is_current = False
            close_code = self._ws.close_code if self._ws is not None else None
            for ctx in list(self._context_data.values()):
                if not ctx.waiter.done():
                    if isinstance(close_code, int):
                        ctx.op.ws_close_code = close_code
                    ctx.waiter.set_exception(_dropped_error())

    async def _recv_loop_inner(self) -> None:
        assert self._ws is not None
        while not self._closed and not self._ws.closed:
            msg = await self._ws.receive()
            if msg.type != aiohttp.WSMsgType.TEXT:
                break
            data = json.loads(msg.data)

            if data.get("session_closed"):
                break

            context_id = data.get("context_id")

            # Error handling
            if data.get("error"):
                if context_id:
                    self._fail_context(context_id, data)
                    continue
                # A session-level error (MODEL_UNAVAILABLE, INSUFFICIENT_CREDITS,
                # ...) carries no context id: it fails every pending turn with
                # the server's own error, and this socket is done. Retries
                # must use a fresh connection.
                self._is_current = False
                for pending_id in list(self._context_data):
                    self._fail_context(pending_id, data)
                await self._ws.close()
                break

            ctx = self._context_data.get(context_id) if context_id else None

            # Messages for unknown/cleaned-up contexts are silently discarded.
            # This is the core barge-in fix.  After a barge-in there's
            # always a small tail of in-flight frames (audio + the
            # context_closed confirmation) — log only the first per
            # ctx_id so the second/third frames don't spam the log.
            if ctx is None:
                if context_id and context_id not in self._logged_late_drops:
                    self._logged_late_drops.append(context_id)
                    logger.debug(
                        "[recv] dropped late frame(s) for closed ctx=%s "
                        "(further drops for this ctx will be silenced)",
                        context_id,
                    )
                continue

            # Any routed message counts as activity for the idle-timeout
            # watchdog in _run.  Update before per-field processing so
            # audio/chunk_complete/context_closed all refresh the deadline.
            ctx.last_activity_at = time.monotonic()

            if data.get("audio"):
                audio_bytes = base64.b64decode(data["audio"])
                ctx.op.record_chunk(len(audio_bytes))
                if ctx.stream is not None:
                    ctx.stream._mark_started()
                ctx.emitter.push(audio_bytes)

            if data.get("word_timestamps"):
                ctx.emitter.push_timed_transcript(
                    _word_timestamps_to_timed(data["word_timestamps"])
                )

            if data.get("chunk_complete"):
                ctx.emitter.flush()
                # For non-streaming (chunked) contexts, close after the
                # server confirms generation is done — avoids the race
                # where close_context arrives before audio is generated.
                if ctx.stream is None:
                    self._input_queue.put_nowait(_CloseContext(context_id))

            if data.get("context_closed"):
                # Sole terminal signal. The server sends this only after it
                # has drained every audio frame for the context, so the
                # waiter resolves at the correct point for both streaming
                # and chunked contexts.
                if not ctx.waiter.done():
                    ctx.waiter.set_result(None)
                self._cleanup_context(context_id)

    def _fail_context(self, context_id: str, data: dict[str, Any]) -> None:
        """Reject *context_id*'s waiter with the server's error frame."""
        ctx = self._context_data.get(context_id)
        if ctx is not None and not ctx.waiter.done():
            ctx.op.mark_server_error(classify_ws_frame(data))
            ctx.waiter.set_exception(
                _api_status_error_from_ingress_payload(data, request_id=context_id)
            )
        self._cleanup_context(context_id)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._input_queue.put_nowait(None)
        for ctx in self._context_data.values():
            if not ctx.waiter.done():
                # The owning stream settles its operation: LiveKit retries
                # this on a fresh connection (an options change recycling the
                # socket), or the stream is cancelled on shutdown.
                ctx.waiter.set_exception(APIConnectionError("Connection closed"))
        self._context_data.clear()
        if self._ws and not self._ws.closed:
            try:
                await self._ws.send_str(json.dumps({"close_socket": True}))
            except Exception:
                pass
            await self._ws.close()
        if self._send_task:
            await utils.aio.gracefully_cancel(self._send_task)
        if self._recv_task:
            await utils.aio.gracefully_cancel(self._recv_task)
