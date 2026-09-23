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

"""KugelAudio TTS plugin for LiveKit Agents.

Uses a single persistent WebSocket to /ws/tts/multi with context IDs.
Each synthesize() or stream() call gets a unique context_id; the shared
connection multiplexes all contexts.  On barge-in (cancellation), the
context is deregistered locally and late server messages are silently
discarded.  A context's waiter resolves only on ``context_closed`` — the
server sends that frame after draining every audio frame, so the tail
is never clipped.

The transport itself (the WebSocket, its send/recv loops, the wire
options and frame dataclasses) lives in :mod:`kugelaudio.livekit._connection`.
This module holds the LiveKit-facing classes only.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import replace
from dataclasses import fields as dataclass_fields
from typing import Any

import aiohttp
from kugelaudio._diagnostics import (
    INTEGRATION_LIVEKIT,
    OP_MULTI_CONTEXT,
    STAGE_HANDSHAKE,
    STAGE_SENDING_REQUEST,
    TRANSPORT_WEBSOCKET,
    Operation,
)
from kugelaudio.client import _build_diagnostics, _parse_api_key, _resolve_region_url
from kugelaudio.models import clamp_cfg_scale
from kugelaudio.exceptions import ValidationError
from livekit.agents import (
    APIConnectionError,
    APIError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from ._connection import (
    _Connection,
    _diagnostic_error,
    _SynthesizeContent,
    _TTSOptions,
    _validate_language,
    _validate_speed,
    _validate_temperature,
    _wait_for_context_idle,
)
from .models import (
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    SUPPORTED_SAMPLE_RATES,
    TTSModels,
)

# Backwards-compatible re-exports: the transport layer moved to
# ``._connection`` but these names have always been importable from
# ``kugelaudio.livekit.tts`` (``__init__`` and the test-suite rely on it).
from ._connection import (  # noqa: F401
    MAX_SPEED,
    MAX_TEMPERATURE,
    MIN_SPEED,
    MIN_TEMPERATURE,
    SUPPORTED_LANGUAGES,
    _api_status_error_from_handshake,
    _api_status_error_from_ingress_payload,
    _append_sdk_query,
    _CloseContext,
    _ContextData,
    _word_timestamps_to_timed,
)

logger = logging.getLogger("kugelaudio.livekit")


class TTS(tts.TTS):
    """KugelAudio Text-to-Speech plugin for LiveKit Agents.

    This plugin integrates KugelAudio's TTS API with LiveKit's agent framework,
    providing high-quality voice synthesis with streaming support.

    Example:
        ```python
        from kugelaudio.livekit import TTS

        tts = TTS(api_key="your-api-key")

        # Use with VoicePipelineAgent
        agent = VoicePipelineAgent(
            tts=tts,
            ...
        )
        ```

    Or via the livekit.plugins namespace (after registering):
        ```python
        from kugelaudio.livekit import register_plugin
        register_plugin()

        from livekit.plugins import kugelaudio
        tts = kugelaudio.TTS()
        ```
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: TTSModels | str = DEFAULT_MODEL,
        voice_id: int | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        cfg_scale: float = 2.0,
        max_new_tokens: int = 2048,
        normalize: bool = True,
        word_timestamps: bool = False,
        language: str | None = None,
        speed: float | None = None,
        temperature: float | None = None,
        region: str | None = None,
        base_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        telemetry: bool | None = None,
    ) -> None:
        """Create a new KugelAudio TTS instance.

        Args:
            api_key: KugelAudio API key. If not provided, reads from
                KUGELAUDIO_API_KEY environment variable. Prefix with
                ``"eu-"`` to select the direct EU endpoint; the prefix is
                stripped before auth.
            model: TTS model to use. "kugel-3" is current; legacy
                ids are accepted as aliases.
            voice_id: Voice ID to use. If None, uses server default.
            sample_rate: Output sample rate in Hz. Supported rates: 24000 (native),
                22050, 16000, 8000. Lower rates use server-side resampling with
                minimal latency impact (~0.1ms per 100ms of audio).
            cfg_scale: Classifier-free guidance scale. Clamped to [1.2, 2.5]. Defaults to 2.0.
            max_new_tokens: Maximum tokens to generate. Defaults to 2048.
            normalize: Apply loudness normalization to the output audio. Defaults
                to True.
            word_timestamps: Request per-chunk word-level timestamps for aligned
                transcript / barge-in. Defaults to False (matches the HTTP SDK).
                Enable only when your model supports server-side alignment;
                ``kugel-2.5`` and ``kugel-2-turbo`` may return a post-processing
                error when this is True.
            language: ISO 639-1 language code for text normalization (e.g. "de",
                "en", "fr"). When unset, the server uses the voice's primary
                language (English if none). Supported: de, en, fr, es, it, pt, nl, pl,
                sv, da, no, fi, cs, hu, ro, el, uk, bg, tr, vi, ar, hi, zh, ja, ko,
                sk, sl, hr, sr, ru, he, fa, ur, bn, ta, yue, th, id, ms.
            speed: Playback speed multiplier, range [0.8, 1.2]. ``None``
                (default) leaves the server default of 1.0 in place. Values
                outside the range raise ``ValueError`` — the API rejects
                them, it does not clamp. Uses pitch-preserving
                time-stretching.

                Note: on the ``/ws/tts/multi`` socket this plugin uses,
                ``speed`` is applied **session-wide**, not per context. All
                contexts share one socket and the server applies the last
                value it received (last-writer-wins), so a value set here —
                or later via ``update_options`` — binds for every context
                started after it.
            temperature: Sampling variance, range [0.0, 1.0]; 0 is the most
                stable, 1 the most varied. ``None`` (default) leaves the
                engine default in place. Values outside the range raise
                ``ValueError``; the API rejects them, it does not clamp.
                Session-wide on ``/ws/tts/multi``, exactly like ``speed``.
            region: API endpoint region. Use ``"eu"`` for the direct EU endpoint.
                Overrides any prefix detected from the API key. Ignored when
                *base_url* is set.
            base_url: API base URL. Overrides region selection entirely. Defaults
                to the default geo-routed API endpoint if no region is specified.
            http_session: Optional aiohttp session to reuse.
            telemetry: Opt in or out of anonymous client-error diagnostics,
                exactly like ``KugelAudio(telemetry=...)``. ``None`` (default)
                enables them only on the hosted ``*.kugelaudio.com`` API; the
                ``KUGELAUDIO_TELEMETRY`` environment variable overrides this
                argument.
        """
        # ``aligned_transcript`` only exists on newer livekit-agents
        # TTSCapabilities; older 1.0.x raises TypeError on unknown kwargs.
        capability_kwargs: dict[str, Any] = {"streaming": True}
        if "aligned_transcript" in {
            f.name for f in dataclass_fields(tts.TTSCapabilities)
        }:
            capability_kwargs["aligned_transcript"] = word_timestamps
        super().__init__(
            capabilities=tts.TTSCapabilities(**capability_kwargs),
            sample_rate=sample_rate,
            num_channels=1,
        )

        kugelaudio_api_key = api_key or os.environ.get("KUGELAUDIO_API_KEY")
        if not kugelaudio_api_key:
            raise ValueError(
                "KUGELAUDIO_API_KEY must be set or api_key must be provided"
            )

        cfg_scale = clamp_cfg_scale(cfg_scale)

        clean_key, detected_region = _parse_api_key(kugelaudio_api_key)

        if base_url:
            resolved_url = base_url.rstrip("/")
        else:
            effective_region = region or detected_region
            try:
                resolved_url = _resolve_region_url(effective_region)
            except ValidationError as exc:
                raise ValueError(str(exc)) from exc

        self._opts = _TTSOptions(
            model=model,
            voice_id=voice_id,
            sample_rate=sample_rate,
            cfg_scale=cfg_scale,
            max_new_tokens=max_new_tokens,
            normalize=normalize,
            word_timestamps=word_timestamps,
            language=language,
            speed=speed,
            temperature=temperature,
            api_key=clean_key,
            base_url=resolved_url,
        )

        # This plugin owns its socket rather than a KugelAudio client, so it
        # owns the reporter such a client would build for this key and URL.
        # Enablement: KUGELAUDIO_TELEMETRY, then *telemetry*, then the host.
        self._diagnostics = _build_diagnostics(
            resolved_url, clean_key, telemetry, INTEGRATION_LIVEKIT
        )
        self._session = http_session
        self._current_connection: _Connection | None = None
        self._connection_lock = asyncio.Lock()
        self._closed: bool = False
        self._acquisitions: set[asyncio.Task[_Connection]] = set()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "KugelAudio"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _release_stranded_acquire(
        self, acquire_task: asyncio.Future[bool]
    ) -> None:
        """Abandon an in-flight ``_connection_lock`` acquisition.

        ``asyncio.Lock.acquire()`` can be granted in the very loop iteration
        in which our deadline expires; cancelling an already-completed task
        is a no-op, so without this the lock would stay held forever and
        every later caller would deadlock.
        """
        acquire_task.cancel()
        try:
            await acquire_task
        except asyncio.CancelledError:
            # Cancellation won the race: the lock was never handed to us and
            # asyncio.Lock has already woken the next waiter.
            return
        # The acquire completed anyway — we own the lock, so give it back.
        self._connection_lock.release()

    async def _discard_raced_connect(
        self, conn: _Connection, connect_task: asyncio.Future[None]
    ) -> None:
        """Abandon an in-flight ``_Connection.connect``, closing a WS that
        completed in the cancellation race so the socket is not leaked."""
        connect_task.cancel()
        try:
            await connect_task
        except asyncio.CancelledError:
            return
        except (APITimeoutError, APIStatusError, APIConnectionError) as exc:
            # The connect failed on its own inside the race window. We are
            # about to raise APITimeoutError for the deadline, so record the
            # real cause rather than dropping it.
            logger.debug("connect failed while being abandoned: %s", exc)
            return
        await conn.aclose()

    async def _ensure_connection(self, timeout: float = 10.0) -> _Connection:
        """Acquire within one total budget; shutdown owns pending work."""
        if self._closed:
            raise APIConnectionError("KugelAudio TTS is closed")
        task = asyncio.create_task(self._acquire_connection(timeout))
        self._acquisitions.add(task)
        try:
            return await task
        finally:
            self._acquisitions.discard(task)

    async def _acquire_connection(self, timeout: float) -> _Connection:
        """Get or create the persistent /ws/tts/multi connection.

        *timeout* bounds the **total** wait — acquiring the shared connection
        lock *plus* the WebSocket connect — so a caller with a short budget
        never inherits another caller's long one. Before this was enforced, a
        2.5 s synthesis call queued behind a 10 s ``prewarm()`` took 12.5 s to
        fail against an unreachable API.

        Raises:
            APITimeoutError: The lock could not be acquired, or the socket
                could not be opened, within *timeout* seconds.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        acquire_task: asyncio.Future[bool] = asyncio.ensure_future(
            self._connection_lock.acquire()
        )
        try:
            done, _pending = await asyncio.wait({acquire_task}, timeout=timeout)
        except asyncio.CancelledError:
            await self._release_stranded_acquire(acquire_task)
            raise
        if not done:
            await self._release_stranded_acquire(acquire_task)
            raise APITimeoutError()
        acquire_task.result()  # surface any unexpected acquisition failure

        try:
            if (
                self._current_connection
                and self._current_connection.is_current
                and not self._current_connection._closed
            ):
                return self._current_connection

            remaining = deadline - loop.time()
            if remaining <= 0:
                raise APITimeoutError()

            session = self._ensure_session()
            conn = _Connection(self._opts, session)
            # Establishing the socket is its own diagnostics operation, owned
            # here rather than in ``connect`` so the budget timeout below is
            # reported as a failure, not as the cancellation it looks like.
            op = self._diagnostics.operation(
                OP_MULTI_CONTEXT, TRANSPORT_WEBSOCKET, STAGE_HANDSHAKE
            )
            try:
                await self._connect_within(conn, remaining)
            except BaseException as exc:
                op.fail(_diagnostic_error(exc))
                raise
            op.succeed()

            self._current_connection = conn
            return conn
        finally:
            self._connection_lock.release()

    async def _connect_within(self, conn: _Connection, remaining: float) -> None:
        # ``connect``'s own timeout only bounds the TCP connect
        # (aiohttp ``sock_connect``); the WS upgrade handshake is
        # unbounded, so wrap the whole thing in the remaining budget.
        connect_task: asyncio.Future[None] = asyncio.ensure_future(
            conn.connect(remaining)
        )
        try:
            done, _pending = await asyncio.wait({connect_task}, timeout=remaining)
        except asyncio.CancelledError:
            await self._discard_raced_connect(conn, connect_task)
            raise
        if not done:
            await self._discard_raced_connect(conn, connect_task)
            raise APITimeoutError()
        connect_task.result()  # re-raises APIStatusError / APIConnectionError

    def prewarm(self, timeout: float = 10.0) -> None:
        """Pre-warm the WebSocket connection to reduce TTFA on first synthesis.

        Call this from LiveKit's ``before_tts_cb`` or during agent setup
        to eagerly establish the WebSocket connection and authenticate,
        eliminating ~35-290ms of handshake latency from the first request.

        This is a synchronous method (as required by the LiveKit TTS interface)
        that schedules the async connection establishment in the background.

        Args:
            timeout: Budget in seconds for the background connect. Defaults
                to 10.0. A concurrent synthesis call is *not* charged this
                budget: ``_ensure_connection`` bounds lock wait plus connect
                by the caller's own timeout, so a short-timeout request
                fails on its own schedule instead of queueing behind
                prewarm.

        Example:
            ```python
            tts = TTS(api_key="your-api-key")
            tts.prewarm()  # establishes WS connection in background
            ```
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("prewarm() called outside event loop, skipping")
            return

        async def _do_prewarm() -> None:
            try:
                await self._ensure_connection(timeout=timeout)
                logger.info("KugelAudio TTS connection pre-warmed")
            except Exception:
                logger.warning(
                    "Failed to prewarm KugelAudio connection, "
                    "will retry on first synthesis",
                    exc_info=True,
                )

        loop.create_task(_do_prewarm())

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice_id: NotGivenOr[int | None] = NOT_GIVEN,
        cfg_scale: NotGivenOr[float] = NOT_GIVEN,
        max_new_tokens: NotGivenOr[int] = NOT_GIVEN,
        normalize: NotGivenOr[bool] = NOT_GIVEN,
        word_timestamps: NotGivenOr[bool] = NOT_GIVEN,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        speed: NotGivenOr[float | None] = NOT_GIVEN,
        temperature: NotGivenOr[float | None] = NOT_GIVEN,
    ) -> None:
        """Update TTS options dynamically.

        Any change recycles the shared WebSocket: the current connection is
        marked non-current and closed, so the next synthesis opens a fresh
        socket carrying the new options.

        Args:
            model: TTS model to use.
            voice_id: Voice ID to use.
            cfg_scale: Classifier-free guidance scale. Clamped to [1.2, 2.5].
            max_new_tokens: Maximum tokens to generate.
            normalize: Apply loudness normalization to the output audio.
            word_timestamps: Enable or disable word-level timestamps.
            language: ISO 639-1 language code for text normalization (e.g. "de").
            speed: Playback speed multiplier, range [0.8, 1.2]; ``None``
                restores the server default of 1.0. Out-of-range values
                raise ``ValueError`` (the API rejects rather than clamps)
                and leave the current setting untouched.

                On ``/ws/tts/multi`` speed is **session-wide**, not
                per-context: every context shares one socket and the server
                keeps the last value it was given (last-writer-wins). The
                connection recycling described above is what makes this
                deterministic — the new speed binds for contexts started
                after the change, while contexts already running on the old
                socket keep the old rate.
            temperature: Sampling variance, range [0.0, 1.0]; ``None``
                restores the engine default. Session-wide like ``speed``,
                with the same reject-don't-clamp and recycling semantics.

        Raises:
            ValueError: If *speed* is outside [0.8, 1.2], *temperature* is
                outside [0.0, 1.0], or *language* is not a supported
                ISO 639-1 code.
        """
        changed = False
        if is_given(model) and model != self._opts.model:
            self._opts.model = model
            changed = True
        if is_given(voice_id) and voice_id != self._opts.voice_id:
            self._opts.voice_id = voice_id
            changed = True
        if is_given(cfg_scale) and cfg_scale != self._opts.cfg_scale:
            cfg_scale = clamp_cfg_scale(cfg_scale)
            self._opts.cfg_scale = cfg_scale
            changed = True
        if is_given(max_new_tokens) and max_new_tokens != self._opts.max_new_tokens:
            self._opts.max_new_tokens = max_new_tokens
            changed = True
        if is_given(normalize) and normalize != self._opts.normalize:
            self._opts.normalize = normalize
            changed = True
        if is_given(word_timestamps) and word_timestamps != self._opts.word_timestamps:
            self._opts.word_timestamps = word_timestamps
            changed = True
        if is_given(language):
            validated = _validate_language(language)
            if validated != self._opts.language:
                self._opts.language = validated
                changed = True
        if is_given(speed):
            # Validate before assigning: a rejected value must leave the
            # previous speed in place rather than half-applying the update.
            validated_speed = _validate_speed(speed)
            if validated_speed != self._opts.speed:
                self._opts.speed = validated_speed
                changed = True
        if is_given(temperature):
            validated_temperature = _validate_temperature(temperature)
            if validated_temperature != self._opts.temperature:
                self._opts.temperature = validated_temperature
                changed = True
        if changed and self._current_connection:
            old_conn = self._current_connection
            old_conn.mark_non_current()
            self._current_connection = None
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(old_conn.aclose())
            except RuntimeError:
                pass

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        """Synthesize text to speech (non-streaming).

        Args:
            text: Text to synthesize.
            conn_options: Connection options.

        Returns:
            ChunkedStream that yields audio frames.
        """
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SynthesizeStream":
        """Create a streaming TTS session.

        Args:
            conn_options: Connection options.

        Returns:
            SynthesizeStream for streaming text input.
        """
        return SynthesizeStream(tts=self, conn_options=conn_options)

    async def aclose(self) -> None:
        """Close the TTS instance and release resources."""
        self._closed = True
        acquisitions = tuple(self._acquisitions)
        for task in acquisitions:
            task.cancel()
        if acquisitions:
            # Callers still observe their own failure; gather drains cancellation
            # cleanup, including lock and successful-upgrade races, before return.
            await asyncio.gather(*acquisitions, return_exceptions=True)
        if self._current_connection:
            await self._current_connection.aclose()
            self._current_connection = None
        # Bounded (<=1 s) wait for the delivery thread, off the event loop.
        await asyncio.to_thread(self._diagnostics.close)


class _StreamDiagnostics:
    """One diagnostics operation per stream, across LiveKit's ``_run`` retries.

    LiveKit's base ``_main_task`` calls ``_run`` again after an ``APIError``
    until ``max_retry`` attempts are spent, so an attempt's failure is only
    the stream's failure when no retry follows. A retried attempt bumps
    ``retry_count``; the final failure is reported once (``retry_exhausted``
    when a retry happened). Class-level defaults, because tests build streams
    with ``__new__`` and skip ``__init__``.
    """

    _tts: TTS
    _conn_options: APIConnectOptions
    _op: Operation | None = None
    _attempts: int = 0

    def _begin_attempt(self) -> Operation:
        self._attempts += 1
        op = self._op
        if op is None:
            op = self._tts._diagnostics.operation(
                OP_MULTI_CONTEXT, TRANSPORT_WEBSOCKET, STAGE_SENDING_REQUEST
            )
            self._op = op
        elif self._attempts > 1:
            op.record_retry()
            op.set_stage(STAGE_SENDING_REQUEST)
        return op

    def _attempt_failed(self, exc: BaseException) -> None:
        # RISK: mirrors livekit-agents' ``_main_task``, which retries EVERY
        # APIError (it does not read ``retryable``) until ``max_retry``. If a
        # future livekit-agents skips non-retryable errors, a final failure
        # would go unreported here; the JS plugin checks ``retryable`` because
        # the JS base class does.
        will_retry = (
            isinstance(exc, APIError) and self._attempts <= self._conn_options.max_retry
        )
        if not will_retry and self._op is not None:
            self._op.fail(_diagnostic_error(exc))


class ChunkedStream(_StreamDiagnostics, tts.ChunkedStream):
    """Non-streaming TTS synthesis using WebSocket."""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        context_id = utils.shortuuid()

        output_emitter.initialize(
            request_id=context_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            frame_size_ms=20,
        )

        connection = await self._tts._ensure_connection(
            timeout=self._conn_options.timeout
        )
        op = self._begin_attempt()
        waiter: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        ctx = connection.register_context(context_id, output_emitter, waiter)
        if ctx is None:
            # register_context rejected (connection closed) — waiter already
            # has the exception; await it so the caller sees it.
            try:
                await waiter
            except Exception as exc:
                self._attempt_failed(exc)
                raise
            finally:
                output_emitter.flush()
            return
        ctx.op = op

        try:
            connection.send_content(
                _SynthesizeContent(context_id, self._input_text, flush=True)
            )
            await _wait_for_context_idle(
                ctx, waiter, idle_threshold=self._conn_options.timeout
            )
            op.succeed()
        except asyncio.CancelledError:
            op.cancel()
            connection.request_close_context(context_id, immediate=True)
            connection._cleanup_context(context_id)
            raise
        except asyncio.TimeoutError:
            timeout_error = APITimeoutError()
            self._attempt_failed(timeout_error)
            connection.request_close_context(context_id, immediate=True)
            connection._cleanup_context(context_id)
            raise timeout_error from None
        except Exception as exc:
            self._attempt_failed(exc)
            raise
        finally:
            output_emitter.flush()


class SynthesizeStream(_StreamDiagnostics, tts.SynthesizeStream):
    """Streaming TTS synthesis over the shared /ws/tts/multi connection.

    Each stream gets its own context_id.  Text tokens are forwarded to
    the shared connection; audio arrives via the recv loop and is routed
    back by context_id.
    """

    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        context_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=context_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=context_id)

        connection = await self._tts._ensure_connection(
            timeout=self._conn_options.timeout
        )
        op = self._begin_attempt()
        waiter: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        ctx = connection.register_context(
            context_id, output_emitter, waiter, stream=self
        )
        if ctx is None:
            # Connection closed before register — waiter is already failed.
            try:
                await waiter
            except Exception as exc:
                self._attempt_failed(exc)
                raise
            finally:
                output_emitter.end_segment()
            return
        ctx.op = op

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    connection.send_content(
                        _SynthesizeContent(context_id, "", flush=True)
                    )
                    continue
                if not data:
                    continue
                connection.send_content(_SynthesizeContent(context_id, data))
            # Input channel closed — flush + close context
            connection.send_content(_SynthesizeContent(context_id, "", flush=True))
            connection.request_close_context(context_id)

        input_t = asyncio.create_task(_input_task())

        def _on_input_done(t: asyncio.Task) -> None:
            # Without this, an exception in _input_task would be stored on
            # the task but never observed — _run would hang on `await waiter`
            # because no flush/close was sent to the server.
            if waiter.done():
                return
            exc = t.exception() if not t.cancelled() else None
            if exc is not None:
                waiter.set_exception(exc)

        input_t.add_done_callback(_on_input_done)

        try:
            await _wait_for_context_idle(
                ctx, waiter, idle_threshold=self._conn_options.timeout
            )
            op.succeed()
        except asyncio.CancelledError:
            # Barge-in / caller cancelled: tell the server to cancel
            # in-flight generation (immediate=True) instead of letting
            # it drain.  Without this the server keeps generating audio
            # for several seconds after the user has moved on.  Barge-in is
            # a cancellation of this context's turn only, never an event.
            op.cancel()
            connection.request_close_context(context_id, immediate=True)
            connection._cleanup_context(context_id)
            raise
        except asyncio.TimeoutError:
            timeout_error = APITimeoutError()
            self._attempt_failed(timeout_error)
            connection.request_close_context(context_id, immediate=True)
            connection._cleanup_context(context_id)
            raise timeout_error from None
        except Exception as exc:
            self._attempt_failed(exc)
            raise
        finally:
            output_emitter.end_segment()
            await utils.aio.gracefully_cancel(input_t)
