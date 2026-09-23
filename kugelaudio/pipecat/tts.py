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

"""KugelAudio TTS service for Pipecat.

Internally delegates to the core KugelAudio Python SDK for WebSocket
connection management, pooling, keepalive, and streaming. This avoids
duplicating low-level transport logic and ensures the PipeCat integration
inherits all SDK-level performance optimisations (connection reuse,
pre-warming, language-skip, etc.).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, AsyncGenerator, Callable, Optional

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
try:
    from pipecat.frames.frames import InterruptionFrame as _PipecatInterruptionFrame
except ImportError:  # Pipecat 0.0.x
    from pipecat.frames.frames import StartInterruptionFrame as _PipecatInterruptionFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService

from kugelaudio import KugelAudio
from kugelaudio._diagnostics import INTEGRATION_PIPECAT
from kugelaudio.client import _parse_api_key, _resolve_region_url
from kugelaudio.models import WordTimestamp, clamp_cfg_scale
from kugelaudio.exceptions import (
    ConnectionError as KugelAudioConnectionError,
    KugelAudioError,
    ValidationError,
)
from kugelaudio.streaming import MultiContextSession

from .models import (
    DEFAULT_CFG_SCALE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    SUPPORTED_SAMPLE_RATES,
    TTSModels,
)

logger = logging.getLogger("kugelaudio.pipecat")


# Pipecat 1.x introduced TTSSettings + AIService.start().validate_complete(),
# which logs ERROR for any TTSSettings field still set to the NOT_GIVEN
# sentinel. Subclasses are expected to pass an initialized TTSSettings via
# super().__init__(settings=...). On Pipecat 0.x this module/class doesn't
# exist; we fall back to the old kwarg-only path. See
# pipecat/services/settings.py:TTSSettings.validate_complete.
try:  # noqa: SIM105 — explicit branch is clearer than contextlib.suppress
    from pipecat.services.settings import TTSSettings as _PipecatTTSSettings
except ImportError:  # Pipecat < 1.0
    _PipecatTTSSettings = None  # type: ignore[assignment]


@lru_cache(maxsize=None)
def _frame_supports_context_id(frame_type: type[Frame]) -> bool:
    """Return whether the installed Pipecat frame accepts context_id."""
    try:
        return "context_id" in inspect.signature(frame_type).parameters
    except (TypeError, ValueError):
        return False


@lru_cache(maxsize=None)
def _base_run_tts_takes_context_id() -> bool:
    """Whether the installed Pipecat base ``TTSService.run_tts`` accepts a
    ``context_id`` (the 1.x AudioContext API).

    True on Pipecat 1.x and the later 0.0.x line (>= ~0.0.63): the base owns
    per-turn context identity and drives our ``on_turn_context_created`` /
    ``on_audio_context_completed`` hooks, so the wrapper closes each turn's
    context there.

    False on the legacy 0.x surface (``run_tts(text)`` only): the base never
    hands us a per-turn id, so we mirror the same per-turn lifecycle by watching
    ``LLMFullResponse{Start,End}Frame`` in ``process_frame`` instead.
    """
    try:
        return "context_id" in inspect.signature(TTSService.run_tts).parameters
    except (TypeError, ValueError):
        return False


def _tts_started_frame(context_id: Optional[str]) -> TTSStartedFrame:
    if context_id and _frame_supports_context_id(TTSStartedFrame):
        return TTSStartedFrame(context_id=context_id)
    return TTSStartedFrame()


def _tts_stopped_frame(context_id: Optional[str]) -> TTSStoppedFrame:
    if context_id and _frame_supports_context_id(TTSStoppedFrame):
        return TTSStoppedFrame(context_id=context_id)
    return TTSStoppedFrame()


def _tts_audio_frame(
    *,
    audio: bytes,
    sample_rate: int,
    num_channels: int,
    context_id: Optional[str],
) -> TTSAudioRawFrame:
    kwargs: dict[str, Any] = {
        "audio": audio,
        "sample_rate": sample_rate,
        "num_channels": num_channels,
    }
    if context_id and _frame_supports_context_id(TTSAudioRawFrame):
        kwargs["context_id"] = context_id
    return TTSAudioRawFrame(**kwargs)


def _is_ws_connection_closed_error(exc: BaseException) -> bool:
    """Return whether an exception is a raw websockets connection close."""
    return "ConnectionClosed" in type(exc).__name__


def _error_frame_message(exc: BaseException) -> str:
    """Format SDK errors for Pipecat's string-only ErrorFrame."""
    if isinstance(exc, KugelAudioError):
        details: list[str] = []
        if exc.error_code:
            details.append(exc.error_code)
        if exc.status_code is not None:
            details.append(f"status={exc.status_code}")
        suffix = f" ({', '.join(details)})" if details else ""
        return f"KugelAudio error{suffix}: {exc.message}"
    return f"KugelAudio error: {exc}"


@dataclass
class _TTSOptions:
    model: TTSModels | str
    voice_id: int
    sample_rate: int
    cfg_scale: float
    max_new_tokens: int
    api_key: str
    base_url: str
    language: str | None = None
    normalize: bool = True
    project_id: int | None = None
    # None = the project's default dictionaries; [] = explicit opt-out.
    dictionary_ids: list[int] | None = None


class KugelAudioTTSService(TTSService):
    """KugelAudio Text-to-Speech service for Pipecat.

    Integrates KugelAudio's TTS API with Pipecat's pipeline framework,
    providing high-quality voice synthesis via WebSocket streaming.

    Internally uses the core KugelAudio Python SDK for all WebSocket
    transport, connection pooling, and streaming logic.

    Example:
        ```python
        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(
            api_key="your-api-key",
            voice_id=280,
            model="kugel-3",
            sample_rate=24000,
        )

        # Use in a Pipecat pipeline
        pipeline = Pipeline([..., tts, ...])
        ```
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: TTSModels | str = DEFAULT_MODEL,
        voice_id: int,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        language: Optional[str] = None,
        normalize: bool = True,
        project_id: Optional[int] = None,
        dictionary_ids: Optional[list[int]] = None,
        region: Optional[str] = None,
        base_url: Optional[str] = None,
        on_word_timestamps: Optional[
            Callable[[str, list[WordTimestamp]], None]
        ] = None,
        telemetry: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """Create a new KugelAudio TTS service for Pipecat.

        Args:
            api_key: KugelAudio API key. If not provided, reads from
                KUGELAUDIO_API_KEY environment variable. Prefix with
                ``"eu-"`` to select the direct EU endpoint; the prefix is
                stripped before auth.
            model: TTS model to use. "kugel-3" is current; legacy
                ids are accepted as aliases.
            voice_id: Voice ID to use for synthesis.
            sample_rate: Output sample rate in Hz. Supported rates: 24000 (native),
                22050, 16000, 8000. Lower rates use server-side resampling.
            cfg_scale: Classifier-free guidance scale. Clamped to [1.2, 2.5]. Defaults to 2.0.
            max_new_tokens: Maximum tokens to generate. Defaults to 2048.
            language: ISO 639-1 language code (e.g., 'en', 'de'). When unset,
                the server uses the voice's primary language (English if none).
            normalize: Apply text normalization. Defaults to True.
            project_id: Project whose pronunciation dictionaries apply at
                synthesis. Required for any dictionary to apply and for a
                non-empty *dictionary_ids*; ``None`` (default) sends no
                project.
            dictionary_ids: Per-request dictionary selection. ``None``
                (default) applies all active dictionaries of the project;
                ``[]`` disables dictionaries; a list of IDs applies exactly
                those. A non-empty list without *project_id* raises
                ``ValueError``.
            region: API endpoint region. Use ``"eu"`` for the direct EU endpoint.
                Overrides any prefix detected from the API key. Ignored when
                *base_url* is set.
            base_url: API base URL. Overrides region selection entirely. Defaults
                to the default geo-routed API endpoint if no region is specified.
            on_word_timestamps: Optional synchronous callback receiving the
                KugelAudio context ID and each batch of server word timestamps.
                Providing it enables word timestamps for the multi-context
                session. When omitted, timestamp generation remains disabled.
            telemetry: Opt in or out of anonymous client-error diagnostics,
                exactly like ``KugelAudio(telemetry=...)``. ``None`` (default)
                enables them only on the hosted ``*.kugelaudio.com`` API; the
                ``KUGELAUDIO_TELEMETRY`` environment variable overrides this
                argument.
            **kwargs: Additional arguments passed to Pipecat TTSService.
        """
        # On Pipecat 1.x the parent expects an initialized TTSSettings; if we
        # leave model / voice / language as NOT_GIVEN, AIService.start() logs
        # an ERROR per field. Build it from our typed wrapper params here so
        # the user doesn't have to construct one themselves. On Pipecat 0.x
        # _PipecatTTSSettings is None and we fall back to the legacy path.
        super_kwargs: dict[str, Any] = dict(kwargs)
        if _PipecatTTSSettings is not None and "settings" not in super_kwargs:
            # Drop a stray ``voice=`` kwarg if the caller passed both ways —
            # we already cover voice via the dedicated ``voice_id`` param,
            # and Pipecat 1.x routes voice through TTSSettings, not kwargs.
            super_kwargs.pop("voice", None)
            super_kwargs["settings"] = _PipecatTTSSettings(
                model=str(model) if model else None,
                voice=str(voice_id) if voice_id is not None else None,
                language=language,
            )
        super().__init__(sample_rate=sample_rate, **super_kwargs)

        kugelaudio_api_key = api_key or os.environ.get("KUGELAUDIO_API_KEY")
        if not kugelaudio_api_key:
            raise ValueError(
                "KUGELAUDIO_API_KEY must be set or api_key must be provided"
            )

        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"Unsupported sample rate: {sample_rate}. "
                f"Supported rates: {SUPPORTED_SAMPLE_RATES}"
            )

        if dictionary_ids and project_id is None:
            # The server rejects this on every synthesis; fail at
            # construction instead of on the first turn.
            raise ValueError(
                "dictionary_ids requires project_id: the server rejects a "
                "dictionary selection without its project."
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
            language=language,
            normalize=normalize,
            project_id=project_id,
            dictionary_ids=dictionary_ids,
            api_key=clean_key,
            base_url=resolved_url,
        )

        # ``api_url`` too, so diagnostics (and any HTTP call) go to the same
        # host as the audio, and every event is stamped ``integration=pipecat``.
        self._client = KugelAudio(
            api_key=clean_key,
            api_url=resolved_url,
            tts_url=resolved_url,
            telemetry=telemetry,
            _integration=INTEGRATION_PIPECAT,
        )
        self._on_word_timestamps = on_word_timestamps
        self._multi_session: Optional[MultiContextSession] = None
        self._multi_session_lock = asyncio.Lock()
        self._multi_session_needs_reset = False
        # Legacy Pipecat 0.x calls run_tts(text) with no context_id, so the base
        # never tells us where a turn starts or ends. We give each LLM turn its
        # OWN server-side context — matching the modern per-turn path — by
        # watching the LLM-response boundary frames in process_frame: the
        # context is pre-created on LLMFullResponseStartFrame (the create
        # round-trip overlaps the LLM's token generation, off the TTFA critical
        # path) and closed on LLMFullResponseEndFrame, so a turn's KV cache is
        # freed at the turn boundary and never bleeds into the next utterance.
        # All run_tts calls within one response (a multi-sentence turn) share
        # this id. If run_tts ever fires without a preceding start frame (a raw
        # TextFrame / static say()), _resolve mints one lazily; with no turn
        # boundary to act on it is simply reused until the next start/cleanup,
        # so it can never accumulate against the server's per-session cap.
        self._legacy_turn_context_id: Optional[str] = None
        # Pipecat 1.x (AudioContext API) calls run_tts(text, context_id) with a
        # FRESH context_id per turn and owns its identity. Unlike the wrapper-
        # owned 0.x path above, we must close each of those server-side contexts
        # when its turn ends — otherwise every turn leaks a /ws/tts/multi context
        # against the per-session cap and the call drops mid-conversation
        # (KUG-1087). We track the caller-supplied ids here and close them from
        # the Pipecat turn/audio completion hooks below.
        self._caller_contexts: set[str] = set()
        # Server-side contexts already create_context'd before run_tts(). Pipecat
        # 1.x mints the turn id on LLMFullResponseStartFrame — we provision
        # during on_turn_context_created so the ~30-45ms WS round-trip overlaps
        # LLM time-to-first-token instead of sitting on the TTFA critical path.
        self._provisioned_contexts: set[str] = set()
        self._closed = False
        # `prewarm()` is synchronous for ergonomic pipeline construction, but
        # its connection attempt is still owned by this service. Keeping the
        # task makes teardown able to cancel a slow handshake instead of
        # leaving it alive after Pipecat has closed the call.
        self._prewarm_task: asyncio.Task[None] | None = None

    def _create_multi_session(self) -> MultiContextSession:
        return self._client.tts.multi_context_session(
            default_voice_id=self._opts.voice_id,
            model_id=self._opts.model,
            sample_rate=self._opts.sample_rate,
            cfg_scale=self._opts.cfg_scale,
            max_new_tokens=self._opts.max_new_tokens,
            normalize=self._opts.normalize,
            language=self._opts.language,
            project_id=self._opts.project_id,
            dictionary_ids=self._opts.dictionary_ids,
            word_timestamps=self._on_word_timestamps is not None,
            on_word_timestamps=self._on_word_timestamps,
        )

    async def _get_multi_session(self) -> MultiContextSession:
        if self._multi_session_needs_reset:
            await self._close_multi_session()
            self._multi_session_needs_reset = False
        # An idle WS may have been silently dropped by an upstream proxy / NAT
        # between turns of a voice conversation. Detect that here so we don't
        # ride a corpse forever; without this check, every subsequent run_tts
        # raises ConnectionClosedError("no close frame received or sent").
        if self._multi_session is not None and not self._multi_session.is_alive:
            logger.info(
                "KugelAudio multi-context session WS is dead; reconnecting"
            )
            await self._close_multi_session()
        if self._multi_session is None:
            session = self._create_multi_session()
            await session.connect()
            self._multi_session = session
        return self._multi_session

    async def _close_multi_session(self) -> None:
        session = self._multi_session
        self._multi_session = None
        self._multi_session_needs_reset = False
        # Closing the WS frees every server-side context with it, so forget our
        # legacy turn context (the next connection starts with none) and drop
        # the caller-context bookkeeping rather than closing them one by one.
        self._legacy_turn_context_id = None
        self._caller_contexts.clear()
        self._provisioned_contexts.clear()
        if session is not None:
            await session.close()

    async def _resolve_turn_context_id(
        self, session: MultiContextSession, caller_context_id: Optional[str]
    ) -> str:
        """Pick the context_id for this turn.

        When the caller supplies one (Pipecat 1.x AudioContext API), honour it
        verbatim — the caller owns context creation and teardown. On legacy 0.x
        use the turn context pre-created by ``_legacy_begin_turn`` on the LLM
        start frame; if run_tts fired without one (a raw TextFrame / static
        say()), mint it lazily here and reuse it until the next turn boundary.
        """
        if caller_context_id:
            return caller_context_id

        if self._legacy_turn_context_id is None:
            self._legacy_turn_context_id = f"kugelaudio-{uuid.uuid4()}"
        return self._legacy_turn_context_id

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Drive the legacy 0.x per-turn context lifecycle off the LLM-response
        boundary frames.

        On the 1.x AudioContext surface the base hands run_tts a per-turn
        context_id and calls our on_turn_context_created /
        on_audio_context_completed hooks, so we must NOT also act here. On legacy
        0.x the base never does either, so the LLM start/end frames flowing
        through this processor are our only turn-boundary signal.

        Ordering is load-bearing: the legacy base ``TTSService.process_frame``
        synthesizes the turn's final buffered sentence *while handling*
        ``LLMFullResponseEndFrame`` (inside the ``super()`` call below). So we
        close the turn's context AFTER ``super()`` runs — closing it first would
        leave that last fragment to ``run_tts``, which would mint a fresh context
        not shared with the rest of the turn and not freed at the turn boundary.
        The start frame triggers no synthesis, so begin runs before ``super()``.
        """
        legacy = not _base_run_tts_takes_context_id()
        if legacy and isinstance(frame, LLMFullResponseStartFrame):
            await self._legacy_begin_turn()
        await super().process_frame(frame, direction)
        if legacy and isinstance(frame, LLMFullResponseEndFrame):
            await self._legacy_end_turn()

    async def _legacy_begin_turn(self) -> None:
        """Legacy 0.x turn start: close the previous turn's context (covers an
        interrupted turn whose end frame never arrived) and pre-create this
        turn's context so the create round-trip overlaps the LLM's token
        generation instead of the first run_tts chunk's TTFA."""
        async with self._multi_session_lock:
            stale = self._legacy_turn_context_id
            self._legacy_turn_context_id = None
            try:
                session = await self._get_multi_session()
            except Exception:
                # KEEP-JUSTIFIED: best-effort prefetch — can't reach the server
                # to provision now; the next run_tts opens the session and
                # lazily creates the context on send().
                logger.warning(
                    "Could not pre-provision legacy turn context; "
                    "will create lazily on first run_tts",
                    exc_info=True,
                )
                return
            if stale is not None:
                # immediate=True: the prior turn is finished and its audio long
                # since played; free the slot now.
                await self._close_context_best_effort(session, stale, immediate=True)
            ctx = f"kugelaudio-{uuid.uuid4()}"
            try:
                await session.create_context(ctx)
            except Exception:
                # KEEP-JUSTIFIED: best-effort prefetch; run_tts creates the
                # context lazily on send() if this fails.
                logger.debug(
                    "Legacy pre-provision failed for %s; lazy create on send",
                    ctx,
                    exc_info=True,
                )
            self._legacy_turn_context_id = ctx

    async def _legacy_end_turn(self) -> None:
        """Legacy 0.x turn end: close this turn's context so its KV cache is
        freed at the turn boundary (the on_audio_context_completed analogue).

        Drains rather than hard-closes (immediate=False) so a final sentence
        still streaming when the LLM end frame arrives isn't truncated.
        """
        async with self._multi_session_lock:
            ctx = self._legacy_turn_context_id
            self._legacy_turn_context_id = None
            if ctx is None or self._multi_session is None:
                return
            await self._close_context_best_effort(
                self._multi_session, ctx, immediate=False
            )

    async def _close_context_best_effort(
        self, session: MultiContextSession, context_id: str, *, immediate: bool
    ) -> None:
        """Close one server-side context, swallowing transport failures."""
        try:
            async for _ in session.close_context(context_id, immediate=immediate):
                pass
            logger.debug(
                "Closed server context %s immediate=%s", context_id, immediate
            )
        except Exception:
            # KEEP-JUSTIFIED: best-effort context release. A dead WS is surfaced
            # by the next run_tts reconnect path and the server's idle reaper is
            # the backstop; there is no correctness value in propagating a
            # cleanup failure here.
            logger.debug("Failed to close context %s", context_id, exc_info=True)

    async def _close_caller_context(self, context_id: Optional[str]) -> None:
        """Close a single caller-supplied (Pipecat 1.x) server-side context.

        Idempotent and best-effort: the SDK's ``close_context`` is a no-op for
        an unknown/already-closed id, and a dead WS is handled by the existing
        reconnect path on the next ``run_tts``.
        """
        if not context_id:
            return
        async with self._multi_session_lock:
            self._caller_contexts.discard(context_id)
            self._provisioned_contexts.discard(context_id)
            session = self._multi_session
            if session is None:
                return
            try:
                # immediate=True: the turn is over, free the slot now and don't
                # drain-then-discard a tail (the audio already played).
                async for _ in session.close_context(context_id, immediate=True):
                    pass
                logger.debug(
                    "Closed caller context %s immediate=True", context_id
                )
            except Exception:
                # KEEP-JUSTIFIED: best-effort server-side context release on turn
                # end. If the WS is already gone the next run_tts surfaces it via
                # the reconnect path, and the server's inactivity reaper is the
                # backstop. No correctness value in propagating a cleanup failure.
                logger.debug(
                    "Failed to close caller context %s", context_id, exc_info=True
                )

    async def _provision_caller_context(self, context_id: str) -> None:
        """Eagerly create a Pipecat 1.x turn context on the multi WS session.

        Called from ``on_turn_context_created`` (fires on LLMFullResponseStartFrame,
        before the first ``run_tts`` chunk) so the create_context round-trip
        overlaps LLM latency instead of adding to measured TTFA.
        """
        if not context_id or context_id in self._provisioned_contexts:
            return
        try:
            async with self._multi_session_lock:
                session = await self._get_multi_session()
                await session.create_context(context_id)
            self._provisioned_contexts.add(context_id)
            logger.debug("Pre-provisioned KugelAudio context %s", context_id)
        except Exception:
            # KEEP-JUSTIFIED: best-effort prefetch — run_tts still creates the
            # context lazily on send() if this fails.
            logger.warning(
                "Failed to pre-provision context %s; will retry on send",
                context_id,
                exc_info=True,
            )

    async def on_turn_context_created(self, context_id: str) -> None:
        """Pipecat 1.x: a new turn opened a new context_id.

        A new turn means every *previous* turn's context is finished, so close
        any still-open caller contexts (covers Pipecat builds that don't emit
        ``on_audio_context_completed``), then track this turn's id.
        """
        parent = getattr(super(), "on_turn_context_created", None)
        if parent is not None:
            await parent(context_id)
        for stale in list(self._caller_contexts):
            if stale != context_id:
                await self._close_caller_context(stale)
        if context_id:
            self._caller_contexts.add(context_id)
        await self._provision_caller_context(context_id)

    async def on_audio_context_interrupted(self, context_id: str) -> None:
        """Pipecat 1.x / late 0.x: barge-in cut a turn short — drop its server-side
        context immediately so generation can't bleed into the next utterance."""
        parent = getattr(super(), "on_audio_context_interrupted", None)
        if parent is not None:
            await parent(context_id)
        await self._close_caller_context(context_id)

    async def on_audio_context_completed(self, context_id: str) -> None:
        """Pipecat 1.x: a turn's audio finished playing — close its server-side
        context so it stops counting against the per-session cap."""
        parent = getattr(super(), "on_audio_context_completed", None)
        if parent is not None:
            await parent(context_id)
        await self._close_caller_context(context_id)

    async def _handle_interruption(
        self, frame: _PipecatInterruptionFrame, direction: FrameDirection
    ) -> None:
        """Close server-side contexts on barge-in.

        Pipecat 1.x / late 0.x: ``on_audio_context_interrupted`` handles contexts
        that already have a local audio queue; sweep ``_caller_contexts`` after
        the base handler for pre-provision-only orphans.

        Legacy 0.0.x (``run_tts(text)``): close the wrapper-owned turn context
        immediately instead of waiting for the next ``LLMFullResponseStartFrame``.
        """
        await super()._handle_interruption(frame, direction)
        if not _base_run_tts_takes_context_id():
            async with self._multi_session_lock:
                ctx = self._legacy_turn_context_id
                self._legacy_turn_context_id = None
                if ctx is not None and self._multi_session is not None:
                    await self._close_context_best_effort(
                        self._multi_session, ctx, immediate=True
                    )
        else:
            await self._close_remaining_caller_contexts()

    async def _close_remaining_caller_contexts(self) -> None:
        """Drop every tracked server context (idempotent)."""
        for stale in list(self._caller_contexts):
            await self._close_caller_context(stale)

    async def _close_multi_session_locked(self) -> None:
        async with self._multi_session_lock:
            await self._close_multi_session()

    async def _invoke_super_set_voice(self, voice: str) -> None:
        """Call Pipecat's set_voice, awaiting when the installed version is async."""
        result = super().set_voice(voice)
        if inspect.isawaitable(result):
            await result

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as KugelAudio service supports metrics generation.
        """
        return True

    def prewarm(self) -> None:
        """Pre-establish the WebSocket connection to reduce TTFB on first synthesis.

        Call this during pipeline setup to eagerly establish the connection,
        eliminating ~100-220ms of handshake latency from the first request.

        Example:
            ```python
            tts = KugelAudioTTSService(api_key="your-api-key", language="en")
            tts.prewarm()
            ```
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("prewarm() called outside event loop, skipping")
            return
        if self._closed:
            logger.warning("prewarm() called after cleanup, skipping")
            return
        if self._prewarm_task is not None and not self._prewarm_task.done():
            return

        async def _do_prewarm() -> None:
            try:
                async with self._multi_session_lock:
                    await self._get_multi_session()
                logger.info("KugelAudio PipeCat TTS connection pre-warmed")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Failed to prewarm KugelAudio connection, "
                    "will retry on first synthesis",
                    exc_info=True,
                )
            finally:
                if self._prewarm_task is asyncio.current_task():
                    self._prewarm_task = None

        self._prewarm_task = loop.create_task(_do_prewarm())

    async def set_model(self, model: str) -> None:
        """Set the TTS model.

        Closes any existing connection since the model is part of the
        WebSocket URL routing.

        Args:
            model: The model identifier to use.
        """
        if model != self._opts.model:
            async with self._multi_session_lock:
                await self._close_multi_session()
        self._opts.model = model
        await super().set_model(model)

    async def set_voice(self, voice: str) -> None:
        """Set the voice ID.

        A no-op when the voice is unchanged. A redundant ``set_voice`` used to
        close the persistent multi-context WebSocket (the voice is part of the
        WS routing, so the next ``run_tts`` had to reconnect) and to update
        Pipecat's settings, which recreates the audio context. On the turn path
        that cost ~0.5s before the first byte and discarded the prewarmed
        handshake; skipping it keeps both the connection and the context.

        Args:
            voice: The voice identifier (integer as string).
        """
        requested_voice_id = int(voice) if voice else None
        if requested_voice_id == self._opts.voice_id:
            return
        previous_voice_id = self._opts.voice_id
        self._opts.voice_id = requested_voice_id
        if self._opts.voice_id != previous_voice_id:
            async with self._multi_session_lock:
                await self._close_multi_session()
        await self._invoke_super_set_voice(voice)

    async def cleanup(self) -> None:
        """Clean up resources when the service is stopped."""
        self._closed = True
        prewarm = self._prewarm_task
        self._prewarm_task = None
        if prewarm is not None and not prewarm.done():
            prewarm.cancel()
            try:
                await prewarm
            except asyncio.CancelledError:
                pass
        async with self._multi_session_lock:
            await self._close_multi_session()
        await self._client.aclose()
        # TTSService owns Pipecat's input-frame task. Omitting this left that
        # task dangling after every prewarmed realtime call even though the
        # KugelAudio WebSocket itself had been closed.
        await super().cleanup()

    async def run_tts(
        self, text: str, context_id: Optional[str] = None
    ) -> AsyncGenerator[Frame, None]:
        """Synthesize text to speech via the core KugelAudio SDK.

        Delegates to ``client.tts.multi_context_session()`` so Pipecat context
        IDs map onto KugelAudio's ``/ws/tts/multi`` endpoint.

        Args:
            text: The text to synthesize into speech.
            context_id: Pipecat TTS context ID. Pipecat 0.x calls this method
                with only ``text``; Pipecat 1.x passes this second argument for
                context tracking.

        Yields:
            Frame: TTSStartedFrame, TTSAudioRawFrame chunks, and TTSStoppedFrame.
        """
        logger.debug(f"Generating TTS [{text}]")
        # Pipecat 1.x supplies a per-turn context_id; record it so the turn/audio
        # completion hooks can close the server-side context (KUG-1087). The 0.x
        # path (context_id is None) is handled by the wrapper-owned stable
        # context and must not be tracked here.
        if context_id is not None:
            self._caller_contexts.add(context_id)
        error_frame: Optional[ErrorFrame] = None

        await self.start_ttfb_metrics()
        await self.start_tts_usage_metrics(text)
        yield _tts_started_frame(context_id)

        # Try once, then retry once on a connection drop. The retry path
        # exists because a NAT/proxy can quietly drop the WS during the
        # user's think-time between turns, and we'd rather give the
        # conversation one transparent reconnect than a hard error frame.
        #
        # A barge-in cancels this generator mid-stream (CancelledError, which is
        # not an Exception and propagates out cleanly). The interrupted turn's
        # legacy context is left open and closed by the next turn's start frame
        # (_legacy_begin_turn's stale close) so its tail can't bleed into the
        # next utterance; a caller-supplied (1.x) context is the caller's to
        # manage. Either way there is nothing to do in a finally here — which
        # also avoids awaiting during cancellation unwind.
        for attempt in (1, 2):
            first_chunk = True
            yielded_audio = False
            try:
                async with self._multi_session_lock:
                    session = await self._get_multi_session()
                    effective_context_id = await self._resolve_turn_context_id(
                        session, context_id
                    )
                    t0 = time.perf_counter()
                    async for item in session.send(
                        effective_context_id,
                        text,
                        flush=True,
                    ):
                        if first_chunk:
                            ttfa_ms = (time.perf_counter() - t0) * 1000
                            logger.info(f"KugelAudio TTFA: {ttfa_ms:.0f}ms")
                            first_chunk = False
                        await self.stop_ttfb_metrics()
                        yielded_audio = True
                        yield _tts_audio_frame(
                            audio=item.audio,
                            sample_rate=self._opts.sample_rate,
                            num_channels=1,
                            context_id=context_id,
                        )
                error_frame = None
                break
            except Exception as e:
                # WS is dead. If nothing was streamed yet on this attempt
                # and we have a retry budget, drop the cached session and
                # try once more with a fresh connection. After that, give
                # up loudly.
                if isinstance(
                    e, KugelAudioConnectionError
                ) or _is_ws_connection_closed_error(e):
                    self._multi_session_needs_reset = True
                    if attempt == 1 and not yielded_audio:
                        logger.warning(
                            "KugelAudio WS dropped; reconnecting and retrying once"
                        )
                        continue
                    logger.error(f"TTS error: {e}")
                    error_frame = ErrorFrame(error=_error_frame_message(e))
                    break

                logger.error(f"TTS error: {e}")
                error_frame = ErrorFrame(error=_error_frame_message(e))
                break

        await self.stop_ttfb_metrics()
        logger.debug(f"Finished TTS [{text}]")
        if error_frame is not None:
            yield error_frame
        yield _tts_stopped_frame(context_id)
