"""KugelAudio speech to text for Pipecat, with server-side turn detection.

The service keeps one streaming connection open for the whole call and sends
the user's audio as it arrives. The server decides when a turn ends, so the
service itself emits the user started / stopped speaking frames; no local VAD
or turn analyzer is needed.

Example:
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
    from kugelaudio.pipecat import KugelAudioSTTService

    stt = KugelAudioSTTService(language="de")  # reads KUGELAUDIO_API_KEY
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=ExternalUserTurnStrategies()
        ),
    )
    pipeline = Pipeline([transport.input(), stt, aggregators.user(), llm, tts,
                         transport.output(), aggregators.assistant()])
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from functools import lru_cache
from typing import Any, Optional

from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from kugelaudio import KugelAudio
from kugelaudio._diagnostics import INTEGRATION_PIPECAT
from kugelaudio.client import _parse_api_key, _resolve_region_url
from kugelaudio.exceptions import ConnectionError as KugelAudioConnectionError
from kugelaudio.exceptions import RateLimitError, ValidationError
from kugelaudio.models import StreamingTranscriptionEvent

from .enhance import _stream_resampler
from .tts import _error_frame_message, _is_ws_connection_closed_error

try:
    from pipecat.frames.frames import InterruptionFrame as _PipecatInterruptionFrame
except ImportError:  # Pipecat 0.0.x
    from pipecat.frames.frames import StartInterruptionFrame as _PipecatInterruptionFrame

# Pipecat >= 1.11: a service with its own turn detection proposes turn
# boundaries and the user aggregator's ExternalUserTurnStrategies decides and
# interrupts. The proposed stop is a control frame, so it stays ordered after
# the final transcript; UserStoppedSpeakingFrame is a system frame that
# overtakes it, and the aggregator then waits ~0.5 s for the text.
try:
    from pipecat.frames.frames import (
        ProposedUserStartedSpeakingFrame as _TurnStartFrame,
        ProposedUserStoppedSpeakingFrame as _TurnStopFrame,
    )

    _PROPOSES_TURNS = True
except ImportError:  # Pipecat < 1.11: announce the turn and interrupt here
    _TurnStartFrame = UserStartedSpeakingFrame  # type: ignore[misc,assignment]
    _TurnStopFrame = UserStoppedSpeakingFrame  # type: ignore[misc,assignment]
    _PROPOSES_TURNS = False

# Pipecat >= 0.0.104 keeps model and language in an STTSettings store and logs
# an error at start for any field left unset.
try:
    from pipecat.services.settings import STTSettings as _PipecatSTTSettings
except ImportError:  # Pipecat < 0.0.104
    _PipecatSTTSettings = None  # type: ignore[assignment,misc]

logger = logging.getLogger("kugelaudio.pipecat")

DEFAULT_STT_SAMPLE_RATE = 16000

# Waits between reconnect attempts after a retryable failure; one more failure
# in a row than there are entries is reported as fatal.
_RECONNECT_DELAYS_S = (0.5, 1.0, 2.0)
# How long EndFrame waits for the server to send the last turn's final.
_END_TIMEOUT_S = 3.0


@lru_cache(maxsize=None)
def _frame_fields(frame_type: type) -> frozenset[str]:
    return frozenset(f.name for f in dataclasses.fields(frame_type))


@lru_cache(maxsize=None)
def _push_error_is_modern() -> bool:
    """Whether ``push_error`` takes ``force_treat_as_permanent`` (Pipecat >= 1.8)."""
    try:
        params = inspect.signature(FrameProcessor.push_error).parameters
    except (TypeError, ValueError):
        return False
    return "force_treat_as_permanent" in params


def _recommends_turn_strategies() -> bool:
    """Whether STT metadata can recommend user turn strategies (Pipecat >= 1.11)."""
    try:
        from pipecat.frames.frames import STTMetadataFrame
    except ImportError:
        return False
    return "user_turn_strategies" in _frame_fields(STTMetadataFrame)


def _transcript_language(language: str | list[str] | None) -> Optional[Language]:
    """The configured language as a Pipecat ``Language``.

    ``None`` means unknown: no language, a list to detect among, ``"auto"``,
    or a code Pipecat has no ``Language`` for. The server does not report the
    language of a final, so it is never guessed.
    """
    if not isinstance(language, str) or language == "auto":
        return None
    try:
        return Language(language)
    except ValueError:
        return None


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(
        exc, (RateLimitError, KugelAudioConnectionError, OSError)
    ) or _is_ws_connection_closed_error(exc)


class KugelAudioSTTService(STTService):
    """KugelAudio streaming speech to text for Pipecat.

    Transcribes the user's audio over one connection per call. The server
    ends each turn itself (``turn_detection="kugelturn"`` by default) and the
    service turns that into Pipecat frames: ``UserStartedSpeakingFrame`` and an
    interruption when a turn starts, ``InterimTranscriptionFrame`` for each
    partial, then ``TranscriptionFrame`` and ``UserStoppedSpeakingFrame`` when
    the turn ends. Revisions that arrive after a final are ignored. From
    Pipecat 1.11 the turn frames are the ``Proposed...`` variants, which the
    user aggregator resolves into the turn and the interruption.

    On Pipecat 1.x before 1.11, pass
    ``user_turn_strategies=ExternalUserTurnStrategies()`` to the user
    aggregator; from 1.11 the service recommends it itself. Do not add a VAD
    analyzer for turn taking.

    Args:
        api_key: KugelAudio API key; defaults to ``KUGELAUDIO_API_KEY``. Prefix
            with ``"eu-"`` to select the direct EU endpoint.
        region: API endpoint region (``"eu"`` for the direct EU endpoint).
            Overrides a prefix on the key; ignored when ``base_url`` is set.
        base_url: API base URL; overrides region selection.
        model: Speech-to-text model ID. Omit it to use the server's default.
        language: A code such as ``"de"``, a list of codes to detect among, or
            ``"auto"``. Omit it for German or English.
        turn_detection: ``"kugelturn"`` (server default), ``"vad"`` or
            ``"manual"``.
        turn_detection_level: With ``"kugelturn"``: ``"responsive"`` or
            ``"conservative"``.
        boosted_phrases: Names, brands and domain terms to spell as written.
        project_id: The project whose dictionaries apply.
        dictionary_ids: The project dictionaries to apply; ``[]`` applies none.
        sample_rate: Rate in Hz the audio is sent at; pipeline audio at another
            rate is resampled to it. Defaults to 16000.
        **kwargs: Passed to Pipecat's ``STTService``.

    Options left as ``None`` are not sent, so the server's defaults apply.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        language: str | list[str] | None = None,
        turn_detection: Optional[str] = None,
        turn_detection_level: Optional[str] = None,
        boosted_phrases: Optional[list[str]] = None,
        project_id: Optional[int] = None,
        dictionary_ids: Optional[list[int]] = None,
        sample_rate: int = DEFAULT_STT_SAMPLE_RATE,
        **kwargs: Any,
    ) -> None:
        if _PipecatSTTSettings is not None and "settings" not in kwargs:
            kwargs["settings"] = _PipecatSTTSettings(
                model=model,
                language=language if isinstance(language, str) else None,
            )
        super().__init__(**kwargs)

        key = api_key or os.environ.get("KUGELAUDIO_API_KEY")
        if not key:
            raise ValueError("KUGELAUDIO_API_KEY must be set or api_key must be provided")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        clean_key, detected_region = _parse_api_key(key)
        if base_url:
            resolved_url = base_url.rstrip("/")
        else:
            try:
                resolved_url = _resolve_region_url(region or detected_region)
            except ValidationError as exc:
                raise ValueError(str(exc)) from exc
        self._client = KugelAudio(
            api_key=clean_key, api_url=resolved_url, _integration=INTEGRATION_PIPECAT
        )

        options: dict[str, Any] = {
            "model": model,
            "language": language,
            "turn_detection": turn_detection,
            "turn_detection_level": turn_detection_level,
            "boosted_phrases": boosted_phrases,
            "project_id": project_id,
            "dictionary_ids": dictionary_ids,
        }
        # Only the options the caller set are sent; the server's defaults apply
        # to the rest.
        self._stream_options: dict[str, Any] = {
            k: v for k, v in options.items() if v is not None
        }
        self._wire_rate = sample_rate
        self._language = _transcript_language(language)

        self._queue: Optional[asyncio.Queue[Optional[bytes]]] = None
        self._receiver: Optional[asyncio.Task[None]] = None
        self._resampler: Optional[BaseAudioResampler] = None
        self._accepting = False
        self._input_ended = False
        self._turn_started = False
        self._alternatives_logged = False

    def can_generate_metrics(self) -> bool:
        return True

    @property
    def supports_ttfs(self) -> bool:
        """The server ends the turn, so there is no speech-end-to-final wait."""
        return False

    def service_metadata_frame(self) -> Frame:
        """Recommend turn strategies that leave turn detection to the server."""
        frame = super().service_metadata_frame()
        if _recommends_turn_strategies():
            from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies

            frame.user_turn_strategies = ExternalUserTurnStrategies()
        return frame

    def language_to_service_language(self, language: Language) -> str | None:
        """The API's two-letter code: ``Language.EN_US`` is sent as ``"en"``."""
        return str(language.value).split("-")[0].lower()

    async def _update_settings(self, delta: Any) -> Any:
        """Apply an ``STTUpdateSettingsFrame``.

        A changed model or language only takes effect on a new connection, so
        the open stream is replaced; a turn in progress is ended without its
        final. Options the update does not name (a language list, for one)
        stay as they were.
        """
        result = await super()._update_settings(delta)
        if _PipecatSTTSettings is None:
            # Pipecat < 0.0.104: the delta is a plain mapping of new values.
            updates = {k: delta[k] for k in ("model", "language") if k in delta}
        else:
            updates = {
                k: getattr(self._settings, k, None)
                for k in ("model", "language")
                if k in (result or {})
            }
        await self._apply_stream_updates(updates)
        return result

    async def _apply_stream_updates(self, updates: dict[str, object]) -> None:
        if not updates:
            return
        for name, value in updates.items():
            if isinstance(value, Language):
                value = self.language_to_service_language(value)
            if isinstance(value, str) and value:
                self._stream_options[name] = value
            else:
                self._stream_options.pop(name, None)
        self._language = _transcript_language(self._stream_options.get("language"))
        if self._receiver is not None:
            await self._close_stream(graceful=False)
            await self._abandon_turn()
            self._open_stream()

    async def process_audio_frame(self, frame: Any, direction: FrameDirection) -> None:
        """Refuse multi-channel audio: the API transcribes mono PCM16 only."""
        channels = getattr(frame, "num_channels", 1)
        if channels != 1:
            if self._accepting:
                await self._report(
                    ValidationError(
                        "KugelAudio speech to text takes mono audio; got a frame "
                        f"with {channels} channels."
                    ),
                    fatal=True,
                )
            return
        await super().process_audio_frame(frame, direction)

    # -- lifecycle ---------------------------------------------------------

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        self._open_stream()

    async def stop(self, frame: EndFrame) -> None:
        await self._close_stream(graceful=True)
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame) -> None:
        await self._close_stream(graceful=False)
        await super().cancel(frame)

    async def cleanup(self) -> None:
        await self._close_stream(graceful=False)
        await self._client.aclose()
        await super().cleanup()

    def _open_stream(self) -> None:
        if self._receiver is not None:
            return
        self._queue = asyncio.Queue()
        self._resampler = (
            _stream_resampler() if self.sample_rate != self._wire_rate else None
        )
        self._accepting = True
        self._input_ended = False
        self._receiver = asyncio.get_running_loop().create_task(self._receive_loop())

    async def _close_stream(self, *, graceful: bool) -> None:
        """End the input (the SDK then sends ``end_of_speech``) and close."""
        self._accepting = False
        self._input_ended = True
        if self._queue is not None:
            self._queue.put_nowait(None)
        task, self._receiver = self._receiver, None
        if task is None:
            return
        if graceful:
            try:
                await asyncio.wait_for(asyncio.shield(task), _END_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.warning(
                    "KugelAudio STT: no close from the server within %.0f s of the "
                    "end of input; closing the connection.",
                    _END_TIMEOUT_S,
                )
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    # -- audio in ----------------------------------------------------------

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Optional[Frame], None]:
        """Queue the audio for the open stream; transcripts arrive separately."""
        if self._accepting and self._queue is not None:
            if self._resampler is not None:
                audio = await self._resampler.resample(
                    audio, self.sample_rate, self._wire_rate
                )
            if audio:
                self._queue.put_nowait(audio)
        yield None

    async def _audio(self, queue: asyncio.Queue[Optional[bytes]]) -> AsyncIterator[bytes]:
        while True:
            chunk = await queue.get()
            if chunk is None:
                return
            yield chunk

    # -- transcripts out ---------------------------------------------------

    async def _receive_loop(self) -> None:
        """Run the stream, reconnecting after a retryable failure."""
        assert self._queue is not None  # set by _open_stream
        failures = 0
        while True:
            received = False
            error: Optional[Exception] = None
            try:
                events = self._client.asr.stream(
                    self._audio(self._queue),
                    sample_rate=self._wire_rate,
                    **self._stream_options,
                )
                async with aclosing(events):
                    async for event in events:
                        received = True
                        await self._on_event(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reported below, never dropped
                error = exc
            await self._abandon_turn()
            if self._input_ended:
                if error is not None:
                    # KEEP-JUSTIFIED: the pipeline is ending (EndFrame/CancelFrame);
                    # nothing downstream can act on a failed close any more.
                    logger.warning(
                        "KugelAudio STT: stream failed after the input ended: %s",
                        _error_frame_message(error),
                    )
                return
            if error is not None and not _is_retryable(error):
                await self._report(error, fatal=True)
                return
            failures = 0 if received else failures + 1
            if failures > len(_RECONNECT_DELAYS_S):
                await self._report(
                    error
                    or KugelAudioConnectionError(
                        "The speech-to-text stream kept closing; giving up."
                    ),
                    fatal=True,
                )
                return
            delay = _RECONNECT_DELAYS_S[max(failures - 1, 0)]
            if error is not None:
                await self._report(error, fatal=False)
                retry_after = getattr(error, "retry_after", None)
                if retry_after:
                    delay = max(delay, float(retry_after))
            logger.info("KugelAudio STT stream ended; reconnecting in %.1f s", delay)
            await asyncio.sleep(delay)

    async def _on_event(self, event: StreamingTranscriptionEvent) -> None:
        if event.type != "partial":
            if not self._alternatives_logged:
                self._alternatives_logged = True
                logger.debug(
                    "KugelAudio STT: ignoring %r frames (the final already ended the turn)",
                    event.type,
                )
            return
        text = event.partial_text
        has_text = bool(text.strip())
        if not event.is_final:
            if not has_text:
                return
            if not self._turn_started:
                await self._start_turn()
            await self.push_frame(
                InterimTranscriptionFrame(text, self._user(), time_now_iso8601(), self._language)
            )
            return
        if not has_text and not self._turn_started:
            return
        if not self._turn_started:
            await self._start_turn()
        if has_text:
            await self.push_frame(self._final_frame(text, event))
        await self._end_turn()

    def _final_frame(self, text: str, event: StreamingTranscriptionEvent) -> TranscriptionFrame:
        extra: dict[str, Any] = {}
        fields = _frame_fields(TranscriptionFrame)
        if "result" in fields:
            extra["result"] = event
        if "finalized" in fields:
            extra["finalized"] = True
        return TranscriptionFrame(text, self._user(), time_now_iso8601(), self._language, **extra)

    def _user(self) -> str:
        return getattr(self, "_user_id", "") or ""

    async def _broadcast(self, frame_type: type[Frame]) -> None:
        if hasattr(self, "broadcast_frame"):
            await self.broadcast_frame(frame_type)
        else:  # Pipecat 0.0.x
            await self.push_frame(frame_type())

    async def _start_turn(self) -> None:
        # RISK: during stop() this processor's process task is awaiting the
        # receiver; a Pipecat that cancels the process task on interruption
        # would then drop the EndFrame. Hence the _input_ended guard below.
        self._turn_started = True
        await self._broadcast(_TurnStartFrame)
        if _PROPOSES_TURNS or self._input_ended:
            # The aggregator interrupts on a proposal; a pipeline that is
            # ending has nothing left to interrupt.
            return
        if hasattr(self, "broadcast_interruption"):
            await self.broadcast_interruption()
        elif getattr(self, "interruptions_allowed", False):  # Pipecat 0.0.x
            await self.push_frame(_PipecatInterruptionFrame())

    async def _end_turn(self) -> None:
        self._turn_started = False
        await self._broadcast(_TurnStopFrame)

    async def _abandon_turn(self) -> None:
        """Close a turn the connection ended before its final."""
        if self._turn_started:
            await self._end_turn()

    async def _report(self, exc: Exception, *, fatal: bool) -> None:
        message = _error_frame_message(exc)
        if fatal:
            self._accepting = False
            logger.error("KugelAudio STT stopped: %s", message)
        else:
            logger.warning("KugelAudio STT: %s", message)
        if _push_error_is_modern():
            await self.push_error(message, exception=exc, force_treat_as_permanent=fatal)
            return
        frame = ErrorFrame(error=message, fatal=fatal)
        if hasattr(self, "push_error_frame"):
            await self.push_error_frame(frame)
        else:  # Pipecat 0.0.x
            await self.push_frame(frame, FrameDirection.UPSTREAM)


__all__ = ["KugelAudioSTTService", "DEFAULT_STT_SAMPLE_RATE"]
