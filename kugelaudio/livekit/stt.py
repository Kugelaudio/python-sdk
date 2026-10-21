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

"""KugelAudio speech-to-text plugin for LiveKit Agents.

The server ends each user turn itself (KugelTurn by default), so the stream
signals start of speech, interim and final transcripts and end of speech, and
``AgentSession(turn_handling={"turn_detection": "stt"})`` lets those turns
drive the agent. The WebSocket is the SDK's own
:meth:`kugelaudio.asr.ASRResource.stream`; this module only maps its events
onto LiveKit's.
"""

from __future__ import annotations

import asyncio
import logging
import os
import weakref
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import Literal

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    stt,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given
from websockets.exceptions import WebSocketException

from kugelaudio._diagnostics import INTEGRATION_LIVEKIT
from kugelaudio.client import KugelAudio, _parse_api_key, _resolve_region_url
from kugelaudio.exceptions import (
    AuthenticationError,
    InsufficientCreditsError,
    KugelAudioError,
    RateLimitError,
    ValidationError,
)
from kugelaudio.exceptions import ConnectionError as KugelAudioConnectionError
from kugelaudio.models import StreamingTranscriptionEvent

logger = logging.getLogger("kugelaudio.livekit")

DEFAULT_STT_SAMPLE_RATE = 16000
_AUTO_LANGUAGE = "auto"

TurnDetection = Literal["kugelturn", "vad", "manual"]
TurnDetectionLevel = Literal["responsive", "conservative"]


@dataclass(frozen=True, slots=True)
class _STTOptions:
    """The constructor options; ``None`` means "not set, not sent"."""

    sample_rate: int
    model: str | None
    language: str | list[str] | None
    turn_detection: TurnDetection | None
    turn_detection_level: TurnDetectionLevel | None
    boosted_phrases: list[str] | None
    project_id: int | None
    dictionary_ids: list[int] | None


def _transcript_language(language: str | Sequence[str] | None) -> str:
    """The language a transcript is labelled with: the configured code when it
    is exactly one code other than ``"auto"``, else ``""`` (unknown)."""
    if isinstance(language, str):
        code = language
    elif language is not None and len(language) == 1:
        code = language[0]
    else:
        return ""
    return "" if code == _AUTO_LANGUAGE else code


def _speech_data(language: str, text: str, *, end_time: float = 0.0) -> stt.SpeechData:
    """A transcript alternative; ``language`` may be ``""`` (unknown)."""
    # livekit-agents < 1.5 types ``language`` as str; newer versions type it
    # LanguageCode and normalize a str to one in ``__post_init__``.
    return stt.SpeechData(
        language=language,  # type: ignore[arg-type]  # see above
        text=text,
        end_time=end_time,
    )


def _api_error(exc: KugelAudioError) -> APIError:
    """The LiveKit error for an SDK error, with the right retryability.

    Auth, credits and validation fail the same way on every retry; a rate
    limit or a dropped connection may not.
    """
    # RISK: livekit-agents' RecognizeStream retries every APIError up to
    # max_retry without reading ``retryable`` (1.1 .. 1.6.9), so a
    # non-retryable error is still retried; it fails the same way each time
    # and the last attempt surfaces it.
    message = exc.message
    if isinstance(exc, KugelAudioConnectionError):
        return APIConnectionError(message, retryable=True)
    if isinstance(exc, RateLimitError):
        return APIStatusError(
            message, status_code=429, request_id=exc.request_id, retryable=True
        )
    if isinstance(
        exc, (AuthenticationError, InsufficientCreditsError, ValidationError)
    ):
        return APIStatusError(
            message,
            status_code=exc.status_code if exc.status_code is not None else 400,
            request_id=exc.request_id,
            retryable=False,
        )
    # Any other server error: a 4xx (not found, ...) is not retryable, a 5xx
    # or an unknown status (model temporarily unavailable, ...) is.
    status = exc.status_code if exc.status_code is not None else -1
    return APIStatusError(
        message,
        status_code=status,
        request_id=exc.request_id,
        retryable=not 400 <= status < 500,
    )


@dataclass(slots=True)
class _TurnMapper:
    """Maps the server's transcript events onto LiveKit speech events.

    Pure: no I/O, no clock. One instance per stream, so a turn that a
    reconnect splits is still started only once.
    """

    language: str
    _started: bool = False
    _alternatives_logged: bool = False

    def feed(
        self, event: StreamingTranscriptionEvent, request_id: str
    ) -> list[stt.SpeechEvent]:
        """The LiveKit events for one server event, in order (often none)."""
        if event.type != "partial":
            # The final already committed the turn to the agent; a late
            # revision (``alternatives``) cannot un-commit it.
            if not self._alternatives_logged:
                self._alternatives_logged = True
                logger.debug(
                    "Ignoring speech-to-text %r frames; the final already ended the turn.",
                    event.type,
                )
            return []
        text = event.partial_text
        if not event.is_final:
            if not text:
                return []
            out = self._start(request_id)
            out.append(
                self._event(stt.SpeechEventType.INTERIM_TRANSCRIPT, text, request_id)
            )
            return out
        if not text and not self._started:
            return []
        out = self._start(request_id)
        out.append(self._event(stt.SpeechEventType.FINAL_TRANSCRIPT, text, request_id))
        out.append(
            stt.SpeechEvent(
                type=stt.SpeechEventType.END_OF_SPEECH, request_id=request_id
            )
        )
        self._started = False
        return out

    def _start(self, request_id: str) -> list[stt.SpeechEvent]:
        if self._started:
            return []
        self._started = True
        return [
            stt.SpeechEvent(
                type=stt.SpeechEventType.START_OF_SPEECH, request_id=request_id
            )
        ]

    def _event(
        self, kind: stt.SpeechEventType, text: str, request_id: str
    ) -> stt.SpeechEvent:
        return stt.SpeechEvent(
            type=kind,
            request_id=request_id,
            alternatives=[_speech_data(self.language, text)],
        )


class STT(stt.STT):
    """KugelAudio speech-to-text for LiveKit Agents.

    Streams the user's audio to KugelAudio and lets the server's end of turn
    drive the conversation.

    Example:
        ```python
        from kugelaudio.livekit import STT
        from livekit.agents import AgentSession
        from livekit.plugins import silero

        session = AgentSession(
            stt=STT(language="en"),
            turn_handling={"turn_detection": "stt"},
            vad=silero.VAD.load(),
            llm=llm,
            tts=tts,
        )
        ```
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        region: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        language: str | list[str] | None = None,
        turn_detection: TurnDetection | None = None,
        turn_detection_level: TurnDetectionLevel | None = None,
        boosted_phrases: list[str] | None = None,
        project_id: int | None = None,
        dictionary_ids: list[int] | None = None,
        sample_rate: int = DEFAULT_STT_SAMPLE_RATE,
        telemetry: bool | None = None,
    ) -> None:
        """Create a KugelAudio STT. Options left as ``None`` are not sent, so
        the server's defaults apply.

        Args:
            api_key: KugelAudio API key. If not provided, reads the
                ``KUGELAUDIO_API_KEY`` environment variable. Prefix with
                ``"eu-"`` to select the direct EU endpoint; the prefix is
                stripped before auth.
            region: API endpoint region. Use ``"eu"`` for the direct EU
                endpoint. Overrides any prefix detected from the API key.
                Ignored when *base_url* is set.
            base_url: API base URL. Overrides region selection entirely.
            model: Speech-to-text model ID, sent as given. Omit it to use the
                API's default model.
            language: A code such as ``"de"``, a list of codes to detect
                among, or ``"auto"``. Transcripts carry it when it is one code
                other than ``"auto"``, else an empty (unknown) language.
            turn_detection: Who ends a turn: ``"kugelturn"`` (the server
                default), ``"vad"`` or ``"manual"``.
            turn_detection_level: With ``"kugelturn"``: ``"responsive"``
                (the server default) or ``"conservative"``.
            boosted_phrases: Names, brands and domain terms to spell as written.
            project_id: The project whose dictionaries apply. Omit it to use
                the API key's project.
            dictionary_ids: The project dictionaries to apply; ``[]`` applies
                none. Omit it to apply every active dictionary.
            sample_rate: Rate in Hz the audio is resampled to before sending.
                Defaults to 16000.
            telemetry: Opt in or out of anonymous client-error diagnostics,
                exactly like ``KugelAudio(telemetry=...)``. The
                ``KUGELAUDIO_TELEMETRY`` environment variable overrides it.

        Raises:
            ValueError: No API key, or an unknown *region*.
        """
        # ``offline_recognize`` only exists on newer livekit-agents
        # STTCapabilities; older 1.x raises TypeError on unknown kwargs.
        if "offline_recognize" in {
            f.name for f in dataclass_fields(stt.STTCapabilities)
        }:
            capabilities = stt.STTCapabilities(
                streaming=True, interim_results=True, offline_recognize=True
            )
        else:
            capabilities = stt.STTCapabilities(streaming=True, interim_results=True)
        super().__init__(capabilities=capabilities)

        kugelaudio_api_key = api_key or os.environ.get("KUGELAUDIO_API_KEY")
        if not kugelaudio_api_key:
            raise ValueError(
                "KUGELAUDIO_API_KEY must be set or api_key must be provided"
            )
        clean_key, detected_region = _parse_api_key(kugelaudio_api_key)
        if base_url:
            resolved_url = base_url.rstrip("/")
        else:
            try:
                resolved_url = _resolve_region_url(region or detected_region)
            except ValidationError as exc:
                raise ValueError(str(exc)) from exc

        self._opts = _STTOptions(
            sample_rate=sample_rate,
            model=model,
            language=language,
            turn_detection=turn_detection,
            turn_detection_level=turn_detection_level,
            boosted_phrases=boosted_phrases,
            project_id=project_id,
            dictionary_ids=dictionary_ids,
        )
        # The client owns the HTTP pool and the diagnostics reporter; each
        # stream opens its own WebSocket through ``client.asr.stream``.
        self._client = KugelAudio(
            api_key=clean_key,
            api_url=resolved_url,
            telemetry=telemetry,
            _integration=INTEGRATION_LIVEKIT,
        )
        self._streams: weakref.WeakSet[SpeechStream] = weakref.WeakSet()

    @property
    def model(self) -> str:
        return self._opts.model or "unknown"

    @property
    def provider(self) -> str:
        return "KugelAudio"

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """Open a streaming recognition session.

        Args:
            language: Overrides the constructor's ``language`` for this stream.
            conn_options: Retry and timeout options.
        """
        stream = SpeechStream(
            stt=self,
            language=language if is_given(language) else self._opts.language,
            conn_options=conn_options,
        )
        self._streams.add(stream)
        return stream

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Transcribe one complete recording over HTTP (``client.asr.transcribe``)."""
        configured = language if is_given(language) else self._opts.language
        if isinstance(configured, list):
            if len(configured) != 1:
                raise ValueError(
                    "recognize() takes one language code or 'auto'; "
                    f"got the list {configured!r}."
                )
            configured = configured[0]
        frame = rtc.combine_audio_frames(buffer)
        _require_mono(frame)
        wav = frame.to_wav_bytes()
        try:
            response = await asyncio.to_thread(
                self._client.asr.transcribe,
                wav,
                language=configured,
                model=self._opts.model,
                boosted_phrases=self._opts.boosted_phrases,
                project_id=self._opts.project_id,
                dictionary_ids=self._opts.dictionary_ids,
            )
        except KugelAudioError as exc:
            raise _api_error(exc) from exc
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=utils.shortuuid(),
            alternatives=[
                _speech_data(
                    response.language, response.text, end_time=response.duration_s
                )
            ],
        )

    async def aclose(self) -> None:
        """Close every open stream and the client's connections."""
        streams = list(self._streams)
        self._streams.clear()
        await asyncio.gather(*(s.aclose() for s in streams))
        await self._client.aclose()


def _require_mono(frame: rtc.AudioFrame) -> None:
    if frame.num_channels != 1:
        raise ValueError(
            "KugelAudio speech to text takes mono audio; got a frame "
            f"with {frame.num_channels} channels."
        )


class SpeechStream(stt.RecognizeStream):
    """One live recognition session over the speech-to-text WebSocket.

    Audio pushed to the stream is resampled to the STT's ``sample_rate`` and
    sent as mono PCM16. Flushes are ignored: the server ends turns itself, and
    ``end_of_speech`` would end the connection. :meth:`end_input` ends the
    audio, which sends ``end_of_speech``; the stream ends when the server has
    sent the last turn.
    """

    def __init__(
        self,
        *,
        stt: STT,
        language: str | list[str] | None,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            stt=stt, conn_options=conn_options, sample_rate=stt._opts.sample_rate
        )
        self._kugel_stt = stt
        self._language = language
        self._turns = _TurnMapper(language=_transcript_language(language))
        self._unbilled_samples = 0
        self._input_exhausted = False

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Refuse a multi-channel frame before the base class resamples it.

        LiveKit's resampler is mono and would read interleaved stereo as one
        channel at twice the length.
        """
        _require_mono(frame)
        super().push_frame(frame)

    async def _audio(self) -> AsyncIterator[bytes]:
        """Mono PCM16 chunks from the input channel; ends when the input ends."""
        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                continue
            self._unbilled_samples += item.samples_per_channel
            yield item.data.tobytes()
        self._input_exhausted = True

    def _usage_event(self, request_id: str) -> stt.SpeechEvent | None:
        """Usage for the audio sent since the last one; ``None`` when there is none."""
        if not self._unbilled_samples:
            return None
        seconds = self._unbilled_samples / self._kugel_stt._opts.sample_rate
        self._unbilled_samples = 0
        return stt.SpeechEvent(
            type=stt.SpeechEventType.RECOGNITION_USAGE,
            request_id=request_id,
            recognition_usage=stt.RecognitionUsage(audio_duration=seconds),
        )

    def _emit(self, event: StreamingTranscriptionEvent, request_id: str) -> None:
        for out in self._turns.feed(event, request_id):
            self._event_ch.send_nowait(out)
        if event.type == "partial" and event.is_final:
            usage = self._usage_event(request_id)
            if usage is not None:
                self._event_ch.send_nowait(usage)

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        opts = self._kugel_stt._opts
        try:
            events = self._kugel_stt._client.asr.stream(
                self._audio(),
                sample_rate=opts.sample_rate,
                language=self._language,
                model=opts.model,
                turn_detection=opts.turn_detection,
                turn_detection_level=opts.turn_detection_level,
                boosted_phrases=opts.boosted_phrases,
                project_id=opts.project_id,
                dictionary_ids=opts.dictionary_ids,
            )
            try:
                async for event in events:
                    self._emit(event, request_id)
            finally:
                # Close the socket now, also on cancellation, not at GC.
                if isinstance(events, AsyncGenerator):
                    await events.aclose()
        except KugelAudioError as exc:
            raise _api_error(exc) from exc
        except (OSError, WebSocketException) as exc:
            # A refused connect or DNS failure reaches us untyped from websockets.
            raise APIConnectionError(
                f"KugelAudio speech-to-text connection failed: {type(exc).__name__}."
            ) from exc
        if not self._input_exhausted:
            raise APIConnectionError(
                "KugelAudio speech to text ended the stream before the input ended."
            )
        usage = self._usage_event(request_id)
        if usage is not None:
            self._event_ch.send_nowait(usage)


__all__ = ["DEFAULT_STT_SAMPLE_RATE", "STT", "SpeechStream"]
