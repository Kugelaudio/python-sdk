"""Speech to text: transcribe a recording, or stream live audio.

Exposed as ``client.asr`` on :class:`kugelaudio.KugelAudio`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import threading
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from kugelaudio._sdk_metadata import sdk_query_string
from kugelaudio.audio import AudioStream
from kugelaudio.enhance import _closed_error, _connect_async, _handshake_error
from kugelaudio.exceptions import (
    ConnectionError as KugelAudioConnectionError,
)
from kugelaudio.exceptions import (
    ValidationError,
    classify_ws_frame,
    ws_handshake_error_types,
)
from kugelaudio.models import StreamingTranscriptionEvent, TranscriptionResponse

if TYPE_CHECKING:
    from kugelaudio.client import KugelAudio

logger = logging.getLogger("kugelaudio.asr")

_TRANSCRIBE_PATH = "/v1/audio/transcriptions"
_STREAM_PATH = "/v1/audio/transcriptions/stream"
_SENDER_THREAD_NAME = "kugelaudio-asr-send"
_SENDER_JOIN_TIMEOUT_S = 5.0
_NORMAL_CLOSE = 1000
_TRANSCRIPT_FRAMES = frozenset({"partial", "alternatives"})
_END_OF_SPEECH = json.dumps({"type": "end_of_speech"})


@dataclass(frozen=True, slots=True)
class _StreamOptions:
    """The caller's ``config`` options; ``None`` means "not set, not sent"."""

    language: str | list[str] | None
    model: str | None
    turn_detection: str | None
    turn_detection_level: str | None
    boosted_phrases: list[str] | None
    project_id: int | None
    dictionary_ids: list[int] | None

    def config_frame(self, sample_rate: int) -> dict[str, object]:
        """The ``config`` frame: ``sample_rate`` plus only the options set.

        ``dictionary_ids=[]`` is set (it means "apply none") and is sent as ``[]``.
        """
        frame: dict[str, object] = {"type": "config", "sample_rate": sample_rate}
        if self.model is not None:
            frame["model"] = self.model
        if self.language is not None:
            frame["language"] = (
                self.language if isinstance(self.language, str) else list(self.language)
            )
        if self.turn_detection is not None:
            frame["turn_detection"] = self.turn_detection
        if self.turn_detection_level is not None:
            frame["turn_detection_level"] = self.turn_detection_level
        if self.boosted_phrases is not None:
            frame["boosted_phrases"] = list(self.boosted_phrases)
        if self.project_id is not None:
            frame["project_id"] = self.project_id
        if self.dictionary_ids is not None:
            frame["dictionary_ids"] = list(self.dictionary_ids)
        return frame


def _stream_rate(audio: object, sample_rate: int | None, *, allow_async: bool) -> int:
    """Validate the stream input and return its sample rate."""
    if isinstance(audio, AudioStream):
        if sample_rate is not None and sample_rate != audio.sample_rate:
            raise ValidationError(
                f"sample_rate={sample_rate} does not match the audio's "
                f"{audio.sample_rate} Hz; omit sample_rate to use it."
            )
        return audio.sample_rate
    if isinstance(audio, (str, os.PathLike, bytes, bytearray, memoryview)):
        raise ValidationError(
            "To transcribe a WAV file live, pass load_audio_stream(path). Raw "
            "PCM16 goes in as an iterable of bytes chunks with sample_rate=."
        )
    iterable = hasattr(audio, "__iter__") or (
        allow_async and hasattr(audio, "__aiter__")
    )
    if not iterable:
        raise ValidationError(
            "Speech-to-text stream audio must be an AudioStream or an iterable "
            "of mono PCM16 bytes chunks."
        )
    if sample_rate is None:
        raise ValidationError(
            "sample_rate is required when streaming raw PCM16 chunks to speech to text."
        )
    return sample_rate


def _max_bytes(sample_rate: int) -> int:
    """Bytes in 1 s of mono PCM16, the most one ``audio_chunk`` carries."""
    return sample_rate * 2


def _audio_messages(chunk: object, max_bytes: int) -> Iterator[str]:
    """Validate one input chunk; yield its ``audio_chunk`` frames of at most 1 s."""
    if not isinstance(chunk, (bytes, bytearray, memoryview)):
        raise ValidationError(
            "Speech-to-text stream chunks must be bytes (mono PCM16), "
            f"got {type(chunk).__name__}."
        )
    data = bytes(chunk)
    for start in range(0, len(data), max_bytes):
        encoded = base64.b64encode(data[start : start + max_bytes]).decode("ascii")
        yield json.dumps({"type": "audio_chunk", "audio_b64": encoded})


def _closed_normally(e: Any) -> bool:
    """Whether the server closed with 1000, its close after the last turn."""
    rcvd = getattr(e, "rcvd", None)
    return rcvd is not None and rcvd.code == _NORMAL_CLOSE


class _FrameReader:
    """Turns server frames into events; raises on error frames."""

    def __init__(self) -> None:
        self._skipped: set[str] = set()

    def event(self, message: object) -> StreamingTranscriptionEvent | None:
        """The event for a transcript frame; ``None`` for a frame of another type."""
        if not isinstance(message, str):
            raise KugelAudioConnectionError(
                "Unexpected binary frame from the speech-to-text stream."
            )
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            raise KugelAudioConnectionError(
                "Unexpected non-JSON frame from the speech-to-text stream."
            ) from e
        if not isinstance(data, dict):
            raise KugelAudioConnectionError(
                "Unexpected frame from the speech-to-text stream: not a JSON object."
            )
        if "error" in data:
            raise classify_ws_frame(data)
        frame_type = data.get("type")
        if frame_type in _TRANSCRIPT_FRAMES:
            return StreamingTranscriptionEvent.from_dict(data)
        kind = str(frame_type)
        if kind not in self._skipped:
            self._skipped.add(kind)
            # KEEP-JUSTIFIED: a frame type newer than this SDK carries no
            # transcript; skipping it keeps the stream usable, and the warning
            # (once per type and stream) says what was skipped.
            logger.warning(
                "Skipping speech-to-text stream frames of unknown type %r.", kind
            )
        return None


def _connect_sync(url: str, timeout: float) -> Any:
    import websockets
    from websockets.exceptions import InvalidURI
    from websockets.sync.client import connect

    try:
        return connect(url, compression=None, open_timeout=timeout)
    except ws_handshake_error_types(websockets) as e:
        raise _handshake_error(e) from e
    except InvalidURI:
        # Its text holds the URL, whose query carries the API key.
        raise ValidationError(
            "The KugelAudio API URL does not form a valid WebSocket URL."
        ) from None


def _run_stream_sync(
    url: str,
    config: dict[str, object],
    max_bytes: int,
    audio: Iterable[bytes],
    timeout: float,
) -> Iterator[StreamingTranscriptionEvent]:
    from websockets.exceptions import ConnectionClosed

    ws = _connect_sync(url, timeout)
    stop = threading.Event()
    failures: list[BaseException] = []

    def pump() -> None:
        try:
            for chunk in audio:
                for message in _audio_messages(chunk, max_bytes):
                    if stop.is_set():
                        return
                    ws.send(message)
            ws.send(_END_OF_SPEECH)
        except ConnectionClosed:
            return  # the receive side reports why the server closed
        except BaseException as e:  # noqa: BLE001 - re-raised by the receive loop
            failures.append(e)
            ws.close()

    sender = threading.Thread(target=pump, name=_SENDER_THREAD_NAME, daemon=True)
    reader = _FrameReader()
    try:
        ws.send(json.dumps(config))
        sender.start()
        while True:
            try:
                message = ws.recv()
            except ConnectionClosed as e:
                # A send-side failure closes the socket itself (with 1000).
                if failures:
                    raise failures[0] from None
                if _closed_normally(e):
                    return
                raise _closed_error(e) from e
            event = reader.event(message)
            if event is not None:
                yield event
    except ConnectionClosed as e:
        if not _closed_normally(e):
            raise _closed_error(e) from e
    finally:
        stop.set()
        ws.close()
        if sender.is_alive():
            sender.join(_SENDER_JOIN_TIMEOUT_S)


async def _pump_async(ws: Any, audio: Any, max_bytes: int) -> None:
    """Send every input chunk, then ``end_of_speech``."""
    from websockets.exceptions import ConnectionClosed

    try:
        if hasattr(audio, "__aiter__"):
            async for chunk in audio:
                for message in _audio_messages(chunk, max_bytes):
                    await ws.send(message)
        else:
            for chunk in audio:
                for message in _audio_messages(chunk, max_bytes):
                    await ws.send(message)
        await ws.send(_END_OF_SPEECH)
    except ConnectionClosed:
        return  # the receive side reports why the server closed


async def _run_stream_async(
    url: str, config: dict[str, object], max_bytes: int, audio: Any
) -> AsyncIterator[StreamingTranscriptionEvent]:
    from websockets.exceptions import ConnectionClosed

    ws = await _connect_async(url)
    sender: asyncio.Task[None] | None = None
    receiver: asyncio.Future[Any] | None = None
    reader = _FrameReader()
    try:
        await ws.send(json.dumps(config))
        sender = asyncio.ensure_future(_pump_async(ws, audio, max_bytes))
        while True:
            if receiver is None:
                receiver = asyncio.ensure_future(ws.recv())
            watch = {receiver} if sender.done() else {receiver, sender}
            await asyncio.wait(watch, return_when=asyncio.FIRST_COMPLETED)
            if sender.done() and not sender.cancelled():
                failure = sender.exception()
                if failure is not None:
                    raise failure
            if not receiver.done():
                continue
            done, receiver = receiver, None
            event = reader.event(done.result())
            if event is not None:
                yield event
    except ConnectionClosed as e:
        if not _closed_normally(e):
            raise _closed_error(e) from e
    finally:
        pending = [t for t in (sender, receiver) if t is not None]
        for task in pending:
            task.cancel()
        # Retrieve every outcome so no "exception was never retrieved" leaks.
        await asyncio.gather(*pending, return_exceptions=True)
        await ws.close()


class ASRResource:
    """Speech to text through the public KugelAudio API.

    :meth:`transcribe` uploads one complete recording. :meth:`stream` (async)
    and :meth:`stream_sync` (blocking) transcribe live audio over a WebSocket
    and yield each partial, final and revision as it arrives.

    Example:
        result = client.asr.transcribe(Path("call.wav").read_bytes(), language="de")
        print(result.text)

        async for event in client.asr.stream(load_audio_stream("call.wav")):
            if event.is_final:
                print(event.partial_text)
    """

    def __init__(self, client: KugelAudio):
        self._client = client

    def _stream_url(self) -> str:
        base = self._client._api_url.replace("https://", "wss://").replace(
            "http://", "ws://"
        )
        query = urlencode({"api_key": self._client._api_key})
        return f"{base}{_STREAM_PATH}?{query}&{sdk_query_string()}"

    def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        model: str | None = None,
        boosted_phrases: list[str] | None = None,
        project_id: int | None = None,
        dictionary_ids: list[int] | None = None,
    ) -> TranscriptionResponse:
        """Transcribe one complete recording.

        Args:
            model: The speech-to-text model ID, sent as given. Omit it to use
                the API's default model.
            boosted_phrases: Your vocabulary for this request: names, brands
                and domain terms to spell as written. The server refuses a
                list over its limits instead of shortening it.
            project_id: The project whose dictionaries apply. Omit it to use
                the API key's project.
            dictionary_ids: The project dictionaries whose words are applied
                as phrases after ``boosted_phrases``. Omit it to apply every
                active dictionary of the project; ``[]`` applies none.

        Example:
            result = client.asr.transcribe(Path("call.wav").read_bytes(), language="de")
            print(result.text)
        """
        if not audio:
            raise ValidationError("ASR audio must not be empty")
        fields: list[tuple[str, tuple[str | None, Any, str]]] = [
            ("file", (filename, audio, content_type)),
        ]
        if model is not None:
            fields.append(("model", (None, model, "text/plain")))
        if language is not None:
            fields.append(("language", (None, language, "text/plain")))
        for phrase in boosted_phrases or []:
            fields.append(("boosted_phrases", (None, phrase, "text/plain")))
        if project_id is not None:
            fields.append(("project_id", (None, str(project_id), "text/plain")))
        if dictionary_ids is not None:
            # Repeated form fields cannot carry an empty list: one empty
            # ``dictionary_ids`` field is the multipart spelling of ``[]``.
            values = [str(i) for i in dictionary_ids] or [""]
            for value in values:
                fields.append(("dictionary_ids", (None, value, "text/plain")))
        response = self._client._request_multipart("POST", _TRANSCRIBE_PATH, fields)
        return TranscriptionResponse.from_dict(response)

    def stream(
        self,
        audio: AudioStream | Iterable[bytes] | AsyncIterable[bytes],
        *,
        sample_rate: int | None = None,
        language: str | list[str] | None = None,
        model: str | None = None,
        turn_detection: str | None = None,
        turn_detection_level: str | None = None,
        boosted_phrases: list[str] | None = None,
        project_id: int | None = None,
        dictionary_ids: list[int] | None = None,
    ) -> AsyncIterator[StreamingTranscriptionEvent]:
        """Transcribe live audio; iterate the transcript events as they arrive.

        Input is sent from its own task while you iterate (in messages of at
        most 1 s of audio, then ``end_of_speech``), and iteration ends when
        the server closes after the last turn. Breaking out of the loop closes
        the connection and cancels that task (wrap the call in
        ``contextlib.aclosing`` to make the cleanup immediate). Options left
        as ``None`` are not sent, so the server's defaults apply.

        Args:
            audio: An :class:`AudioStream` from :func:`load_audio_stream`, or
                any iterable or async iterable of raw mono PCM16 ``bytes``
                chunks (longer than 1 s are split).
            sample_rate: Sample rate of raw chunks in Hz. Required for raw
                chunks; taken from an :class:`AudioStream`, and must match it
                if given.
            language: A code such as ``"de"``, a list of codes to detect
                among, or ``"auto"``. Omit it for German or English.
            model: The speech-to-text model ID, sent as given. Omit it to use
                the API's default model.
            turn_detection: Who ends a turn: ``"kugelturn"`` (the default),
                ``"vad"`` or ``"manual"`` (only ``end_of_speech``).
            turn_detection_level: With ``"kugelturn"``: ``"responsive"``
                (the default) or ``"conservative"``.
            boosted_phrases: Names, brands and domain terms to spell as written.
            project_id: The project whose dictionaries apply.
            dictionary_ids: The project dictionaries to apply; ``[]`` applies none.

        Yields:
            A :class:`StreamingTranscriptionEvent` per partial, final and
            alternatives frame. ``event.revises_final`` marks the turn's
            revision, which also carries ``event.language``.

        Raises:
            ValidationError: Bad input before connecting, or a refused option
                (an error frame with code 422).
            AuthenticationError, RateLimitError, InsufficientCreditsError:
                A refused connection or an error frame mid-stream.
            KugelAudioConnectionError: Any close other than the normal 1000.

        Example:
            audio = load_audio_stream("call.wav")  # mono 16-bit WAV
            async for event in client.asr.stream(audio):
                if event.revises_final:
                    print("revised:", event.partial_text, event.language)
                elif event.is_final:
                    print("final:", event.partial_text, event.turn_end_reason)
        """
        rate = _stream_rate(audio, sample_rate, allow_async=True)
        options = _StreamOptions(
            language,
            model,
            turn_detection,
            turn_detection_level,
            boosted_phrases,
            project_id,
            dictionary_ids,
        )
        return _run_stream_async(
            self._stream_url(), options.config_frame(rate), _max_bytes(rate), audio
        )

    def stream_sync(
        self,
        audio: AudioStream | Iterable[bytes],
        *,
        sample_rate: int | None = None,
        language: str | list[str] | None = None,
        model: str | None = None,
        turn_detection: str | None = None,
        turn_detection_level: str | None = None,
        boosted_phrases: list[str] | None = None,
        project_id: int | None = None,
        dictionary_ids: list[int] | None = None,
    ) -> Iterator[StreamingTranscriptionEvent]:
        """Blocking :meth:`stream`, same arguments; input is sent from a thread.

        ``audio`` must be an :class:`AudioStream` or a regular iterable.
        Breaking out of the loop closes the connection and stops the sender
        thread.

        Example:
            audio = load_audio_stream("call.wav")
            for event in client.asr.stream_sync(audio, language="de"):
                if event.is_final:
                    print(event.partial_text)
        """
        rate = _stream_rate(audio, sample_rate, allow_async=False)
        options = _StreamOptions(
            language,
            model,
            turn_detection,
            turn_detection_level,
            boosted_phrases,
            project_id,
            dictionary_ids,
        )
        return _run_stream_sync(
            self._stream_url(),
            options.config_frame(rate),
            _max_bytes(rate),
            audio,
            self._client._timeout,
        )


__all__ = ["ASRResource"]
