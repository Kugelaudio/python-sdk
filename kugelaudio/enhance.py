"""Speech enhancement: remove background noise, or keep only one voice.

Exposed as ``client.enhance`` on :class:`kugelaudio.KugelAudio`. One rule
everywhere: enhancement removes noise; passing ``speaker=`` keeps only that
voice.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import threading
import wave
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)
from urllib.parse import urlencode, urljoin

import httpx

from kugelaudio._sdk_metadata import sdk_query_string
from kugelaudio.audio import Audio, AudioStream
from kugelaudio.exceptions import (
    AuthenticationError,
    InsufficientCreditsError,
    KugelAudioError,
    ValidationError,
    classify_http_response,
    classify_ws_close,
    classify_ws_frame,
    classify_ws_handshake_error,
    ws_handshake_error_types,
)
from kugelaudio.exceptions import (
    ConnectionError as KugelAudioConnectionError,
)

if TYPE_CHECKING:
    from kugelaudio.client import KugelAudio

TASK_NOISE_REMOVAL = "noise_removal"
TASK_TARGET_SPEAKER_EXTRACTION = "target_speaker_extraction"
ENHANCED_SAMPLE_RATE = 24000

_ENHANCE_PATH = "/v1/audio/enhance"
_ENHANCE_STREAM_PATH = "/v1/audio/enhance/stream"
_SENDER_THREAD_NAME = "kugelaudio-enhance-send"
_SENDER_JOIN_TIMEOUT_S = 5.0

# Close codes the enhancement stream uses (see the public API docs).
_WS_CLOSE_INVALID = 4400
_WS_CLOSE_UNAUTHORIZED = 4401
_WS_CLOSE_INSUFFICIENT_CREDITS = 4402
_WS_CLOSE_IDLE = 4408
_WS_CLOSE_BUSY = 4429
_WS_CLOSE_UNAVAILABLE = 4503

_MultipartFields = list[tuple[str, tuple[Optional[str], Any, str]]]


@dataclass
class EnhancedAudio:
    """Enhanced audio: mono PCM16 at 24 kHz.

    Example:
        result = await client.enhance.generate(load_audio("call.wav"), model="clarity-1")
        print(result.duration)
        result.save("clean.wav")
    """

    audio: bytes
    """Raw mono PCM16 (little-endian) samples."""
    sample_rate: int = ENHANCED_SAMPLE_RATE

    @classmethod
    def from_wav(cls, data: bytes) -> EnhancedAudio:
        """Parse a mono PCM16 WAV file into an :class:`EnhancedAudio`."""
        try:
            with wave.open(io.BytesIO(data), "rb") as reader:
                if reader.getnchannels() != 1 or reader.getsampwidth() != 2:
                    raise KugelAudioError(
                        "Unexpected enhancement response: expected mono PCM16 WAV."
                    )
                return cls(
                    audio=reader.readframes(reader.getnframes()),
                    sample_rate=reader.getframerate(),
                )
        except (wave.Error, EOFError) as e:
            raise KugelAudioError(
                f"Unexpected enhancement response: not a WAV file ({e})."
            ) from e

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.audio) // 2 / self.sample_rate

    @property
    def wav(self) -> bytes:
        """The audio as WAV file bytes."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(self.sample_rate)
            writer.writeframes(self.audio)
        return buffer.getvalue()

    def save(self, path: str | os.PathLike) -> None:
        """Write the audio to ``path`` as a WAV file."""
        Path(path).write_bytes(self.wav)


# ------------------------------------------------------------------ HTTP


def _check_model(model: Any) -> None:
    if not isinstance(model, str) or not model.strip():
        raise ValidationError(
            'model must be a non-empty string, e.g. model="clarity-1".'
        )


def _require_audio(value: object, name: str) -> None:
    if not isinstance(value, Audio):
        raise TypeError(
            f"{name} must be an Audio, got {type(value).__name__}: pass "
            f'load_audio(...), e.g. {name}=load_audio("{name}.wav").'
        )


def _multipart(audio: Audio, model: str, speaker: Audio | None) -> _MultipartFields:
    _check_model(model)
    _require_audio(audio, "audio")
    task = TASK_NOISE_REMOVAL
    if speaker is not None:
        _require_audio(speaker, "speaker")
        task = TASK_TARGET_SPEAKER_EXTRACTION
    fields: _MultipartFields = [
        ("file", (audio.filename, audio.data, "audio/wav")),
        ("model", (None, model, "text/plain")),
        ("task", (None, task, "text/plain")),
    ]
    if speaker is not None:
        fields.append(("speaker", (speaker.filename, speaker.data, "audio/wav")))
    return fields


def _parse_response(response: httpx.Response) -> EnhancedAudio:
    if response.status_code >= 400:
        raise classify_http_response(response)
    return EnhancedAudio.from_wav(response.content)


def _transport_error(e: httpx.HTTPError, url: str, timeout: float) -> KugelAudioError:
    if isinstance(e, httpx.TimeoutException):
        return KugelAudioConnectionError(
            f"Request to POST {_ENHANCE_PATH} timed out after {timeout}s."
        )
    return KugelAudioConnectionError(
        f"Could not reach KugelAudio at {url}: {e}. Check network connectivity."
    )


# ------------------------------------------------------------- WebSocket


def _stream_rate(audio: Any, sample_rate: int | None, allow_async: bool) -> int:
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
            "To stream a WAV file, pass load_audio_stream(path). Raw PCM16 goes "
            "in as an iterable of bytes chunks with sample_rate=."
        )
    iterable = hasattr(audio, "__iter__") or (
        allow_async and hasattr(audio, "__aiter__")
    )
    if not iterable:
        raise ValidationError(
            "Enhancement stream audio must be an AudioStream or an iterable of "
            "PCM16 bytes chunks."
        )
    if sample_rate is None:
        raise ValidationError(
            "sample_rate is required when streaming raw PCM16 chunks."
        )
    return sample_rate


def _split(chunk: Any, max_bytes: int) -> Iterator[bytes]:
    """Validate one input chunk and split it into messages of at most 1 s."""
    if not isinstance(chunk, (bytes, bytearray, memoryview)):
        raise ValidationError(
            "Enhancement stream chunks must be bytes (mono PCM16), "
            f"got {type(chunk).__name__}."
        )
    data = bytes(chunk)
    for start in range(0, len(data), max_bytes):
        yield data[start : start + max_bytes]


def _classify_error_frame(data: dict) -> KugelAudioError:
    """Map a stream ``{"type":"error","code","message"}`` frame."""
    return classify_ws_frame(
        {"error_code": data.get("code"), "error": data.get("message")}
    )


def _classify_close(code: int | None, reason: str | None) -> KugelAudioError:
    detail = f" ({reason})" if reason else ""
    if code == _WS_CLOSE_UNAUTHORIZED:
        return AuthenticationError()
    if code == _WS_CLOSE_INSUFFICIENT_CREDITS:
        return InsufficientCreditsError()
    if code == _WS_CLOSE_INVALID:
        return ValidationError(f"The enhancement stream rejected the request{detail}.")
    if code == _WS_CLOSE_BUSY:
        return KugelAudioConnectionError(
            f"Speech enhancement is busy{detail}. Retry shortly."
        )
    if code == _WS_CLOSE_IDLE:
        return KugelAudioConnectionError(
            f"The enhancement stream closed after 30 s without audio{detail}."
        )
    if code == _WS_CLOSE_UNAVAILABLE:
        return KugelAudioConnectionError(
            f"Speech enhancement is temporarily unavailable{detail}. Retry shortly."
        )
    return classify_ws_close(code, reason)


def _closed_error(e: Any) -> KugelAudioError:
    rcvd = getattr(e, "rcvd", None)
    return _classify_close(
        rcvd.code if rcvd is not None else None,
        rcvd.reason if rcvd is not None else None,
    )


def _handshake_error(e: Exception) -> KugelAudioError:
    typed = classify_ws_handshake_error(e)
    if typed is not None:
        return typed
    return KugelAudioConnectionError(f"KugelAudio WebSocket handshake failed: {e}.")


def _check_ready(message: Any) -> None:
    data = json.loads(message) if isinstance(message, str) else {}
    if data.get("type") == "error":
        raise _classify_error_frame(data)
    if data.get("type") != "ready":
        raise KugelAudioConnectionError(
            "Unexpected first message from the enhancement stream."
        )


def _is_done(message: str) -> bool:
    """Handle a text frame: raise on ``error``, return True on ``done``."""
    data = json.loads(message)
    if data.get("type") == "error":
        raise _classify_error_frame(data)
    return data.get("type") == "done"


_END = json.dumps({"type": "end"})
_NOT_READY = "The enhancement stream did not become ready in time."


def _run_stream_sync(
    url: str, config: dict, audio: Iterable[bytes], timeout: float
) -> Iterator[bytes]:
    import websockets
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import connect

    try:
        ws = connect(url, compression=None, open_timeout=timeout)
    except ws_handshake_error_types(websockets) as e:
        raise _handshake_error(e) from e
    stop = threading.Event()
    failures: list[BaseException] = []
    max_bytes = config["sample_rate_hz"] * 2

    def pump() -> None:
        try:
            for chunk in audio:
                for message in _split(chunk, max_bytes):
                    if stop.is_set():
                        return
                    ws.send(message)
            ws.send(_END)
        except ConnectionClosed:
            return  # the receive side reports why the server closed
        except BaseException as e:  # noqa: BLE001 - re-raised by the receive loop
            failures.append(e)
            ws.close()

    sender = threading.Thread(target=pump, name=_SENDER_THREAD_NAME, daemon=True)
    try:
        ws.send(json.dumps(config))
        try:
            _check_ready(ws.recv(timeout=timeout))
        except TimeoutError as e:
            raise KugelAudioConnectionError(_NOT_READY) from e
        sender.start()
        while True:
            try:
                message = ws.recv()
            except ConnectionClosed as e:
                if failures:
                    raise failures[0] from None
                raise _closed_error(e) from e
            if isinstance(message, (bytes, bytearray)):
                yield bytes(message)
            elif _is_done(message):
                return
    except ConnectionClosed as e:
        raise _closed_error(e) from e
    finally:
        stop.set()
        ws.close()
        if sender.is_alive():
            sender.join(_SENDER_JOIN_TIMEOUT_S)


async def _pump_async(ws: Any, audio: Any, max_bytes: int) -> None:
    """Send every input chunk, then ``end``."""
    from websockets.exceptions import ConnectionClosed

    try:
        if hasattr(audio, "__aiter__"):
            async for chunk in audio:
                for message in _split(chunk, max_bytes):
                    await ws.send(message)
        else:
            for chunk in audio:
                for message in _split(chunk, max_bytes):
                    await ws.send(message)
        await ws.send(_END)
    except ConnectionClosed:
        return  # the receive side reports why the server closed


async def _run_stream_async(
    url: str, config: dict, audio: Any, timeout: float
) -> AsyncIterator[bytes]:
    import websockets
    from websockets.exceptions import ConnectionClosed

    try:
        ws = await websockets.connect(url, compression=None)
    except ws_handshake_error_types(websockets) as e:
        raise _handshake_error(e) from e
    sender: asyncio.Task[None] | None = None
    receiver: asyncio.Future[Any] | None = None
    try:
        await ws.send(json.dumps(config))
        try:
            _check_ready(await asyncio.wait_for(ws.recv(), timeout))
        except asyncio.TimeoutError as e:
            raise KugelAudioConnectionError(_NOT_READY) from e
        sender = asyncio.ensure_future(
            _pump_async(ws, audio, config["sample_rate_hz"] * 2)
        )
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
            message = done.result()
            if isinstance(message, (bytes, bytearray)):
                yield bytes(message)
            elif _is_done(message):
                return
    except ConnectionClosed as e:
        raise _closed_error(e) from e
    finally:
        pending = [t for t in (sender, receiver) if t is not None]
        for task in pending:
            task.cancel()
        # Retrieve every outcome so no "exception was never retrieved" leaks.
        await asyncio.gather(*pending, return_exceptions=True)
        await ws.close()


# -------------------------------------------------------------- resource


class EnhanceResource:
    """Speech enhancement through the public KugelAudio API.

    Async first: :meth:`generate` and :meth:`stream` are async, and
    :meth:`generate_sync` / :meth:`stream_sync` are their blocking twins.
    Enhancement removes background noise; passing ``speaker=`` (a clean
    2-8 s sample of one voice, from :func:`load_audio`) keeps only that
    voice. Results are mono PCM16 at 24 kHz with the same duration as the
    input.

    Example:
        audio = load_audio("meeting.wav")
        speaker = load_audio("speaker.wav")
        result = await client.enhance.generate(audio, model="clarity-1", speaker=speaker)
        result.save("clean.wav")

        stream_in = load_audio_stream("meeting.wav")
        async for chunk in client.enhance.stream(stream_in, model="clarity-1"):
            ...
    """

    def __init__(self, client: KugelAudio):
        self._client = client
        # Test hook: transport for the per-call async HTTP client.
        self._async_transport: httpx.AsyncBaseTransport | None = None

    def _url(self) -> str:
        return urljoin(self._client._api_url + "/", _ENHANCE_PATH.lstrip("/"))

    def _stream_args(
        self, model: str, sample_rate: int, speaker: Audio | None
    ) -> tuple[str, dict]:
        config: dict = {
            "type": "config",
            "model": model,
            "task": TASK_NOISE_REMOVAL,
            "sample_rate_hz": sample_rate,
            "encoding": "pcm_s16le",
        }
        if speaker is not None:
            _require_audio(speaker, "speaker")
            config["task"] = TASK_TARGET_SPEAKER_EXTRACTION
            config["speaker_wav_b64"] = base64.b64encode(speaker.data).decode("ascii")
        base = self._client._api_url.replace("https://", "wss://").replace(
            "http://", "ws://"
        )
        query = urlencode({"api_key": self._client._api_key})
        return f"{base}{_ENHANCE_STREAM_PATH}?{query}&{sdk_query_string()}", config

    async def generate(
        self,
        audio: Audio,
        *,
        model: str,
        speaker: Audio | None = None,
    ) -> EnhancedAudio:
        """Enhance a recording: remove noise, or keep only ``speaker``'s voice.

        Args:
            audio: The recording, from :func:`load_audio` (any supported WAV:
                PCM 16/24/32-bit or float, mono or stereo, 8-48 kHz, at most
                300 s).
            model: Enhancement model, e.g. ``"clarity-1"``.
            speaker: Optional clean 2-8 s sample of one voice, from
                :func:`load_audio`. When given, only that voice is kept and
                every other sound removed.

        Returns:
            The enhanced audio (mono PCM16, 24 kHz, same duration).

        Example:
            audio = load_audio("meeting.wav")
            result = await client.enhance.generate(audio, model="clarity-1")
            result.save("clean.wav")
        """
        fields = _multipart(audio, model, speaker)
        url = self._url()
        try:
            async with httpx.AsyncClient(
                timeout=self._client._timeout,
                headers=self._client._http_client.headers,
                transport=self._async_transport,
            ) as http:
                response = await http.post(url, files=fields)
        except httpx.HTTPError as e:
            raise _transport_error(e, url, self._client._timeout) from e
        return _parse_response(response)

    def generate_sync(
        self,
        audio: Audio,
        *,
        model: str,
        speaker: Audio | None = None,
    ) -> EnhancedAudio:
        """Blocking :meth:`generate`, same arguments.

        Example:
            result = client.enhance.generate_sync(load_audio("call.wav"), model="clarity-1")
            result.save("clean.wav")
        """
        fields = _multipart(audio, model, speaker)
        url = self._url()
        try:
            response = self._client._http_client.request("POST", url, files=fields)
        except httpx.HTTPError as e:
            raise _transport_error(e, url, self._client._timeout) from e
        return _parse_response(response)

    def stream(
        self,
        audio: AudioStream | Iterable[bytes] | AsyncIterable[bytes],
        *,
        model: str,
        sample_rate: int | None = None,
        speaker: Audio | None = None,
    ) -> AsyncIterator[bytes]:
        """Enhance audio in real time; iterate the enhanced chunks as they arrive.

        Input is sent from its own task while you iterate, and iteration ends
        once the last input has been enhanced. Breaking out of the loop closes
        the connection and cancels that task (wrap the call in
        ``contextlib.aclosing`` to make the cleanup immediate).

        Args:
            audio: An :class:`AudioStream` from :func:`load_audio_stream`, or
                any iterable or async iterable of raw mono PCM16 ``bytes``
                chunks (ideally at most 1 s each; longer chunks are split).
            model: Enhancement model, e.g. ``"clarity-1"``.
            sample_rate: Sample rate of raw chunks (8-48 kHz). Required for
                raw chunks; taken from an :class:`AudioStream`, and must match
                it if given.
            speaker: Optional clean 2-8 s sample of one voice, from
                :func:`load_audio`. When given, only that voice is kept.

        Yields:
            Enhanced mono PCM16 at 24 kHz.

        Example:
            stream_in = load_audio_stream("meeting.wav")
            async for chunk in client.enhance.stream(stream_in, model="clarity-1"):
                play(chunk)
        """
        _check_model(model)
        rate = _stream_rate(audio, sample_rate, allow_async=True)
        url, config = self._stream_args(model, rate, speaker)
        return _run_stream_async(url, config, audio, self._client._timeout)

    def stream_sync(
        self,
        audio: AudioStream | Iterable[bytes],
        *,
        model: str,
        sample_rate: int | None = None,
        speaker: Audio | None = None,
    ) -> Iterator[bytes]:
        """Blocking :meth:`stream`, same arguments; input is sent from a thread.

        ``audio`` must be an :class:`AudioStream` or a regular iterable.
        Breaking out of the loop closes the connection and stops the sender
        thread.

        Example:
            stream_in = load_audio_stream("meeting.wav")
            for chunk in client.enhance.stream_sync(stream_in, model="clarity-1"):
                print(len(chunk), "bytes")
        """
        _check_model(model)
        rate = _stream_rate(audio, sample_rate, allow_async=False)
        url, config = self._stream_args(model, rate, speaker)
        return _run_stream_sync(url, config, audio, self._client._timeout)


__all__ = ["EnhanceResource", "EnhancedAudio"]
