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
import logging
import os
import threading
import time
import wave
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from contextlib import aclosing
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
    ConnectionError as KugelAudioConnectionError,
)
from kugelaudio.exceptions import (
    KugelAudioError,
    RateLimitError,
    ValidationError,
    classify_http_response,
    classify_ws_close,
    classify_ws_frame,
    classify_ws_handshake_error,
    ws_handshake_error_types,
)

if TYPE_CHECKING:
    from kugelaudio.client import KugelAudio

logger = logging.getLogger("kugelaudio.enhance")

TASK_NOISE_REMOVAL = "noise_removal"
TASK_TARGET_SPEAKER_EXTRACTION = "target_speaker_extraction"
ENHANCED_SAMPLE_RATE = 24000

_ENHANCE_PATH = "/v1/audio/enhance"
_ENHANCE_STREAM_PATH = "/v1/audio/enhance/stream"
_SENDER_THREAD_NAME = "kugelaudio-enhance-send"
_SENDER_JOIN_TIMEOUT_S = 5.0

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


def _log_prewarm_failure(e: httpx.TransportError, url: str) -> None:
    # KEEP-JUSTIFIED: warming is an optimisation; the first real request
    # connects on its own and raises its own typed error if the network is down.
    logger.warning(
        "Could not prewarm the enhancement connection to %s (%s: %s); the first "
        "request will connect instead.",
        url,
        type(e).__name__,
        e,
    )


# ------------------------------------------------------------- WebSocket


def _stream_config(model: str, sample_rate: int, speaker: Audio | None) -> dict:
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
    return config


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


def _closed_error(e: Any) -> KugelAudioError:
    """The typed error for a server close, through the same route as TTS."""
    rcvd = getattr(e, "rcvd", None)
    return classify_ws_close(
        rcvd.code if rcvd is not None else None,
        rcvd.reason if rcvd is not None else None,
    )


def _handshake_error(e: Exception) -> KugelAudioError:
    typed = classify_ws_handshake_error(e)
    if typed is not None:
        return typed
    return KugelAudioConnectionError(f"KugelAudio WebSocket handshake failed: {e}.")


def _check_ready(message: Any) -> dict:
    """The ``ready`` frame; raises on an ``error`` frame or anything else."""
    data = json.loads(message) if isinstance(message, str) else {}
    if data.get("type") == "error":
        raise classify_ws_frame(data)
    if data.get("type") != "ready":
        raise KugelAudioConnectionError(
            "Unexpected first message from the enhancement stream."
        )
    return data


def _is_done(message: str) -> bool:
    """Handle a text frame: raise on ``error``, return True on ``done``."""
    data = json.loads(message)
    if data.get("type") == "error":
        raise classify_ws_frame(data)
    return data.get("type") == "done"


_END = json.dumps({"type": "end"})
_NOT_READY = "The enhancement stream did not become ready in time."
# The server closes a session after 60 s without a config and at 1 h; a socket
# is not reused past these, a little before the server's limits.
_SESSION_IDLE_S = 55.0
_SESSION_LIFETIME_S = 3540.0
# What a server without session mode sends on a socket left without a config
# for 30 s, before it closes it: that socket is stale, not the new config wrong.
_IDLE_FRAME_CODE = 408


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


async def _connect_async(url: str) -> Any:
    import websockets
    from websockets.exceptions import InvalidURI

    try:
        return await websockets.connect(url, compression=None)
    except ws_handshake_error_types(websockets) as e:
        raise _handshake_error(e) from e
    except InvalidURI:
        # Its text holds the URL, whose query carries the API key.
        raise ValidationError(
            "The KugelAudio API URL does not form a valid WebSocket URL."
        ) from None


class _StaleSocket(Exception):
    """A reused session socket turned out closed before it took the config."""


async def _begin_async(ws: Any, config: dict, timeout: float, *, reused: bool) -> dict:
    """Send ``config`` and wait for ``ready``; returns it.

    On a ``reused`` socket, a close (the server's idle, lifetime or restart
    close) or an old server's idle-timeout frame raises :class:`_StaleSocket`.
    """
    from websockets.exceptions import ConnectionClosed

    try:
        await ws.send(json.dumps(config))
        message = await asyncio.wait_for(ws.recv(), timeout)
    except ConnectionClosed as e:
        if reused:
            raise _StaleSocket from e
        raise _closed_error(e) from e
    except asyncio.TimeoutError as e:
        raise KugelAudioConnectionError(_NOT_READY) from e
    if reused and isinstance(message, str):
        frame = json.loads(message)
        if frame.get("type") == "error" and frame.get("code") == _IDLE_FRAME_CODE:
            raise _StaleSocket
    return _check_ready(message)


async def _exchange_async(ws: Any, config: dict, audio: Any) -> AsyncIterator[bytes]:
    """After ``ready``: send the audio and ``end`` from a task, yield the
    enhanced chunks until ``done``. Leaves the socket open."""
    from websockets.exceptions import ConnectionClosed

    sender: asyncio.Task[None] | None = None
    receiver: asyncio.Future[Any] | None = None
    try:
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


async def _run_stream_async(
    url: str, config: dict, audio: Any, timeout: float
) -> AsyncIterator[bytes]:
    ws = await _connect_async(url)
    try:
        await _begin_async(ws, config, timeout, reused=False)
        async with aclosing(_exchange_async(ws, config, audio)) as chunks:
            async for chunk in chunks:
                yield chunk
    finally:
        await ws.close()


def _is_open(ws: Any) -> bool:
    return getattr(ws, "close_code", None) is None


class EnhanceSession:
    """One warm WebSocket that carries enhancement streams one after another.

    From :meth:`EnhanceResource.session`. Use it as an async context manager,
    or call :meth:`connect` ahead of the first audio (e.g. while an agent
    starts) and :meth:`aclose` when done. Each :meth:`stream` sends its own
    config, so the task, speaker and sample rate may change between streams;
    it is admitted and billed as one request of its own. Streams run one at a
    time: a second :meth:`stream` waits until the first has finished.

    The socket is replaced without an error when the server closed it (after
    60 s without a stream, at its one-hour lifetime, or during a restart). A
    server without session support gets one connection per stream instead.

    Example:
        async with client.enhance.session() as session:
            async for chunk in session.stream(first, model="clarity-1"):
                play(chunk)
            async for chunk in session.stream(second, model="clarity-1"):
                play(chunk)
    """

    def __init__(self, resource: EnhanceResource) -> None:
        self._resource = resource
        self._ws: Any = None
        self._opened_at = 0.0
        self._idle_since = 0.0
        self._lock = asyncio.Lock()
        self._one_shot = False
        self._closed = False
        self._connecting: asyncio.Future[Any] | None = None
        # Whether the server kept this session's socket for several streams:
        # None until a stream or a refusal showed it.
        self._honored: bool | None = None

    async def __aenter__(self) -> EnhanceSession:
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def connect(self) -> None:
        """Open the session's socket now, so the next :meth:`stream` skips the
        connection setup. Does nothing while an open socket is reusable.

        Raises the typed handshake errors of :meth:`EnhanceResource.stream`
        (e.g. :class:`~kugelaudio.AuthenticationError`).
        """
        async with self._lock:
            await self._socket()

    def stream(
        self,
        audio: AudioStream | Iterable[bytes] | AsyncIterable[bytes],
        *,
        model: str,
        sample_rate: int | None = None,
        speaker: Audio | None = None,
    ) -> AsyncIterator[bytes]:
        """Enhance one audio on the session's socket; same arguments and
        output as :meth:`EnhanceResource.stream`.

        Breaking out of the loop ends this audio and closes the socket (the
        next stream opens a new one).
        """
        _check_model(model)
        rate = _stream_rate(audio, sample_rate, allow_async=True)
        config = _stream_config(model, rate, speaker)
        return self._run(config, audio)

    async def aclose(self) -> None:
        """Close the socket, and stop a connect still in flight; the session
        cannot stream afterwards."""
        self._closed = True
        connecting = self._connecting
        if connecting is not None:
            connecting.cancel()
            await asyncio.wait({connecting})
        await self._drop()

    async def _run(self, config: dict, audio: Any) -> AsyncIterator[bytes]:
        timeout = self._resource._client._timeout
        async with self._lock:
            ws, reused = await self._socket()
            if ws is None:
                url = self._resource._stream_url(session=False)
                async with aclosing(
                    _run_stream_async(url, config, audio, timeout)
                ) as chunks:
                    async for chunk in chunks:
                        yield chunk
                return
            try:
                try:
                    ready = await _begin_async(ws, config, timeout, reused=reused)
                except _StaleSocket:
                    await self._drop()
                    ws, _ = await self._socket()
                    if ws is None:
                        raise KugelAudioConnectionError(
                            "The enhancement session could not be reopened."
                        ) from None
                    ready = await _begin_async(ws, config, timeout, reused=False)
            except RateLimitError:
                # A rate-limit refusal of the config leaves the socket open.
                if not _is_open(ws):
                    await self._drop()
                raise
            except BaseException:
                # Refused, failed or cancelled mid-config: the server's state
                # for this socket is unknown, so it is not reused.
                await self._drop()
                raise
            self._honored = "request_id" in ready
            if not self._honored:
                # The server ignored session=1: it closes after this audio.
                logger.info(
                    "The enhancement server does not keep sessions open; using "
                    "one connection per stream."
                )
                self._one_shot = True
            finished = False
            try:
                async with aclosing(_exchange_async(ws, config, audio)) as chunks:
                    async for chunk in chunks:
                        yield chunk
                finished = True
            finally:
                if finished and not self._one_shot:
                    self._idle_since = time.monotonic()
                else:
                    await self._drop()

    async def _socket(self) -> tuple[Any, bool]:
        """The session socket and whether it was already open, opening one
        when needed; ``(None, False)`` when this stream must connect on its own."""
        if self._closed:
            raise KugelAudioError("The enhancement session is closed.")
        if self._one_shot:
            return None, False
        if self._ws is not None and self._reusable():
            return self._ws, True
        await self._drop()
        url = self._resource._stream_url(session=True)
        try:
            ws = await self._connect(url)
        except (RateLimitError, ValidationError) as e:
            if isinstance(e, RateLimitError) and e.retry_after is not None:
                raise
            # The open-session limit (429 without Retry-After) or a server
            # that does not know sessions (400): stream on a connection of its own.
            if isinstance(e, ValidationError):
                self._one_shot = True
            self._honored = False
            logger.info(
                "The enhancement session was refused (%s); streaming on a "
                "connection of its own.",
                e,
            )
            return None, False
        self._ws = ws
        self._opened_at = self._idle_since = time.monotonic()
        return ws, False

    async def _connect(self, url: str) -> Any:
        """Open a socket that :meth:`aclose` can stop while it connects; raises
        the session-closed error when the session closed meanwhile."""
        connecting = asyncio.ensure_future(_connect_async(url))
        self._connecting = connecting
        try:
            await asyncio.wait({connecting})
        except BaseException:
            # This stream was cancelled: stop the connect, close what it opened.
            connecting.cancel()
            await asyncio.wait({connecting})
            if not connecting.cancelled() and connecting.exception() is None:
                await connecting.result().close()
            raise
        finally:
            self._connecting = None
        if self._closed:
            if not connecting.cancelled() and connecting.exception() is None:
                await connecting.result().close()
            raise KugelAudioError("The enhancement session is closed.")
        return connecting.result()

    def _reusable(self) -> bool:
        now = time.monotonic()
        return (
            _is_open(self._ws)
            and now - self._idle_since < _SESSION_IDLE_S
            and now - self._opened_at < _SESSION_LIFETIME_S
        )

    async def _drop(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
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

    def _url(self) -> str:
        return urljoin(self._client._api_url + "/", _ENHANCE_PATH.lstrip("/"))

    def _stream_url(self, *, session: bool) -> str:
        base = self._client._api_url.replace("https://", "wss://").replace(
            "http://", "ws://"
        )
        params = {"api_key": self._client._api_key}
        if session:
            params["session"] = "1"
        return f"{base}{_ENHANCE_STREAM_PATH}?{urlencode(params)}&{sdk_query_string()}"

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
            response = await self._client._async_http.get().post(url, files=fields)
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

    async def prewarm(self) -> None:
        """Open the connection :meth:`generate` uses, so the first request
        skips the connection setup (TCP and TLS).

        Sends one ``GET`` to the enhancement path, which only accepts ``POST``:
        the server refuses it (405) before authentication or any processing,
        so it is not billed and does not count against rate limits.
        Safe to call any number of times. An idle connection is kept for up to
        50 s, so call it shortly before the first request. A network error is
        logged and not raised: the first real request then connects as usual.

        Example:
            await client.enhance.prewarm()
            result = await client.enhance.generate(load_audio("call.wav"), model="clarity-1")
        """
        url = self._url()
        try:
            await self._client._async_http.get().get(url)
        except httpx.TransportError as e:
            _log_prewarm_failure(e, url)

    def prewarm_sync(self) -> None:
        """Blocking :meth:`prewarm`: opens the connection :meth:`generate_sync` uses.

        Example:
            client.enhance.prewarm_sync()
            result = client.enhance.generate_sync(load_audio("call.wav"), model="clarity-1")
        """
        url = self._url()
        try:
            self._client._http_client.get(url)
        except httpx.TransportError as e:
            _log_prewarm_failure(e, url)

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
        config = _stream_config(model, rate, speaker)
        url = self._stream_url(session=False)
        return _run_stream_async(url, config, audio, self._client._timeout)

    def session(self) -> EnhanceSession:
        """A session: one warm WebSocket for many :meth:`stream`-style calls.

        Opening a connection costs a TCP, TLS and WebSocket handshake plus the
        server's admission; a session pays that once, and :meth:`EnhanceSession.connect`
        pays it before the first audio. Each stream on it is still one request
        (rate limits, billing). Use it as an async context manager.

        Example:
            async with client.enhance.session() as session:
                for path in ("first.wav", "second.wav"):
                    async for chunk in session.stream(
                        load_audio_stream(path), model="clarity-1"
                    ):
                        play(chunk)
        """
        return EnhanceSession(self)

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
        config = _stream_config(model, rate, speaker)
        url = self._stream_url(session=False)
        return _run_stream_sync(url, config, audio, self._client._timeout)


__all__ = ["EnhanceResource", "EnhanceSession", "EnhancedAudio"]
