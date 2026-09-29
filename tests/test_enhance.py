"""Tests for ``client.enhance`` and ``load_audio_stream``.

HTTP runs against ``httpx.MockTransport``; the stream runs against a local
websockets server (in its own thread) speaking the public stream protocol.
"""

from __future__ import annotations

import asyncio
import base64
import email
import gc
import io
import json
import struct
import threading
import time
import wave
from collections.abc import Iterator
from typing import Any, Callable

import httpx
import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from kugelaudio import (
    Audio,
    AudioStream,
    AuthenticationError,
    EnhancedAudio,
    InsufficientCreditsError,
    KugelAudio,
    KugelAudioConnectionError,
    KugelAudioError,
    RateLimitError,
    ValidationError,
    load_audio,
    load_audio_stream,
)


def _pcm(value: int, n: int, width: int = 2) -> bytes:
    return value.to_bytes(width, "little", signed=True) * n


def _wav(
    n_samples: int = 2400,
    rate: int = 24000,
    fill: int = 7,
    channels: int = 1,
    width: int = 2,
    frames: bytes | None = None,
) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(frames or _pcm(fill, n_samples * channels, width))
    return buffer.getvalue()


MODEL = "clarity-1"
INPUT_WAV = _wav(1600, rate=16000, fill=1)
SPEAKER_WAV = _wav(3200, rate=16000, fill=2)
OUTPUT_WAV = _wav(2400, rate=24000, fill=7)


@pytest.fixture
def files(tmp_path):
    audio = tmp_path / "call.wav"
    audio.write_bytes(INPUT_WAV)
    speaker = tmp_path / "speaker.wav"
    speaker.write_bytes(SPEAKER_WAV)
    return audio, speaker


# ---------------------------------------------------------------- HTTP


def _parts(request: httpx.Request) -> dict[str, tuple[str | None, bytes]]:
    """Parse a multipart body into ``{field: (filename, payload)}``."""
    head = f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
    message = email.message_from_bytes(head + request.content)
    out: dict[str, tuple[str | None, bytes]] = {}
    for part in message.get_payload():
        name = part.get_param("name", header="content-disposition")
        out[name] = (part.get_filename(), part.get_payload(decode=True))
    return out


Handler = Callable[[httpx.Request], httpx.Response]


def _ok(requests: list[httpx.Request]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            content=OUTPUT_WAV,
            headers={"content-type": "audio/wav", "X-Audio-Duration-Seconds": "0.1"},
        )

    return handler


def _http_client(handler: Handler) -> KugelAudio:
    client = KugelAudio(api_key="sk-test", api_url="https://api.example.test")
    transport = httpx.MockTransport(handler)
    client._http_client = httpx.Client(
        transport=transport, headers=client._http_client.headers
    )
    client.enhance._async_transport = transport
    return client


def _check_noise_removal(requests: list[httpx.Request], filename: str) -> None:
    (request,) = requests
    assert request.method == "POST"
    assert str(request.url) == "https://api.example.test/v1/audio/enhance"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert request.headers["x-api-key"] == "sk-test"
    parts = _parts(request)
    assert set(parts) == {"file", "model", "task"}
    assert parts["model"] == (None, b"clarity-1")
    assert parts["file"] == (filename, INPUT_WAV)
    assert parts["task"][1] == b"noise_removal"


def _check_speaker(requests: list[httpx.Request], names: tuple[str, str]) -> None:
    (request,) = requests
    parts = _parts(request)
    assert set(parts) == {"file", "model", "task", "speaker"}
    assert parts["model"] == (None, b"clarity-1")
    assert parts["file"] == (names[0], INPUT_WAV)
    assert parts["speaker"] == (names[1], SPEAKER_WAV)
    assert parts["task"][1] == b"target_speaker_extraction"


def _check_result(result: EnhancedAudio) -> None:
    assert isinstance(result, EnhancedAudio)
    assert result.sample_rate == 24000
    assert result.audio == _pcm(7, 2400)
    assert result.duration == pytest.approx(0.1)


def test_generate_sync_from_path(files):
    requests: list[httpx.Request] = []
    result = _http_client(_ok(requests)).enhance.generate_sync(
        load_audio(str(files[0])), model=MODEL
    )
    _check_noise_removal(requests, "call.wav")
    _check_result(result)


def test_generate_sync_from_bytes():
    requests: list[httpx.Request] = []
    result = _http_client(_ok(requests)).enhance.generate_sync(
        load_audio(INPUT_WAV), model=MODEL
    )
    _check_noise_removal(requests, "audio.wav")
    _check_result(result)


def test_generate_sync_with_speaker_paths(files):
    requests: list[httpx.Request] = []
    result = _http_client(_ok(requests)).enhance.generate_sync(
        load_audio(files[0]), model=MODEL, speaker=load_audio(files[1])
    )
    _check_speaker(requests, ("call.wav", "speaker.wav"))
    _check_result(result)


def test_generate_sync_with_speaker_bytes():
    requests: list[httpx.Request] = []
    _http_client(_ok(requests)).enhance.generate_sync(
        load_audio(INPUT_WAV), model=MODEL, speaker=load_audio(SPEAKER_WAV)
    )
    _check_speaker(requests, ("audio.wav", "audio.wav"))


async def test_generate(files):
    requests: list[httpx.Request] = []
    result = await _http_client(_ok(requests)).enhance.generate(
        load_audio(files[0]), model=MODEL
    )
    _check_noise_removal(requests, "call.wav")
    _check_result(result)


async def test_generate_with_speaker():
    requests: list[httpx.Request] = []
    result = await _http_client(_ok(requests)).enhance.generate(
        load_audio(INPUT_WAV), model=MODEL, speaker=load_audio(SPEAKER_WAV)
    )
    _check_speaker(requests, ("audio.wav", "audio.wav"))
    _check_result(result)


def test_save_writes_valid_wav(tmp_path):
    result = EnhancedAudio.from_wav(OUTPUT_WAV)
    target = tmp_path / "out.wav"
    result.save(target)
    with wave.open(str(target), "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getframerate() == 24000
        assert reader.getnframes() == 2400
        assert reader.readframes(2400) == result.audio
    assert target.read_bytes() == result.wav


def _error_body(status: int, code: str, message: str) -> dict:
    """The server's error body: ``{error, error_code, code}``."""
    return {"error": message, "error_code": code, "code": status}


_RPM = "Rate limit exceeded (10 requests per minute)"
_CONCURRENCY = "Concurrent generation limit reached (2)"
_NOT_ENABLED = "Speech enhancement is not enabled for this organization."
_AT_CAPACITY = "Speech enhancement is at capacity. Please try again shortly."


@pytest.mark.parametrize(
    ("status", "body", "headers", "expected", "retry_after"),
    [
        (
            400,
            _error_body(400, "VALIDATION_ERROR", "bad wav"),
            {},
            ValidationError,
            None,
        ),
        (
            401,
            _error_body(401, "UNAUTHORIZED", "bad key"),
            {},
            AuthenticationError,
            None,
        ),
        (
            402,
            _error_body(402, "INSUFFICIENT_CREDITS", "Insufficient credits"),
            {},
            InsufficientCreditsError,
            None,
        ),
        (
            403,
            _error_body(403, "UNAUTHORIZED", _NOT_ENABLED),
            {},
            AuthenticationError,
            None,
        ),
        (
            429,
            _error_body(429, "RATE_LIMITED", _RPM),
            {"Retry-After": "42"},
            RateLimitError,
            42,
        ),
        (429, _error_body(429, "RATE_LIMITED", _CONCURRENCY), {}, RateLimitError, None),
        (
            503,
            _error_body(503, "MODEL_UNAVAILABLE", _AT_CAPACITY),
            {"Retry-After": "5"},
            KugelAudioConnectionError,
            5,
        ),
    ],
    ids=["400", "401", "402", "403-not-enabled", "429-rpm", "429-concurrency", "503"],
)
async def test_http_errors_map_to_sdk_exceptions(
    status, body, headers, expected, retry_after
):
    headers = {**headers, "x-request-id": "req-http"}
    client = _http_client(
        lambda request: httpx.Response(status, json=body, headers=headers)
    )
    with pytest.raises(expected) as sync_err:
        client.enhance.generate_sync(load_audio(INPUT_WAV), model=MODEL)
    with pytest.raises(expected) as async_err:
        await client.enhance.generate(
            load_audio(INPUT_WAV), model=MODEL, speaker=load_audio(SPEAKER_WAV)
        )
    for err in (sync_err.value, async_err.value):
        assert err.status_code == status
        assert err.error_code == body["error_code"]
        assert err.retry_after == retry_after
        assert err.request_id == "req-http"
        assert body["error"] in err.message


def test_non_wav_response_raises():
    client = _http_client(lambda request: httpx.Response(200, content=b"not a wav"))
    with pytest.raises(Exception, match="not a WAV"):
        client.enhance.generate_sync(load_audio(INPUT_WAV), model=MODEL)


@pytest.mark.parametrize(
    "raw", ["call.wav", b"RIFF", "PATH"], ids=["str", "bytes", "path"]
)
async def test_raw_inputs_raise_type_error_before_any_request(raw, files):
    raw = files[0] if raw == "PATH" else raw
    requests: list[httpx.Request] = []
    enhance = _http_client(_ok(requests)).enhance
    audio = load_audio(INPUT_WAV)
    hint = r"load_audio\(\.\.\.\)"
    with pytest.raises(TypeError, match=hint):
        await enhance.generate(audio=raw, model=MODEL)
    with pytest.raises(TypeError, match=hint):
        await enhance.generate(audio, model=MODEL, speaker=raw)
    with pytest.raises(TypeError, match=hint):
        enhance.generate_sync(audio=raw, model=MODEL)
    with pytest.raises(TypeError, match=hint):
        enhance.stream(load_audio_stream(INPUT_WAV), model=MODEL, speaker=raw)
    with pytest.raises(TypeError, match=hint):
        enhance.stream_sync(load_audio_stream(INPUT_WAV), model=MODEL, speaker=raw)
    assert requests == []


# ------------------------------------------------------------- load_audio


def test_load_audio_path_and_bytes(files):
    from_path = load_audio(files[0])
    assert isinstance(from_path, Audio)
    assert from_path.data == INPUT_WAV
    assert from_path.filename == "call.wav"
    assert from_path.duration == pytest.approx(0.1)
    from_bytes = load_audio(SPEAKER_WAV)
    assert (from_bytes.data, from_bytes.filename) == (SPEAKER_WAV, "audio.wav")
    assert from_bytes.duration == pytest.approx(0.2)


def test_load_audio_float_wav_has_no_duration():
    samples = struct.pack("<4f", 0.0, 0.5, -0.5, 0.0)
    fmt = struct.pack("<HHIIHH", 3, 1, 16000, 64000, 4, 32)
    data = (
        b"RIFF"
        + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(samples))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(samples))
        + samples
    )
    audio = load_audio(data)
    assert audio.data == data
    assert audio.duration is None


def test_load_audio_fails_early(tmp_path):
    with pytest.raises(ValidationError, match="empty"):
        load_audio(b"")
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(ValidationError, match="empty"):
        load_audio(empty)
    with pytest.raises(ValidationError, match="Cannot read"):
        load_audio(tmp_path / "missing.wav")
    with pytest.raises(TypeError):
        load_audio(42)  # type: ignore[arg-type]


# ------------------------------------------------------ load_audio_stream


def test_load_audio_stream_mono_path_chunks_at_100_ms_by_default(tmp_path):
    # Small chunks keep the first enhanced audio close to the start of the input.
    path = tmp_path / "call.wav"
    path.write_bytes(_wav(2000, rate=8000, fill=3))  # 0.25 s
    stream = load_audio_stream(path)
    assert isinstance(stream, AudioStream)
    assert stream.sample_rate == 8000
    assert stream.duration == pytest.approx(0.25)
    chunks = list(stream)
    assert [len(chunk) for chunk in chunks] == [1600, 1600, 800]  # 100 ms = 800 samples
    assert b"".join(chunks) == _pcm(3, 2000)
    assert list(stream) == chunks  # iterating twice works


def test_load_audio_stream_bytes_and_chunk_seconds():
    stream = load_audio_stream(INPUT_WAV, chunk_seconds=0.04)
    assert stream.sample_rate == 16000
    assert stream.duration == pytest.approx(0.1)
    assert [len(chunk) for chunk in stream] == [1280, 1280, 640]


def test_load_audio_stream_downmixes_stereo():
    frames = (_pcm(100, 1) + _pcm(300, 1) + _pcm(-3, 1) + _pcm(-5, 1)) * 2
    stream = load_audio_stream(_wav(2, rate=16000, channels=2, frames=frames))
    assert stream.sample_rate == 16000
    assert b"".join(stream) == (_pcm(200, 1) + _pcm(-4, 1)) * 2
    assert stream.duration == pytest.approx(4 / 16000)


@pytest.mark.parametrize("width", [1, 3], ids=["8-bit", "24-bit"])
def test_load_audio_stream_rejects_other_sample_widths(width):
    with pytest.raises(ValidationError, match=r"16-bit PCM WAV.*enhance\.generate"):
        load_audio_stream(_wav(100, rate=16000, width=width, fill=1))


@pytest.mark.parametrize("chunk_seconds", [0, -1, 1.5])
def test_load_audio_stream_rejects_bad_chunk_seconds(chunk_seconds):
    with pytest.raises(ValidationError, match="chunk_seconds"):
        load_audio_stream(INPUT_WAV, chunk_seconds=chunk_seconds)


def test_load_audio_stream_rejects_non_wav_and_empty():
    with pytest.raises(ValidationError, match="Cannot read the WAV"):
        load_audio_stream(b"not a wav file")
    with pytest.raises(ValidationError, match="empty"):
        load_audio_stream(_wav(0, rate=16000))


# ---------------------------------------------------------------- stream


# A post-accept refusal: an optional error frame, then a close code.
_Refusal = tuple[dict | None, int]


def _frame(status: int, code: str, message: str, **extra: Any) -> dict:
    """A server error frame in the shared ``{type, error, error_code, code}`` shape."""
    return {
        "type": "error",
        "request_id": "req-ws",
        **_error_body(status, code, message),
        **extra,
    }


class _Server:
    """Local server speaking the enhancement stream protocol.

    Echoes each binary frame reversed; answers ``end`` with ``done``.
    ``handshake`` refuses the upgrade with an HTTP response; ``on_config``
    answers the config with a refusal instead of ``ready``; ``on_audio``
    answers the first audio message with its echo and then a refusal.
    """

    def __init__(self) -> None:
        self.handshake: tuple[int, dict[str, str]] | None = None
        self.on_config: _Refusal | None = None
        self.on_audio: _Refusal | None = None
        self.config: dict | None = None
        self.path: str | None = None
        self.received: list[bytes] = []
        self.disconnected = threading.Event()

    def process_request(self, connection: Any, request: Any) -> Any:
        if self.handshake is None:
            return None
        status, headers = self.handshake
        response = connection.respond(status, json.dumps({"error": "refused"}))
        for name, value in headers.items():
            response.headers[name] = value
        self.disconnected.set()
        return response

    def handler(self, ws: Any) -> None:
        try:
            self._serve(ws)
        except ConnectionClosed:
            pass
        finally:
            self.disconnected.set()

    @staticmethod
    def _refuse(ws: Any, refusal: _Refusal) -> None:
        frame, close_code = refusal
        if frame is not None:
            ws.send(json.dumps(frame))
        ws.close(close_code)

    def _serve(self, ws: Any) -> None:
        self.path = ws.request.path
        self.config = json.loads(ws.recv())
        if self.on_config is not None:
            self._refuse(ws, self.on_config)
            return
        ws.send(
            json.dumps(
                {"type": "ready", "sample_rate_hz": 24000, "encoding": "pcm_s16le"}
            )
        )
        for message in ws:
            if isinstance(message, bytes):
                self.received.append(message)
                ws.send(message[::-1])
                if self.on_audio is not None:
                    self._refuse(ws, self.on_audio)
                    return
            elif json.loads(message).get("type") == "end":
                ws.send(json.dumps({"type": "done", "duration_s": 0.25}))
                ws.close(1000)
                return

    def wait_disconnected(self) -> None:
        assert self.disconnected.wait(5), "the SDK did not close the socket"


@pytest.fixture
def ws_env() -> Iterator[tuple[_Server, KugelAudio]]:
    server = _Server()
    ws_server = serve(
        server.handler,
        "127.0.0.1",
        0,
        compression=None,
        process_request=server.process_request,
    )
    thread = threading.Thread(target=ws_server.serve_forever, daemon=True)
    thread.start()
    port = ws_server.socket.getsockname()[1]
    try:
        yield server, KugelAudio(api_key="sk-test", api_url=f"http://127.0.0.1:{port}")
    finally:
        ws_server.shutdown()
        thread.join(5)


def _sender_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "kugelaudio-enhance-send"]


def _assert_cleaned_up(server: _Server) -> None:
    server.wait_disconnected()
    assert _sender_threads() == []


def _endless() -> Iterator[bytes]:
    while True:
        time.sleep(0.005)
        yield b"\x01\x02"


def _check_noise_config(server: _Server, rate: int) -> None:
    assert server.config == {
        "type": "config",
        "model": "clarity-1",
        "task": "noise_removal",
        "sample_rate_hz": rate,
        "encoding": "pcm_s16le",
    }
    assert server.path is not None
    assert server.path.startswith("/v1/audio/enhance/stream?api_key=sk-test&sdk=")


def test_stream_sync_audio_stream_input(ws_env, tmp_path):
    server, client = ws_env
    path = tmp_path / "call.wav"
    path.write_bytes(_wav(20000, rate=8000, fill=3))
    chunks = list(client.enhance.stream_sync(load_audio_stream(path), model=MODEL))
    _check_noise_config(server, 8000)
    assert [len(message) for message in server.received] == [1600] * 25  # 100 ms chunks
    assert chunks == [message[::-1] for message in server.received]
    _assert_cleaned_up(server)


def test_stream_sync_iterable_with_sample_rate_splits_long_chunks(ws_env):
    server, client = ws_env
    long_chunk = bytes(range(10)) * 2000  # 20000 bytes > 1 s at 8 kHz
    chunks = list(
        client.enhance.stream_sync(
            [b"\x01\x02\x03\x04", long_chunk], model=MODEL, sample_rate=8000
        )
    )
    _check_noise_config(server, 8000)
    assert server.received == [
        b"\x01\x02\x03\x04",
        long_chunk[:16000],
        long_chunk[16000:],
    ]
    assert chunks == [message[::-1] for message in server.received]


def test_stream_sync_speaker_config(ws_env, files):
    server, client = ws_env
    audio = load_audio_stream(INPUT_WAV)
    chunks = list(
        client.enhance.stream_sync(
            audio, model=MODEL, sample_rate=16000, speaker=load_audio(files[1])
        )
    )
    assert server.config is not None
    assert server.config["task"] == "target_speaker_extraction"
    assert server.config["sample_rate_hz"] == 16000
    assert base64.b64decode(server.config["speaker_wav_b64"]) == SPEAKER_WAV
    assert server.received == [_pcm(1, 1600)]
    assert len(chunks) == 1


@pytest.mark.parametrize(
    ("audio", "kwargs", "match"),
    [
        ([b"\x00\x00"], {}, "sample_rate is required"),
        (load_audio_stream(INPUT_WAV), {"sample_rate": 8000}, "does not match"),
        ("call.wav", {"sample_rate": 16000}, "load_audio_stream"),
        (INPUT_WAV, {"sample_rate": 16000}, "load_audio_stream"),
        (42, {"sample_rate": 16000}, "iterable"),
    ],
    ids=["no-rate", "rate-mismatch", "path", "bytes", "not-iterable"],
)
@pytest.mark.parametrize("method", ["stream_sync", "stream"])
def test_stream_rejects_bad_input_before_connecting(audio, kwargs, match, method):
    # Nothing listens on port 9: a connection attempt would fail differently.
    client = KugelAudio(api_key="sk-test", api_url="http://127.0.0.1:9")
    with pytest.raises(ValidationError, match=match):
        getattr(client.enhance, method)(audio, model=MODEL, **kwargs)


def test_stream_sync_rejects_async_iterable_in_sync_stream():
    async def chunks():
        yield b"\x00\x00"

    client = KugelAudio(api_key="sk-test", api_url="http://127.0.0.1:9")
    source = chunks()
    with pytest.raises(ValidationError, match="iterable"):
        client.enhance.stream_sync(source, model=MODEL, sample_rate=16000)
    asyncio.run(source.aclose())


_RATE_LIMITED_FRAME = _frame(429, "RATE_LIMITED", _RPM, retry_after=7)

# (server setup, expected class, status_code, retry_after, request_id).
_REFUSALS: dict[
    str, tuple[dict, type[KugelAudioError], int, int | None, str | None]
] = {
    "handshake-429": (
        {"handshake": (429, {"Retry-After": "7", "X-Request-Id": "req-hs"})},
        RateLimitError,
        429,
        7,
        "req-hs",
    ),
    "handshake-429-concurrency": (
        {"handshake": (429, {"X-Request-Id": "req-hs"})},
        RateLimitError,
        429,
        None,
        "req-hs",
    ),
    "handshake-402": (
        {"handshake": (402, {})},
        InsufficientCreditsError,
        402,
        None,
        None,
    ),
    "handshake-403": ({"handshake": (403, {})}, AuthenticationError, 401, None, None),
    "handshake-503": (
        {"handshake": (503, {"Retry-After": "5"})},
        KugelAudioConnectionError,
        503,
        5,
        None,
    ),
    "frame-then-4029": (
        {"on_config": (_RATE_LIMITED_FRAME, 4029)},
        RateLimitError,
        429,
        7,
        "req-ws",
    ),
    "close-4029-only": ({"on_config": (None, 4029)}, RateLimitError, 429, None, None),
    "frame-then-4000": (
        {"on_config": (_frame(400, "VALIDATION_ERROR", "bad rate"), 4000)},
        ValidationError,
        400,
        None,
        "req-ws",
    ),
    "close-4001": ({"on_config": (None, 4001)}, AuthenticationError, 401, None, None),
    "close-4003": (
        {"on_config": (None, 4003)},
        InsufficientCreditsError,
        402,
        None,
        None,
    ),
    "close-4500": (
        {"on_config": (None, 4500)},
        KugelAudioConnectionError,
        503,
        None,
        None,
    ),
    "close-4000": (
        {"on_config": (None, 4000)},
        KugelAudioConnectionError,
        503,
        None,
        None,
    ),
    "close-1011": (
        {"on_config": (None, 1011)},
        KugelAudioConnectionError,
        503,
        None,
        None,
    ),
}


def _check_refusal(err: KugelAudioError, case: str) -> None:
    _, expected, status, retry_after, request_id = _REFUSALS[case]
    assert type(err) is expected
    assert err.status_code == status
    assert err.retry_after == retry_after
    assert err.request_id == request_id


@pytest.mark.parametrize("case", list(_REFUSALS))
def test_stream_sync_refusal_raises_typed_error_before_first_chunk(ws_env, case):
    server, client = ws_env
    for name, value in _REFUSALS[case][0].items():
        setattr(server, name, value)
    received: list[bytes] = []
    with pytest.raises(KugelAudioError) as err:
        for chunk in client.enhance.stream_sync(
            [b"\x00\x00"], model=MODEL, sample_rate=16000
        ):
            received.append(chunk)  # noqa: PERF402 - keep chunks seen before the error
    _check_refusal(err.value, case)
    assert received == []
    assert server.received == []
    _assert_cleaned_up(server)


@pytest.mark.parametrize(
    ("refusal", "expected", "retry_after"),
    [
        ((_RATE_LIMITED_FRAME, 4029), RateLimitError, 7),
        ((None, 4029), RateLimitError, None),
        (
            (_frame(503, "MODEL_UNAVAILABLE", "backend failed"), 4500),
            KugelAudioConnectionError,
            None,
        ),
        ((None, 4003), InsufficientCreditsError, None),
    ],
    ids=["frame-then-4029", "close-4029-only", "frame-then-4500", "close-4003"],
)
def test_stream_sync_refusal_mid_session_raises(ws_env, refusal, expected, retry_after):
    server, client = ws_env
    server.on_audio = refusal
    received: list[bytes] = []
    with pytest.raises(expected) as err:
        for chunk in client.enhance.stream_sync(
            _endless(), model=MODEL, sample_rate=16000
        ):
            received.append(chunk)  # noqa: PERF402 - keep chunks seen before the error
    assert err.value.retry_after == retry_after
    assert received == [b"\x02\x01"]
    _assert_cleaned_up(server)


def test_stream_sync_early_break_leaves_no_thread(ws_env):
    server, client = ws_env
    for chunk in client.enhance.stream_sync(_endless(), model=MODEL, sample_rate=16000):
        assert chunk == b"\x02\x01"
        break
    _assert_cleaned_up(server)


def test_stream_sync_garbage_collected_leaves_no_thread(ws_env):
    server, client = ws_env
    chunks = client.enhance.stream_sync(_endless(), model=MODEL, sample_rate=16000)
    assert next(chunks) == b"\x02\x01"
    del chunks
    gc.collect()
    _assert_cleaned_up(server)


def test_stream_sync_consumer_exception_leaves_no_thread(ws_env):
    server, client = ws_env
    with pytest.raises(RuntimeError, match="consumer failed"):
        for _chunk in client.enhance.stream_sync(
            _endless(), model=MODEL, sample_rate=16000
        ):
            raise RuntimeError("consumer failed")
    _assert_cleaned_up(server)


def test_stream_sync_send_side_failure_surfaces(ws_env):
    server, client = ws_env

    def failing_input() -> Iterator[bytes]:
        yield b"\x00\x01"
        time.sleep(0.01)
        raise OSError("microphone unplugged")

    with pytest.raises(OSError, match="microphone unplugged"):
        for _chunk in client.enhance.stream_sync(
            failing_input(), model=MODEL, sample_rate=16000
        ):
            pass
    _assert_cleaned_up(server)


def test_stream_sync_rejects_non_bytes_chunk(ws_env):
    server, client = ws_env
    with pytest.raises(ValidationError, match="must be bytes"):
        list(client.enhance.stream_sync(["not bytes"], model=MODEL, sample_rate=16000))
    _assert_cleaned_up(server)


# ------------------------------------------------------- stream (async)


@pytest.fixture
async def async_env(ws_env):
    """``ws_env`` plus checks that no task leaked and no error was logged."""
    loop = asyncio.get_running_loop()
    errors: list[dict] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    baseline = asyncio.all_tasks()
    server, _ = ws_env
    try:
        yield ws_env
        # Cleanup after an early exit runs via the async-generator finalizer.
        for _ in range(500):
            leaked = asyncio.all_tasks() - baseline - {asyncio.current_task()}
            if server.disconnected.is_set() and not leaked:
                break
            await asyncio.sleep(0.01)
        assert server.disconnected.is_set(), "the SDK did not close the socket"
        assert not leaked, f"leaked tasks: {leaked}"
        gc.collect()
        await asyncio.sleep(0)
        assert errors == [], errors
    finally:
        loop.set_exception_handler(previous_handler)


async def _endless_async():
    while True:
        await asyncio.sleep(0.005)
        yield b"\x01\x02"


async def test_stream_audio_stream_input(async_env, tmp_path):
    server, client = async_env
    path = tmp_path / "call.wav"
    path.write_bytes(_wav(20000, rate=8000, fill=3))
    chunks = [
        chunk
        async for chunk in client.enhance.stream(load_audio_stream(path), model=MODEL)
    ]
    _check_noise_config(server, 8000)
    assert [len(message) for message in server.received] == [1600] * 25  # 100 ms chunks
    assert chunks == [message[::-1] for message in server.received]


async def test_stream_async_iterable_with_speaker(async_env, files):
    server, client = async_env
    source = [bytes([i, i + 1]) for i in range(5)]

    async def chunks_in():
        for chunk in source:
            await asyncio.sleep(0)
            yield chunk

    chunks = [
        chunk
        async for chunk in client.enhance.stream(
            chunks_in(), model=MODEL, sample_rate=16000, speaker=load_audio(files[1])
        )
    ]
    assert chunks == [chunk[::-1] for chunk in source]
    assert server.config is not None
    assert server.config["task"] == "target_speaker_extraction"
    assert base64.b64decode(server.config["speaker_wav_b64"]) == SPEAKER_WAV


@pytest.mark.parametrize("case", list(_REFUSALS))
async def test_stream_refusal_raises_typed_error(async_env, case):
    server, client = async_env
    for name, value in _REFUSALS[case][0].items():
        setattr(server, name, value)
    received: list[bytes] = []
    with pytest.raises(KugelAudioError) as err:
        async for chunk in client.enhance.stream(
            [b"\x00\x00"], model=MODEL, sample_rate=16000
        ):
            received.append(chunk)
    _check_refusal(err.value, case)
    assert received == []


@pytest.mark.parametrize(
    ("refusal", "expected", "retry_after"),
    [
        ((_RATE_LIMITED_FRAME, 4029), RateLimitError, 7),
        ((None, 4029), RateLimitError, None),
        (
            (_frame(503, "MODEL_UNAVAILABLE", "backend failed"), 4500),
            KugelAudioConnectionError,
            None,
        ),
    ],
    ids=["frame-then-4029", "close-4029-only", "frame-then-4500"],
)
async def test_stream_refusal_mid_session(async_env, refusal, expected, retry_after):
    server, client = async_env
    server.on_audio = refusal
    received: list[bytes] = []
    with pytest.raises(expected) as err:
        async for chunk in client.enhance.stream(
            _endless_async(), model=MODEL, sample_rate=16000
        ):
            received.append(chunk)
    assert err.value.retry_after == retry_after
    assert received == [b"\x02\x01"]


async def test_stream_early_break_cancels_sender(async_env):
    _, client = async_env
    async for chunk in client.enhance.stream(
        _endless_async(), model=MODEL, sample_rate=16000
    ):
        assert chunk == b"\x02\x01"
        break
    # async_env asserts the socket closed and the send task is gone.


async def test_stream_consumer_exception_cancels_sender(async_env):
    _, client = async_env
    with pytest.raises(RuntimeError, match="consumer failed"):
        async for _chunk in client.enhance.stream(
            _endless_async(), model=MODEL, sample_rate=16000
        ):
            raise RuntimeError("consumer failed")


async def test_stream_send_side_failure_surfaces(async_env):
    _, client = async_env

    async def failing_input():
        yield b"\x00\x01"
        await asyncio.sleep(0.01)
        raise OSError("microphone unplugged")

    with pytest.raises(OSError, match="microphone unplugged"):
        async for _chunk in client.enhance.stream(
            failing_input(), model=MODEL, sample_rate=16000
        ):
            pass


# ------------------------------------------------------------------ model


@pytest.mark.parametrize(
    "method", ["generate", "generate_sync", "stream", "stream_sync"]
)
def test_model_is_required_keyword(method):
    client = KugelAudio(api_key="sk-test", api_url="http://127.0.0.1:9")
    source = (
        load_audio(INPUT_WAV)
        if method.startswith("generate")
        else load_audio_stream(INPUT_WAV)
    )
    with pytest.raises(TypeError, match="model"):
        getattr(client.enhance, method)(source)
    with pytest.raises(TypeError):
        getattr(client.enhance, method)(source, MODEL)  # keyword-only


@pytest.mark.parametrize("model", ["", "   ", None])
async def test_empty_model_rejected_before_any_request(model):
    requests: list[httpx.Request] = []
    enhance = _http_client(_ok(requests)).enhance
    audio = load_audio_stream(INPUT_WAV)
    with pytest.raises(ValidationError, match="model"):
        enhance.generate_sync(load_audio(INPUT_WAV), model=model)
    with pytest.raises(ValidationError, match="model"):
        await enhance.generate(load_audio(INPUT_WAV), model=model)
    with pytest.raises(ValidationError, match="model"):
        enhance.stream_sync(audio, model=model)
    with pytest.raises(ValidationError, match="model"):
        enhance.stream(audio, model=model)
    assert requests == []
