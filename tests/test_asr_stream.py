"""Tests for ``client.asr.stream`` and ``client.asr.stream_sync``.

The stream runs against a local websockets server (in its own thread) speaking
the public speech-to-text stream protocol
(docs/api-reference/endpoints/speech-to-text.mdx, "Stream live audio").
"""

from __future__ import annotations

import asyncio
import base64
import gc
import io
import json
import threading
import time
import wave
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from kugelaudio import (
    AuthenticationError,
    InsufficientCreditsError,
    KugelAudio,
    KugelAudioConnectionError,
    KugelAudioError,
    RateLimitError,
    ServerRestartingError,
    StreamingTranscriptionEvent,
    ValidationError,
    load_audio_stream,
)
from kugelaudio.client import ASRResource

PARTIAL = {
    "type": "partial",
    "partial_text": "Ich möchte",
    "is_final": False,
    "audio_end_s": 4.0,
}
FINAL = {
    "type": "partial",
    "partial_text": "Ich möchte einen Flug",
    "is_final": True,
    "model": "test-model",
    "model_revision": "rev",
    "turn_end_reason": "client_end_of_speech",
    "word_alternatives": [],
}
REVISION = {
    "type": "alternatives",
    "partial_text": "Ich möchte einen Flug nach Berlin buchen.",
    "revises_final": True,
    "language": "de",
    "model": "test-model",
    "model_revision": "rev",
    "word_alternatives": [],
}


def _wav(n_samples: int, rate: int, fill: int = 3) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(fill.to_bytes(2, "little", signed=True) * n_samples)
    return buffer.getvalue()


def _error(status: int, code: str, message: str) -> dict:
    """The ingress error frame: ``{error, error_code, code, request_id}``, no ``type``."""
    return {
        "error": message,
        "error_code": code,
        "code": status,
        "request_id": "req-ws",
    }


# What the server does after the first audio_chunk: frames to send, then a
# close code (None keeps reading).
_Script = tuple[list[dict], int | None]


class _Server:
    """Local server speaking the speech-to-text stream protocol.

    Sends nothing until it has audio. ``on_audio`` runs after the first
    ``audio_chunk``; on ``end_of_speech`` it sends ``after_end`` and closes
    with ``close_code``. ``handshake`` refuses the upgrade with an HTTP status.
    """

    def __init__(self) -> None:
        self.handshake: tuple[int, dict[str, str]] | None = None
        self.on_audio: _Script | None = None
        self.after_end: list[dict] = [FINAL]
        self.close_code = 1000
        self.path: str | None = None
        self.config: dict | None = None
        self.messages: list[dict] = []
        self.disconnected = threading.Event()

    @property
    def audio(self) -> list[bytes]:
        return [
            base64.b64decode(m["audio_b64"])
            for m in self.messages
            if m["type"] == "audio_chunk"
        ]

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

    def _serve(self, ws: Any) -> None:
        self.path = ws.request.path
        self.config = json.loads(ws.recv())
        script = self.on_audio
        for raw in ws:
            assert isinstance(raw, str), "audio must go as JSON audio_chunk frames"
            message = json.loads(raw)
            self.messages.append(message)
            if message["type"] == "audio_chunk" and script is not None:
                frames, close_code = script
                script = None
                for frame in frames:
                    ws.send(json.dumps(frame))
                if close_code is not None:
                    ws.close(close_code)
                    return
            elif message["type"] == "end_of_speech":
                for frame in self.after_end:
                    ws.send(json.dumps(frame))
                ws.close(self.close_code)
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
    return [t for t in threading.enumerate() if t.name == "kugelaudio-asr-send"]


def _assert_cleaned_up(server: _Server) -> None:
    server.wait_disconnected()
    for _ in range(100):
        if not _sender_threads():
            break
        time.sleep(0.01)
    assert _sender_threads() == []


def _collect(client: KugelAudio, mode: str, audio: Any, **kwargs: Any) -> list:
    """Every event of one stream, through ``stream_sync`` or ``stream``."""
    if mode == "sync":
        return list(client.asr.stream_sync(audio, **kwargs))

    async def run() -> list:
        return [event async for event in client.asr.stream(audio, **kwargs)]

    return asyncio.run(run())


def _endless() -> Iterator[bytes]:
    while True:
        time.sleep(0.005)
        yield b"\x01\x02"


async def _endless_async() -> AsyncIterator[bytes]:
    while True:
        await asyncio.sleep(0.005)
        yield b"\x01\x02"


MODES = pytest.mark.parametrize("mode", ["sync", "async"])


def test_asr_resource_moved_but_still_reachable_from_client() -> None:
    from kugelaudio.asr import ASRResource as Moved

    assert ASRResource is Moved
    assert isinstance(KugelAudio(api_key="sk-test").asr, Moved)


# ---------------------------------------------------------------- config


@MODES
def test_config_carries_only_sample_rate_when_nothing_is_set(ws_env, mode):
    server, client = ws_env
    _collect(client, mode, [b"\x00\x00"], sample_rate=16000)
    assert server.config == {"type": "config", "sample_rate": 16000}
    assert server.path is not None
    assert server.path.startswith(
        "/v1/audio/transcriptions/stream?api_key=sk-test&sdk="
    )
    _assert_cleaned_up(server)


@MODES
def test_config_carries_every_set_option(ws_env, mode):
    server, client = ws_env
    _collect(
        client,
        mode,
        [b"\x00\x00"],
        sample_rate=8000,
        language=["de", "fr"],
        model="test-model",
        turn_detection="kugelturn",
        turn_detection_level="conservative",
        boosted_phrases=["Kratzenstein"],
        project_id=42,
        dictionary_ids=[],
    )
    assert server.config == {
        "type": "config",
        "sample_rate": 8000,
        "language": ["de", "fr"],
        "model": "test-model",
        "turn_detection": "kugelturn",
        "turn_detection_level": "conservative",
        "boosted_phrases": ["Kratzenstein"],
        "project_id": 42,
        "dictionary_ids": [],
    }


def test_config_passes_a_single_language_code_through(ws_env):
    server, client = ws_env
    _collect(
        client,
        "sync",
        [b"\x00\x00"],
        sample_rate=16000,
        language="de",
        dictionary_ids=[3],
    )
    assert server.config is not None
    assert server.config["language"] == "de"
    assert server.config["dictionary_ids"] == [3]


# ----------------------------------------------------------------- audio


@MODES
def test_chunks_split_at_one_second_base64_then_end_of_speech(ws_env, mode):
    server, client = ws_env
    long_chunk = bytes(range(10)) * 2000  # 20000 bytes, 1.25 s at 8 kHz
    _collect(client, mode, [b"\x01\x02\x03\x04", long_chunk], sample_rate=8000)
    assert server.audio == [b"\x01\x02\x03\x04", long_chunk[:16000], long_chunk[16000:]]
    assert [m["type"] for m in server.messages] == ["audio_chunk"] * 3 + [
        "end_of_speech"
    ]
    assert server.messages[-1] == {"type": "end_of_speech"}


@MODES
def test_audio_stream_input_takes_its_sample_rate(ws_env, tmp_path, mode):
    server, client = ws_env
    path = tmp_path / "call.wav"
    path.write_bytes(_wav(20000, rate=8000))
    _collect(client, mode, load_audio_stream(path))
    assert server.config == {"type": "config", "sample_rate": 8000}
    assert [len(chunk) for chunk in server.audio] == [1600] * 25  # 100 ms chunks


async def test_async_iterable_input(ws_env):
    server, client = ws_env

    async def chunks() -> AsyncIterator[bytes]:
        for i in range(3):
            await asyncio.sleep(0)
            yield bytes([i, i])

    events = [event async for event in client.asr.stream(chunks(), sample_rate=16000)]
    assert server.audio == [b"\x00\x00", b"\x01\x01", b"\x02\x02"]
    assert [event.partial_text for event in events] == [FINAL["partial_text"]]


# ---------------------------------------------------------------- events


@MODES
def test_events_parsed_and_stream_ends_on_close_1000(ws_env, mode, caplog):
    server, client = ws_env
    server.after_end = [PARTIAL, {"type": "speech_started"}, FINAL, REVISION]
    with caplog.at_level("WARNING", logger="kugelaudio.asr"):
        events = _collect(client, mode, [b"\x00\x00"], sample_rate=16000)
    assert all(isinstance(event, StreamingTranscriptionEvent) for event in events)
    partial, final, revision = events
    assert (partial.type, partial.is_final, partial.revises_final) == (
        "partial",
        False,
        False,
    )
    assert final.is_final and final.model == "test-model"
    assert final.turn_end_reason == "client_end_of_speech" and not final.revises_final
    assert final.language is None
    assert revision.type == "alternatives" and revision.revises_final
    assert revision.partial_text == REVISION["partial_text"]
    assert revision.language == "de"
    assert "speech_started" in caplog.text  # skipped, not silently
    _assert_cleaned_up(server)


def test_events_arrive_while_input_is_still_being_sent(ws_env):
    server, client = ws_env
    server.on_audio = ([PARTIAL], None)
    for event in client.asr.stream_sync(_endless(), sample_rate=16000):
        assert event.partial_text == PARTIAL["partial_text"]
        break
    _assert_cleaned_up(server)


# ---------------------------------------------------------------- errors


_FRAME_ERRORS = {
    "422-validation": (
        ([_error(422, "VALIDATION_ERROR", "Unsupported language 'ja'")], None),
        ValidationError,
    ),
    "402-credits-then-4003": (
        ([_error(402, "INSUFFICIENT_CREDITS", "Balance spent")], 4003),
        InsufficientCreditsError,
    ),
    "429-rate-limited": (
        ([_error(429, "RATE_LIMITED", "Slow down")], None),
        RateLimitError,
    ),
}


@pytest.mark.parametrize("case", list(_FRAME_ERRORS))
@MODES
def test_error_frame_raises_typed_error(ws_env, case, mode):
    server, client = ws_env
    server.on_audio, expected = _FRAME_ERRORS[case]
    source = _endless() if mode == "sync" else _endless_async()
    with pytest.raises(KugelAudioError) as err:
        _collect(client, mode, source, sample_rate=16000)
    assert type(err.value) is expected
    assert err.value.request_id == "req-ws"
    _assert_cleaned_up(server)


_CLOSES = {
    "1011": (1011, KugelAudioConnectionError),
    "1003": (1003, KugelAudioConnectionError),
    "1012": (1012, ServerRestartingError),
    "4003": (4003, InsufficientCreditsError),
}


@pytest.mark.parametrize("case", list(_CLOSES))
@MODES
def test_abnormal_close_raises_typed_error(ws_env, case, mode):
    server, client = ws_env
    code, expected = _CLOSES[case]
    server.on_audio = ([PARTIAL], code)
    source = _endless() if mode == "sync" else _endless_async()
    with pytest.raises(KugelAudioError) as err:
        _collect(client, mode, source, sample_rate=16000)
    assert type(err.value) is expected
    _assert_cleaned_up(server)


@MODES
def test_close_after_end_of_speech_other_than_1000_raises(ws_env, mode):
    server, client = ws_env
    server.close_code = 1011
    with pytest.raises(KugelAudioConnectionError):
        _collect(client, mode, [b"\x00\x00"], sample_rate=16000)


@pytest.mark.parametrize(
    ("handshake", "expected", "retry_after"),
    [
        ((401, {}), AuthenticationError, None),
        ((429, {"Retry-After": "7"}), RateLimitError, 7),
        ((402, {}), InsufficientCreditsError, None),
    ],
    ids=["401", "429", "402"],
)
@MODES
def test_refused_handshake_raises_typed_error(
    ws_env, handshake, expected, retry_after, mode
):
    server, client = ws_env
    server.handshake = handshake
    with pytest.raises(KugelAudioError) as err:
        _collect(client, mode, [b"\x00\x00"], sample_rate=16000)
    assert type(err.value) is expected
    assert err.value.retry_after == retry_after
    assert "sk-test" not in str(err.value)


@pytest.mark.parametrize(
    ("audio", "kwargs", "match"),
    [
        ([b"\x00\x00"], {}, "sample_rate is required"),
        (
            load_audio_stream(_wav(1600, rate=16000)),
            {"sample_rate": 8000},
            "does not match",
        ),
        ("call.wav", {"sample_rate": 16000}, "load_audio_stream"),
        (b"RIFF", {"sample_rate": 16000}, "load_audio_stream"),
        (42, {"sample_rate": 16000}, "iterable"),
    ],
    ids=["no-rate", "rate-mismatch", "path", "bytes", "not-iterable"],
)
@pytest.mark.parametrize("method", ["stream_sync", "stream"])
def test_bad_input_rejected_before_connecting(audio, kwargs, match, method):
    # Nothing listens on port 9: a connection attempt would fail differently.
    client = KugelAudio(api_key="sk-test", api_url="http://127.0.0.1:9")
    with pytest.raises(ValidationError, match=match):
        getattr(client.asr, method)(audio, **kwargs)


def test_stream_sync_rejects_an_async_iterable():
    async def chunks() -> AsyncIterator[bytes]:
        yield b"\x00\x00"

    client = KugelAudio(api_key="sk-test", api_url="http://127.0.0.1:9")
    source = chunks()
    with pytest.raises(ValidationError, match="iterable"):
        client.asr.stream_sync(source, sample_rate=16000)
    asyncio.run(source.aclose())


@MODES
def test_non_bytes_chunk_raises(ws_env, mode):
    server, client = ws_env
    with pytest.raises(ValidationError, match="must be bytes"):
        _collect(client, mode, ["not bytes"], sample_rate=16000)
    _assert_cleaned_up(server)


@MODES
def test_send_side_failure_surfaces(ws_env, mode):
    server, client = ws_env

    def failing() -> Iterator[bytes]:
        yield b"\x00\x01"
        time.sleep(0.01)
        raise OSError("microphone unplugged")

    async def failing_async() -> AsyncIterator[bytes]:
        yield b"\x00\x01"
        await asyncio.sleep(0.01)
        raise OSError("microphone unplugged")

    source = failing() if mode == "sync" else failing_async()
    with pytest.raises(OSError, match="microphone unplugged"):
        _collect(client, mode, source, sample_rate=16000)
    _assert_cleaned_up(server)


def test_invalid_api_url_does_not_leak_the_key():
    client = KugelAudio(
        api_key="sk-secret", api_url="ftp://example.test"
    )  # not ws:// or wss://
    with pytest.raises(ValidationError) as err:
        list(client.asr.stream_sync([b"\x00\x00"], sample_rate=16000))
    assert "sk-secret" not in str(err.value)


# --------------------------------------------------------------- cleanup


def test_stream_sync_early_break_leaves_no_thread(ws_env):
    server, client = ws_env
    server.on_audio = ([PARTIAL], None)
    for _event in client.asr.stream_sync(_endless(), sample_rate=16000):
        break
    _assert_cleaned_up(server)


def test_stream_sync_garbage_collected_leaves_no_thread(ws_env):
    server, client = ws_env
    server.on_audio = ([PARTIAL], None)
    events = client.asr.stream_sync(_endless(), sample_rate=16000)
    assert next(events).partial_text == PARTIAL["partial_text"]
    del events
    gc.collect()
    _assert_cleaned_up(server)


async def test_stream_early_break_cancels_sender_and_closes(ws_env):
    server, client = ws_env
    server.on_audio = ([PARTIAL], None)
    loop = asyncio.get_running_loop()
    errors: list[dict] = []
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    baseline = asyncio.all_tasks()
    async for event in client.asr.stream(_endless_async(), sample_rate=16000):
        assert event.partial_text == PARTIAL["partial_text"]
        break
    # Cleanup after an early exit runs via the async-generator finalizer.
    leaked: set = set()
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


# ----------------------------------------------------------------- model


def test_event_revises_final_defaults_false_and_is_parsed():
    assert StreamingTranscriptionEvent.from_dict(PARTIAL).revises_final is False
    revision = StreamingTranscriptionEvent.from_dict(REVISION)
    assert revision.revises_final is True
    assert revision.language == "de"
    # The positional constructor keeps its order; the new field comes last.
    assert StreamingTranscriptionEvent("t", True).revises_final is False
