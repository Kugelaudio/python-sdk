"""Tests for ``client.enhance.session()`` and the real-time enhancer on it.

A local websockets server (in its own thread) speaks the enhancement stream
protocol, in session mode when the URL carries ``session=1``, and counts the
connections it accepts, so socket reuse is measured on the server side.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from kugelaudio import KugelAudio, KugelAudioError, RateLimitError
from kugelaudio._enhance_live import LiveEnhancer, LiveEnhancerState

MODEL = "clarity-1"
RATE = 16000
CHUNK = b"\x01\x00\x02\x00"


class SessionServer:
    """Echoes each audio message reversed (or, with ``reply_sample``, answers
    it with that sample value for the same duration at 24 kHz, as Clarity
    does); answers ``end`` with ``done``.

    In session mode (``session=1`` and ``supports_sessions``) ``ready`` and
    ``done`` carry a per-audio ``request_id`` and the socket stays open for the
    next config; otherwise it closes with 1000 after ``done``, as a server
    without session mode does.
    """

    def __init__(self) -> None:
        self.supports_sessions = True
        # A handshake with session=1 is refused with this status and headers.
        self.refuse_sessions: tuple[int, dict[str, str]] | None = None
        # Close with this code after each done (a draining server: 1012).
        self.close_after_audio: int | None = None
        # An old server's idle rule: no config for this long -> 408 frame, 4000.
        self.idle_frame_after_s: float | None = None
        # The session rule: no config for this long -> close 1000 (60 s live).
        self.session_idle_close_s: float | None = None
        self.reply_sample: int | None = None
        # Configs (1-based) refused with a rate-limit frame; the socket stays open.
        self.rate_limited_configs: set[int] = set()
        # Seconds every handshake waits before it is answered.
        self.handshake_delay_s = 0.0
        self.open = 0
        self.handshakes = 0
        self.connections = 0
        self.paths: list[str] = []
        self.configs: list[dict] = []
        self.lock = threading.Lock()

    def process_request(self, connection: Any, request: Any) -> Any:
        with self.lock:
            self.handshakes += 1
        if self.handshake_delay_s:
            time.sleep(self.handshake_delay_s)
        if self.refuse_sessions is not None and "session=1" in request.path:
            status, headers = self.refuse_sessions
            body = {"error": "Open session limit reached (2)", "error_code": "RATE_LIMITED", "code": status}
            response = connection.respond(status, json.dumps(body))
            for name, value in headers.items():
                response.headers[name] = value
            return response
        return None

    def handler(self, ws: Any) -> None:
        with self.lock:
            self.connections += 1
            self.open += 1
            self.paths.append(ws.request.path)
        try:
            self._serve(ws, "session=1" in ws.request.path and self.supports_sessions)
        except ConnectionClosed:
            pass
        finally:
            with self.lock:
                self.open -= 1

    def _serve(self, ws: Any, session: bool) -> None:
        while True:
            try:
                message = ws.recv(
                    timeout=self.session_idle_close_s if session else self.idle_frame_after_s
                )
            except TimeoutError:
                if session:
                    ws.close(1000)
                    return
                frame = {"type": "error", "error": "no audio received for 30 seconds",
                         "error_code": "VALIDATION_ERROR", "code": 408, "request_id": "idle"}
                ws.send(json.dumps(frame))
                ws.close(4000)
                return
            data = json.loads(message)
            if data["type"] == "close":
                ws.close(1000)
                return
            with self.lock:
                self.configs.append(data)
                number = len(self.configs)
            audio_id = f"audio-{number}"
            if number in self.rate_limited_configs:
                frame = {"type": "error", "error": "Rate limit exceeded (1 requests per minute)",
                         "error_code": "RATE_LIMITED", "code": 429, "request_id": audio_id,
                         "retry_after": 3}
                ws.send(json.dumps(frame))
                continue
            ready: dict[str, object] = {"type": "ready", "sample_rate_hz": 24000, "encoding": "pcm_s16le"}
            if session:
                ready["request_id"] = audio_id
            ws.send(json.dumps(ready))
            rate = data["sample_rate_hz"]
            for message in ws:
                if isinstance(message, bytes):
                    if self.reply_sample is None:
                        ws.send(message[::-1])
                    else:
                        samples = len(message) // 2 * 24000 // rate
                        ws.send(self.reply_sample.to_bytes(2, "little", signed=True) * samples)
                    continue
                if json.loads(message).get("type") == "end":
                    done: dict[str, object] = {"type": "done", "duration_s": 0.1}
                    if session:
                        done["request_id"] = audio_id
                    ws.send(json.dumps(done))
                    break
            else:
                return
            if not session:
                ws.close(1000)
                return
            if self.close_after_audio is not None:
                ws.close(self.close_after_audio)
                return

    def rates(self) -> list[int]:
        return [c["sample_rate_hz"] for c in self.configs]


@pytest.fixture
def env() -> Iterator[tuple[SessionServer, KugelAudio]]:
    server = SessionServer()
    ws_server = serve(server.handler, "127.0.0.1", 0, compression=None,
                      process_request=server.process_request)
    thread = threading.Thread(target=ws_server.serve_forever, daemon=True)
    thread.start()
    port = ws_server.socket.getsockname()[1]
    try:
        yield server, KugelAudio(api_key="sk-test", api_url=f"http://127.0.0.1:{port}")
    finally:
        ws_server.shutdown()
        thread.join(5)


async def enhance(session: Any, chunks: list[bytes], rate: int = RATE) -> list[bytes]:
    return [c async for c in session.stream(chunks, model=MODEL, sample_rate=rate)]


async def wait_for(condition: Any, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "condition not met in time"
        await asyncio.sleep(0.01)


# ------------------------------------------------------------------ session


async def test_a_session_carries_many_streams_on_one_socket(env) -> None:
    server, client = env
    async with client.enhance.session() as session:
        outputs = [await enhance(session, [CHUNK, bytes([i, 0])]) for i in range(4)]
    assert outputs == [[CHUNK[::-1], bytes([0, i])] for i in range(4)]
    assert server.connections == 1
    assert len(server.configs) == 4
    assert "session=1" in server.paths[0]


async def test_connect_opens_the_socket_before_the_first_audio(env) -> None:
    server, client = env
    session = client.enhance.session()
    await session.connect()
    await wait_for(lambda: server.connections == 1)
    assert server.configs == []
    await session.connect()  # reuses the open socket
    assert await enhance(session, [CHUNK]) == [CHUNK[::-1]]
    await session.aclose()
    assert server.handshakes == server.connections == 1


async def test_each_stream_may_change_the_sample_rate(env) -> None:
    server, client = env
    async with client.enhance.session() as session:
        await enhance(session, [CHUNK], rate=16000)
        await enhance(session, [CHUNK], rate=48000)
    assert server.rates() == [16000, 48000]
    assert server.connections == 1


async def test_a_socket_the_server_closed_is_replaced_without_an_error(env) -> None:
    server, client = env
    server.close_after_audio = 1012  # draining: finish the audio, then close
    async with client.enhance.session() as session:
        for _ in range(3):
            assert await enhance(session, [CHUNK]) == [CHUNK[::-1]]
    assert server.connections == 3


async def test_an_old_servers_idle_close_is_replaced_without_an_error(env) -> None:
    server, client = env
    server.supports_sessions = False
    server.idle_frame_after_s = 0.05
    async with client.enhance.session() as session:
        await wait_for(lambda: server.connections == 1)
        await asyncio.sleep(0.2)  # the server sent its 408 frame and closed
        assert await enhance(session, [CHUNK]) == [CHUNK[::-1]]
    assert server.connections == 2


async def test_a_server_without_sessions_gets_one_connection_per_stream(env) -> None:
    server, client = env
    server.supports_sessions = False
    async with client.enhance.session() as session:
        for _ in range(3):
            assert await enhance(session, [CHUNK]) == [CHUNK[::-1]]
    assert server.connections == 3
    # Once the server showed it has no sessions, streams connect without asking.
    assert ["session=1" in path for path in server.paths] == [True, False, False]


async def test_a_refused_session_streams_on_a_connection_of_its_own(env) -> None:
    server, client = env
    server.refuse_sessions = (429, {})  # the open-session limit: no Retry-After
    async with client.enhance.session() as session:
        assert await enhance(session, [CHUNK]) == [CHUNK[::-1]]
    assert server.connections == 1
    assert "session=1" not in server.paths[0]


async def test_a_session_refused_with_retry_after_raises(env) -> None:
    server, client = env
    server.refuse_sessions = (429, {"Retry-After": "7"})
    with pytest.raises(RateLimitError) as err:
        async with client.enhance.session():
            pass  # pragma: no cover
    assert err.value.retry_after == 7


async def test_a_rate_limited_config_raises_and_keeps_the_socket(env) -> None:
    server, client = env
    server.rate_limited_configs = {2}
    async with client.enhance.session() as session:
        await enhance(session, [CHUNK])
        with pytest.raises(RateLimitError) as err:
            await enhance(session, [CHUNK])
        assert err.value.retry_after == 3
        assert await enhance(session, [CHUNK]) == [CHUNK[::-1]]
    assert server.connections == 1


async def test_breaking_out_of_a_stream_closes_the_socket(env) -> None:
    server, client = env

    async def endless():
        while True:
            await asyncio.sleep(0.005)
            yield CHUNK

    async with client.enhance.session() as session:
        async for _ in session.stream(endless(), model=MODEL, sample_rate=RATE):
            break
        assert await enhance(session, [CHUNK]) == [CHUNK[::-1]]
    assert server.connections == 2


async def test_a_closed_session_refuses_to_stream(env) -> None:
    _, client = env
    session = client.enhance.session()
    await session.aclose()
    with pytest.raises(Exception, match="session is closed"):
        await enhance(session, [CHUNK])


async def test_aclose_during_a_connect_leaves_no_socket_open(env) -> None:
    server, client = env
    server.handshake_delay_s = 0.3
    session = client.enhance.session()
    streaming = asyncio.ensure_future(enhance(session, [CHUNK]))
    await asyncio.sleep(0.1)  # the handshake is in flight
    await session.aclose()
    with pytest.raises(KugelAudioError, match="session is closed"):
        await streaming
    await asyncio.sleep(0.5)  # past the handshake delay
    assert server.open == 0 and server.configs == []


async def test_aclose_during_connect_ends_connect_with_the_closed_error(env) -> None:
    server, client = env
    server.handshake_delay_s = 0.3
    session = client.enhance.session()
    connecting = asyncio.ensure_future(session.connect())
    await asyncio.sleep(0.1)
    await session.aclose()
    with pytest.raises(KugelAudioError, match="session is closed"):
        await connecting
    await asyncio.sleep(0.5)
    assert server.open == 0


# ------------------------------------------------- real-time enhancer on it


def frames(rate: int, count: int) -> list[bytes]:
    return [b"\x01\x00" * (rate // 50)] * count


async def push_all(enhancer: LiveEnhancer, rate: int, count: int) -> None:
    for frame in frames(rate, count):
        enhancer.push(frame, rate)
        await asyncio.sleep(0.005)


async def test_enhancer_prewarm_opens_the_socket_and_streams_reuse_it(env) -> None:
    server, client = env
    enhancer = LiveEnhancer(client, idle_close_s=0.1)
    enhancer.prewarm()
    await wait_for(lambda: server.connections == 1)
    assert server.configs == []

    await push_all(enhancer, 16000, 10)
    await wait_for(lambda: enhancer.state is LiveEnhancerState.LIVE)
    # A new sample rate: a new config on the same socket.
    await push_all(enhancer, 48000, 10)
    await wait_for(lambda: server.rates() == [16000, 48000])
    # Muted: the audio ends after idle_close_s, the socket stays warm.
    await asyncio.sleep(0.3)
    await push_all(enhancer, 48000, 5)
    await wait_for(lambda: len(server.configs) == 3)
    await enhancer.aclose()
    assert server.connections == 1
    assert enhancer.failures == 0


async def test_enhancer_reconnects_when_the_server_closes_between_audios(env) -> None:
    server, client = env
    server.close_after_audio = 1012
    enhancer = LiveEnhancer(client, idle_close_s=0.1)
    await push_all(enhancer, 16000, 10)
    await wait_for(lambda: enhancer.state is LiveEnhancerState.LIVE)
    await asyncio.sleep(0.3)  # idle: the audio ends and the server closes 1012
    await push_all(enhancer, 16000, 10)
    await wait_for(lambda: len(server.configs) == 2 and enhancer.state is LiveEnhancerState.LIVE)
    await enhancer.aclose()
    assert server.connections == 2
    assert enhancer.failures == 0


async def test_enhancer_aclose_while_its_prewarm_connects_leaves_no_socket_open(env) -> None:
    server, client = env
    server.handshake_delay_s = 0.3
    enhancer = LiveEnhancer(client)
    enhancer.prewarm()
    await push_all(enhancer, 16000, 5)  # the stream waits for the same connect
    enhancer.pause()  # Pipecat's FilterEnableFrame(False), then stop()
    await enhancer.aclose()
    await asyncio.sleep(0.5)
    assert server.open == 0 and server.configs == []


async def test_a_bad_api_url_never_puts_the_key_in_the_error() -> None:
    client = KugelAudio(api_key="sk-secret-key", api_url="ftp://example.com")
    session = client.enhance.session()
    with pytest.raises(KugelAudioError) as err:
        await enhance(session, [CHUNK])
    assert "sk-secret-key" not in str(err.value)
    assert err.value.__cause__ is None and err.value.__suppress_context__
