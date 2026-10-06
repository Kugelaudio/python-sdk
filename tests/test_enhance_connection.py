"""Connection reuse and prewarming (the warm-up request) for ``client.enhance``.

Runs against a real local HTTP/1.1 server that counts accepted TCP
connections, so reuse is measured on the socket, not inferred from a mock.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from kugelaudio import KugelAudio, load_audio

from .test_enhance import INPUT_WAV, MODEL, OUTPUT_WAV

WARMUP = "POST /v1/audio/enhance/warmup"
ENHANCE = "POST /v1/audio/enhance"


class _CountingServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.connections = 0
        self.requests: list[str] = []
        self.warmup_auth: list[str | None] = []
        self.warmup_status = 202

    def process_request(self, request: socket.socket, client_address: object) -> None:
        self.connections += 1
        super().process_request(request, client_address)  # type: ignore[arg-type]  # stdlib stub


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _CountingServer

    def do_POST(self) -> None:
        request = f"POST {self.path}"
        self.server.requests.append(request)
        self.rfile.read(int(self.headers.get("content-length", 0)))
        if request == WARMUP:
            # What the enhance service answers on its warm-up route.
            self.server.warmup_auth.append(self.headers.get("authorization"))
            ok = self.server.warmup_status == 202
            body = b'{"status":"warming"}' if ok else b'{"detail":"Invalid API key"}'
            self._reply(self.server.warmup_status, "application/json", body)
            return
        self._reply(200, "audio/wav", OUTPUT_WAV)

    def _reply(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def server() -> Iterator[_CountingServer]:
    srv = _CountingServer()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _client(server: _CountingServer) -> KugelAudio:
    host, port = server.server_address[:2]
    return KugelAudio(api_key="sk-test", api_url=f"http://{host}:{port}")


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def test_sequential_generate_calls_share_one_connection(server) -> None:
    client = _client(server)
    audio = load_audio(INPUT_WAV)
    for _ in range(3):
        await client.enhance.generate(audio, model=MODEL)
    await client.aclose()
    assert server.requests == [ENHANCE] * 3
    assert server.connections == 1


def test_sequential_generate_sync_calls_share_one_connection(server) -> None:
    with _client(server) as client:
        audio = load_audio(INPUT_WAV)
        for _ in range(3):
            client.enhance.generate_sync(audio, model=MODEL)
    assert server.requests == [ENHANCE] * 3
    assert server.connections == 1


async def test_prewarm_posts_the_warmup_and_generate_reuses_its_connection(server) -> None:
    client = _client(server)
    await client.enhance.prewarm()
    assert server.connections == 1
    await client.enhance.prewarm()  # idempotent: the same connection again
    await client.enhance.generate(load_audio(INPUT_WAV), model=MODEL)
    await client.aclose()
    assert server.requests == [WARMUP, WARMUP, ENHANCE]
    assert server.warmup_auth == ["Bearer sk-test", "Bearer sk-test"]
    assert server.connections == 1


def test_prewarm_sync_posts_the_warmup_and_generate_sync_reuses_its_connection(server) -> None:
    with _client(server) as client:
        client.enhance.prewarm_sync()
        client.enhance.generate_sync(load_audio(INPUT_WAV), model=MODEL)
    assert server.requests == [WARMUP, ENHANCE]
    assert server.warmup_auth == ["Bearer sk-test"]
    assert server.connections == 1


async def test_prewarm_logs_a_refused_warmup_and_returns(server, caplog) -> None:
    server.warmup_status = 401
    client = _client(server)
    with caplog.at_level(logging.WARNING, logger="kugelaudio.enhance"):
        await client.enhance.prewarm()
    await client.aclose()
    (record,) = [r for r in caplog.records if r.name == "kugelaudio.enhance"]
    assert record.levelno == logging.WARNING and "401" in record.getMessage()


def test_prewarm_sync_logs_a_refused_warmup_and_returns(server, caplog) -> None:
    server.warmup_status = 401
    with _client(server) as client, caplog.at_level(logging.WARNING, logger="kugelaudio.enhance"):
        client.enhance.prewarm_sync()
    (record,) = [r for r in caplog.records if r.name == "kugelaudio.enhance"]
    assert record.levelno == logging.WARNING and "401" in record.getMessage()


def test_each_event_loop_gets_its_own_connection(server) -> None:
    # A connection is bound to the loop that opened it: asyncio.run twice on
    # one client must reconnect, not reuse a socket from a closed loop.
    client = _client(server)
    audio = load_audio(INPUT_WAV)
    asyncio.run(client.enhance.generate(audio, model=MODEL))
    asyncio.run(client.enhance.generate(audio, model=MODEL))
    client.close()
    assert server.requests == [ENHANCE, ENHANCE]
    assert server.connections == 2


async def test_aclose_closes_the_shared_connection(server) -> None:
    client = _client(server)
    await client.enhance.prewarm()
    http = client._async_http.get()
    await client.aclose()
    assert http.is_closed


async def test_close_from_the_loop_closes_the_shared_connection(server) -> None:
    client = _client(server)
    await client.enhance.prewarm()
    http = client._async_http.get()
    client.close()
    await asyncio.sleep(0.05)
    assert http.is_closed


async def test_prewarm_logs_and_returns_on_a_network_error(caplog) -> None:
    client = KugelAudio(api_key="sk-test", api_url=f"http://127.0.0.1:{_closed_port()}")
    with caplog.at_level(logging.WARNING, logger="kugelaudio.enhance"):
        await client.enhance.prewarm()
    await client.aclose()
    assert "Could not prewarm" in caplog.text


def test_prewarm_sync_logs_and_returns_on_a_network_error(caplog) -> None:
    with KugelAudio(
        api_key="sk-test", api_url=f"http://127.0.0.1:{_closed_port()}"
    ) as client, caplog.at_level(logging.WARNING, logger="kugelaudio.enhance"):
        client.enhance.prewarm_sync()
    assert "Could not prewarm" in caplog.text


async def test_prewarm_swallows_a_timeout_through_the_transport_hook() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    client = KugelAudio(api_key="sk-test", api_url="https://api.example.test")
    client._async_transport = httpx.MockTransport(handler)
    await client.enhance.prewarm()
    await client.aclose()
