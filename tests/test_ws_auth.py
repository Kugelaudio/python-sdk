"""Every socket upgrade carries the key in an Authorization header, never in the URL."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import websockets

from kugelaudio import AuthenticationError, KugelAudio
from kugelaudio._ws_auth import bearer_headers
from kugelaudio.enhance import _connect_async
from kugelaudio.streaming import MultiContextSession, StreamingSession

KEY = "sk-secret-key"
HEADER = {"Authorization": f"Bearer {KEY}"}


def _header_kwarg(kwargs: dict[str, Any]) -> dict[str, str]:
    name = "additional_headers" if "additional_headers" in kwargs else "extra_headers"
    return kwargs[name]


def _assert_header_only(url: str, kwargs: dict[str, Any]) -> None:
    assert KEY not in url
    assert "api_key" not in url
    assert _header_kwarg(kwargs) == HEADER



def test_rest_request_sends_key_in_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    def handle_request(
        _transport: httpx.HTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        assert KEY not in str(request.url)
        assert "api_key" not in request.url.params
        assert request.headers["Authorization"] == HEADER["Authorization"]
        return httpx.Response(200, json={"voices": [], "total": 0, "limit": 50, "offset": 0})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
    with KugelAudio(api_key=KEY, api_url="https://ingress.example") as client:
        assert not client.voices.list().voices


def test_bearer_headers() -> None:
    assert bearer_headers(KEY) == HEADER


@pytest.mark.parametrize("module", ["asr", "enhance"])
def test_sync_stream_paths_pass_additional_headers(
    monkeypatch: pytest.MonkeyPatch, module: str
) -> None:
    import websockets.sync.client as sync_client

    seen: dict[str, Any] = {}

    def fake_connect(url: str, **kwargs: Any) -> Any:
        seen.update(kwargs)
        raise OSError("stop after the connect call")

    monkeypatch.setattr(sync_client, "connect", fake_connect)
    if module == "asr":
        from kugelaudio.asr import _connect_sync

        call = lambda: _connect_sync("ws://h/x", KEY, 1.0)  # noqa: E731
    else:
        from kugelaudio.enhance import _run_stream_sync

        call = lambda: list(_run_stream_sync("ws://h/x", KEY, {}, [], 1.0))  # noqa: E731
    with pytest.raises(OSError, match="stop after the connect call"):
        call()
    assert seen["additional_headers"] == HEADER
    assert "extra_headers" not in seen


@pytest.mark.asyncio
@pytest.mark.parametrize("session_cls", [StreamingSession, MultiContextSession])
async def test_streaming_sessions_send_header(session_cls: type) -> None:
    session = session_cls(api_key=KEY, tts_url="https://tts.example")
    connect = AsyncMock(return_value=MagicMock())
    with patch.object(websockets, "connect", connect):
        await session._connect_socket()
    url = connect.call_args.args[0]
    _assert_header_only(url, connect.call_args.kwargs)


@pytest.mark.asyncio
async def test_pooled_tts_socket_sends_header() -> None:
    client = KugelAudio(api_key=KEY, keepalive_ping_interval=None)
    connect = AsyncMock(return_value=MagicMock())
    with patch.object(websockets, "connect", connect):
        await client.tts._get_ws_connection("kugel-3.5")
    _assert_header_only(connect.call_args.args[0], connect.call_args.kwargs)


@pytest.mark.asyncio
async def test_livekit_connection_sends_header() -> None:
    pytest.importorskip("livekit.agents")
    from kugelaudio.livekit.tts import _Connection, _TTSOptions

    opts = _TTSOptions(
        model="kugel-1-turbo",
        voice_id=None,
        sample_rate=24000,
        cfg_scale=2.0,
        max_new_tokens=2048,
        api_key=KEY,
        base_url="https://x",
    )
    session = MagicMock()
    session.ws_connect = AsyncMock(return_value=MagicMock())
    conn = _Connection(opts, session)
    with patch("asyncio.create_task"):
        await conn.connect()
    url = session.ws_connect.call_args.args[0]
    assert KEY not in url
    assert session.ws_connect.call_args.kwargs["headers"] == HEADER


@pytest.mark.asyncio
@pytest.mark.parametrize("cross_origin", [False, True])
async def test_async_redirect_scopes_bearer_header(cross_origin: bool) -> None:
    received: list[bytes | None] = []
    destination: str

    async def handle_request(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            lines = (await reader.readuntil(b"\r\n\r\n")).split(b"\r\n")
            authorization = next(
                (
                    line.partition(b":")[2].strip()
                    for line in lines
                    if line.lower().startswith(b"authorization:")
                ),
                None,
            )
            received.append(authorization)
            if lines[0].split()[1] == b"/start":
                writer.write(
                    f"HTTP/1.1 302 Found\r\nLocation: {destination}\r\n"
                    f"Content-Length: 0\r\nConnection: close\r\n\r\n".encode()
                )
            else:
                writer.write(
                    b"HTTP/1.1 401 Unauthorized\r\n"
                    b"Content-Length: 0\r\nConnection: close\r\n\r\n"
                )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async with (
        await asyncio.start_server(handle_request, "127.0.0.1", 0) as origin,
        await asyncio.start_server(handle_request, "127.0.0.1", 0) as target,
    ):
        origin_url = f"ws://127.0.0.1:{origin.sockets[0].getsockname()[1]}"
        target_url = f"ws://127.0.0.1:{target.sockets[0].getsockname()[1]}"
        destination = f"{target_url}/end" if cross_origin else "/end"
        with pytest.raises(AuthenticationError):
            await _connect_async(f"{origin_url}/start", KEY)

    bearer = HEADER["Authorization"].encode()
    assert received == [bearer, None if cross_origin else bearer]
