"""Pooled /ws/tts socket: one request at a time.

Server frames carry no request id, so audio left on the pooled socket by one
``stream_async`` call is read as the next call's audio, and two calls reading
the same socket at once collide. Each request must own the socket until its
final frame.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from collections.abc import AsyncIterator
from contextlib import aclosing

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import ServerConnection, serve

from kugelaudio import KugelAudio
from kugelaudio.models import AudioChunk

FRAMES = 3


def _audio_of(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def _handler(ws: ServerConnection) -> None:
    """Answers each request in order: FRAMES audio frames, then final."""
    try:
        for msg in ws:
            req = json.loads(msg)
            if "text" not in req:
                continue
            for idx in range(FRAMES):
                ws.send(json.dumps({"audio": _audio_of(req["text"]), "enc": "pcm_s16le",
                                    "idx": idx, "sr": 24000, "samples": 1}))
            ws.send(json.dumps({"final": True, "chunks": FRAMES, "total_samples": FRAMES}))
    except ConnectionClosed:
        return


@pytest.fixture
async def client() -> AsyncIterator[KugelAudio]:
    ws_server = serve(_handler, "127.0.0.1", 0, compression=None)
    thread = threading.Thread(target=ws_server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{ws_server.socket.getsockname()[1]}"
    kugel = KugelAudio(api_key="sk-test", api_url=url, keepalive_ping_interval=None)
    try:
        yield kugel
    finally:
        await kugel.aclose()
        ws_server.shutdown()
        thread.join(5)


async def _texts(client: KugelAudio, text: str, stop_after: int | None = None) -> list[str]:
    got: list[str] = []
    async with aclosing(client.tts.stream_async(text, language="en")) as items:
        async for item in items:
            if isinstance(item, AudioChunk):
                got.append(item.audio.decode())
                if stop_after is not None and len(got) >= stop_after:
                    break
    return got


async def test_a_stream_stopped_early_does_not_leak_into_the_next(client: KugelAudio) -> None:
    assert await _texts(client, "first", stop_after=1) == ["first"]
    assert await _texts(client, "second") == ["second"] * FRAMES


async def test_two_streams_at_once_each_get_their_own_audio(client: KugelAudio) -> None:
    await client.tts.connect_async()
    a, b = await asyncio.gather(_texts(client, "first"), _texts(client, "second"))
    assert a == ["first"] * FRAMES
    assert b == ["second"] * FRAMES


async def test_a_finished_stream_keeps_the_pooled_socket(client: KugelAudio) -> None:
    assert await _texts(client, "first") == ["first"] * FRAMES
    pooled = client.tts._ws_connection
    assert await _texts(client, "second") == ["second"] * FRAMES
    assert client.tts._ws_connection is pooled
