"""Pooled ``/ws/tts`` streams on one client never hear each other (ENG-3312).

The pooled socket's frames carry no request id; ingress answers pipelined
requests strictly in order. A stream that stops reading early (barge-in,
``break``) leaves the rest of its turn on the socket, and concurrent streams
read from one socket: either must never hand one request's audio to another.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from collections import deque
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from websockets.exceptions import ConnectionClosed

from kugelaudio import KugelAudio
from kugelaudio.models import AudioChunk


SLOW_PREFIX = "Slow"


class FifoTtsSocket:
    """Ingress ``/ws/tts``: one turn at a time, 3 audio frames then final.

    Each turn's audio bytes are the request text's first byte. Like
    ``websockets``, a second concurrent ``recv`` is refused and a closed
    socket refuses ``recv``. A text starting with ``SLOW_PREFIX`` paces its
    frames so another request can finish first.
    """

    def __init__(self) -> None:
        self.frames: deque[str] = deque()
        self.arrived = asyncio.Event()
        self.receiving = False
        self.open = True
        self.closed = False
        self.pace_s = 0.0
        self.loop: asyncio.AbstractEventLoop | None = None
        """The event loop that opened the socket."""
        self.closed_from_foreign_loop = False

    async def send(self, data: str) -> None:
        text = json.loads(data)["text"]
        if text.startswith(SLOW_PREFIX):
            self.pace_s = 0.05
        audio = base64.b64encode(bytes([ord(text[0])]) * 8).decode()
        for idx in range(3):
            self.frames.append(
                json.dumps(
                    {"audio": audio, "enc": "pcm_s16le", "idx": idx, "sr": 24000, "samples": 4}
                )
            )
        self.frames.append(json.dumps({"final": True, "chunks": 3, "total_samples": 12}))
        self.arrived.set()

    async def recv(self) -> str:
        if self.receiving:
            raise RuntimeError("cannot call recv while another coroutine is already running recv")
        self.receiving = True
        try:
            while not self.frames:
                self.arrived.clear()
                await self.arrived.wait()
            await asyncio.sleep(self.pace_s)
            if self.closed:
                raise ConnectionClosed(None, None)
            return self.frames.popleft()
        finally:
            self.receiving = False

    async def close(self) -> None:
        if asyncio.get_running_loop() is not self.loop:
            self.closed_from_foreign_loop = True
        self.open = False
        self.closed = True

    async def ping(self) -> None:
        return None


def _heard(chunks: list[AudioChunk]) -> set[int]:
    return {b for chunk in chunks for b in chunk.audio}


@pytest.fixture
def pooled() -> Iterator[tuple[KugelAudio, list[FifoTtsSocket], list[FifoTtsSocket]]]:
    """A client whose ``websockets.connect`` opens fakes, never the network:
    awaited (the pool) or entered (a request's own socket)."""
    client = KugelAudio(api_key="test_key", keepalive_ping_interval=None)
    pooled_sockets: list[FifoTtsSocket] = []
    own_sockets: list[FifoTtsSocket] = []

    class FakeConnect:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.socket = FifoTtsSocket()

        def __await__(self):  # type: ignore[no-untyped-def]  # awaitable protocol
            return self._pooled().__await__()

        async def _pooled(self) -> FifoTtsSocket:
            self.socket.loop = asyncio.get_running_loop()
            pooled_sockets.append(self.socket)
            return self.socket

        async def __aenter__(self) -> FifoTtsSocket:
            self.socket.loop = asyncio.get_running_loop()
            own_sockets.append(self.socket)
            return self.socket

        async def __aexit__(self, *_exc: object) -> None:
            await self.socket.close()

    with patch("websockets.connect", FakeConnect):
        yield client, pooled_sockets, own_sockets


async def _collect(client: KugelAudio, text: str, *, stop_after: int | None = None) -> list[AudioChunk]:
    chunks: list[AudioChunk] = []
    stream = client.tts.stream_async(text, language="en")
    try:
        async for item in stream:
            if isinstance(item, AudioChunk):
                chunks.append(item)
                await asyncio.sleep(0)
                if stop_after is not None and len(chunks) >= stop_after:
                    break
    finally:
        await stream.aclose()
    return chunks


@pytest.mark.asyncio
async def test_a_stream_abandoned_mid_turn_does_not_feed_the_next_request(pooled) -> None:
    client, sockets, fresh = pooled
    first = await _collect(client, "A", stop_after=1)
    second = await _collect(client, "B")
    assert _heard(first) == {ord("A")}
    assert _heard(second) == {ord("B")}
    assert len(second) == 3
    # The abandoned turn's socket was dropped; the next request pooled anew.
    assert len(sockets) == 2 and sockets[0].closed and not sockets[1].closed
    assert not fresh


@pytest.mark.asyncio
async def test_concurrent_streams_on_one_client_hear_only_their_own_audio(pooled) -> None:
    client, sockets, fresh = pooled
    results = await asyncio.gather(
        _collect(client, "A"), _collect(client, "B"), return_exceptions=True
    )
    for text, result in zip("AB", results, strict=True):
        assert not isinstance(result, BaseException), (text, result)
        assert _heard(result) == {ord(text)}, text
        assert len(result) == 3
    # One stream read the pooled socket; the other had one of its own.
    assert len(sockets) == 1 and len(fresh) == 1


def test_concurrent_sync_streams_never_close_each_others_socket(pooled) -> None:
    """Each sync ``stream()`` runs on its own thread and loop: one finishing
    first must not close the socket another is still reading (ENG-3312)."""
    client, sockets, fresh = pooled
    heard: dict[str, list[AudioChunk]] = {}
    errors: dict[str, BaseException] = {}

    def run(text: str) -> None:
        try:
            heard[text] = [
                item for item in client.tts.stream(text, language="en")
                if isinstance(item, AudioChunk)
            ]
        except BaseException as exc:  # noqa: BLE001  # reported by the assertion below
            errors[text] = exc

    slow = threading.Thread(target=run, args=(f"{SLOW_PREFIX} A",))
    slow.start()
    while not (sockets or fresh):
        threading.Event().wait(0.005)
    quick = threading.Thread(target=run, args=("B",))
    quick.start()
    quick.join(5)
    slow.join(5)

    assert not errors, errors
    assert _heard(heard[f"{SLOW_PREFIX} A"]) == {ord("S")}
    assert _heard(heard["B"]) == {ord("B")}
    assert not any(s.closed_from_foreign_loop for s in sockets + fresh)
