"""Turn boundaries of ``/ws/tts/stream`` against a production-like server.

The local server behaves like the production endpoint: a frame with
``flush: true`` ends the turn on its own (``final``, then ``session_closed``
with usage), and a ``close`` with no turn open ends an empty session
(``session_closed`` with 0 s). Sending that ``close`` after a flushed turn
used to overwrite the turn's usage with zeros and leave the empty
``session_closed`` in the socket, where it ended the next turn before its
audio arrived.
"""

from __future__ import annotations

import base64
import json
import struct
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import aclosing
from typing import Any

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import ServerConnection, serve

from kugelaudio import KugelAudio
from kugelaudio.models import AudioChunk
from kugelaudio.streaming import StreamingSession

FRAMES_PER_TURN = 5
SAMPLES_PER_FRAME = 240
SAMPLE_RATE = 24000
TURN_SECONDS = FRAMES_PER_TURN * SAMPLES_PER_FRAME / SAMPLE_RATE


def _audio_frame(idx: int, turn: int) -> dict[str, Any]:
    """Every sample carries the turn number, so a chunk names its turn."""
    pcm = struct.pack(f"<{SAMPLES_PER_FRAME}h", *([turn] * SAMPLES_PER_FRAME))
    return {
        "audio": base64.b64encode(pcm).decode("ascii"),
        "enc": "pcm_s16le",
        "idx": idx,
        "sr": SAMPLE_RATE,
        "samples": SAMPLES_PER_FRAME,
    }


def _usage(seconds: float, characters: int, cost: float) -> dict[str, Any]:
    return {
        "session_closed": True,
        "total_audio_seconds": seconds,
        "usage": {
            "audio_seconds": seconds,
            "characters": characters,
            "cost_cents": cost,
            "currency": "eur",
        },
    }


class ProductionLikeServer:
    """Ends a turn on flush or close; an idle close ends an empty session."""

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    def closes_sent(self) -> int:
        return sum(1 for m in self.seen if m.get("close") is True)

    def handler(self, ws: ServerConnection) -> None:
        turn = 0
        buffered = False
        try:
            for raw in ws:
                message: dict[str, Any] = json.loads(raw)
                self.seen.append(message)
                if isinstance(message.get("text"), str):
                    buffered = True
                if message.get("close") is True and not buffered:
                    ws.send(json.dumps(_usage(0.0, 0, 0.0)))
                elif message.get("flush") is True or message.get("close") is True:
                    turn += 1
                    buffered = False
                    self._speak_turn(ws, turn)
        except ConnectionClosed:
            return

    @staticmethod
    def _speak_turn(ws: ServerConnection, turn: int) -> None:
        ws.send(json.dumps({"generation_started": True, "chunk_id": 0}))
        for idx in range(FRAMES_PER_TURN):
            ws.send(json.dumps(_audio_frame(idx, turn)))
        ws.send(json.dumps({"chunk_complete": True, "chunk_id": 0}))
        ws.send(json.dumps({"final": True, "total_audio_seconds": TURN_SECONDS}))
        ws.send(json.dumps(_usage(TURN_SECONDS, 10, 0.5)))


@pytest.fixture
def env() -> Iterator[tuple[ProductionLikeServer, StreamingSession]]:
    server = ProductionLikeServer()
    ws_server = serve(server.handler, "127.0.0.1", 0, compression=None)
    thread = threading.Thread(target=ws_server.serve_forever, daemon=True)
    thread.start()
    port = ws_server.socket.getsockname()[1]
    client = KugelAudio(api_key="sk-test", api_url=f"http://127.0.0.1:{port}")
    try:
        yield server, client.tts.streaming_session(voice_id=7, language="en")
    finally:
        ws_server.shutdown()
        thread.join(5)


def _turn_of(chunk: AudioChunk) -> int:
    return int(struct.unpack_from("<h", chunk.audio)[0])


async def _collect(chunks: AsyncIterator[AudioChunk]) -> list[AudioChunk]:
    return [c async for c in chunks]


async def _read_first_then_stop(session: StreamingSession, text: str) -> AudioChunk:
    """Read one chunk of a flushed turn and stop, like a caller that breaks."""
    async with aclosing(session.send(text, flush=True)) as audio:
        async for chunk in audio:
            return chunk
    raise AssertionError("the flushed turn produced no audio")


async def test_drain_and_end_session_after_a_flushed_turn_keep_its_usage(env) -> None:
    server, session = env
    audio = await _collect(session.send("Turn one.", flush=True))
    assert len(audio) == FRAMES_PER_TURN
    assert session.last_usage is not None
    usage = session.last_usage.audio_seconds
    assert usage > 0

    tail = await _collect(session.drain())
    assert tail == [], "the flushed turn had no tail"
    assert session.last_usage.audio_seconds == usage, (
        "drain must not replace the turn's usage with an empty session's"
    )
    stats = await session.end_session()
    assert stats["total_audio_seconds"] == usage

    nxt = await _collect(session.send("Turn two.", flush=True))
    assert len(nxt) == FRAMES_PER_TURN
    assert all(_turn_of(c) == 2 for c in nxt)
    stats = await session.close()
    assert stats["total_audio_seconds"] == usage, "close returns turn two's stats"
    assert server.closes_sent() == 0, "no turn was left open to close"


async def test_stopping_a_flushed_turn_then_draining_yields_its_tail_only(env) -> None:
    server, session = env
    first = await _read_first_then_stop(session, "A long turn.")
    assert _turn_of(first) == 1

    tail = await _collect(session.drain())
    assert len(tail) == FRAMES_PER_TURN - 1
    assert all(_turn_of(c) == 1 for c in tail)
    assert session.last_usage is not None and session.last_usage.audio_seconds > 0
    assert (await session.end_session())["total_audio_seconds"] > 0

    nxt = await _collect(session.send("Follow-up.", flush=True))
    assert len(nxt) == FRAMES_PER_TURN
    assert all(_turn_of(c) == 2 for c in nxt)
    assert session.last_usage.audio_seconds > 0
    assert server.closes_sent() == 0
    await session.close()


async def test_stopping_a_flushed_turn_then_end_session_leaves_no_stale_frames(
    env,
) -> None:
    server, session = env
    await _read_first_then_stop(session, "A long turn.")
    stats = await session.end_session()
    assert stats["total_audio_seconds"] > 0, stats

    nxt = await _collect(session.send("Follow-up.", flush=True))
    assert len(nxt) == FRAMES_PER_TURN
    assert all(_turn_of(c) == 2 for c in nxt)
    assert server.closes_sent() == 0
    await session.close()


async def test_drain_closes_an_unflushed_turn_every_time(env) -> None:
    server, session = env
    early = await _collect(session.send("Buffered text", flush=False))
    assert early == []
    tail = await _collect(session.drain())
    assert len(tail) == FRAMES_PER_TURN
    assert session.last_usage is not None and session.last_usage.audio_seconds > 0
    assert server.closes_sent() == 1
    # A second drain has nothing left to end.
    assert await _collect(session.drain()) == []
    assert server.closes_sent() == 1

    # drain, then a new turn, then drain again: the new turn is closed too.
    await _collect(session.send("More buffered text", flush=False))
    tail = await _collect(session.drain())
    assert len(tail) == FRAMES_PER_TURN
    assert all(_turn_of(c) == 2 for c in tail)
    assert server.closes_sent() == 2
    await session.close()
