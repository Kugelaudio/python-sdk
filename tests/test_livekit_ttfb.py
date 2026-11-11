"""TTFB in LiveKit's ``TTSMetrics`` for the KugelAudio plugin (ENG-2479).

LiveKit Agents computes a streaming ``TTSMetrics.ttfb`` as the time from
``SynthesizeStream._mark_started()`` to the first audio frame, and documents
the mark as "first text sent to the provider". These tests drive LiveKit's
real stream classes against a fake ``/ws/tts/multi`` that answers each text
only after a fixed delay, so the true time to first audio is known.
"""

from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("livekit.agents")

SERVER_DELAY_S = 0.2
# 100 ms of 24 kHz mono int16 silence.
_PCM = base64.b64encode(b"\x00\x00" * 2400).decode()


class _DelayedAudioWS:
    """Fake multi-context socket: audio for a context arrives SERVER_DELAY_S
    after its first text; ``context_closed`` follows that audio."""

    def __init__(self) -> None:
        self.closed = False
        self.close_code = None
        self._out: asyncio.Queue[dict] = asyncio.Queue()
        self._audio: dict[str, asyncio.Task] = {}

    async def send_str(self, raw: str) -> None:
        msg = json.loads(raw)
        cid = msg.get("context_id")
        if cid is None:
            return
        if msg.get("close_context"):
            asyncio.get_running_loop().create_task(self._close(cid))
        elif msg.get("text") and cid not in self._audio:
            self._audio[cid] = asyncio.get_running_loop().create_task(
                self._answer(cid)
            )

    async def _answer(self, cid: str) -> None:
        await asyncio.sleep(SERVER_DELAY_S)
        await self._out.put({"context_id": cid, "audio": _PCM})
        await self._out.put({"context_id": cid, "chunk_complete": True})

    async def _close(self, cid: str) -> None:
        if cid in self._audio:
            await self._audio[cid]
        await self._out.put({"context_id": cid, "context_closed": True})

    async def receive(self):
        import aiohttp

        data = await self._out.get()
        return MagicMock(type=aiohttp.WSMsgType.TEXT, data=json.dumps(data))

    async def close(self) -> None:
        self.closed = True


def _tts():
    from kugelaudio.livekit import TTS

    session = MagicMock()
    session.ws_connect = AsyncMock(return_value=_DelayedAudioWS())
    tts = TTS(api_key="k", http_session=session, telemetry=False)
    metrics: list = []
    tts.on("metrics_collected", metrics.append)
    return tts, metrics


async def _drain(stream) -> None:
    async with stream:
        async for _ in stream:
            pass


async def test_streaming_ttfb_counts_from_the_first_text_sent():
    tts, metrics = _tts()
    stream = tts.stream()
    stream.push_text("Hallo Welt.")
    stream.end_input()
    await _drain(stream)
    await tts.aclose()

    (m,) = metrics
    assert m.streamed is True
    assert m.ttfb >= SERVER_DELAY_S * 0.9, m.ttfb
    assert m.ttfb < SERVER_DELAY_S + 1.0, m.ttfb


async def test_chunked_ttfb_still_counts_the_server_delay():
    tts, metrics = _tts()
    await _drain(tts.synthesize("Hallo Welt."))
    await tts.aclose()

    (m,) = metrics
    assert m.streamed is False
    assert m.ttfb >= SERVER_DELAY_S * 0.9, m.ttfb
    assert m.ttfb < SERVER_DELAY_S + 1.0, m.ttfb
