"""Live testbench for barge-in cancellation via the LiveKit SDK.

Exercises the real ``kugelaudio.livekit.TTS`` → ``SynthesizeStream`` path
against a running local TTS server, then simulates a barge-in by calling
``stream.aclose()`` mid-generation.

Invariant the test verifies:
    After aclose(), the server stops streaming audio for the cancelled
    context within ~1s.  Without the ``immediate=True`` close flag on
    CancelledError, the server kept generating for 5–10s after the
    client already gave up.

How it works:
    Wraps the underlying ``_Connection._ws.receive`` to record every
    raw incoming frame with its arrival timestamp.  After ``aclose()``
    we count how many audio frames for the cancelled ctx_id arrived
    after the cancel.  Independent of the SDK's debug logging.

Skip conditions: no API key, or server not reachable.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

# Skip collection if optional livekit deps (livekit-agents, aiohttp) are missing.
pytest.importorskip("livekit.agents")
aiohttp = pytest.importorskip("aiohttp")

from kugelaudio.livekit import TTS  # noqa: E402

_BASE_URL = os.environ.get(
    "KUGELAUDIO_BASE_URL", "http://localhost:8000",
)


def _read_master_key() -> str | None:
    p = Path(__file__).resolve().parents[3] / "tts" / ".env.local"
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        if line.startswith("TTS_MASTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    return None


_API_KEY = os.environ.get("KUGELAUDIO_API_KEY") or _read_master_key()


async def _install_sniffer(tts_instance, sniffer) -> None:
    """Wait for the SDK to open its connection, then attach the sniffer."""
    for _ in range(80):  # up to ~4s
        conn = tts_instance._current_connection
        if conn is not None and conn._ws is not None:
            await sniffer.install(conn)
            return
        await asyncio.sleep(0.05)


class _WsFrameSniffer:
    """Wrap _Connection._ws.receive to record every incoming WS frame.

    Independent of SDK logging — sees raw frames as they arrive,
    including post-cancel residuals and the ``context_closed``
    confirmation.
    """

    def __init__(self) -> None:
        self.records: list[tuple[float, dict]] = []
        self.t0 = time.monotonic()

    async def install(self, conn) -> None:
        # Wait for _ws to exist (created in connect()).
        for _ in range(40):
            if conn._ws is not None:
                break
            await asyncio.sleep(0.05)
        assert conn._ws is not None, "connection never opened ws"

        orig_receive = conn._ws.receive

        async def _wrapped():
            msg = await orig_receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    self.records.append(
                        (time.monotonic() - self.t0, json.loads(msg.data))
                    )
                except Exception:
                    pass
            return msg

        conn._ws.receive = _wrapped  # type: ignore[assignment]

    def audio_frames_after(self, since_s: float, ctx_id: str) -> int:
        return sum(
            1 for (t, d) in self.records
            if t >= since_s
            and d.get("context_id") == ctx_id
            and d.get("audio")
        )

    def last_audio_at(self, ctx_id: str) -> float | None:
        latest = None
        for t, d in self.records:
            if d.get("context_id") == ctx_id and d.get("audio"):
                latest = t
        return latest


@pytest.mark.asyncio
async def test_barge_in_stops_server_within_one_second() -> None:
    if not _API_KEY:
        pytest.skip("no TTS_MASTER_API_KEY / KUGELAUDIO_API_KEY")

    # Reachability probe — skip cleanly if serve is down.
    probe_url = _BASE_URL.replace("ws://", "http://").replace("wss://", "https://")
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(probe_url, timeout=aiohttp.ClientTimeout(total=2)) as r:
                _ = r.status
    except Exception as e:
        pytest.skip(f"TTS server not reachable at {probe_url}: {e}")

    sniffer = _WsFrameSniffer()
    http_session = aiohttp.ClientSession()

    tts_instance = TTS(
        api_key=_API_KEY,
        base_url=_BASE_URL,
        model="kugel-2-turbo",
        voice_id=268,
        sample_rate=24000,
        http_session=http_session,
    )

    # Hook the underlying connection's WS receive — wraps before the
    # first message is dispatched.
    sniffer_install = asyncio.create_task(_install_sniffer(tts_instance, sniffer))

    # Long German text (~30 numbers).  Full generation is ~15–20s of audio.
    long_text = (
        "Eins, zwei, drei, vier, fünf, sechs, sieben, acht, neun, zehn. "
        "Elf, zwölf, dreizehn, vierzehn, fünfzehn, sechzehn, siebzehn. "
        "Achtzehn, neunzehn, zwanzig, einundzwanzig, zweiundzwanzig, "
        "dreiundzwanzig, vierundzwanzig, fünfundzwanzig, sechsundzwanzig, "
        "siebenundzwanzig, achtundzwanzig, neunundzwanzig, dreißig."
    )

    try:
        stream = tts_instance.stream()

        # Producer: push tokens and end input immediately (LiveKit pattern).
        for tok in long_text.split(" "):
            stream.push_text(tok + " ")
        stream.end_input()

        # Consumer: collect audio; break out after ~0.3s worth so we can
        # barge-in while generation is still mid-flight.
        audio_before_cancel_ms = 0.0
        first_audio_wall: float | None = None
        ac_deadline_s = 0.3
        total_chunks = 0

        async def _consume_until_some_audio():
            nonlocal audio_before_cancel_ms, first_audio_wall, total_chunks
            async for ev in stream:
                total_chunks += 1
                if first_audio_wall is None:
                    first_audio_wall = time.monotonic()
                # ev.frame is a rtc.AudioFrame; duration in seconds
                audio_before_cancel_ms += ev.frame.duration * 1000
                if audio_before_cancel_ms >= ac_deadline_s * 1000:
                    return

        try:
            await asyncio.wait_for(
                _consume_until_some_audio(), timeout=20.0,
            )
        except asyncio.TimeoutError:
            pytest.fail(
                "timed out waiting for initial audio frames "
                f"(chunks received: {total_chunks})"
            )

        assert first_audio_wall is not None, "no audio received at all"
        assert total_chunks > 0

        # Resolve the ctx_id the SDK assigned to this stream — needed
        # to filter sniffer records.
        ctx_id = tts_instance._current_connection._context_data and next(
            iter(tts_instance._current_connection._context_data.keys()), None,
        )
        # If the stream's ctx already cleaned up, look at sniffer records
        # for any audio frame we saw.
        if ctx_id is None:
            for _, d in sniffer.records:
                if d.get("audio") and d.get("context_id"):
                    ctx_id = d["context_id"]
                    break
        assert ctx_id, "could not resolve ctx_id for this stream"

        # Simulate barge-in.  aclose cancels _run → CancelledError →
        # SDK sends close_context {"immediate": true}.
        t_cancel = time.monotonic() - sniffer.t0
        await stream.aclose()

        # Give the server a generous window to stop.  Anything > 1s is
        # "late audio that shouldn't have been generated".
        await asyncio.sleep(2.5)

        last_audio_rel = sniffer.last_audio_at(ctx_id)
        late_after_cancel_s = (
            (last_audio_rel - t_cancel) if last_audio_rel is not None
            and last_audio_rel >= t_cancel
            else 0.0
        )
        late_audio_frames = sniffer.audio_frames_after(t_cancel, ctx_id)

        # Report diagnostics — useful whether this passes or fails.
        print(
            f"\nbarge-in report (ctx={ctx_id}):\n"
            f"  chunks consumed before cancel : {total_chunks}\n"
            f"  audio ms before cancel        : {audio_before_cancel_ms:.0f} ms\n"
            f"  late audio frames after cancel: {late_audio_frames}\n"
            f"  last late frame at +          : {late_after_cancel_s:.2f} s "
            f"after cancel\n"
        )

        # Before the fix, late audio continued for 5–10s.  With
        # immediate=True + server upgrade-to-cancel, it should stop
        # within ~1s (just network-pipeline residual).
        assert late_after_cancel_s <= 1.5, (
            f"server kept sending audio for {late_after_cancel_s:.2f}s "
            f"after barge-in (expected ≤ 1.5s). "
            f"immediate=True close is not stopping server-side generation."
        )
    finally:
        sniffer_install.cancel()
        try:
            await sniffer_install
        except (asyncio.CancelledError, Exception):
            pass
        await http_session.close()
