"""End-to-end proof that the Pipecat 1.x per-turn context fix (KUG-1087) works
against the REAL pipecat base TTSService — not just our hand-called hooks.

Unlike ``test_pipecat_context_leak.py`` (which invokes the completion hooks
directly), this drives real ``LLMFullResponseStart/EndFrame`` + ``TextFrame``s
through pipecat's own ``run_test`` pipeline, so the base ``TTSService`` is the
one that assigns a per-turn ``context_id`` and calls our
``on_turn_context_created`` / ``on_audio_context_completed`` overrides. It then
asserts that ``close_context`` actually goes out per turn and that a long
conversation never accumulates contexts.

Requires the pipecat 1.x AudioContext API (run_tts(text, context_id) + the turn
hooks); skipped on the pre-1.0 0.0.x surface.
"""
from __future__ import annotations

import asyncio
import base64
import json

import pytest

pytest.importorskip("pipecat")

from pipecat.services.tts_service import TTSService  # noqa: E402

# Gate: only the 1.x AudioContext API has these. Pre-1.0 pipecat skips.
_HAS_TURN_HOOKS = hasattr(TTSService, "on_turn_context_created") and hasattr(
    TTSService, "on_audio_context_completed"
)
pytestmark = pytest.mark.skipif(
    not _HAS_TURN_HOOKS,
    reason="requires pipecat 1.x AudioContext API (run_tts context_id + turn hooks)",
)

if _HAS_TURN_HOOKS:
    from pipecat.frames.frames import (
        TextFrame,
        LLMFullResponseStartFrame,
        LLMFullResponseEndFrame,
    )
    from pipecat.tests.utils import run_test
    from kugelaudio.pipecat import KugelAudioTTSService


def _created(ctx: str) -> str:
    return json.dumps({"context_created": True, "context_id": ctx})


def _audio(ctx: str, n: int = 40) -> str:
    return json.dumps({
        "audio": base64.b64encode(b"\x00\x01" * n).decode(), "enc": "pcm_s16le",
        "idx": 0, "sr": 24000, "samples": n, "context_id": ctx,
    })


def _chunk_complete(ctx: str) -> str:
    return json.dumps({"chunk_complete": True, "context_id": ctx})


class _FakeWS:
    """Protocol-accurate fake that also tracks server-side context lifetimes."""

    def __init__(self) -> None:
        self._replies: list[str] = []
        self._created: set[str] = set()
        self.open: set[str] = set()
        self.max_open = 0
        self.total_created = 0
        self.total_closed = 0
        self._state = "OPEN"

    @property
    def state(self):
        from types import SimpleNamespace
        return SimpleNamespace(name=self._state)

    @property
    def close_code(self):
        return None

    async def send(self, data: str) -> None:
        m = json.loads(data)
        ctx = m.get("context_id")
        if m.get("close_context"):
            if ctx in self.open:
                self.open.discard(ctx)
                self.total_closed += 1
            self._replies.append(json.dumps({"context_closed": True, "context_id": ctx}))
        elif m.get("close") or m.get("close_socket") or m.get("end_session"):
            self._replies.append(json.dumps({"session_closed": True, "total_audio_seconds": 0.0}))
        elif "text" in m and ctx is not None:
            if ctx not in self._created:
                self._created.add(ctx)
                self.open.add(ctx)
                self.total_created += 1
                self.max_open = max(self.max_open, len(self.open))
                self._replies.append(_created(ctx))
            else:
                self._replies.append(_audio(ctx))
                self._replies.append(_chunk_complete(ctx))

    async def recv(self) -> str:
        if self._replies:
            return self._replies.pop(0)
        await asyncio.sleep(0.005)
        raise asyncio.TimeoutError()

    async def close(self) -> None:
        self._state = "CLOSED"


def _install_fake(monkeypatch) -> _FakeWS:
    ws = _FakeWS()

    async def _connect(url, **kw):
        return ws

    import websockets
    monkeypatch.setattr(websockets, "connect", _connect)
    return ws


def _service() -> "KugelAudioTTSService":
    return KugelAudioTTSService(
        api_key="ka_test_dummy", voice_id=42,
        model="kugel-1-turbo", sample_rate=24000, language="en",
    )


@pytest.mark.asyncio
async def test_base_drives_per_turn_close(monkeypatch) -> None:
    """The real base assigns a per-turn context_id and our overrides close it:
    two turns -> two distinct contexts, both closed."""
    ws = _install_fake(monkeypatch)
    tts = _service()

    seen: list[str] = []
    orig = tts.run_tts

    async def _spy(text, context_id=None):
        seen.append(context_id)
        async for f in orig(text, context_id):
            yield f

    monkeypatch.setattr(tts, "run_tts", _spy)

    frames = [
        LLMFullResponseStartFrame(), TextFrame("First turn."), LLMFullResponseEndFrame(),
        LLMFullResponseStartFrame(), TextFrame("Second turn."), LLMFullResponseEndFrame(),
    ]
    await run_test(tts, frames_to_send=frames, expected_down_frames=None)

    distinct = list(dict.fromkeys(seen))
    assert all(c is not None for c in seen) and seen, "run_tts must get a per-turn context_id (1.x path)"
    assert len(distinct) == 2, f"expected one context per turn, got {distinct}"
    assert ws.total_created == 2
    assert ws.total_closed == 2, "every turn's server context must be closed"
    assert ws.max_open <= 2


@pytest.mark.asyncio
async def test_long_conversation_does_not_pile_up(monkeypatch) -> None:
    """40-turn conversation: contexts must not accumulate — max concurrent stays
    well under the server's cap no matter how long the call runs."""
    ws = _install_fake(monkeypatch)
    tts = _service()

    n_turns = 40
    frames = []
    for i in range(n_turns):
        frames.append(LLMFullResponseStartFrame())
        frames.append(TextFrame(f"Agent turn {i}."))
        if i % 3 == 0:
            frames.append(TextFrame("Second sentence in the same turn."))
        frames.append(LLMFullResponseEndFrame())

    await run_test(tts, frames_to_send=frames, expected_down_frames=None)

    assert ws.total_created == n_turns, f"one context per turn; got {ws.total_created}"
    assert ws.max_open <= 2, f"contexts piled up: max concurrent {ws.max_open}"
    assert ws.total_closed >= n_turns - 1, "essentially all contexts closed"
    assert len(ws.open) <= 1
