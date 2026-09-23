"""Pipecat 1.x per-turn context leak fix (KUG-1087).

Pipecat 1.x (AudioContext API) calls ``run_tts(text, context_id)`` with a FRESH
``context_id`` per turn and signals completion via ``on_audio_context_completed``
/ ``on_turn_context_created``. The wrapper must close each caller-supplied
server-side context when its turn ends — otherwise every turn leaks a
``/ws/tts/multi`` context against the per-session cap and the call drops
mid-conversation (the INGRESS-S / Callbook AI symptom).

Like ``test_pipecat_ws_regression.py``, this exercises the REAL
``MultiContextSession`` + REAL ``KugelAudioTTSService`` with only the
``websockets.connect`` boundary mocked, so the actual ``close_context`` wire
message is asserted — not a method mock.
"""
from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import TTSAudioRawFrame  # noqa: E402


def _created(ctx: str) -> str:
    return json.dumps({"context_created": True, "context_id": ctx})


def _audio(ctx: str, n: int = 80) -> str:
    pcm = b"\x00\x01" * n
    return json.dumps({
        "audio": base64.b64encode(pcm).decode("ascii"),
        "enc": "pcm_s16le", "idx": 0, "sr": 24000, "samples": n,
        "context_id": ctx,
    })


def _chunk_complete(ctx: str) -> str:
    return json.dumps({"chunk_complete": True, "context_id": ctx})


class _FakeWS:
    """Protocol-accurate fake: replies to init/create sends with
    ``context_created``, to real text sends with audio + ``chunk_complete``, and
    to ``close_context`` with ``context_closed`` — id-agnostic, so it works for
    both caller-supplied ids and the wrapper's own minted id."""

    def __init__(self) -> None:
        self._replies: list[str] = []
        self._created: set[str] = set()
        self.sent: list[dict] = []
        self._state = "OPEN"

    @property
    def state(self) -> Any:
        from types import SimpleNamespace
        return SimpleNamespace(name=self._state)

    @property
    def close_code(self) -> int | None:
        return None

    async def send(self, data: str) -> None:
        parsed = json.loads(data)
        self.sent.append(parsed)
        ctx = parsed.get("context_id")
        if parsed.get("close_context"):
            self._replies.append(json.dumps(
                {"context_closed": True, "context_id": ctx}
            ))
        elif parsed.get("close") or parsed.get("close_socket") or parsed.get("end_session"):
            self._replies.append(json.dumps(
                {"session_closed": True, "total_audio_seconds": 0.0}
            ))
        elif "text" in parsed and ctx is not None:
            if ctx not in self._created:
                # init " " / create_context → server acks creation
                self._created.add(ctx)
                self._replies.append(_created(ctx))
            else:
                # real text send (flush) → one audio chunk then completion
                self._replies.append(_audio(ctx))
                self._replies.append(_chunk_complete(ctx))

    async def recv(self) -> str:
        if self._replies:
            return self._replies.pop(0)
        await asyncio.sleep(0.01)
        raise asyncio.TimeoutError()

    async def close(self) -> None:
        self._state = "CLOSED"


def _closed_context_ids(ws: _FakeWS) -> list[str]:
    return [m["context_id"] for m in ws.sent if m.get("close_context")]


def _make_service(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, _FakeWS]:
    from kugelaudio.pipecat import KugelAudioTTSService
    import websockets as ws_module

    ws = _FakeWS()

    async def _fake_connect(url: str, **kwargs: Any) -> _FakeWS:
        return ws

    monkeypatch.setattr(ws_module, "connect", _fake_connect)
    tts = KugelAudioTTSService(
        api_key="ka_test_dummy",
        model="kugel-1-turbo",
        voice_id=42,
        sample_rate=24000,
        language="en",
    )
    return tts, ws


async def _drive_turn(tts: Any, text: str, context_id: str | None) -> list[Any]:
    return [f async for f in tts.run_tts(text, context_id)]


@pytest.mark.asyncio
async def test_turn_context_created_prefetches_server_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """create_context runs on LLMFullResponseStartFrame, not on run_tts."""
    tts, ws = _make_service(monkeypatch)

    await tts.on_turn_context_created("ctx-1")

    init_msgs = [
        m for m in ws.sent
        if m.get("context_id") == "ctx-1" and m.get("text") == " "
    ]
    assert len(init_msgs) == 1
    assert "ctx-1" in tts._provisioned_contexts

    frames = await _drive_turn(tts, "Hello there.", "ctx-1")
    assert any(isinstance(f, TTSAudioRawFrame) for f in frames)

    init_msgs = [
        m for m in ws.sent
        if m.get("context_id") == "ctx-1" and m.get("text") == " "
    ]
    assert len(init_msgs) == 1, "run_tts must not create_context twice"

    await tts.cleanup()


@pytest.mark.asyncio
async def test_audio_context_completed_closes_caller_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tts, ws = _make_service(monkeypatch)

    frames = await _drive_turn(tts, "Hello there.", "ctx-1")
    assert any(isinstance(f, TTSAudioRawFrame) for f in frames)
    assert "ctx-1" in tts._caller_contexts  # tracked while open

    # Pipecat 1.x: the turn's audio finished playing.
    await tts.on_audio_context_completed("ctx-1")

    assert "ctx-1" in _closed_context_ids(ws), \
        "server-side context must be closed when the turn's audio completes"
    assert "ctx-1" not in tts._caller_contexts

    await tts.cleanup()


@pytest.mark.asyncio
async def test_new_turn_closes_previous_caller_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback path (builds without on_audio_context_completed): a new turn's
    context_created closes every prior turn's context."""
    tts, ws = _make_service(monkeypatch)

    await _drive_turn(tts, "First turn.", "ctx-1")
    assert "ctx-1" in tts._caller_contexts

    await tts.on_turn_context_created("ctx-2")

    assert "ctx-1" in _closed_context_ids(ws)
    assert "ctx-1" not in tts._caller_contexts
    assert "ctx-2" in tts._caller_contexts

    await tts.cleanup()


@pytest.mark.asyncio
async def test_many_turns_do_not_accumulate_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The leak scenario: many turns, each with a fresh caller context_id.
    With per-turn closing, at most one caller context is ever open, so the
    server's per-session cap is never approached regardless of its value."""
    tts, ws = _make_service(monkeypatch)

    for i in range(12):
        ctx = f"turn-{i}"
        await _drive_turn(tts, f"Turn {i}.", ctx)
        await tts.on_audio_context_completed(ctx)
        assert len(tts._caller_contexts) == 0, (
            f"after turn {i}, expected 0 open caller contexts, "
            f"got {tts._caller_contexts}"
        )

    # Every turn's context was closed server-side.
    assert _closed_context_ids(ws) == [f"turn-{i}" for i in range(12)]

    await tts.cleanup()


@pytest.mark.asyncio
async def test_pipecat_0x_unframed_run_tts_not_tracked_or_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy 0.x fallback: run_tts(text) driven directly with no surrounding
    LLM frames has no turn boundary to act on, so the lazily-minted context is
    reused and never tracked as a caller context or closed per call. (The
    per-turn close path is exercised by the frame-driven test below.)"""
    tts, ws = _make_service(monkeypatch)

    await _drive_turn(tts, "First.", None)
    await _drive_turn(tts, "Second.", None)

    assert tts._caller_contexts == set()
    assert _closed_context_ids(ws) == [], \
        "unframed legacy run_tts must not close its context per call"

    await tts.cleanup()


def _created_context_ids(ws: _FakeWS) -> list[str]:
    """Server-side contexts the wrapper opened: a text send for a context_id the
    fake hadn't seen before is what the real server treats as create_context."""
    seen: list[str] = []
    for m in ws.sent:
        ctx = m.get("context_id")
        if "text" in m and ctx is not None and ctx not in seen:
            seen.append(ctx)
    return seen


@pytest.mark.asyncio
async def test_pipecat_0x_frame_driven_closes_each_turn_no_accumulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy 0.x frame-driven lifecycle over a long conversation: each LLM turn
    pre-creates its OWN context on the start frame and closes it on the end
    frame, so the server never holds more than one open context at a time — no
    accumulation against the per-session cap, and each turn's KV is freed at its
    boundary. This is the 0.x analogue of the 1.x per-turn close proof, against
    the real MultiContextSession + protocol-accurate WS.
    """
    tts, ws = _make_service(monkeypatch)

    n_turns = 30
    for i in range(n_turns):
        await tts._legacy_begin_turn()  # LLM start frame: pre-create this turn's ctx
        ctx = tts._legacy_turn_context_id
        assert ctx is not None
        frames = await _drive_turn(tts, f"Agent turn {i}.", None)
        assert any(isinstance(f, TTSAudioRawFrame) for f in frames), \
            f"turn {i} produced no audio"
        await tts._legacy_end_turn()  # LLM end frame: close this turn's ctx
        assert tts._legacy_turn_context_id is None

    created = _created_context_ids(ws)
    closed = _closed_context_ids(ws)
    # One distinct context per turn, and each one closed at its turn boundary.
    assert len(created) == n_turns, f"expected one context per turn, got {len(created)}"
    assert len(set(created)) == n_turns, "every turn must use a distinct context"
    assert closed == created, "each turn's context must be closed at its end frame"
    assert tts._caller_contexts == set()  # legacy path doesn't use caller bookkeeping

    await tts.cleanup()
    assert ws.state.name == "CLOSED"


@pytest.mark.asyncio
async def test_pipecat_0x_real_base_closes_each_turn_at_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy 0.x, REAL base, end-to-end: drive LLMFullResponseStart / Text /
    End frames through pipecat's own ``run_test`` so the base
    ``TTSService.process_frame`` — not our hand-called helpers — synthesizes
    each turn's final buffered sentence while handling LLMFullResponseEndFrame.

    The TextFrame has no terminal punctuation, so the base holds it in its
    aggregator and renders it only on the end frame — the exact path the
    close-before-synthesize ordering bug hit. We assert at the WIRE level that
    each turn opens and closes exactly ONE server context: a second context per
    turn would mean the final fragment spilled onto a fresh context (the bug),
    visible as ``len(created) == 2 * n_turns``.

    Legacy-only: on the AudioContext surface the base owns context identity and
    the e2e proof lives in test_pipecat_context_leak_e2e.py instead.
    """
    from kugelaudio.pipecat.tts import _base_run_tts_takes_context_id

    if _base_run_tts_takes_context_id():
        pytest.skip("legacy run_tts(text) surface only")

    from pipecat.frames.frames import (  # noqa: E402
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        TextFrame,
    )
    from pipecat.tests.utils import run_test  # noqa: E402

    tts, ws = _make_service(monkeypatch)

    n_turns = 8
    frames: list[Any] = []
    for i in range(n_turns):
        frames.append(LLMFullResponseStartFrame())
        # No sentence terminator -> the base aggregator holds this and only
        # synthesizes it on the end frame.
        frames.append(TextFrame(f"Agent turn {i} without a sentence terminator"))
        frames.append(LLMFullResponseEndFrame())

    await run_test(tts, frames_to_send=frames, expected_down_frames=None)

    created = _created_context_ids(ws)
    closed = _closed_context_ids(ws)
    assert len(created) == n_turns, (
        f"expected one server context per turn, got {len(created)} — a second "
        "context per turn means the final fragment spilled onto a fresh context"
    )
    assert len(set(created)) == n_turns, "every turn must use a distinct context"
    assert closed == created, "each turn's context must be closed at its end frame"

    await tts.cleanup()
