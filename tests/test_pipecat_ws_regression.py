"""End-to-end regression test for the pipecat WS dead-recovery bug.

Locks in the fix from PR #695 / KUG-616: when the underlying WebSocket of a
cached MultiContextSession dies between conversation turns (proxy idle drop,
NAT timeout, server restart), the next ``KugelAudioTTSService.run_tts`` call
must transparently reconnect and produce audio frames — not an ``ErrorFrame``
with ``no close frame received or sent``, which is what users were seeing in
production (KUG-616, KUG-618).

Unlike the existing tests in ``test_pipecat.py::TestTTSReconnectOnDeadWS`` —
which mock at the ``MultiContextSession`` level — this test mocks at the
``websockets.connect`` boundary. It therefore exercises the *real*
``MultiContextSession`` (handshake message, ``context_created`` ack, audio
recv loop, ``chunk_complete`` handling, ``ConnectionClosed`` recovery, dead-
WS guard) and the *real* ``KugelAudioTTSService.run_tts`` retry loop. If
either side regresses, this test fails.

The test does NOT need network access or live TTS credentials.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import pytest

# Skip the whole module when pipecat isn't installed (kugelaudio[pipecat] extra).
pytest.importorskip("pipecat")

from pipecat.frames.frames import (  # noqa: E402
    ErrorFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)


# ---------------------------------------------------------------------------
# Fake websocket implementations — driven entirely by scripted message lists.
# ---------------------------------------------------------------------------


class _ConnectionClosedError(Exception):
    """Stand-in for websockets.ConnectionClosedError. The SDK detects it by
    type-name string match (so its handling is robust across websockets
    minor versions); we mirror that here. The exact phrase ``no close frame
    received or sent`` is what users see in production logs and what we want
    to lock out from coming back.
    """

    def __init__(self) -> None:
        super().__init__("no close frame received or sent")
        self.code: int | None = None  # mimic the websockets attr
        self.reason: str | None = None


_ConnectionClosedError.__name__ = "ConnectionClosedError"


class _ScriptedWS:
    """Mock WS that replays a list of outgoing server messages then dies.

    A `_ScriptedWS` is one TCP connection's lifetime. After all scripted
    messages are consumed, the next ``recv()`` raises
    ``_ConnectionClosedError`` — simulating the proxy idle drop. The SDK's
    fix must detect this (via ``state == "CLOSED"`` *or* the raised
    exception) and either raise a typed ``KugelAudioConnectionError`` or
    let the wrapper trigger a reconnect.
    """

    def __init__(self, replies: list[str], die_after_replies: bool = False) -> None:
        # ``replies`` is a queue of JSON strings the "server" sends. We hand
        # them out in order on each ``recv()``. When empty + ``die_after_replies``,
        # ``recv()`` raises to mimic a closed transport.
        self._replies: list[str] = list(replies)
        self._die = die_after_replies
        self.sent: list[str] = []
        # Mirror websockets API: ``state`` is an enum-like with ``.name``.
        self._state_name = "OPEN"

    @property
    def state(self) -> Any:
        # Exposes ``.name`` like websockets does.
        from types import SimpleNamespace
        return SimpleNamespace(name=self._state_name)

    @property
    def close_code(self) -> int | None:
        return None

    async def send(self, data: str) -> None:
        if self._state_name != "OPEN":
            # Match how websockets behaves on send-after-close: raises the
            # same ConnectionClosed type the SDK now catches.
            raise _ConnectionClosedError()
        self.sent.append(data)
        # Auto-reply to a clean ``close_socket`` so MultiContextSession.close()
        # doesn't sit on its 30 s drain timeout during test teardown. A real
        # server emits ``{"session_closed": true, ...}`` here.
        try:
            parsed = json.loads(data)
        except (TypeError, ValueError):
            return
        if parsed.get("close_socket") or parsed.get("close") or parsed.get("end_session"):
            self._replies.append(json.dumps(
                {"session_closed": True, "total_audio_seconds": 0.0}
            ))

    async def recv(self) -> str:
        if self._replies:
            return self._replies.pop(0)
        if self._die:
            self._state_name = "CLOSED"
            raise _ConnectionClosedError()
        # No more replies but not "dying" — block briefly and let the caller
        # time out as it would on a healthy idle WS.
        await asyncio.sleep(60)
        raise asyncio.TimeoutError()

    async def close(self) -> None:
        self._state_name = "CLOSED"

    def kill(self) -> None:
        """Force the next operation on this WS to raise as if the TCP socket
        was abruptly closed by an upstream proxy. Mirrors what
        repro_pipecat_dead_ws.py does to a real socket via transport.abort()."""
        self._state_name = "CLOSED"


def _audio_msg(context_id: str, num_samples: int = 100) -> str:
    """Build the JSON shape the server sends for one audio chunk."""
    pcm = b"\x00\x01" * num_samples
    return json.dumps({
        "audio": base64.b64encode(pcm).decode("ascii"),
        "enc": "pcm_s16le",
        "idx": 0,
        "sr": 24000,
        "samples": num_samples,
        "context_id": context_id,
    })


def _context_created_msg(context_id: str) -> str:
    return json.dumps({"context_created": True, "context_id": context_id})


def _chunk_complete_msg(context_id: str) -> str:
    return json.dumps({"chunk_complete": True, "context_id": context_id})


def _build_turn_replies(context_id: str) -> list[str]:
    """The server's reply sequence for one ``send()``-with-flush turn:
    a ``context_created`` (on the implicit first send), one audio chunk,
    then ``chunk_complete``. Matches the real /ws/tts/multi protocol."""
    return [
        _context_created_msg(context_id),
        _audio_msg(context_id),
        _chunk_complete_msg(context_id),
    ]


# ---------------------------------------------------------------------------
# The regression test
# ---------------------------------------------------------------------------


class TestPipecatWSDeadRecoveryRegression:
    """Locks in the user-reported KUG-616 bug fix end-to-end.

    Exercises the *real* MultiContextSession (not a mock) wired into the
    *real* KugelAudioTTSService, with only the ``websockets.connect`` boundary
    mocked. If anyone regresses the dead-WS recovery (in either layer), the
    user's exact error string — ``no close frame received or sent`` — would
    re-appear and this test fails.
    """

    @pytest.mark.asyncio
    async def test_dead_ws_between_turns_self_heals(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kugelaudio.pipecat import KugelAudioTTSService
        from kugelaudio import streaming as ka_streaming
        import websockets as ws_module

        # Two scripted "TCP connections" — the second one is what the wrapper
        # must transparently establish after we kill the first.
        turn1_ws = _ScriptedWS(_build_turn_replies("ctx-turn-1"), die_after_replies=True)
        turn2_ws = _ScriptedWS(_build_turn_replies("ctx-turn-2"), die_after_replies=False)
        ws_queue: list[_ScriptedWS] = [turn1_ws, turn2_ws]

        async def _fake_connect(url: str, **kwargs: Any) -> _ScriptedWS:
            # The SDK passes ping_interval/ping_timeout/close_timeout (per
            # the keepalive fix). Accept and ignore — we just hand back the
            # next scripted WS.
            assert ws_queue, "wrapper attempted a 3rd connect — only 2 expected"
            return ws_queue.pop(0)

        monkeypatch.setattr(ws_module, "connect", _fake_connect)

        tts = KugelAudioTTSService(
            api_key="ka_test_dummy",
            model="kugel-1-turbo",
            voice_id=42,
            sample_rate=24000,
            language="en",
        )

        # ---- Turn 1: healthy WS ----
        frames_t1 = [f async for f in tts.run_tts("First turn.", "ctx-turn-1")]
        assert any(isinstance(f, TTSStartedFrame) for f in frames_t1)
        assert any(isinstance(f, TTSAudioRawFrame) for f in frames_t1), \
            "turn 1 must produce audio over the healthy mocked WS"
        assert any(isinstance(f, TTSStoppedFrame) for f in frames_t1)
        assert not any(isinstance(f, ErrorFrame) for f in frames_t1)

        # ---- Force the proxy-drop condition ----
        # The cached session's underlying WS is now exhausted+closed. Without
        # the fix, _get_multi_session would reuse it and the next send() would
        # crash inside `await self._ws.send(...)` with the user's exact
        # ConnectionClosedError("no close frame received or sent").
        assert tts._multi_session is turn1_ws_session(tts), \
            "wrapper should have cached the turn-1 session"
        turn1_ws.kill()

        # ---- Turn 2: dead cached WS, must self-heal ----
        frames_t2 = [f async for f in tts.run_tts("Second turn.", "ctx-turn-2")]

        # Critical assertions — the regression-locking ones:
        error_frames = [f for f in frames_t2 if isinstance(f, ErrorFrame)]
        assert error_frames == [], (
            "Regression: KugelAudio surfaced an ErrorFrame after a dead WS — "
            "this is the KUG-616 bug returning. ErrorFrame contents: "
            f"{[f.error for f in error_frames]}"
        )
        audio_frames_t2 = [f for f in frames_t2 if isinstance(f, TTSAudioRawFrame)]
        assert len(audio_frames_t2) >= 1, (
            "Regression: turn 2 produced no audio after a transparent reconnect"
        )

        # The wrapper must have established a SECOND ws connection — not
        # ridden the dead corpse. (ws_queue is now empty.)
        assert ws_queue == [], (
            "wrapper did not call websockets.connect a second time — it must "
            "have reused the dead WS, which is the KUG-616 bug"
        )
        # And the wrapper's cached session is the new one, not the dead one.
        assert tts._multi_session is not None
        assert tts._multi_session._ws is turn2_ws

        await tts.cleanup()


def turn1_ws_session(tts: Any) -> Any:
    """Return the wrapper's currently-cached session, asserting one exists."""
    sess = tts._multi_session
    assert sess is not None, "expected a cached session after turn 1"
    return sess


# ---------------------------------------------------------------------------
# Diagnostics opt-out matrix over a real socket (real stdlib sender). Lives in
# this CI-listed file so the python-sdk-pipecat unit runs it.
# ---------------------------------------------------------------------------

from .diagnostics_wire_support import (  # noqa: E402,F401  (fixtures)
    OPT_OUT_MATRIX,
    diagnostics_stub,
    hosted_probe,
)


@pytest.mark.real_diagnostics_sender
@pytest.mark.parametrize(("env", "option", "hosted", "expect_posts"), OPT_OUT_MATRIX)
async def test_pipecat_diagnostics_opt_out_matrix_on_the_wire(
    diagnostics_stub, hosted_probe, monkeypatch, env, option, hosted, expect_posts
):
    from kugelaudio.exceptions import AuthenticationError
    from kugelaudio.pipecat import KugelAudioTTSService

    if env is not None:
        monkeypatch.setenv("KUGELAUDIO_TELEMETRY", env)
    url = hosted_probe(diagnostics_stub) if hosted else diagnostics_stub.url()
    service = KugelAudioTTSService(
        api_key="wire-key", base_url=url, voice_id=1, language="de", telemetry=option
    )
    with pytest.raises(AuthenticationError):
        await service._get_multi_session()
    await service.cleanup()

    assert bool(diagnostics_stub.posts) is expect_posts
    for post in diagnostics_stub.posts:
        labels = {
            item["value"]["stringValue"]
            for record in post.records
            for item in record["attributes"]
            if item["key"] == "kugel.integration"
        }
        assert labels == {"pipecat"}
