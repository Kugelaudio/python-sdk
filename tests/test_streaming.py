"""Unit tests for streaming session classes.

Tests the SDK-side streaming logic with mocked WebSocket connections
to verify protocol compatibility with the server.
"""

import asyncio
import base64
import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kugelaudio.exceptions import (
    AuthenticationError,
    ConnectionError as KugelAudioConnectionError,
    KugelAudioError,
    RateLimitError,
    ServerRestartingError,
    ValidationError,
)
from kugelaudio.models import AudioChunk, SessionUsage, StreamConfig
from kugelaudio.streaming import (
    _POLL_TIMEOUT_S,
    MultiContextSession,
    StreamingSession,
    StreamingSessionSync,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audio_msg(idx: int = 0, sr: int = 24000, num_samples: int = 100) -> str:
    """Create a JSON audio message like the server sends."""
    pcm = b"\x00\x01" * num_samples  # 2 bytes per sample
    return json.dumps(
        {
            "audio": base64.b64encode(pcm).decode("ascii"),
            "enc": "pcm_s16le",
            "idx": idx,
            "sr": sr,
            "samples": num_samples,
        }
    )


def _make_audio_msg_with_ctx(
    context_id: str, idx: int = 0, sr: int = 24000, num_samples: int = 100
) -> str:
    """Create a JSON audio message with context_id."""
    pcm = b"\x00\x01" * num_samples
    return json.dumps(
        {
            "audio": base64.b64encode(pcm).decode("ascii"),
            "enc": "pcm_s16le",
            "idx": idx,
            "sr": sr,
            "samples": num_samples,
            "context_id": context_id,
        }
    )


class MockWebSocket:
    """Fake WebSocket that replays a list of messages then raises TimeoutError."""

    def __init__(self, messages: List[str]):
        self._messages = list(messages)
        self._sent: List[str] = []
        self._closed = False

    async def recv(self) -> str:
        if not self._messages:
            raise asyncio.TimeoutError()
        message = self._messages.pop(0)
        if isinstance(message, tuple):
            delay, message = message
            if delay:
                await asyncio.sleep(delay)
        return message

    async def send(self, data: str) -> None:
        self._sent.append(data)

    async def close(self) -> None:
        self._closed = True


class DelayedMockWebSocket(MockWebSocket):
    """Fake WebSocket with per-message delays before recv returns."""

    def __init__(self, messages: List[tuple[float, str]]):
        self._messages: List[str] = []
        self._sent: List[str] = []
        self._closed = False
        self._delayed_messages = list(messages)

    async def recv(self) -> str:
        if not self._delayed_messages:
            raise asyncio.TimeoutError()
        delay, message = self._delayed_messages.pop(0)
        if delay:
            await asyncio.sleep(delay)
        return message


# ---------------------------------------------------------------------------
# StreamConfig tests
# ---------------------------------------------------------------------------


class TestStreamConfigServerCompat:
    """Verify StreamConfig.to_dict() produces fields the server expects."""

    def test_config_omits_optional_none_fields(self):
        cfg = StreamConfig()
        d = cfg.to_dict()
        assert "voice_id" not in d
        assert "model_id" not in d
        assert "language" not in d


# ---------------------------------------------------------------------------
# StreamingSession tests
# ---------------------------------------------------------------------------


class TestStreamingSessionSend:
    """Test that StreamingSession.send() merges config into the first msg."""

    @pytest.mark.asyncio
    async def test_first_message_contains_config(self):
        """First send() should embed StreamConfig fields in the message."""
        ws = MockWebSocket([])  # no server replies — will timeout immediately

        session = StreamingSession(
            api_key="test",
            tts_url="http://localhost:8000",
            config=StreamConfig(voice_id=42, model_id="kugel-1-turbo"),
        )
        session._ws = ws
        session._is_started = True

        # Consume the async generator (no audio expected)
        async for _ in session.send("Hello"):
            pass

        assert len(ws._sent) == 1
        sent = json.loads(ws._sent[0])
        # Config fields merged into the text message
        assert sent["text"] == "Hello"
        assert sent["voice_id"] == 42
        assert sent["model_id"] == "kugel-1-turbo"

    @pytest.mark.asyncio
    async def test_second_message_omits_config(self):
        """Subsequent send() calls should NOT re-send config fields."""
        ws = MockWebSocket([])

        session = StreamingSession(
            api_key="test",
            tts_url="http://localhost:8000",
            config=StreamConfig(voice_id=42),
        )
        session._ws = ws
        session._is_started = True

        # First send — includes config
        async for _ in session.send("Hello"):
            pass
        # Second send — should be plain
        async for _ in session.send("World"):
            pass

        assert len(ws._sent) == 2
        second = json.loads(ws._sent[1])
        assert second["text"] == "World"
        assert "voice_id" not in second


class TestStreamingSessionReceive:
    """Test _receive_until_idle handles server messages correctly."""

    @pytest.mark.asyncio
    async def test_receives_audio_and_stops_on_chunk_complete(self):
        """Should yield audio and break on chunk_complete."""
        ws = MockWebSocket(
            [
                json.dumps(
                    {"generation_started": True, "chunk_id": 0, "text": "Hello"}
                ),
                _make_audio_msg(idx=0),
                _make_audio_msg(idx=1),
                json.dumps({"chunk_complete": True, "chunk_id": 0}),
            ]
        )

        session = StreamingSession(
            api_key="test",
            tts_url="http://localhost:8000",
        )
        session._ws = ws
        session._is_started = True
        session._config_sent = True

        chunks = []
        async for chunk in session.send("Hello."):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert all(isinstance(c, AudioChunk) for c in chunks)

    @pytest.mark.asyncio
    async def test_stops_on_session_closed(self):
        """Should break on session_closed too."""
        ws = MockWebSocket(
            [
                _make_audio_msg(idx=0),
                json.dumps({"session_closed": True}),
            ]
        )

        session = StreamingSession(
            api_key="test",
            tts_url="http://localhost:8000",
        )
        session._ws = ws
        session._is_started = True
        session._config_sent = True

        chunks = []
        async for chunk in session.send("Hi."):
            chunks.append(chunk)

        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_error_message_raises(self):
        """Server error messages should raise KugelAudioError."""
        ws = MockWebSocket(
            [
                json.dumps({"error": "something went wrong"}),
            ]
        )

        session = StreamingSession(
            api_key="test",
            tts_url="http://localhost:8000",
        )
        session._ws = ws
        session._is_started = True
        session._config_sent = True

        with pytest.raises(KugelAudioError, match="something went wrong"):
            async for _ in session.send("test"):
                pass

    @pytest.mark.asyncio
    async def test_ingress_missing_voice_frame_raises_validation_error(self):
        """Ingress WS error frames include code/error_code; preserve both."""
        ws = MockWebSocket(
            [
                json.dumps(
                    {
                        "error": "voice_id is required",
                        "error_code": "MISSING_VOICE_ID",
                        "code": 400,
                    }
                ),
            ]
        )

        session = StreamingSession(
            api_key="test",
            tts_url="http://localhost:8000",
        )
        session._ws = ws
        session._is_started = True
        session._config_sent = True

        with pytest.raises(ValidationError) as exc_info:
            async for _ in session.send("test"):
                pass

        assert exc_info.value.status_code == 400
        assert exc_info.value.error_code == "MISSING_VOICE_ID"


class DelayedMockWebSocket:
    """MockWebSocket that sleeps before returning each message.

    Used to exercise the first-frame timeout path: with a delay greater
    than ``_POLL_TIMEOUT_S`` (50 ms), the short poll will time out before
    the message arrives unless the caller opted into the longer timeout.
    """

    def __init__(self, messages: List[str], delay: float = 0.1) -> None:
        self._messages = list(messages)
        self._delay = delay
        self._sent: List[str] = []

    async def recv(self) -> str:
        if not self._messages:
            raise asyncio.TimeoutError()
        await asyncio.sleep(self._delay)
        return self._messages.pop(0)

    async def send(self, data: str) -> None:
        self._sent.append(data)

    async def close(self) -> None:
        pass


class TestStreamingSessionFirstFrameTimeout:
    """Regressions for KUG-422 / KUG-421: the first audio frame may arrive
    after the short poll timeout (TTFA is typically 100–500 ms). When the
    caller knows the server will produce audio (``flush=True`` or an
    explicit ``flush()``), the iterator must wait long enough.
    """

    @pytest.mark.asyncio
    async def test_send_with_flush_waits_for_first_frame(self):
        """send(text, flush=True) must collect audio even if TTFA > 50 ms."""
        ws = DelayedMockWebSocket(
            [
                _make_audio_msg(idx=0),
                json.dumps({"chunk_complete": True, "chunk_id": 0}),
            ],
            delay=0.1,  # > _POLL_TIMEOUT_S
        )

        session = StreamingSession(
            api_key="test", tts_url="http://localhost:8000"
        )
        session._ws = ws
        session._is_started = True
        session._config_sent = True

        chunks = []
        async for chunk in session.send("Hello.", flush=True):
            chunks.append(chunk)

        assert len(chunks) == 1, (
            "send(flush=True) must wait for the first audio frame; got "
            f"{len(chunks)} chunks (KUG-422)"
        )

    @pytest.mark.asyncio
    async def test_send_without_flush_polls_briefly(self):
        """send(text, flush=False) returns quickly when no audio is produced.

        A partial token that does not cross a sentence boundary triggers no
        synthesis; we must not block the caller for the full 30 s.
        """
        ws = DelayedMockWebSocket([], delay=0)  # no messages
        session = StreamingSession(
            api_key="test", tts_url="http://localhost:8000"
        )
        session._ws = ws
        session._is_started = True
        session._config_sent = True

        import time

        t0 = time.perf_counter()
        chunks = [c async for c in session.send("tok", flush=False)]
        elapsed = time.perf_counter() - t0

        assert chunks == []
        assert elapsed < 0.5, (
            f"send(flush=False) polled too long ({elapsed:.2f}s); "
            "open-ended send must not wait the full receive timeout"
        )

    @pytest.mark.asyncio
    async def test_flush_waits_for_first_frame(self):
        """flush() must collect audio even if TTFA exceeds the poll timeout."""
        ws = DelayedMockWebSocket(
            [
                _make_audio_msg(idx=0),
                _make_audio_msg(idx=1),
                json.dumps({"chunk_complete": True, "chunk_id": 0}),
            ],
            delay=0.1,
        )

        session = StreamingSession(
            api_key="test", tts_url="http://localhost:8000"
        )
        session._ws = ws
        session._is_started = True
        session._config_sent = True

        chunks = []
        async for chunk in session.flush():
            chunks.append(chunk)

        assert len(chunks) == 2, (
            "flush() must wait for the first audio frame; tail audio was "
            f"dropped (got {len(chunks)} chunks) — KUG-421"
        )


class _FakeConnectionClosed(Exception):
    """Mimics websockets.exceptions.ConnectionClosed (has .code / .reason)."""

    def __init__(self, code: Optional[int], reason: str = "") -> None:
        super().__init__(f"ConnectionClosed({code}, {reason!r})")
        self.code = code
        self.reason = reason


class _WSRaisingClose:
    """WebSocket stub whose recv() raises a ConnectionClosed with a given code."""

    def __init__(self, code: Optional[int], reason: str = "") -> None:
        self._exc = _FakeConnectionClosed(code, reason)
        self._sent: List[str] = []

    async def recv(self) -> str:
        raise self._exc

    async def send(self, data: str) -> None:
        self._sent.append(data)

    async def close(self) -> None:
        pass


class _WSScripted:
    """WebSocket stub whose recv() replays a script of messages / exceptions."""

    def __init__(self, script: List[Any]) -> None:
        self._script = list(script)
        self._sent: List[str] = []
        self._closed = False

    async def recv(self) -> str:
        if not self._script:
            raise asyncio.TimeoutError()
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def send(self, data: str) -> None:
        self._sent.append(data)

    async def close(self) -> None:
        self._closed = True


def _patch_reconnect(monkeypatch, session, new_ws, *, mark_started: bool):
    """Make the restart replay instant and hand it *new_ws* instead of a real
    socket. Returns the list of reconnects performed."""
    import kugelaudio.streaming as streaming_mod

    monkeypatch.setattr(streaming_mod, "_RESTART_RECONNECT_DELAY_S", 0)
    # The delay comes from the close's retry_after hint, so zero the whole
    # lookup rather than only the fallback constant.
    monkeypatch.setattr(streaming_mod, "_restart_delay_s", lambda err: 0)
    connects: List[Any] = []

    async def fake_connect() -> None:
        connects.append(new_ws)
        session._ws = new_ws
        if mark_started:
            session._is_started = True

    # The socket seam under both connect() and the in-turn recovery reconnect.
    monkeypatch.setattr(session, "_connect_socket", fake_connect)
    return connects


class TestStreamingSessionConnectionClose:
    """Normal WS closes end the stream; error closes raise typed exceptions."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("code", [1000, 1001, 1006, None])
    async def test_normal_close_ends_cleanly(self, code: Optional[int]):
        """Non-error close codes (incl. 1000) should end without raising."""
        session = StreamingSession(
            api_key="test", tts_url="http://localhost:8000"
        )
        session._ws = _WSRaisingClose(code)
        session._is_started = True
        session._config_sent = True

        chunks = []
        async for chunk in session.send("hi"):
            chunks.append(chunk)
        assert chunks == []

    @pytest.mark.asyncio
    async def test_error_close_4001_raises_authentication(self):
        session = StreamingSession(
            api_key="test", tts_url="http://localhost:8000"
        )
        session._ws = _WSRaisingClose(4001, "bad key")
        session._is_started = True
        session._config_sent = True

        with pytest.raises(AuthenticationError):
            async for _ in session.send("hi"):
                pass

    @pytest.mark.asyncio
    @pytest.mark.parametrize("code", [1012, 1013])
    async def test_restart_close_before_audio_replays_turn_transparently(
        self, code: int, monkeypatch
    ):
        """A rolling deploy closes the socket with 1012 (or refuses with
        1013) before any audio for the turn went out: the SDK reconnects
        after the retry delay and resends the turn (config included), so
        the caller sees the audio and no error."""
        session = StreamingSession(
            api_key="test", tts_url="http://localhost:8000"
        )
        old_ws = _WSRaisingClose(code, "server restarting")
        session._ws = old_ws
        session._is_started = True
        new_ws = MockWebSocket(
            [_make_audio_msg(idx=0), json.dumps({"chunk_complete": True})]
        )
        connects = _patch_reconnect(monkeypatch, session, new_ws, mark_started=True)

        chunks = [c async for c in session.send("hi", flush=True)]

        assert len(chunks) == 1
        assert len(connects) == 1
        assert json.loads(old_ws._sent[0])["text"] == "hi"
        replayed = json.loads(new_ws._sent[0])
        assert replayed["text"] == "hi"
        assert replayed["flush"] is True
        assert "sample_rate" in replayed  # config re-sent on the fresh socket
        assert session._ws is new_ws
        assert session._turn_replayed is True

    @pytest.mark.asyncio
    async def test_replay_waits_for_the_close_retry_after_hint(
        self, monkeypatch
    ):
        """The reconnect delay is the server's retry_after, not a constant."""
        import kugelaudio.streaming as streaming_mod

        slept: List[float] = []

        async def fake_sleep(delay: float) -> None:
            slept.append(delay)

        monkeypatch.setattr(streaming_mod.asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(
            streaming_mod,
            "classify_ws_close",
            lambda code, reason=None: ServerRestartingError(retry_after=7),
        )
        session = StreamingSession(
            api_key="test", tts_url="http://localhost:8000"
        )
        session._ws = _WSRaisingClose(1012, "server restarting")
        session._is_started = True
        new_ws = MockWebSocket([json.dumps({"chunk_complete": True})])

        async def fake_connect() -> None:
            session._ws = new_ws
            session._is_started = True

        monkeypatch.setattr(session, "_connect_socket", fake_connect)

        async for _ in session.send("hi", flush=True):
            pass

        assert slept == [7.0]

    def test_restart_delay_falls_back_when_the_close_has_no_hint(self):
        """No retry_after on the wire means the module default applies."""
        import kugelaudio.streaming as streaming_mod

        assert streaming_mod._restart_delay_s(ServerRestartingError()) == 1.0
        assert (
            streaming_mod._restart_delay_s(
                ServerRestartingError(retry_after=None)
            )
            == streaming_mod._RESTART_RECONNECT_DELAY_S
        )
        assert (
            streaming_mod._restart_delay_s(ServerRestartingError(retry_after=-3))
            == 0.0
        )

    @pytest.mark.asyncio
    async def test_replay_resends_every_frame_of_the_turn_in_order(
        self, monkeypatch
    ):
        session = StreamingSession(
            api_key="test", tts_url="http://localhost:8000"
        )
        # First poll idles out (partial text), the flush then hits the close.
        old_ws = _WSScripted(
            [asyncio.TimeoutError(), _FakeConnectionClosed(1012, "restart")]
        )
        session._ws = old_ws
        session._is_started = True
        new_ws = MockWebSocket(
            [_make_audio_msg(idx=0), json.dumps({"chunk_complete": True})]
        )
        _patch_reconnect(monkeypatch, session, new_ws, mark_started=True)

        async for _ in session.send("Hel"):
            pass
        chunks = [c async for c in session.flush()]

        assert len(chunks) == 1
        sent = [json.loads(m) for m in new_ws._sent]
        assert sent[0]["text"] == "Hel" and sent[0]["flush"] is False
        assert "sample_rate" in sent[0] and "sample_rate" not in sent[1]
        assert sent[1] == {"flush": True}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("code", [1012, 1013])
    async def test_restart_close_after_audio_raises_typed_retryable_and_resets(
        self, code: int
    ):
        """Once audio for the turn went out a replay would repeat the start
        of the sentence, so the typed error surfaces and the caller decides
        what to resend on a fresh socket."""
        session = StreamingSession(
            api_key="test", tts_url="http://localhost:8000"
        )
        session._ws = _WSScripted(
            [_make_audio_msg(idx=0), _FakeConnectionClosed(code, "restart")]
        )
        session._is_started = True

        with pytest.raises(ServerRestartingError) as info:
            async for _ in session.send("hi", flush=True):
                pass
        assert isinstance(info.value, KugelAudioConnectionError)
        assert info.value.retry_after == 1
        assert session._ws is None
        assert session._is_started is False
        assert session._config_sent is False
        assert session._turn_frames == []
        assert session._turn_audio is False

    @pytest.mark.asyncio
    async def test_second_restart_in_the_same_turn_raises(self, monkeypatch):
        session = StreamingSession(
            api_key="test", tts_url="http://localhost:8000"
        )
        session._ws = _WSRaisingClose(1012, "restart")
        session._is_started = True
        connects = _patch_reconnect(
            monkeypatch, session, _WSRaisingClose(1012, "restart"), mark_started=True
        )

        with pytest.raises(ServerRestartingError):
            async for _ in session.send("hi", flush=True):
                pass
        assert len(connects) == 1
        assert session._ws is None

    @pytest.mark.asyncio
    async def test_restart_close_during_end_session_is_swallowed(self):
        """end_session() discards in-flight audio by contract, so a restart
        close during its handshake is not an error."""
        session = StreamingSession(
            api_key="test", tts_url="http://localhost:8000"
        )
        session._ws = _WSRaisingClose(1012, "server restarting")
        session._is_started = True
        session._config_sent = True
        session._turn_frames = [{"text": "hi", "flush": False}]

        stats = await session.end_session()
        assert stats == {}
        assert session._ws is None
        assert session._config_sent is False
        assert session._turn_frames == []

    @pytest.mark.asyncio
    async def test_restart_close_during_drain_replays_unspoken_turn(
        self, monkeypatch
    ):
        session = StreamingSession(
            api_key="test", tts_url="http://localhost:8000"
        )
        old_ws = _WSScripted(
            [asyncio.TimeoutError(), _FakeConnectionClosed(1012, "restart")]
        )
        session._ws = old_ws
        session._is_started = True
        async for _ in session.send("hi"):
            pass
        new_ws = MockWebSocket(
            [
                _make_audio_msg(idx=0),
                json.dumps({"session_closed": True, "total_audio_seconds": 0.5}),
            ]
        )
        _patch_reconnect(monkeypatch, session, new_ws, mark_started=True)

        chunks = [c async for c in session.drain()]

        assert len(chunks) == 1
        sent = [json.loads(m) for m in new_ws._sent]
        assert sent[0]["text"] == "hi" and "sample_rate" in sent[0]
        assert sent[1] == {"close": True}
        assert session._session_ended is True
        assert session._session_stats["total_audio_seconds"] == 0.5
        assert session._turn_frames == []

    @pytest.mark.asyncio
    async def test_restart_close_during_idle_drain_ends_quietly(self):
        session = StreamingSession(
            api_key="test", tts_url="http://localhost:8000"
        )
        session._ws = _WSRaisingClose(1012, "server restarting")
        session._is_started = True
        session._config_sent = True

        chunks = [c async for c in session.drain()]
        assert chunks == []
        assert session._ws is None
        assert session._session_ended is False

    @pytest.mark.asyncio
    async def test_restart_close_during_drain_after_audio_raises(self):
        session = StreamingSession(
            api_key="test", tts_url="http://localhost:8000"
        )
        session._ws = _WSScripted(
            [_make_audio_msg(idx=0), _FakeConnectionClosed(1012, "restart")]
        )
        session._is_started = True
        session._config_sent = True

        with pytest.raises(ServerRestartingError):
            async for _ in session.drain():
                pass
        assert session._ws is None

    @pytest.mark.asyncio
    async def test_normal_close_during_end_session_still_swallowed(self):
        session = StreamingSession(
            api_key="test", tts_url="http://localhost:8000"
        )
        session._ws = _WSRaisingClose(1000)
        session._is_started = True
        session._config_sent = True
        await session.end_session()
        assert session._ws is None


class TestStreamingSessionClose:
    """Test close() sends the correct command and reads stats."""

    @pytest.mark.asyncio
    async def test_close_sends_close_and_returns_stats(self):
        ws = MockWebSocket(
            [
                json.dumps(
                    {
                        "session_closed": True,
                        "total_audio_seconds": 3.5,
                        "total_text_chunks": 2,
                        "total_audio_chunks": 10,
                    }
                ),
            ]
        )

        session = StreamingSession(
            api_key="test",
            tts_url="http://localhost:8000",
        )
        session._ws = ws
        session._is_started = True

        stats = await session.close()

        # Should have sent {"close": true}
        assert len(ws._sent) == 1
        assert json.loads(ws._sent[0]) == {"close": True}

        assert stats["session_closed"] is True
        assert stats["total_audio_seconds"] == 3.5
        assert session._ws is None
        assert session._is_started is False

    @pytest.mark.asyncio
    async def test_close_drains_messages_until_session_closed(self):
        """close() must skip intermediate messages (audio, chunk_complete)
        from a final flush and still return the session_closed stats.

        This reproduces KUG-264: when the server flushes remaining text on
        close, it sends audio + chunk_complete before session_closed.
        """
        ws = MockWebSocket(
            [
                # Server flushes remaining buffered text → audio + chunk_complete
                _make_audio_msg(idx=0, num_samples=50),
                json.dumps({"chunk_complete": True, "chunk_id": 0}),
                # Then the actual session_closed stats
                json.dumps(
                    {
                        "session_closed": True,
                        "total_audio_seconds": 2.0,
                        "total_text_chunks": 1,
                        "total_audio_chunks": 3,
                    }
                ),
            ]
        )

        session = StreamingSession(
            api_key="test",
            tts_url="http://localhost:8000",
        )
        session._ws = ws
        session._is_started = True

        stats = await session.close()

        assert stats.get("session_closed") is True
        assert stats["total_audio_seconds"] == 2.0
        assert stats["total_text_chunks"] == 1
        assert session._ws is None


class TestSessionUsage:
    """session_closed exposes typed per-session usage (KUG-1192)."""

    @pytest.mark.asyncio
    async def test_close_populates_typed_usage(self):
        ws = MockWebSocket(
            [
                json.dumps(
                    {
                        "session_closed": True,
                        "total_audio_seconds": 5.4,
                        "usage": {
                            "audio_seconds": 5.4,
                            "characters": 142,
                            "cost_cents": 0.49,
                            "currency": "eur",
                            "model_id": "kugel-3",
                        },
                    }
                ),
            ]
        )
        session = StreamingSession(api_key="t", tts_url="http://localhost:8000")
        session._ws = ws
        session._is_started = True

        await session.close()

        usage = session.last_usage
        assert isinstance(usage, SessionUsage)
        assert usage.audio_seconds == 5.4
        assert usage.characters == 142
        assert usage.cost_cents == 0.49
        assert usage.currency == "eur"
        assert usage.model_id == "kugel-3"
        assert usage.cost_available is True

    @pytest.mark.asyncio
    async def test_cost_unavailable_is_none_not_zero(self):
        ws = MockWebSocket(
            [
                json.dumps(
                    {
                        "session_closed": True,
                        "total_audio_seconds": 2.0,
                        "usage": {
                            "audio_seconds": 2.0,
                            "characters": 30,
                            "cost_cents": None,
                            "cost_unavailable": True,
                            "model_id": "kugel-3",
                        },
                    }
                ),
            ]
        )
        session = StreamingSession(api_key="t", tts_url="http://localhost:8000")
        session._ws = ws
        session._is_started = True

        await session.close()

        usage = session.last_usage
        assert usage is not None
        assert usage.cost_cents is None
        assert usage.cost_available is False
        assert usage.audio_seconds == 2.0

    @pytest.mark.asyncio
    async def test_legacy_server_without_usage_block(self):
        ws = MockWebSocket(
            [json.dumps({"session_closed": True, "total_audio_seconds": 3.0})]
        )
        session = StreamingSession(api_key="t", tts_url="http://localhost:8000")
        session._ws = ws
        session._is_started = True

        await session.close()

        usage = session.last_usage
        assert usage is not None
        assert usage.audio_seconds == 3.0
        assert usage.cost_cents is None
        assert usage.cost_available is False

    @pytest.mark.asyncio
    async def test_multi_per_context_usage_on_context_closed(self):
        # Multi reports usage per context on context_closed (each context is a
        # conversation); session_closed has no usage block.
        ws = MockWebSocket(
            [
                json.dumps({
                    "context_closed": True, "context_id": "narrator",
                    "usage": {"audio_seconds": 4.1, "cost_cents": 0.37,
                              "currency": "eur", "model_id": "kugel-3"},
                }),
                json.dumps({
                    "context_closed": True, "context_id": "character",
                    "usage": {"audio_seconds": 2.0, "cost_cents": None,
                              "cost_unavailable": True, "model_id": "kugel-3"},
                }),
                json.dumps({"session_closed": True, "total_audio_seconds": 6.1}),
            ]
        )
        session = MultiContextSession(api_key="t", tts_url="http://localhost:8000")
        session._ws = ws
        session._is_started = True

        await session.close()

        narrator = session.usage_for("narrator")
        assert isinstance(narrator, SessionUsage)
        assert narrator.audio_seconds == 4.1
        assert narrator.cost_cents == 0.37
        assert narrator.cost_available is True

        character = session.usage_for("character")
        assert character is not None
        assert character.cost_cents is None
        assert character.cost_available is False

        assert set(session.context_usage) == {"narrator", "character"}
        assert session.usage_for("missing") is None


class TestStreamingSessionDrain:
    """drain() must yield tail audio that close() would otherwise discard
    (KUG-421)."""

    @pytest.mark.asyncio
    async def test_drain_yields_tail_audio_then_consumes_session_closed(self):
        ws = MockWebSocket(
            [
                _make_audio_msg(idx=0, num_samples=40),
                _make_audio_msg(idx=1, num_samples=40),
                json.dumps({"chunk_complete": True, "chunk_id": 0}),
                json.dumps(
                    {
                        "session_closed": True,
                        "total_audio_seconds": 1.5,
                        "total_audio_chunks": 2,
                    }
                ),
            ]
        )

        session = StreamingSession(
            api_key="test",
            tts_url="http://localhost:8000",
        )
        session._ws = ws
        session._is_started = True

        chunks: List[AudioChunk] = []
        async for chunk in session.drain():
            chunks.append(chunk)

        assert len(chunks) == 2
        assert json.loads(ws._sent[0]) == {"close": True}
        assert session._session_ended is True
        assert session._session_stats["total_audio_seconds"] == 1.5

    @pytest.mark.asyncio
    async def test_close_after_drain_returns_cached_stats_without_resend(self):
        ws = MockWebSocket(
            [
                _make_audio_msg(idx=0, num_samples=20),
                json.dumps(
                    {
                        "session_closed": True,
                        "total_audio_seconds": 0.5,
                    }
                ),
            ]
        )

        session = StreamingSession(
            api_key="test",
            tts_url="http://localhost:8000",
        )
        session._ws = ws
        session._is_started = True

        async for _ in session.drain():
            pass

        stats = await session.close()

        # drain() sent one {"close": true}; close() must not send another.
        assert [json.loads(m) for m in ws._sent] == [{"close": True}]
        assert stats["session_closed"] is True
        assert stats["total_audio_seconds"] == 0.5
        assert session._ws is None
        assert session._session_ended is False  # reset after consumption

    @pytest.mark.asyncio
    async def test_drain_on_already_closed_session_is_noop(self):
        session = StreamingSession(
            api_key="test",
            tts_url="http://localhost:8000",
        )
        # No ws — drain() should yield nothing and not raise.
        chunks = [chunk async for chunk in session.drain()]
        assert chunks == []


class TestStreamingSessionCancelCurrent:
    """cancel_current() barge-in (KUG-1050)."""

    @pytest.mark.asyncio
    async def test_cancel_sends_cancel_and_waits_for_interrupted(self):
        ws = MockWebSocket(
            [
                # A stale audio frame from the cancelled turn still in flight…
                _make_audio_msg(idx=0, num_samples=20),
                # …then the server acks the barge-in.
                json.dumps({"interrupted": True}),
            ]
        )

        session = StreamingSession(api_key="test", tts_url="http://localhost:8000")
        session._ws = ws
        session._is_started = True
        session._config_sent = True

        await session.cancel_current()

        # Sent exactly the barge-in frame.
        assert [json.loads(m) for m in ws._sent] == [{"cancel": True}]
        # Socket stays open for the next turn, and config will be re-sent.
        assert session._ws is ws
        assert session._is_started is True
        assert session._config_sent is False

    @pytest.mark.asyncio
    async def test_cancel_resolves_on_quiet_timeout_if_no_ack(self):
        # Empty message list → recv() raises TimeoutError (quiet server).
        ws = MockWebSocket([])

        session = StreamingSession(api_key="test", tts_url="http://localhost:8000")
        session._ws = ws
        session._is_started = True
        session._config_sent = True

        await session.cancel_current()  # must not raise

        assert [json.loads(m) for m in ws._sent] == [{"cancel": True}]
        assert session._config_sent is False


# ---------------------------------------------------------------------------
# MultiContextSession tests
# ---------------------------------------------------------------------------


class TestMultiContextStartSession:
    """Test _start_session sends voice_settings correctly."""

    @pytest.mark.asyncio
    async def test_voice_id_nested_in_voice_settings(self):
        """voice_id must be inside voice_settings, not top-level."""
        ws = MockWebSocket(
            [
                json.dumps({"context_created": True, "context_id": "ctx1"}),
            ]
        )

        session = MultiContextSession(
            api_key="test",
            tts_url="http://localhost:8000",
            default_voice_id=42,
            cfg_scale=1.5,
        )
        session._ws = ws

        await session._start_session("ctx1")

        sent = json.loads(ws._sent[0])
        # voice_id should NOT be top-level
        assert "voice_id" not in sent
        # voice_id should be inside voice_settings
        assert "voice_settings" in sent
        assert sent["voice_settings"]["voice_id"] == 42
        assert sent["voice_settings"]["cfg_scale"] == 1.5


    @pytest.mark.asyncio
    async def test_no_voice_settings_when_no_voice(self):
        """If no voice_id is set, voice_settings should be absent."""
        ws = MockWebSocket(
            [
                json.dumps({"context_created": True, "context_id": "ctx1"}),
            ]
        )

        session = MultiContextSession(
            api_key="test",
            tts_url="http://localhost:8000",
            # No default_voice_id
        )
        session._ws = ws

        await session._start_session("ctx1")

        sent = json.loads(ws._sent[0])
        assert "voice_settings" not in sent
        assert "voice_id" not in sent


class TestMultiContextCreateContext:
    """Test create_context also uses voice_settings."""

    @pytest.mark.asyncio
    async def test_create_context_nests_voice_id(self):
        ws = MockWebSocket(
            [
                json.dumps({"context_created": True, "context_id": "ctx2"}),
            ]
        )

        session = MultiContextSession(
            api_key="test",
            tts_url="http://localhost:8000",
            default_voice_id=99,
        )
        session._ws = ws
        session._is_started = True
        session._contexts = {"ctx1"}

        await session.create_context("ctx2", voice_id=77)

        sent = json.loads(ws._sent[0])
        assert "voice_id" not in sent
        assert sent["voice_settings"]["voice_id"] == 77

    @pytest.mark.asyncio
    @pytest.mark.parametrize("project_id", [42, None])
    async def test_session_config_sends_project_id_only_when_set(self, project_id):
        """Session config (incl. project_id) rides the first context's init."""
        ws = MockWebSocket(
            [json.dumps({"context_created": True, "context_id": "ctx1"})]
        )
        session = MultiContextSession(
            api_key="test",
            tts_url="http://localhost:8000",
            language="en",
            project_id=project_id,
        )
        session._ws = ws

        await session._start_session("ctx1")

        sent = json.loads(ws._sent[0])
        if project_id is None:
            assert "project_id" not in sent
        else:
            assert sent["project_id"] == project_id

    @pytest.mark.asyncio
    async def test_create_context_skips_existing(self):
        ws = MockWebSocket([])

        session = MultiContextSession(
            api_key="test",
            tts_url="http://localhost:8000",
        )
        session._ws = ws
        session._is_started = True
        session._contexts = {"ctx1"}

        await session.create_context("ctx1")

        # Should not send anything for an existing context
        assert len(ws._sent) == 0


class TestMultiContextReceiveAudio:
    """Test _receive_audio filters by context_id and handles signals."""

    @pytest.mark.asyncio
    async def test_yields_only_matching_context(self):
        ws = MockWebSocket(
            [
                _make_audio_msg_with_ctx("ctx1", idx=0),
                _make_audio_msg_with_ctx("ctx2", idx=0),  # different context
                _make_audio_msg_with_ctx("ctx1", idx=1),
                json.dumps({"chunk_complete": True, "context_id": "ctx1"}),
            ]
        )

        session = MultiContextSession(
            api_key="test",
            tts_url="http://localhost:8000",
        )
        session._ws = ws
        session._is_started = True
        session._contexts = {"ctx1", "ctx2"}

        chunks = []
        async for chunk in session._receive_audio("ctx1"):
            chunks.append(chunk)

        assert len(chunks) == 2
        # The ctx2 message should have been stashed
        assert len(session._pending_messages) == 1
        assert session._pending_messages[0]["context_id"] == "ctx2"

    @pytest.mark.asyncio
    async def test_flush_waits_for_chunk_complete_despite_audio_gap(self):
        """flush=True waits past short inter-audio gaps until chunk_complete."""
        ws = MockWebSocket(
            [
                (0.0, _make_audio_msg_with_ctx("ctx1", idx=0)),
                (_POLL_TIMEOUT_S * 2, _make_audio_msg_with_ctx("ctx1", idx=1)),
                (0.0, json.dumps({"chunk_complete": True, "context_id": "ctx1"})),
            ]
        )

        session = MultiContextSession(
            api_key="test",
            tts_url="http://localhost:8000",
        )
        session._ws = ws
        session._is_started = True
        session._contexts = {"ctx1"}

        chunks = []
        async for chunk in session.send("ctx1", "hello", flush=True):
            chunks.append(chunk)

        assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_zero_chunk_complete_idle_returns_immediately(self):
        """Callers can opt out of the post-completion multi-chunk wait."""
        ws = MockWebSocket(
            [
                _make_audio_msg_with_ctx("ctx1", idx=0),
                json.dumps({"chunk_complete": True, "context_id": "ctx1"}),
                _make_audio_msg_with_ctx("ctx1", idx=1),
            ]
        )

        session = MultiContextSession(
            api_key="test",
            tts_url="http://localhost:8000",
        )
        session._ws = ws
        session._is_started = True
        session._contexts = {"ctx1"}

        chunks = []
        async for chunk in session.send(
            "ctx1",
            "hello",
            flush=True,
            chunk_complete_idle_timeout=0.0,
        ):
            chunks.append(chunk)

        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_ingress_too_many_contexts_raises_rate_limit_error(self):
        """The /ws/tts/multi context cap is a 429 ingress error frame."""
        ws = MockWebSocket(
            [
                json.dumps(
                    {
                        "error": "Too many concurrent contexts",
                        "error_code": "TOO_MANY_CONTEXTS",
                        "code": 429,
                        "context_id": "ctx1",
                    }
                ),
            ]
        )

        session = MultiContextSession(
            api_key="test",
            tts_url="http://localhost:8000",
        )
        session._ws = ws
        session._is_started = True
        session._contexts = {"ctx1"}

        with pytest.raises(RateLimitError) as exc_info:
            async for _ in session.send("ctx1", "hello", flush=True):
                pass

        assert exc_info.value.status_code == 429
        assert exc_info.value.error_code == "TOO_MANY_CONTEXTS"

    @pytest.mark.asyncio
    async def test_stops_on_context_closed(self):
        ws = MockWebSocket(
            [
                _make_audio_msg_with_ctx("ctx1", idx=0),
                json.dumps({"context_closed": True, "context_id": "ctx1"}),
            ]
        )

        session = MultiContextSession(
            api_key="test",
            tts_url="http://localhost:8000",
        )
        session._ws = ws
        session._is_started = True
        session._contexts = {"ctx1"}

        chunks = []
        async for chunk in session._receive_audio("ctx1", wait_for_close=True):
            chunks.append(chunk)

        assert len(chunks) == 1


class TestMultiContextClose:
    """Test close() sends close_socket and reads stats."""

    @pytest.mark.asyncio
    async def test_close_sends_close_socket(self):
        ws = MockWebSocket(
            [
                json.dumps({"session_closed": True, "total_audio_seconds": 5.0}),
            ]
        )

        session = MultiContextSession(
            api_key="test",
            tts_url="http://localhost:8000",
        )
        session._ws = ws
        session._is_started = True
        session._contexts = {"ctx1"}

        stats = await session.close()

        assert len(ws._sent) == 1
        assert json.loads(ws._sent[0]) == {"close_socket": True}
        assert stats["session_closed"] is True
        assert session._ws is None
        assert session._is_started is False
        assert len(session._contexts) == 0


class _WSSendRaisingClose:
    """WebSocket stub whose send() raises a ConnectionClosed mid-write.

    Models the half-open-connection case: ``recv`` would still work / the
    state still looks OPEN, but the actual ``send`` fails with the raw
    ``ConnectionClosedError("no close frame received or sent")``.
    """

    def __init__(self, code: Optional[int] = None, reason: str = "") -> None:
        self._exc = _FakeConnectionClosed(code, reason)
        self.sends = 0

    async def send(self, data: str) -> None:
        self.sends += 1
        raise self._exc

    async def recv(self) -> str:  # pragma: no cover — send fails first
        raise asyncio.TimeoutError()

    async def close(self) -> None:
        pass


class TestMultiContextSendClassifiesRawClose:
    """A WS drop during the text-send write must surface as a typed error,
    not the raw websockets ConnectionClosedError, and must tear the session
    down so the next send() reconnects. Regression for the Pipecat
    dead-WS-reset bug (KUG pipecat-dead-ws-reset)."""

    @pytest.mark.asyncio
    async def test_send_raw_close_becomes_typed_connection_error(self):
        ws = _WSSendRaisingClose(code=None, reason="no close frame received or sent")
        session = MultiContextSession(api_key="test", tts_url="http://localhost:8000")
        session._ws = ws
        session._is_started = True
        session._contexts = {"ctx1"}

        with pytest.raises(KugelAudioConnectionError):
            async for _ in session.send("ctx1", "hello", flush=True):
                pass

        # Session torn down so the cached WS isn't reused on the next turn.
        assert session._ws is None
        assert session._is_started is False
        assert session.is_alive is False

    @pytest.mark.asyncio
    async def test_send_restart_close_replays_when_nothing_spoken(self, monkeypatch):
        ws = _WSSendRaisingClose(code=1012, reason="server restarting")
        session = MultiContextSession(api_key="test", tts_url="http://localhost:8000")
        session._ws = ws
        session._is_started = True
        session._contexts = {"ctx1"}
        session._context_voices = {"ctx1": None}
        new_ws = MockWebSocket(
            [
                _ctx_created("ctx1"),
                _make_audio_msg_with_ctx("ctx1"),
                json.dumps({"final": True, "context_id": "ctx1"}),
            ]
        )
        _patch_reconnect(monkeypatch, session, new_ws, mark_started=False)

        chunks = [c async for c in session.send("ctx1", "hello", flush=True)]

        assert len(chunks) == 1
        sent = [json.loads(m) for m in new_ws._sent]
        assert sent[0]["context_id"] == "ctx1" and "sample_rate" in sent[0]
        assert sent[1] == {"text": "hello", "context_id": "ctx1", "flush": True}
        assert session._ws is new_ws
        assert session._is_started is True

    @pytest.mark.asyncio
    async def test_send_restart_close_after_audio_classifies_to_server_restarting(self):
        ws = _WSSendRaisingClose(code=1012, reason="server restarting")
        session = MultiContextSession(api_key="test", tts_url="http://localhost:8000")
        session._ws = ws
        session._is_started = True
        session._contexts = {"ctx1"}
        session._turn_audio = {"ctx1"}

        with pytest.raises(ServerRestartingError):
            async for _ in session.send("ctx1", "hello", flush=True):
                pass
        assert session._ws is None
        assert session._is_started is False

    @pytest.mark.asyncio
    async def test_send_error_close_code_classifies_to_typed_error(self):
        # 4001 == WS_CLOSE_UNAUTHORIZED → AuthenticationError, like _receive_audio.
        ws = _WSSendRaisingClose(code=4001, reason="bad key")
        session = MultiContextSession(api_key="test", tts_url="http://localhost:8000")
        session._ws = ws
        session._is_started = True
        session._contexts = {"ctx1"}

        with pytest.raises(AuthenticationError):
            async for _ in session.send("ctx1", "hello", flush=True):
                pass
        assert session._ws is None


def _ctx_created(context_id: str) -> str:
    return json.dumps({"context_created": True, "context_id": context_id})


class TestMultiContextRestartReplay:
    """A rolling-deploy close (1012 / 1013) is absorbed: the SDK reconnects,
    re-creates every context and replays the unspoken turns."""

    @pytest.mark.asyncio
    async def test_restore_does_not_consume_replayed_audio(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class RespondingSocket(MockWebSocket):
            async def send(self, message: str) -> None:
                self._sent.append(message)
                frame = json.loads(message)
                cid = frame["context_id"]
                if frame.get("text") == " ":
                    self._messages.append(_ctx_created(cid))
                elif frame.get("text"):
                    self._messages.extend([
                        _make_audio_msg_with_ctx(cid),
                        json.dumps({"final": True, "context_id": cid}),
                    ])

        session = MultiContextSession(api_key="t", tts_url="http://x")
        session._ws = _WSRaisingClose(1012, "restart")
        session._is_started = True
        session._contexts = {"a", "b"}
        session._context_voices = {"a": 7, "b": 8}
        new_ws = RespondingSocket([])
        _patch_reconnect(monkeypatch, session, new_ws, mark_started=False)

        chunks = [chunk async for chunk in session.send("a", "hello", flush=True)]

        assert len(chunks) == 1
        assert chunks[0].audio == b"\x00\x01" * 100

    @pytest.mark.asyncio
    async def test_recv_restart_before_audio_restores_contexts_and_replays(
        self, monkeypatch
    ):
        s = MultiContextSession(api_key="t", tts_url="http://x", default_voice_id=5)
        s._ws = _WSRaisingClose(1012, "server restarting")
        s._is_started = True
        s._contexts = {"a", "b"}
        s._context_voices = {"a": 7, "b": None}
        s._turn_frames = {"b": [{"text": "pending", "context_id": "b", "flush": False}]}
        new_ws = MockWebSocket(
            [
                _ctx_created("a"),
                _ctx_created("b"),
                _make_audio_msg_with_ctx("a"),
                json.dumps({"final": True, "context_id": "a"}),
            ]
        )
        connects = _patch_reconnect(monkeypatch, s, new_ws, mark_started=False)

        chunks = [c async for c in s.send("a", "hello", flush=True)]

        assert len(chunks) == 1
        assert len(connects) == 1
        sent = [json.loads(m) for m in new_ws._sent]
        # All setup reads finish before any context can produce replay audio.
        assert [m["context_id"] for m in sent] == ["a", "b", "a", "b"]
        assert sent[0]["voice_settings"]["voice_id"] == 7
        assert "sample_rate" in sent[0]  # session config rides on the first context
        assert sent[2] == {"text": "hello", "context_id": "a", "flush": True}
        assert sent[1]["voice_settings"]["voice_id"] == 5
        assert sent[3]["text"] == "pending"
        assert s._contexts == {"a", "b"}
        assert s._context_voices == {"a": 7, "b": None}
        assert s._ws is new_ws
        assert "b" in s._turn_replayed
        assert "a" not in s._turn_replayed  # final ended a's turn

    @pytest.mark.asyncio
    async def test_context_with_audio_in_flight_is_restored_but_not_replayed(
        self, monkeypatch
    ):
        s = MultiContextSession(api_key="t", tts_url="http://x")
        s._ws = _WSRaisingClose(1013, "try again later")
        s._is_started = True
        s._contexts = {"a", "b"}
        s._context_voices = {"a": None, "b": None}
        s._turn_frames = {"b": [{"text": "half spoken", "context_id": "b", "flush": True}]}
        s._turn_audio = {"b"}
        new_ws = MockWebSocket(
            [
                _ctx_created("a"),
                _ctx_created("b"),
                _make_audio_msg_with_ctx("a"),
                json.dumps({"final": True, "context_id": "a"}),
            ]
        )
        _patch_reconnect(monkeypatch, s, new_ws, mark_started=False)

        chunks = [c async for c in s.send("a", "hello", flush=True)]

        assert len(chunks) == 1
        sent = [json.loads(m) for m in new_ws._sent]
        # b keeps its socket-level context but its half-spoken turn is dropped.
        assert [m["context_id"] for m in sent] == ["a", "b", "a"]
        assert sent[2] == {"text": "hello", "context_id": "a", "flush": True}
        assert "b" not in s._turn_replayed
        assert s._turn_frames.get("b") is None

    @pytest.mark.asyncio
    async def test_recv_restart_after_audio_raises(self):
        s = MultiContextSession(api_key="t", tts_url="http://x")
        s._ws = _WSScripted(
            [_make_audio_msg_with_ctx("a"), _FakeConnectionClosed(1012, "restart")]
        )
        s._is_started = True
        s._contexts = {"a"}
        s._context_voices = {"a": None}

        with pytest.raises(ServerRestartingError):
            async for _ in s.send("a", "hello", flush=True):
                pass
        assert s._ws is None
        assert s.is_alive is False
        assert s._turn_frames == {}
        assert s._context_voices == {}

    @pytest.mark.asyncio
    async def test_second_restart_in_the_same_turn_raises(self, monkeypatch):
        s = MultiContextSession(api_key="t", tts_url="http://x")
        s._ws = _WSRaisingClose(1012, "restart")
        s._is_started = True
        s._contexts = {"a"}
        s._context_voices = {"a": None}
        new_ws = _WSScripted([_ctx_created("a"), _FakeConnectionClosed(1012, "restart")])
        connects = _patch_reconnect(monkeypatch, s, new_ws, mark_started=False)

        with pytest.raises(ServerRestartingError):
            async for _ in s.send("a", "hello", flush=True):
                pass
        assert len(connects) == 1
        assert s._ws is None

    @pytest.mark.asyncio
    async def test_idle_restart_close_recovers_on_next_send(self, monkeypatch):
        """Nobody was receiving when the server closed the idle socket for a
        deploy; the next send() reconnects and restores the contexts."""
        s = MultiContextSession(api_key="t", tts_url="http://x")
        dead = _StatefulMockWS(state_name="CLOSED")
        dead.close_code = 1012
        s._ws = dead
        s._is_started = True
        s._contexts = {"a"}
        s._context_voices = {"a": 3}
        new_ws = MockWebSocket(
            [
                _ctx_created("a"),
                _make_audio_msg_with_ctx("a"),
                json.dumps({"final": True, "context_id": "a"}),
            ]
        )
        _patch_reconnect(monkeypatch, s, new_ws, mark_started=False)

        chunks = [c async for c in s.send("a", "hi", flush=True)]

        assert len(chunks) == 1
        sent = [json.loads(m) for m in new_ws._sent]
        assert sent[0]["voice_settings"]["voice_id"] == 3
        assert sent[1] == {"text": "hi", "context_id": "a", "flush": True}
        assert s.is_alive is True

    @pytest.mark.asyncio
    async def test_close_context_forgets_replay_state(self):
        s = MultiContextSession(api_key="t", tts_url="http://x")
        s._ws = MockWebSocket([json.dumps({"context_closed": True, "context_id": "a"})])
        s._is_started = True
        s._contexts = {"a"}
        s._context_voices = {"a": 3}
        s._turn_frames = {"a": [{"text": "x", "context_id": "a", "flush": False}]}
        s._turn_audio = {"a"}

        async for _ in s.close_context("a"):
            pass
        assert s._context_voices == {}
        assert s._turn_frames == {}
        assert s._turn_audio == set()


# ---------------------------------------------------------------------------
# Word timestamp tests
# ---------------------------------------------------------------------------


def _make_word_timestamps_msg(
    words: list[dict],
    chunk_id: int | None = None,
    context_id: str | None = None,
) -> str:
    """Create a word_timestamps JSON message like the server sends."""
    msg: dict[str, Any] = {"word_timestamps": words}
    if chunk_id is not None:
        msg["chunk_id"] = chunk_id
    if context_id is not None:
        msg["context_id"] = context_id
    return json.dumps(msg)


_SAMPLE_TIMESTAMPS = [
    {"word": "Hello", "start_ms": 0, "end_ms": 300,
     "char_start": 0, "char_end": 5, "score": 0.95},
    {"word": "world", "start_ms": 350, "end_ms": 700,
     "char_start": 6, "char_end": 11, "score": 0.90},
]


class TestStreamingSessionWordTimestamps:
    """Test that StreamingSession handles word_timestamps messages."""

    @pytest.mark.asyncio
    async def test_word_timestamps_stored(self):
        """word_timestamps message should populate last_word_timestamps."""
        from kugelaudio.models import WordTimestamp

        messages = [
            _make_audio_msg(idx=0),
            _make_word_timestamps_msg(_SAMPLE_TIMESTAMPS, chunk_id=0),
            json.dumps({"chunk_complete": True, "chunk_id": 0}),
            json.dumps({"session_closed": True}),
        ]
        ws = MockWebSocket(messages)
        session = StreamingSession(
            api_key="test",
            tts_url="http://localhost:8000",
            config=StreamConfig(word_timestamps=True),
        )
        session._ws = ws
        session._is_started = True

        async for _ in session.send("Hello world"):
            pass

        assert len(session.last_word_timestamps) == 2
        assert session.last_word_timestamps[0].word == "Hello"
        assert session.last_word_timestamps[0].start_ms == 0
        assert session.last_word_timestamps[0].end_ms == 300
        assert session.last_word_timestamps[1].word == "world"

    @pytest.mark.asyncio
    async def test_word_timestamps_callback_invoked(self):
        """on_word_timestamps callback should be called with parsed list."""
        from kugelaudio.models import WordTimestamp

        received: list[list[WordTimestamp]] = []

        messages = [
            _make_audio_msg(idx=0),
            _make_word_timestamps_msg(_SAMPLE_TIMESTAMPS, chunk_id=0),
            json.dumps({"chunk_complete": True, "chunk_id": 0}),
            json.dumps({"session_closed": True}),
        ]
        ws = MockWebSocket(messages)
        session = StreamingSession(
            api_key="test",
            tts_url="http://localhost:8000",
            config=StreamConfig(word_timestamps=True),
            on_word_timestamps=lambda ts: received.append(ts),
        )
        session._ws = ws
        session._is_started = True

        async for _ in session.send("Hello world"):
            pass

        assert len(received) == 1
        assert len(received[0]) == 2
        assert received[0][0].word == "Hello"


class TestMultiContextWordTimestamps:
    """Test that MultiContextSession handles word_timestamps per context."""

    @pytest.mark.asyncio
    async def test_word_timestamps_per_context(self):
        """word_timestamps should be stored per context_id via _receive_audio."""
        from kugelaudio.models import WordTimestamp

        ts_ctx_a = [
            {"word": "Hello", "start_ms": 0, "end_ms": 300,
             "char_start": 0, "char_end": 5, "score": 0.9},
        ]

        messages = [
            _make_audio_msg_with_ctx("ctx-a", idx=0),
            _make_word_timestamps_msg(ts_ctx_a, chunk_id=0, context_id="ctx-a"),
            json.dumps({"chunk_complete": True, "chunk_id": 0,
                        "context_id": "ctx-a"}),
        ]
        ws = MockWebSocket(messages)
        session = MultiContextSession(
            api_key="test",
            tts_url="http://localhost:8000",
            word_timestamps=True,
        )
        session._ws = ws
        session._is_started = True
        session._contexts.add("ctx-a")

        # Use send() which internally calls _receive_audio
        chunks = []
        async for chunk in session.send("ctx-a", "Hello", flush=True):
            chunks.append(chunk)

        assert len(chunks) == 1  # one audio chunk
        ts_a = session.get_word_timestamps("ctx-a")
        assert len(ts_a) == 1
        assert ts_a[0].word == "Hello"
        assert ts_a[0].start_ms == 0
        assert ts_a[0].end_ms == 300

    @pytest.mark.asyncio
    async def test_word_timestamps_callback_multi(self):
        """on_word_timestamps callback with (context_id, timestamps)."""
        from kugelaudio.models import WordTimestamp

        received: list[tuple[str, list[WordTimestamp]]] = []

        ts_data = [
            {"word": "Bye", "start_ms": 0, "end_ms": 200,
             "char_start": 0, "char_end": 3, "score": 0.85},
        ]

        messages = [
            _make_audio_msg_with_ctx("ctx-x", idx=0),
            _make_word_timestamps_msg(ts_data, chunk_id=0, context_id="ctx-x"),
            json.dumps({"chunk_complete": True, "chunk_id": 0,
                        "context_id": "ctx-x"}),
        ]
        ws = MockWebSocket(messages)
        session = MultiContextSession(
            api_key="test",
            tts_url="http://localhost:8000",
            word_timestamps=True,
            on_word_timestamps=lambda cid, ts: received.append((cid, ts)),
        )
        session._ws = ws
        session._is_started = True
        session._contexts.add("ctx-x")

        async for _ in session.send("ctx-x", "Bye", flush=True):
            pass

        assert len(received) == 1
        assert received[0][0] == "ctx-x"
        assert received[0][1][0].word == "Bye"


# ---------------------------------------------------------------------------
# Dead-WS detection / typed-error / reset behavior
# ---------------------------------------------------------------------------


class _ConnectionClosedFake(Exception):
    """Stand-in for websockets.ConnectionClosedError. The SDK detects it by
    string-matching the type name, so the only requirement is the class
    being named exactly 'ConnectionClosedError'."""


_ConnectionClosedFake.__name__ = "ConnectionClosedError"


class _StatefulMockWS:
    """MockWebSocket variant with a settable ``state`` (mirrors websockets API)."""

    def __init__(self, state_name: str = "OPEN") -> None:
        from types import SimpleNamespace
        self.state = SimpleNamespace(name=state_name)
        self.close_code: Optional[int] = None
        self._sent: list[str] = []

    async def recv(self) -> str:
        raise asyncio.TimeoutError()

    async def send(self, data: str) -> None:
        self._sent.append(data)


class TestMultiContextDeadWS:
    """Dead-WS detection, typed errors, and state reset for MultiContextSession."""

    @pytest.mark.asyncio
    async def test_send_on_dead_ws_raises_typed_error(self):
        """send() must NOT call .send() on a closed WS — that would raise the
        opaque ConnectionClosedError('no close frame received or sent') the
        user reported. It should raise a typed KugelAudioConnectionError."""
        s = MultiContextSession(api_key="t", tts_url="http://x")
        s._ws = _StatefulMockWS(state_name="CLOSED")
        s._is_started = True
        s._contexts = {"ctx"}

        with pytest.raises(KugelAudioConnectionError) as exc_info:
            async for _ in s.send("ctx", "hello", flush=True):
                pass

        # Message must hint at reconnection so users know what to do
        assert "Reconnect" in str(exc_info.value) or "connect" in str(exc_info.value)
        # State must be cleared so a follow-up reconnect starts clean
        assert s._ws is None
        assert s._is_started is False

    @pytest.mark.asyncio
    async def test_receive_audio_on_connection_closed_resets_and_raises(self):
        """When the WS dies mid-stream, _receive_audio must drop session state
        AND raise a typed error — not silently 'break' (which leaves the dead
        WS in place for the next send to crash on)."""

        class _DyingMockWS:
            def __init__(self) -> None:
                self.state = type("S", (), {"name": "OPEN"})()
                self.close_code: Optional[int] = None
                self._sent: list[str] = []

            async def recv(self) -> str:
                raise _ConnectionClosedFake()

            async def send(self, data: str) -> None:
                self._sent.append(data)

        ws = _DyingMockWS()
        s = MultiContextSession(api_key="t", tts_url="http://x")
        s._ws = ws
        s._is_started = True
        s._contexts = {"ctx"}

        with pytest.raises(KugelAudioConnectionError):
            async for _ in s._receive_audio("ctx"):
                pass

        # Critical: state must be reset, otherwise the next send() crashes
        # inside ``await self._ws.send(...)`` on the dead handle.
        assert s._ws is None
        assert s._is_started is False
        assert s._contexts == set()

    @pytest.mark.asyncio
    async def test_flush_send_connection_closed_resets_and_raises(self):
        """flush() should use the same typed send path as send()."""

        class ConnectionClosedError(Exception):
            pass

        class _DroppingSendWS:
            state = type("S", (), {"name": "OPEN"})()
            close_code: Optional[int] = None

            async def send(self, _data: str) -> None:
                raise ConnectionClosedError("no close frame received or sent")

            async def recv(self) -> str:
                raise asyncio.TimeoutError()

        s = MultiContextSession(api_key="t", tts_url="http://x")
        s._ws = _DroppingSendWS()
        s._is_started = True
        s._contexts = {"ctx"}

        with pytest.raises(KugelAudioConnectionError):
            async for _ in s.flush("ctx"):
                pass

        assert s._ws is None
        assert s._is_started is False
        assert s._contexts == set()

    @pytest.mark.asyncio
    async def test_close_context_send_connection_closed_resets_and_raises(self):
        """close_context() should not leak raw websockets close exceptions."""

        class ConnectionClosedError(Exception):
            pass

        class _DroppingSendWS:
            state = type("S", (), {"name": "OPEN"})()
            close_code: Optional[int] = None

            async def send(self, _data: str) -> None:
                raise ConnectionClosedError("no close frame received or sent")

            async def recv(self) -> str:
                raise asyncio.TimeoutError()

        s = MultiContextSession(api_key="t", tts_url="http://x")
        s._ws = _DroppingSendWS()
        s._is_started = True
        s._contexts = {"ctx"}

        with pytest.raises(KugelAudioConnectionError):
            async for _ in s.close_context("ctx"):
                pass

        assert s._ws is None
        assert s._is_started is False
        assert s._contexts == set()


class TestMultiContextCloseContextBargeIn:
    """closeContext immediate barge-in on /ws/tts/multi (KUG-1050)."""

    @pytest.mark.asyncio
    async def test_close_context_immediate_sends_barge_in_flag(self):
        ws = MockWebSocket(
            [json.dumps({"context_closed": True, "context_id": "ctx1"})]
        )
        session = MultiContextSession(api_key="t", tts_url="http://localhost:8000")
        session._ws = ws
        session._is_started = True
        session._contexts = {"ctx1"}

        _ = [c async for c in session.close_context("ctx1", immediate=True)]

        assert json.loads(ws._sent[0]) == {
            "close_context": True,
            "context_id": "ctx1",
            "immediate": True,
        }
        assert "ctx1" not in session._contexts


class TestFinalFrame:
    """``final`` end-of-audio frames (KUG-1238, ElevenLabs isFinal parity).

    /ws/tts/stream sends ``{"final": true, ...turn stats}`` after the last
    audio frame of every gracefully completed turn (before session_closed);
    /ws/tts/multi sends ``{"final": true, "context_id": ...}`` once all audio
    admitted before a client flush has been delivered, and again before a
    graceful ``context_closed``.
    """

    @pytest.mark.asyncio
    async def test_stream_final_is_captured_and_session_closed_terminates(self):
        ws = MockWebSocket(
            [
                _make_audio_msg(idx=0),
                json.dumps(
                    {
                        "final": True,
                        "total_audio_seconds": 1.2,
                        "total_text_chunks": 1,
                        "total_audio_chunks": 1,
                    }
                ),
                json.dumps({"session_closed": True, "total_audio_seconds": 1.2}),
            ]
        )
        session = StreamingSession(api_key="t", tts_url="http://localhost:8000")
        session._ws = ws
        session._is_started = True
        session._config_sent = True

        assert session.last_final is None
        chunks = [c async for c in session.send("Hello.", flush=True)]

        assert len(chunks) == 1
        assert session.last_final is not None
        assert session.last_final["total_audio_seconds"] == 1.2
        # session_closed was still consumed as the terminal frame.
        assert session._session_stats.get("session_closed") is True

    @pytest.mark.asyncio
    async def test_multi_flush_exits_deterministically_on_final(self):
        # A poison error frame AFTER final proves the loop breaks on final
        # instead of polling on (and consuming) later frames.
        ws = MockWebSocket(
            [
                json.dumps(
                    {"generation_started": True, "context_id": "a", "chunk_id": 0}
                ),
                _make_audio_msg_with_ctx("a", idx=0),
                json.dumps(
                    {"chunk_complete": True, "chunk_id": 0, "context_id": "a"}
                ),
                json.dumps({"final": True, "context_id": "a"}),
                json.dumps({"error": "must never be consumed"}),
            ]
        )
        session = MultiContextSession(api_key="t", tts_url="http://localhost:8000")
        session._ws = ws
        session._is_started = True
        session._contexts = {"a"}

        chunks = [c async for c in session.send("a", "Hello.", flush=True)]

        assert len(chunks) == 1
        # The poison frame is still queued — final ended the receive loop.
        assert len(ws._messages) == 1

    @pytest.mark.asyncio
    async def test_multi_close_drain_consumes_final_then_context_closed(self):
        ws = MockWebSocket(
            [
                _make_audio_msg_with_ctx("a", idx=0),
                json.dumps({"final": True, "context_id": "a"}),
                json.dumps({"context_closed": True, "context_id": "a"}),
            ]
        )
        session = MultiContextSession(api_key="t", tts_url="http://localhost:8000")
        session._ws = ws
        session._is_started = True
        session._contexts = {"a"}

        chunks = [c async for c in session.close_context("a")]

        assert len(chunks) == 1
        # final did NOT end the close drain early; context_closed did.
        assert not ws._messages
        assert "a" not in session._contexts


# ---------------------------------------------------------------------------
# update_settings — mid-connection generation-param changes (KUG-1166)
# ---------------------------------------------------------------------------


class TestUpdateSettings:
    """SDK-side update_settings: wire shape, ack handling, local-config sync."""

    @pytest.mark.asyncio
    async def test_stream_sends_message_and_returns_effective(self):
        ws = MockWebSocket(
            [json.dumps({"settings_updated": True,
                         "settings": {"cfg_scale": 1.5, "speed": 1.1}})]
        )
        session = StreamingSession(
            api_key="test", tts_url="http://localhost:8000",
            config=StreamConfig(voice_id=7, cfg_scale=2.0),
        )
        session._ws = ws
        session._is_started = True

        effective = await session.update_settings(cfg_scale=1.5, speed=1.1)

        assert len(ws._sent) == 1
        sent = json.loads(ws._sent[0])
        assert sent == {"update_settings": {"cfg_scale": 1.5, "speed": 1.1}}
        assert effective == {"cfg_scale": 1.5, "speed": 1.1}
        # Local config kept in sync so a later end_session()+send() re-send
        # does not revert the change.
        assert session._config.cfg_scale == 1.5
        assert session._config.speed == 1.1

    @pytest.mark.asyncio
    async def test_stream_only_sends_provided_fields(self):
        ws = MockWebSocket([json.dumps({"settings_updated": True, "settings": {}})])
        session = StreamingSession(api_key="t", tts_url="http://localhost:8000")
        session._ws = ws
        session._is_started = True

        await session.update_settings(temperature=0.0)
        sent = json.loads(ws._sent[0])
        assert sent == {"update_settings": {"temperature": 0.0}}

    @pytest.mark.asyncio
    async def test_stream_empty_update_raises_valueerror(self):
        session = StreamingSession(api_key="t", tts_url="http://localhost:8000")
        session._ws = MockWebSocket([])
        session._is_started = True
        with pytest.raises(ValueError, match="at least one parameter"):
            await session.update_settings()

    @pytest.mark.asyncio
    async def test_stream_server_rejection_raises(self):
        ws = MockWebSocket([json.dumps(
            {"error": "Invalid settings update: cfg_scale: out of range",
             "error_code": "VALIDATION_ERROR", "code": 400}
        )])
        session = StreamingSession(
            api_key="t",
            tts_url="http://localhost:8000",
            config=StreamConfig(voice_id=7, cfg_scale=2.0),
        )
        session._ws = ws
        session._is_started = True
        with pytest.raises(KugelAudioError):
            await session.update_settings(cfg_scale=99)
        assert session._config.cfg_scale == 2.0

    @pytest.mark.asyncio
    async def test_stream_discards_stray_audio_before_ack(self):
        ws = MockWebSocket([
            _make_audio_msg(idx=0),  # leftover from a draining turn
            json.dumps({"settings_updated": True, "settings": {"speed": 1.2}}),
        ])
        session = StreamingSession(api_key="t", tts_url="http://localhost:8000")
        session._ws = ws
        session._is_started = True
        effective = await session.update_settings(speed=1.2)
        assert effective == {"speed": 1.2}

    @pytest.mark.asyncio
    async def test_multi_session_scoped_ack(self):
        ws = MockWebSocket([json.dumps(
            {"settings_updated": True, "settings": {"cfg_scale": 2.5, "normalize": False}}
        )])
        session = MultiContextSession(
            api_key="t", tts_url="http://localhost:8000",
            default_voice_id=7, cfg_scale=1.0,
        )
        session._ws = ws

        effective = await session.update_settings(cfg_scale=2.5, normalize=False)

        sent = json.loads(ws._sent[0])
        assert sent == {"update_settings": {"cfg_scale": 2.5, "normalize": False}}
        assert effective == {"cfg_scale": 2.5, "normalize": False}
        # Contexts created after the update inherit the new per-context cfg_scale.
        assert session._cfg_scale == 2.5
        assert session._normalize is False

    @pytest.mark.asyncio
    async def test_multi_server_rejection_does_not_change_local_defaults(self):
        ws = MockWebSocket([json.dumps(
            {"error": "Invalid settings update: cfg_scale: out of range",
             "error_code": "VALIDATION_ERROR", "code": 400}
        )])
        # cfg_scale 2.0 is inside the client-side clamp band [1.2, 2.5] so the
        # constructor stores it verbatim — a value below 1.2 would be clamped
        # up and mask whether the rejected update poisoned the local default.
        session = MultiContextSession(
            api_key="t", tts_url="http://localhost:8000",
            default_voice_id=7, cfg_scale=2.0,
        )
        session._ws = ws

        with pytest.raises(KugelAudioError):
            await session.update_settings(cfg_scale=99)
        assert session._cfg_scale == 2.0

    def test_sync_wrapper_update_settings(self):
        ws = MockWebSocket([json.dumps(
            {"settings_updated": True, "settings": {"temperature": 0.2}}
        )])
        session = StreamingSession(api_key="t", tts_url="http://localhost:8000")
        session._ws = ws
        session._is_started = True
        sync = StreamingSessionSync(session)

        effective = sync.update_settings(temperature=0.2)
        assert effective == {"temperature": 0.2}
        assert json.loads(ws._sent[0]) == {"update_settings": {"temperature": 0.2}}
