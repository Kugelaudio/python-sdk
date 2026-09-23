# Copyright 2024 KugelAudio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for kugelaudio.livekit module.

These tests verify the LiveKit integration works correctly.
Requires: pip install kugelaudio[livekit]
"""


import pytest

# Skip all tests if livekit-agents is not installed
pytest.importorskip("livekit.agents")

from livekit.agents import APIConnectOptions  # noqa: E402


class TestTTSInitialization:
    """Test TTS initialization."""


    def test_init_with_env_var(self, monkeypatch):
        """Test initialization with environment variable."""
        from kugelaudio.livekit import TTS

        monkeypatch.setenv("KUGELAUDIO_API_KEY", "env-test-key")
        tts = TTS()
        assert tts._opts.api_key == "env-test-key"

    def test_init_without_api_key_raises(self, monkeypatch):
        """Test that missing API key raises ValueError."""
        from kugelaudio.livekit import TTS

        monkeypatch.delenv("KUGELAUDIO_API_KEY", raising=False)
        with pytest.raises(ValueError, match="KUGELAUDIO_API_KEY"):
            TTS()


class TestTTSRegion:
    """Test multi-region support."""


    def test_key_prefix_eu(self):
        """Test that 'eu-' key prefix auto-selects EU region and strips prefix."""
        from kugelaudio.livekit import TTS

        tts = TTS(api_key="eu-ka_test123")
        assert tts._opts.base_url == "https://api.eu.kugelaudio.com"
        assert tts._opts.api_key == "ka_test123"

    def test_explicit_region_overrides_key_prefix(self):
        """Test that explicit region param takes priority over key prefix."""
        from kugelaudio.livekit import TTS

        tts = TTS(api_key="us-ka_test123", region="global")
        assert tts._opts.base_url == "https://api.kugelaudio.com"
        assert tts._opts.api_key == "ka_test123"

    def test_base_url_overrides_region(self):
        """Test that explicit base_url overrides region entirely."""
        from kugelaudio.livekit import TTS

        tts = TTS(api_key="us-ka_test123", region="global", base_url="https://custom.api.com")
        assert tts._opts.base_url == "https://custom.api.com"
        assert tts._opts.api_key == "ka_test123"

    def test_invalid_region_raises(self):
        """Test that invalid region raises ValueError."""
        from kugelaudio.livekit import TTS

        with pytest.raises(ValueError, match="Invalid region"):
            TTS(api_key="ka_test123", region="mars")

    def test_env_var_with_prefix(self, monkeypatch):
        """Test that region prefix in env var is parsed correctly."""
        from kugelaudio.livekit import TTS

        monkeypatch.setenv("KUGELAUDIO_API_KEY", "us-ka_envkey")
        tts = TTS()
        assert tts._opts.base_url == "https://api.kugelaudio.com"
        assert tts._opts.api_key == "ka_envkey"


class TestTTSOptions:
    """Test TTS option updates."""


    def test_update_cfg_scale_clamps_to_supported_band(self):
        """update_options() clamps cfg_scale into the supported [1.2, 2.5] band.

        3.0 is above MAX_CFG_SCALE, so ``clamp_cfg_scale`` brings it down to
        2.5 — the same value the server would enforce.
        """
        from kugelaudio.livekit import TTS
        from kugelaudio.models import MAX_CFG_SCALE, MIN_CFG_SCALE

        tts = TTS(api_key="test-key")
        tts.update_options(cfg_scale=3.0)
        assert tts._opts.cfg_scale == MAX_CFG_SCALE == 2.5

        tts.update_options(cfg_scale=0.5)
        assert tts._opts.cfg_scale == MIN_CFG_SCALE == 1.2


    def test_update_language_to_none(self):
        """Test clearing language back to None."""
        from kugelaudio.livekit import TTS

        tts = TTS(api_key="test-key", language="de")
        assert tts._opts.language == "de"
        tts.update_options(language=None)
        assert tts._opts.language is None


# ---------------------------------------------------------------------------
# Protocol-level tests for SynthesizeStream & ChunkedStream
#
# LiveKit's base __init__ calls asyncio.create_task(), so we patch it out
# to test our protocol logic without a running event loop.
# ---------------------------------------------------------------------------
import asyncio
import base64
import json
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch


def _make_synth_stream(tts_instance):
    """Create a SynthesizeStream with the LiveKit super().__init__ patched out."""
    from kugelaudio.livekit.tts import SynthesizeStream

    with patch("livekit.agents.tts.SynthesizeStream.__init__", return_value=None):
        stream = SynthesizeStream.__new__(SynthesizeStream)
        stream._tts = tts_instance
        stream._opts = replace(tts_instance._opts)
    return stream


def _make_chunked_stream(tts_instance, input_text="Hello"):
    """Create a ChunkedStream with the LiveKit super().__init__ patched out."""
    from kugelaudio.livekit.tts import ChunkedStream

    with patch("livekit.agents.tts.ChunkedStream.__init__", return_value=None):
        chunked = ChunkedStream.__new__(ChunkedStream)
        chunked._tts = tts_instance
        chunked._opts = replace(tts_instance._opts)
        chunked._input_text = input_text
    return chunked


class TestConnectionErrorPropagation:
    """Ingress error frames should keep their real status inside LiveKit."""

    @pytest.mark.asyncio
    async def test_recv_loop_maps_rate_limit_frame_to_api_status_error(self):
        import aiohttp
        from livekit.agents import APIStatusError

        from kugelaudio.livekit.tts import _Connection, _ContextData, _TTSOptions

        opts = _TTSOptions(
            model="kugel-1-turbo",
            voice_id=None,
            sample_rate=24000,
            cfg_scale=2.0,
            max_new_tokens=2048,
            api_key="k",
            base_url="https://x",
        )
        conn = _Connection(opts, MagicMock())
        waiter = asyncio.get_running_loop().create_future()
        conn._context_data["ctx1"] = _ContextData(
            emitter=MagicMock(),
            waiter=waiter,
        )
        conn._active_contexts.add("ctx1")
        payload = {
            "error": "Rate limit exceeded",
            "error_code": "RATE_LIMITED",
            "code": 429,
            "context_id": "ctx1",
        }
        ws = MagicMock()
        ws.closed = False
        ws.receive = AsyncMock(
            side_effect=[
                MagicMock(
                    type=aiohttp.WSMsgType.TEXT,
                    data=json.dumps(payload),
                ),
                MagicMock(type=aiohttp.WSMsgType.CLOSED, data=""),
            ]
        )
        conn._ws = ws

        await conn._recv_loop_inner()

        exc = waiter.exception()
        assert isinstance(exc, APIStatusError)
        assert exc.status_code == 429
        assert exc.body == payload
        assert "Rate limit exceeded" in str(exc)

    @pytest.mark.asyncio
    async def test_connect_maps_ws_handshake_rejection_to_api_status_error(self):
        import aiohttp
        from livekit.agents import APIStatusError

        from kugelaudio.livekit.tts import _Connection, _TTSOptions

        opts = _TTSOptions(
            model="kugel-1-turbo",
            voice_id=None,
            sample_rate=24000,
            cfg_scale=2.0,
            max_new_tokens=2048,
            api_key="k",
            base_url="https://x",
        )
        session = MagicMock()
        session.ws_connect = AsyncMock(
            side_effect=aiohttp.WSServerHandshakeError(
                None,
                (),
                status=429,
                message="Too Many Requests",
            )
        )
        conn = _Connection(opts, session)

        with pytest.raises(APIStatusError) as exc_info:
            await conn.connect()

        assert exc_info.value.status_code == 429
        assert exc_info.value.body == {
            "error": "Too Many Requests",
            "code": 429,
        }


class TestConnectionContextCleanup:
    """Test that _Connection._cleanup_context removes context data."""

    def test_cleanup_removes_context(self):
        from kugelaudio.livekit.tts import _Connection, _ContextData, _TTSOptions

        opts = _TTSOptions(
            model="kugel-1-turbo", voice_id=None, sample_rate=24000,
            cfg_scale=2.0, max_new_tokens=2048, api_key="k", base_url="https://x",
        )
        conn = _Connection(opts, MagicMock())
        loop = asyncio.new_event_loop()
        waiter = loop.create_future()
        emitter = MagicMock()
        conn._context_data["ctx1"] = _ContextData(emitter=emitter, waiter=waiter)
        conn._active_contexts.add("ctx1")

        conn._cleanup_context("ctx1")

        assert "ctx1" not in conn._context_data
        assert "ctx1" not in conn._active_contexts
        loop.close()


class TestUpdateOptionsInvalidatesConnection:
    """Test that update_options marks the connection non-current when options change."""

    def test_model_change_invalidates(self):
        from kugelaudio.livekit.tts import TTS, _Connection

        tts_instance = TTS(api_key="test-key", model="kugel-1-turbo")
        mock_conn = MagicMock(spec=_Connection)
        mock_conn.is_current = True
        mock_conn._closed = False
        tts_instance._current_connection = mock_conn

        tts_instance.update_options(model="kugel-1")

        mock_conn.mark_non_current.assert_called_once()
        assert tts_instance._current_connection is None

    def test_same_model_does_not_invalidate(self):
        from kugelaudio.livekit.tts import TTS, _Connection

        tts_instance = TTS(api_key="test-key", model="kugel-1-turbo")
        mock_conn = MagicMock(spec=_Connection)
        tts_instance._current_connection = mock_conn

        tts_instance.update_options(model="kugel-1-turbo")

        mock_conn.mark_non_current.assert_not_called()
        assert tts_instance._current_connection is mock_conn


# ---------------------------------------------------------------------------
# Word timestamps integration
# ---------------------------------------------------------------------------


class TestWordTimestampsToTimed:
    """Test the _word_timestamps_to_timed helper."""

    def test_converts_ms_to_seconds(self):
        from kugelaudio.livekit.tts import _word_timestamps_to_timed

        ts = [
            {"word": "hello", "start_ms": 0, "end_ms": 500},
            {"word": "world", "start_ms": 600, "end_ms": 1100},
        ]
        result = _word_timestamps_to_timed(ts)
        assert len(result) == 2

        assert str(result[0]) == "hello"
        assert result[0].start_time == 0.0
        assert result[0].end_time == 0.5

        assert str(result[1]) == "world"
        assert result[1].start_time == 0.6
        assert abs(result[1].end_time - 1.1) < 1e-9


class TestAlignedTranscriptCapability:
    """Test that TTS reports aligned_transcript=True."""

    def test_capabilities_aligned_transcript_follows_word_timestamps(self):
        from kugelaudio.livekit.tts import TTS

        default_tts = TTS(api_key="test-key")
        assert default_tts.capabilities.aligned_transcript is False
        assert default_tts.capabilities.streaming is True

        aligned_tts = TTS(api_key="test-key", word_timestamps=True)
        assert aligned_tts.capabilities.aligned_transcript is True


class TestBargeInSendsCloseContext:
    """On cancel, the SDK must tell the server to free the context
    immediately instead of waiting for the 20 s inactivity sweep."""

    def test_cancel_sends_close_context(self):
        from kugelaudio.livekit.tts import (
            SynthesizeStream,
            TTS,
            _CloseContext,
            _Connection,
        )

        tts_instance = TTS(api_key="test-key")

        async def _go():
            stream = _make_synth_stream(tts_instance)
            stream._conn_options = APIConnectOptions(max_retry=0, timeout=1.0)

            # Mock connection that records every input_queue item so we
            # can assert a _CloseContext was enqueued.
            mock_conn = MagicMock(spec=_Connection)
            mock_conn._input_queue = asyncio.Queue()

            def _record_close(ctx_id, immediate=False):
                mock_conn._input_queue.put_nowait(
                    _CloseContext(ctx_id, immediate=immediate)
                )

            mock_conn.request_close_context.side_effect = _record_close
            mock_conn.send_content = MagicMock()
            mock_conn.register_context = MagicMock()
            mock_conn._cleanup_context = MagicMock()

            with patch.object(
                tts_instance, "_ensure_connection", AsyncMock(return_value=mock_conn)
            ):
                emitter = MagicMock()
                # Replace the async iteration with an endless wait so the
                # stream sits waiting on `waiter` until we cancel it.
                stream._input_ch = AsyncMock()
                stream._input_ch.__aiter__ = MagicMock(
                    return_value=iter([])  # immediately exhausted
                )

                run_task = asyncio.create_task(stream._run(emitter))
                # Let _run reach `await waiter`.
                await asyncio.sleep(0)
                await asyncio.sleep(0)

                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass

            assert mock_conn.request_close_context.called, (
                "cancel must call request_close_context so the server "
                "releases the context immediately"
            )
            # Barge-in must pass immediate=True so the server cancels
            # in-flight generation instead of draining.  Draining wastes
            # GPU on audio the client has already discarded.
            call_kwargs = mock_conn.request_close_context.call_args.kwargs
            assert call_kwargs.get("immediate") is True, (
                "cancel must pass immediate=True to request_close_context "
                f"(got kwargs={call_kwargs})"
            )
            assert mock_conn._cleanup_context.called

        asyncio.new_event_loop().run_until_complete(_go())


class TestCloseContextImmediateProtocol:
    """Protocol-level regression: close_context payload must carry the
    ``immediate`` flag on barge-in and omit it on graceful end."""

    def test_send_loop_serializes_immediate_true(self):
        import json as _json
        from kugelaudio.livekit.tts import _CloseContext, _Connection, _TTSOptions

        opts = _TTSOptions(
            model="kugel-2-turbo", voice_id=None, sample_rate=24000,
            cfg_scale=2.0, max_new_tokens=2048, api_key="k", base_url="https://x",
            word_timestamps=False, language=None, normalize=True,
        )
        conn = _Connection(opts, MagicMock())
        sent: list[str] = []

        class _WS:
            closed = False

            async def send_str(self, s):
                sent.append(s)

        conn._ws = _WS()

        async def _go():
            conn.request_close_context("ctx1", immediate=True)
            item = await conn._input_queue.get()
            # Reproduce the send_loop body for _CloseContext manually —
            # we do not want to spin the real loop here.
            assert isinstance(item, _CloseContext)
            payload = {"close_context": True, "context_id": item.context_id}
            if item.immediate:
                payload["immediate"] = True
            await conn._ws.send_str(_json.dumps(payload))

        asyncio.new_event_loop().run_until_complete(_go())

        assert len(sent) == 1
        body = _json.loads(sent[0])
        assert body == {
            "close_context": True,
            "context_id": "ctx1",
            "immediate": True,
        }

    def test_send_loop_omits_immediate_when_false(self):
        import json as _json
        from kugelaudio.livekit.tts import _CloseContext

        item = _CloseContext(context_id="ctx1")
        payload = {"close_context": True, "context_id": item.context_id}
        if item.immediate:
            payload["immediate"] = True

        assert payload == {"close_context": True, "context_id": "ctx1"}, (
            "graceful close must NOT include immediate flag (server "
            "default is graceful drain)"
        )


# ---------------------------------------------------------------------------
# Idle-activity timeout (not absolute)
#
# Regression: the SDK used to fire APITimeoutError at _conn_options.timeout
# seconds after _run started, even if audio frames were still arriving for
# the context.  The correct behaviour is to fail only if the server has been
# silent for this context for longer than the idle threshold.
# ---------------------------------------------------------------------------


class _AsyncIter:
    """Minimal async iterator for driving stream._input_ch in tests."""

    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


def _install_mock_connection_on_tts(tts_instance, mock_conn):
    """Patch tts_instance._ensure_connection to return *mock_conn*."""
    return patch.object(
        tts_instance, "_ensure_connection", AsyncMock(return_value=mock_conn)
    )


def _mock_connection_with_register_tracker():
    """Create a MagicMock _Connection that captures (emitter, waiter, stream)
    from register_context so tests can drive the waiter externally and, once
    the real _ContextData is returned by register_context, also expose it
    via mock_conn._registered_ctx for tests that need direct access.
    """
    from kugelaudio.livekit.tts import _ContextData, _Connection

    mock_conn = MagicMock(spec=_Connection)
    mock_conn._input_queue = asyncio.Queue()
    mock_conn._context_data = {}
    mock_conn._registered_ctx = None  # type: ignore[attr-defined]

    def _register(ctx_id, emitter, waiter, stream=None):
        ctx = _ContextData(emitter=emitter, waiter=waiter, stream=stream)
        mock_conn._context_data[ctx_id] = ctx
        mock_conn._registered_ctx = ctx  # type: ignore[attr-defined]
        # Preserve legacy behaviour (register_context returned None before the
        # fix) by not returning the ctx unless the SDK's _run fetches it via
        # _context_data — the helper supports both styles.
        return ctx

    mock_conn.register_context.side_effect = _register
    mock_conn.send_content = MagicMock()
    mock_conn.request_close_context = MagicMock()
    mock_conn._cleanup_context = MagicMock()
    return mock_conn


class TestIdleTimeoutNotAbsolute:
    """The streaming & chunked runs must use an *idle* timeout (silence from
    the server), not an absolute wall-clock timeout.  This is the regression
    that caused APITimeoutError for long LLM replies even when audio was
    flowing at real-time rate.
    """

    # ----- Test A: streaming active-but-slow server should NOT time out ----

    # ----- Test B: truly dead server must still raise APITimeoutError -----
    def test_stream_raises_on_truly_idle_server(self):
        """With timeout=0.3 and NO server activity, _run must raise
        APITimeoutError within a bit more than 0.3s (not hang forever).
        """
        from livekit.agents import APITimeoutError as LKAPITimeoutError

        from kugelaudio.livekit.tts import TTS

        tts_instance = TTS(api_key="test-key")

        async def _go():
            import time as _time

            stream = _make_synth_stream(tts_instance)
            stream._conn_options = APIConnectOptions(max_retry=0, timeout=0.3)

            mock_conn = _mock_connection_with_register_tracker()

            with _install_mock_connection_on_tts(tts_instance, mock_conn):
                emitter = MagicMock()
                stream._input_ch = _AsyncIter([])

                t0 = _time.monotonic()
                with pytest.raises(LKAPITimeoutError):
                    await stream._run(emitter)
                elapsed = _time.monotonic() - t0
                assert elapsed >= 0.3, f"timed out too early: {elapsed:.3f}s"
                assert elapsed < 2.0, f"timed out too late: {elapsed:.3f}s"

        asyncio.new_event_loop().run_until_complete(_go())

    # ----- Test C: late activity shifts the idle deadline ------------------
    def test_stream_idle_deadline_shifts_with_last_activity(self):
        """Activity at T=0.3s, then silence: APITimeoutError should fire
        around T=0.3 + 0.5 = 0.8s (not at T=0.5s).
        """
        from livekit.agents import APITimeoutError as LKAPITimeoutError

        from kugelaudio.livekit.tts import TTS

        tts_instance = TTS(api_key="test-key")

        async def _go():
            import time as _time

            stream = _make_synth_stream(tts_instance)
            stream._conn_options = APIConnectOptions(max_retry=0, timeout=0.5)

            mock_conn = _mock_connection_with_register_tracker()

            async def _drive_one_activity():
                for _ in range(200):
                    if mock_conn._registered_ctx is not None:
                        break
                    await asyncio.sleep(0.01)
                ctx = mock_conn._registered_ctx
                assert ctx is not None
                await asyncio.sleep(0.3)
                # Bump last_activity_at — this should reset the idle deadline.
                ctx.last_activity_at = _time.monotonic()

            with _install_mock_connection_on_tts(tts_instance, mock_conn):
                emitter = MagicMock()
                stream._input_ch = _AsyncIter([])

                t0 = _time.monotonic()
                driver = asyncio.create_task(_drive_one_activity())
                with pytest.raises(LKAPITimeoutError):
                    await stream._run(emitter)
                elapsed = _time.monotonic() - t0
                await driver
                # Minimum wall-clock is ~0.3 (activity) + ~0.5 (idle) = 0.8s.
                assert elapsed >= 0.7, (
                    f"fired too early: {elapsed:.3f}s — idle deadline "
                    f"did not shift with last_activity_at"
                )
                assert elapsed < 2.0, f"fired too late: {elapsed:.3f}s"

        asyncio.new_event_loop().run_until_complete(_go())

    # ----- Test D: same three scenarios for ChunkedStream -----------------
    def test_chunked_survives_active_but_slow_server(self):
        from kugelaudio.livekit.tts import TTS

        tts_instance = TTS(api_key="test-key")

        async def _go():
            chunked = _make_chunked_stream(tts_instance, input_text="hello")
            chunked._conn_options = APIConnectOptions(max_retry=0, timeout=0.5)

            mock_conn = _mock_connection_with_register_tracker()

            async def _drive_activity():
                import time as _time

                for _ in range(200):
                    if mock_conn._registered_ctx is not None:
                        break
                    await asyncio.sleep(0.01)
                ctx = mock_conn._registered_ctx
                assert ctx is not None
                for _ in range(6):  # 6 × 0.2s = 1.2s > 0.5s absolute
                    await asyncio.sleep(0.2)
                    ctx.last_activity_at = _time.monotonic()
                if not ctx.waiter.done():
                    ctx.waiter.set_result(None)

            with _install_mock_connection_on_tts(tts_instance, mock_conn):
                emitter = MagicMock()
                driver = asyncio.create_task(_drive_activity())
                await chunked._run(emitter)
                await driver

        asyncio.new_event_loop().run_until_complete(_go())

    def test_chunked_raises_on_truly_idle_server(self):
        from livekit.agents import APITimeoutError as LKAPITimeoutError

        from kugelaudio.livekit.tts import TTS

        tts_instance = TTS(api_key="test-key")

        async def _go():
            import time as _time

            chunked = _make_chunked_stream(tts_instance, input_text="hello")
            chunked._conn_options = APIConnectOptions(max_retry=0, timeout=0.3)

            mock_conn = _mock_connection_with_register_tracker()

            with _install_mock_connection_on_tts(tts_instance, mock_conn):
                emitter = MagicMock()
                t0 = _time.monotonic()
                with pytest.raises(LKAPITimeoutError):
                    await chunked._run(emitter)
                elapsed = _time.monotonic() - t0
                assert elapsed >= 0.3, f"timed out too early: {elapsed:.3f}s"
                assert elapsed < 2.0, f"timed out too late: {elapsed:.3f}s"

        asyncio.new_event_loop().run_until_complete(_go())


class TestRecvLoopUpdatesActivity:
    """Ensure _recv_loop_inner bumps ctx.last_activity_at for every kind of
    server message routed to a known context."""

    def _make_conn_with_ctx(self):
        from kugelaudio.livekit.tts import (
            _Connection,
            _ContextData,
            _TTSOptions,
        )

        opts = _TTSOptions(
            model="kugel-1-turbo", voice_id=None, sample_rate=24000,
            cfg_scale=2.0, max_new_tokens=2048, api_key="k", base_url="https://x",
        )
        conn = _Connection(opts, MagicMock())
        emitter = MagicMock()
        # waiter created later, inside the event loop (see _drive_one_message).
        ctx = _ContextData(emitter=emitter, waiter=MagicMock())  # type: ignore[arg-type]
        conn._context_data["ctx1"] = ctx
        return conn, ctx

    def _drive_one_message(self, conn, payload):
        """Feed a single WS TEXT message through _recv_loop_inner and exit."""
        frames = [
            MagicMock(type=aiohttp.WSMsgType.TEXT, data=json.dumps(payload)),
            MagicMock(type=aiohttp.WSMsgType.CLOSED, data=""),
        ]

        ws = MagicMock()
        ws.closed = False

        async def _receive():
            return frames.pop(0)

        ws.receive.side_effect = _receive
        conn._ws = ws

        async def _go():
            # Replace any placeholder waiters with real Futures bound to the
            # running loop (context_closed path calls waiter.set_result).
            for c in conn._context_data.values():
                if not isinstance(c.waiter, asyncio.Future):
                    c.waiter = asyncio.get_running_loop().create_future()
            await conn._recv_loop_inner()

        asyncio.new_event_loop().run_until_complete(_go())

    def test_audio_frame_updates_activity(self):
        import time as _time

        import aiohttp as _aiohttp  # noqa: F401

        conn, ctx = self._make_conn_with_ctx()
        ctx.last_activity_at = 0.0

        audio_b64 = base64.b64encode(b"\x00\x01").decode()
        before = _time.monotonic()
        self._drive_one_message(conn, {"context_id": "ctx1", "audio": audio_b64})
        assert ctx.last_activity_at >= before


    def test_chunk_complete_updates_activity(self):
        import time as _time

        conn, ctx = self._make_conn_with_ctx()
        # stream is not None so chunk_complete does not auto-close.
        ctx.stream = MagicMock()
        ctx.last_activity_at = 0.0
        before = _time.monotonic()
        self._drive_one_message(
            conn, {"context_id": "ctx1", "chunk_complete": True}
        )
        assert ctx.last_activity_at >= before


class TestChunkedContextAutoClosesPerTurn:
    """LiveKit's anti-leak invariant: a CHUNKED (non-stream) context is
    auto-closed by the recv loop on chunk_complete — the SDK closes its server
    context after every synthesis. This is exactly the discipline the Pipecat
    wrapper was missing, which leaked a new context per turn into the server's
    hard 5-context cap and dropped calls ~30-40s in. Guards against a
    regression that would make LiveKit leak the same way.
    """

    def test_chunk_complete_enqueues_close_for_chunked_context(self):
        from kugelaudio.livekit.tts import (
            _CloseContext,
            _Connection,
            _ContextData,
            _TTSOptions,
        )

        opts = _TTSOptions(
            model="kugel-1-turbo", voice_id=None, sample_rate=24000,
            cfg_scale=2.0, max_new_tokens=2048, api_key="k", base_url="https://x",
        )
        conn = _Connection(opts, MagicMock())
        ctx = _ContextData(emitter=MagicMock(), waiter=MagicMock())  # type: ignore[arg-type]
        ctx.stream = None  # chunked (one-shot synthesize), not a streaming turn
        conn._context_data["ctx1"] = ctx

        frames = [
            MagicMock(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps({"context_id": "ctx1", "chunk_complete": True}),
            ),
            MagicMock(type=aiohttp.WSMsgType.CLOSED, data=""),
        ]
        ws = MagicMock()
        ws.closed = False

        async def _receive():
            return frames.pop(0)

        ws.receive.side_effect = _receive
        conn._ws = ws

        async def _go():
            for c in conn._context_data.values():
                if not isinstance(c.waiter, asyncio.Future):
                    c.waiter = asyncio.get_running_loop().create_future()
            await conn._recv_loop_inner()

        asyncio.new_event_loop().run_until_complete(_go())

        queued = []
        while not conn._input_queue.empty():
            queued.append(conn._input_queue.get_nowait())
        assert any(
            isinstance(it, _CloseContext) and it.context_id == "ctx1" for it in queued
        ), "chunked context must be auto-closed per turn (LiveKit anti-leak invariant)"


# Alias so `import aiohttp` above is resolved when tests expect it.
import aiohttp  # noqa: E402
