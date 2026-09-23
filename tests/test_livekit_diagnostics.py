"""Diagnostics for the LiveKit plugin's own ``/ws/tts/multi`` socket.

The plugin does not wrap a KugelAudio client, so it owns the reporter such a
client would build (``integration=livekit``). Establishing the socket is one
operation; every context (one synthesis) is its own turn operation.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("livekit.agents")

from .diagnostics_support import FakeSender, attrs_of, live_reporter  # noqa: E402


def _opts():
    from kugelaudio.livekit._connection import _TTSOptions

    return _TTSOptions(
        model="kugel-1-turbo",
        voice_id=None,
        sample_rate=24000,
        cfg_scale=2.0,
        max_new_tokens=2048,
        api_key="k",
        base_url="https://api.kugelaudio.com",
    )


def _text(payload: dict) -> MagicMock:
    import aiohttp

    return MagicMock(type=aiohttp.WSMsgType.TEXT, data=json.dumps(payload))


def _closed() -> MagicMock:
    import aiohttp

    return MagicMock(type=aiohttp.WSMsgType.CLOSED, data="")


def test_the_plugin_reporter_is_labelled_livekit_and_per_instance():
    from kugelaudio import KugelAudio
    from kugelaudio.livekit import TTS

    tts = TTS(api_key="k")
    assert tts._diagnostics.integration == "livekit"
    assert tts._diagnostics.url == "https://api.kugelaudio.com/v1/sdk-diagnostics"
    assert KugelAudio(api_key="k")._diagnostics.integration == "none"


@pytest.mark.parametrize(
    ("env", "option", "enabled"),
    [(None, None, True), (None, False, False), ("1", False, True), ("0", True, False)],
)
def test_telemetry_option_follows_the_contract_precedence(
    monkeypatch, env, option, enabled
):
    """Env beats the option, the option beats the hosted default."""
    from kugelaudio.livekit import TTS

    monkeypatch.delenv("KUGELAUDIO_TELEMETRY", raising=False)
    if env is not None:
        monkeypatch.setenv("KUGELAUDIO_TELEMETRY", env)
    assert TTS(api_key="k", telemetry=option)._diagnostics.enabled is enabled


async def test_a_rejected_handshake_is_one_connect_event_with_the_request_id():
    import aiohttp
    from livekit.agents import APIStatusError
    from multidict import CIMultiDict, CIMultiDictProxy

    from kugelaudio.livekit import TTS

    sender = FakeSender()
    session = MagicMock()
    session.ws_connect = AsyncMock(
        side_effect=aiohttp.WSServerHandshakeError(
            None,
            (),
            status=429,
            message="Too Many Requests",
            headers=CIMultiDictProxy(CIMultiDict({"X-Request-Id": "rid-lk-1"})),
        )
    )
    tts = TTS(api_key="k", http_session=session)
    tts._diagnostics = live_reporter(sender, integration="livekit")

    with pytest.raises(APIStatusError) as exc_info:
        await tts._ensure_connection(timeout=2.0)
    assert exc_info.value.request_id == "rid-lk-1"

    tts._diagnostics.flush(timeout=5.0)
    (event,) = [attrs_of(r) for r in sender.records]
    assert event["kugel.event"] == "connection_failed"
    assert event["kugel.operation"] == "multi_context"
    assert event["kugel.failure_stage"] == "handshake"
    assert event["kugel.error_type"] == "RateLimitError"
    assert event["kugel.error_code"] == "RATE_LIMITED"
    assert event["kugel.http_status"] == "429"
    assert event["kugel.server_request_id"] == "rid-lk-1"
    assert event["kugel.integration"] == "livekit"


def _registered(conn, reporter, context_id):
    """Register *context_id* the way a stream does: its op is assigned."""
    from kugelaudio._diagnostics import STAGE_SENDING_REQUEST

    waiter = asyncio.get_running_loop().create_future()
    ctx = conn.register_context(context_id, MagicMock(), waiter)
    ctx.op = reporter.operation("multi_context", "websocket", STAGE_SENDING_REQUEST)
    return ctx


def _settle(ctx):
    """What the owning stream does with its final attempt's outcome."""
    from kugelaudio.livekit._connection import _diagnostic_error

    exc = ctx.waiter.exception()
    if exc is None:
        ctx.op.succeed()
    else:
        ctx.op.fail(_diagnostic_error(exc))


def _ws(frames, close_code=None):
    ws = MagicMock()
    ws.closed = False
    ws.close_code = close_code
    ws.close = AsyncMock()
    ws.receive = AsyncMock(side_effect=frames)
    return ws


async def test_a_context_error_frame_is_request_failed_for_that_turn_only():
    from kugelaudio.livekit._connection import _Connection

    sender = FakeSender()
    reporter = live_reporter(sender, integration="livekit")
    conn = _Connection(_opts(), MagicMock())
    bad = _registered(conn, reporter, "bad")
    good = _registered(conn, reporter, "good")
    conn._ws = _ws(
        [
            _text({"audio": "AAAA", "context_id": "good"}),
            _text(
                {
                    "error": "slow down",
                    "error_code": "RATE_LIMITED",
                    "code": 429,
                    "request_id": "rid-frame",
                    "context_id": "bad",
                }
            ),
            _text({"context_closed": True, "context_id": "good"}),
            _closed(),
        ]
    )

    await conn._recv_loop_inner()
    _settle(bad)
    _settle(good)

    reporter.flush(timeout=5.0)
    (event,) = [attrs_of(r) for r in sender.records]
    assert event["kugel.event"] == "request_failed"
    assert event["kugel.error_type"] == "RateLimitError"
    assert event["kugel.error_code"] == "RATE_LIMITED"
    assert event["kugel.server_request_id"] == "rid-frame"
    assert reporter.counters() == {"success": 1, "failure": 1, "cancelled": 0}
    assert good.op.audio_chunks == 1


async def test_a_session_error_frame_fails_every_pending_turn_with_its_error():
    from livekit.agents import APIStatusError

    from kugelaudio.livekit._connection import _Connection

    sender = FakeSender()
    reporter = live_reporter(sender, integration="livekit")
    conn = _Connection(_opts(), MagicMock())
    turns = [_registered(conn, reporter, cid) for cid in ("a", "b")]
    conn._ws = _ws(
        [
            _text(
                {
                    "error": "model unavailable",
                    "error_code": "MODEL_UNAVAILABLE",
                    "request_id": "rid-session",
                }
            ),
            _text({"audio": "AAAA", "context_id": "a"}),  # never read
        ]
    )

    await conn._recv_loop_inner()

    assert conn.is_current is False
    conn._ws.close.assert_awaited()
    for ctx in turns:
        assert isinstance(ctx.waiter.exception(), APIStatusError)
        _settle(ctx)
    reporter.flush(timeout=5.0)
    events = [attrs_of(r) for r in sender.records]
    assert len(events) == 2
    for event in events:
        assert event["kugel.event"] == "request_failed"
        assert event["kugel.error_code"] == "MODEL_UNAVAILABLE"
        assert event["kugel.server_request_id"] == "rid-session"


async def test_a_dropped_socket_fails_open_contexts_with_the_close_code():
    from kugelaudio.livekit._connection import _Connection

    sender = FakeSender()
    reporter = live_reporter(sender, integration="livekit")
    conn = _Connection(_opts(), MagicMock())
    ctx = _registered(conn, reporter, "ctx")
    conn._ws = _ws([_closed()], close_code=1006)

    await conn._recv_loop()
    _settle(ctx)

    reporter.flush(timeout=5.0)
    (event,) = [attrs_of(r) for r in sender.records]
    assert event["kugel.error_type"] == "ConnectionError"
    assert event["kugel.ws_close_code"] == "1006"
    assert event["kugel.integration"] == "livekit"


async def test_closing_the_connection_leaves_settlement_to_the_stream():
    """A recycled socket is retried by LiveKit; only the stream may settle."""
    from kugelaudio.livekit._connection import _Connection

    reporter = live_reporter(FakeSender(), integration="livekit")
    conn = _Connection(_opts(), MagicMock())
    ctx = _registered(conn, reporter, "ctx")

    await conn.aclose()

    assert ctx.waiter.exception() is not None
    assert ctx.op.settled is False


# ---------------------------------------------------------------------------
# One operation per stream across LiveKit's _run retries
# ---------------------------------------------------------------------------


def _chunked_stream(tts_instance, max_retry):
    from unittest.mock import patch

    from livekit.agents import APIConnectOptions

    from kugelaudio.livekit.tts import ChunkedStream

    with patch("livekit.agents.tts.ChunkedStream.__init__", return_value=None):
        stream = ChunkedStream.__new__(ChunkedStream)
    stream._tts = tts_instance
    stream._opts = tts_instance._opts
    stream._input_text = "Hallo"
    stream._conn_options = APIConnectOptions(max_retry=max_retry, timeout=1.0)
    return stream


def _scripted_connection(outcomes):
    """A connection whose Nth registered context resolves with outcomes[N]."""
    from kugelaudio.livekit._connection import _ContextData

    conn = MagicMock()

    def register(context_id, emitter, waiter, stream=None):
        outcome = outcomes.pop(0)
        if outcome is None:
            waiter.set_result(None)
        else:
            waiter.set_exception(outcome)
        return _ContextData(emitter=emitter, waiter=waiter, stream=stream)

    conn.register_context.side_effect = register
    return conn


async def test_a_retried_attempt_that_recovers_is_one_success_with_a_retry():
    from unittest.mock import patch

    from livekit.agents import APIConnectionError

    from kugelaudio.livekit import TTS

    sender = FakeSender()
    tts = TTS(api_key="k")
    tts._diagnostics = live_reporter(sender, integration="livekit")
    stream = _chunked_stream(tts, max_retry=3)
    conn = _scripted_connection([APIConnectionError("dropped"), None])

    with patch.object(tts, "_ensure_connection", AsyncMock(return_value=conn)):
        with pytest.raises(APIConnectionError):
            await stream._run(MagicMock())  # attempt 1: LiveKit will retry
        await stream._run(MagicMock())  # attempt 2: succeeds

    tts._diagnostics.flush(timeout=5.0)
    assert sender.records == []
    assert tts._diagnostics.counters() == {"success": 1, "failure": 0, "cancelled": 0}
    assert stream._op.retry_count == 1


async def test_the_final_attempt_reports_retry_exhausted_once():
    from unittest.mock import patch

    from livekit.agents import APIConnectionError

    from kugelaudio.livekit import TTS

    sender = FakeSender()
    tts = TTS(api_key="k")
    tts._diagnostics = live_reporter(sender, integration="livekit")
    stream = _chunked_stream(tts, max_retry=1)
    conn = _scripted_connection(
        [APIConnectionError("dropped"), APIConnectionError("dropped again")]
    )

    with patch.object(tts, "_ensure_connection", AsyncMock(return_value=conn)):
        for _ in range(2):
            with pytest.raises(APIConnectionError):
                await stream._run(MagicMock())

    tts._diagnostics.flush(timeout=5.0)
    (event,) = [attrs_of(r) for r in sender.records]
    assert event["kugel.event"] == "retry_exhausted"
    assert event["kugel.retry_count"] == "1"
    assert tts._diagnostics.counters()["failure"] == 1
