"""Tests for diagnostics operations on real SDK paths.

Covers the operation seam (one event per failed operation), operation scope
(one operation per turn, a separate connect operation, retries scoped per
operation), cancellation, request-id correlation including WS handshake
rejections, and the per-client integration label. Contract:
``services/ingress/docs/sdk-diagnostics-contract.md``.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kugelaudio import KugelAudio
from kugelaudio._diagnostics import (
    EVENT_CONNECTION_FAILED,
    EVENT_REQUEST_FAILED,
    EVENT_SDK_STATS,
    EVENT_STREAM_INTERRUPTED,
    OP_MODELS,
    OP_STREAM,
    OUTCOME_FAILED,
    STAGE_AWAITING_FIRST_AUDIO,
    STAGE_SENDING_REQUEST,
    TRANSPORT_HTTP,
    TRANSPORT_WEBSOCKET,
)
from kugelaudio.exceptions import classify_ws_frame

from .diagnostics_support import (
    FakeResponse,
    FakeSender,
    attrs_of,
    live_reporter,
)


# ---------------------------------------------------------------------------
# Real SDK error paths
# ---------------------------------------------------------------------------


class TestRealErrorPaths:
    def test_http_401_produces_one_request_failed_event(self):
        sender = FakeSender()
        client = KugelAudio(api_key="k")
        client._diagnostics = live_reporter(sender)

        response = FakeResponse(
            401,
            body={"error": "bad key", "error_code": "UNAUTHORIZED"},
            headers={"x-request-id": "req-abc-123"},
        )
        with patch.object(
            client._http_client, "request", return_value=response
        ):
            with pytest.raises(Exception):
                client.models.list()

        client._diagnostics.flush(timeout=5.0)
        records = sender.records
        assert len(records) == 1

        flat = attrs_of(records[0])
        assert flat["kugel.event"] == EVENT_REQUEST_FAILED
        assert flat["kugel.operation"] == OP_MODELS
        assert flat["kugel.transport"] == TRANSPORT_HTTP
        assert flat["kugel.failure_stage"] == STAGE_SENDING_REQUEST
        assert flat["kugel.error_type"] == "AuthenticationError"
        assert flat["kugel.error_code"] == "UNAUTHORIZED"
        assert flat["kugel.http_status"] == "401"
        assert flat["kugel.server_request_id"] == "req-abc-123"
        assert flat["kugel.outcome"] == OUTCOME_FAILED

    async def test_ws_close_1006_produces_one_stream_interrupted_event(self):
        from websockets.exceptions import ConnectionClosedError

        sender = FakeSender()
        client = KugelAudio(api_key="k")
        client._diagnostics = live_reporter(sender)

        # An abnormal closure with no close frame: ``.code`` is 1006.
        closed = ConnectionClosedError(None, None)
        assert closed.code == 1006

        ws = MagicMock()
        ws.send = AsyncMock()
        ws.recv = AsyncMock(side_effect=closed)

        with patch.object(
            client.tts, "_get_ws_connection", AsyncMock(return_value=ws)
        ), patch.object(client.tts, "_close_ws_connection", AsyncMock()):
            with pytest.raises(Exception):
                async for _ in client.tts.stream_async("hallo"):
                    pass

        client._diagnostics.flush(timeout=5.0)
        records = sender.records
        assert len(records) == 1

        flat = attrs_of(records[0])
        assert flat["kugel.event"] == EVENT_STREAM_INTERRUPTED
        assert flat["kugel.operation"] == OP_STREAM
        assert flat["kugel.transport"] == TRANSPORT_WEBSOCKET
        assert flat["kugel.failure_stage"] == STAGE_AWAITING_FIRST_AUDIO
        assert flat["kugel.error_type"] == "ConnectionError"
        assert flat["kugel.ws_close_code"] == "1006"
        assert flat["kugel.audio_chunks"] == "0"

    def test_handshake_failure_is_a_connection_failed_event(self):
        sender = FakeSender()
        reporter = live_reporter(sender)
        from kugelaudio._diagnostics import STAGE_HANDSHAKE

        op = reporter.operation(OP_STREAM, TRANSPORT_WEBSOCKET)
        op.set_stage(STAGE_HANDSHAKE)
        op.fail(OSError("refused"))
        reporter.flush()

        flat = attrs_of(sender.records[0])
        assert flat["kugel.event"] == EVENT_CONNECTION_FAILED
        assert flat["kugel.failure_stage"] == STAGE_HANDSHAKE

    async def test_session_handshake_failure_reports_once(self):
        from kugelaudio.streaming import StreamingSession

        sender = FakeSender()
        reporter = live_reporter(sender)
        session = StreamingSession(
            api_key="k", tts_url="https://api.kugelaudio.com", diagnostics=reporter
        )

        with patch(
            "websockets.connect", AsyncMock(side_effect=OSError("refused"))
        ):
            with pytest.raises(OSError):
                await session.connect()

        reporter.flush(timeout=5.0)
        records = sender.records
        assert len(records) == 1

        flat = attrs_of(records[0])
        assert flat["kugel.event"] == EVENT_CONNECTION_FAILED
        assert flat["kugel.operation"] == "stream_session"
        assert flat["kugel.failure_stage"] == "handshake"
        assert flat["kugel.error_type"] == "OSError"

    async def test_session_error_frame_reports_once_with_request_id(self):
        from kugelaudio.streaming import StreamingSession

        sender = FakeSender()
        reporter = live_reporter(sender)
        session = StreamingSession(
            api_key="k", tts_url="https://api.kugelaudio.com", diagnostics=reporter
        )
        ws = MagicMock()
        ws.send = AsyncMock()
        ws.recv = AsyncMock(
            return_value=json.dumps(
                {
                    "error": "out of credits",
                    "error_code": "INSUFFICIENT_CREDITS",
                    "request_id": "ws-rid-1",
                }
            )
        )
        session._ws = ws
        session._is_started = True

        with pytest.raises(Exception):
            async for _ in session.send("hallo", flush=True):
                pass

        reporter.flush(timeout=5.0)
        records = sender.records
        assert len(records) == 1

        flat = attrs_of(records[0])
        assert flat["kugel.event"] == EVENT_REQUEST_FAILED
        # No audio arrived before the error frame.
        assert flat["kugel.failure_stage"] == STAGE_AWAITING_FIRST_AUDIO
        assert flat["kugel.error_type"] == "InsufficientCreditsError"
        assert flat["kugel.error_code"] == "INSUFFICIENT_CREDITS"
        assert flat["kugel.server_request_id"] == "ws-rid-1"

    def test_one_operation_reports_at_most_one_event(self):
        sender = FakeSender()
        reporter = live_reporter(sender)
        op = reporter.operation(OP_STREAM, TRANSPORT_WEBSOCKET)
        op.fail(RuntimeError("first"))
        op.fail(RuntimeError("second"))
        op.cancel()  # settled already: adds no event and no cancellation
        reporter.flush()

        assert len(sender.records) == 1


# ---------------------------------------------------------------------------
# Cancellation emits nothing, is counted, and is not a failure
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_cancel_emits_no_event_but_increments_the_cancelled_counter(self):
        sender = FakeSender()
        reporter = live_reporter(sender)
        reporter.operation(OP_STREAM, TRANSPORT_WEBSOCKET).cancel()
        reporter.flush()

        assert sender.calls == []
        assert reporter.counters() == {"success": 0, "failure": 0, "cancelled": 1}

    def test_asyncio_cancellation_emits_no_event_and_is_not_a_failure(self):
        sender = FakeSender()
        reporter = live_reporter(sender)
        reporter.operation(OP_STREAM, TRANSPORT_WEBSOCKET).fail(
            asyncio.CancelledError()
        )
        reporter.flush()

        assert sender.calls == []
        assert reporter.counters()["failure"] == 0
        assert reporter.counters()["cancelled"] == 1

    def test_no_record_anywhere_carries_a_cancelled_outcome(self):
        """``failed`` is the only outcome a record can carry."""
        sender = FakeSender()
        reporter = live_reporter(sender)
        for _ in range(5):
            reporter.operation(OP_STREAM, TRANSPORT_WEBSOCKET).cancel()
        reporter.operation(OP_STREAM, TRANSPORT_WEBSOCKET).fail(RuntimeError("x"))
        reporter.close(timeout=5.0)

        outcomes = {
            attrs_of(r).get("kugel.outcome")
            for r in sender.records
            if "kugel.outcome" in attrs_of(r)
        }
        assert outcomes == {OUTCOME_FAILED}

    def test_a_cancelled_operation_stays_settled(self):
        """A cancellation wins the operation; a later failure says nothing."""
        sender = FakeSender()
        reporter = live_reporter(sender)
        op = reporter.operation(OP_STREAM, TRANSPORT_WEBSOCKET)
        op.cancel()
        op.fail(RuntimeError("late"))
        op.cancel()
        reporter.flush()

        assert sender.calls == []
        assert reporter.counters()["cancelled"] == 1

    def test_a_real_failure_still_emits_exactly_one_event(self):
        sender = FakeSender()
        reporter = live_reporter(sender)
        reporter.operation(OP_STREAM, TRANSPORT_WEBSOCKET).fail(RuntimeError("boom"))
        reporter.flush()

        assert len(sender.records) == 1
        assert attrs_of(sender.records[0])["kugel.outcome"] == OUTCOME_FAILED

    def test_closing_a_generator_early_reports_nothing(self):
        sender = FakeSender()
        reporter = live_reporter(sender)
        reporter.operation(OP_STREAM, TRANSPORT_WEBSOCKET).fail(GeneratorExit())
        reporter.flush()

        assert sender.calls == []
        assert reporter.counters()["cancelled"] == 0

    def test_sdk_stats_separates_failures_from_cancellations(self):
        sender = FakeSender()
        reporter = live_reporter(sender)
        reporter.operation(OP_MODELS, TRANSPORT_HTTP).succeed()
        reporter.operation(OP_MODELS, TRANSPORT_HTTP).succeed()
        reporter.operation(OP_STREAM, TRANSPORT_WEBSOCKET).fail(RuntimeError("x"))
        reporter.operation(OP_STREAM, TRANSPORT_WEBSOCKET).cancel()
        reporter.close(timeout=5.0)

        stats = [
            attrs_of(r)
            for r in sender.records
            if attrs_of(r)["kugel.event"] == EVENT_SDK_STATS
        ]
        assert len(stats) == 1
        assert stats[0]["kugel.success_count"] == "2"
        assert stats[0]["kugel.failure_count"] == "1"
        assert stats[0]["kugel.cancelled_count"] == "1"

    def test_sdk_stats_is_emitted_for_cancellations_alone(self):
        """The counter is the only carrier, so it must survive on its own."""
        sender = FakeSender()
        reporter = live_reporter(sender)
        reporter.operation(OP_STREAM, TRANSPORT_WEBSOCKET).cancel()
        reporter.close(timeout=5.0)

        stats = [
            attrs_of(r)
            for r in sender.records
            if attrs_of(r)["kugel.event"] == EVENT_SDK_STATS
        ]
        assert len(stats) == 1
        assert stats[0]["kugel.cancelled_count"] == "1"
        assert stats[0]["kugel.failure_count"] == "0"


# ---------------------------------------------------------------------------
# Request-id correlation
# ---------------------------------------------------------------------------


class TestRequestIdCorrelation:
    def test_ws_error_frame_request_id_lands_on_the_typed_error(self):
        err = classify_ws_frame(
            {
                "error": "rate limited",
                "error_code": "RATE_LIMITED",
                "request_id": "ws-req-42",
            }
        )
        assert err.request_id == "ws-req-42"
        assert "ws-req-42" in str(err)

    def test_ws_error_frame_without_request_id_is_unchanged(self):
        err = classify_ws_frame({"error": "boom", "error_code": "INTERNAL_ERROR"})
        assert err.request_id is None

    def test_ws_error_frame_ignores_a_non_string_request_id(self):
        err = classify_ws_frame({"error": "boom", "request_id": 12345})
        assert err.request_id is None

    def test_http_response_header_request_id_lands_on_the_typed_error(self):
        from kugelaudio.exceptions import classify_http_response

        err = classify_http_response(
            FakeResponse(
                429,
                body={"error": "slow down", "error_code": "RATE_LIMITED"},
                headers={"x-request-id": "http-req-7"},
            )
        )
        assert err.request_id == "http-req-7"

    def test_reported_event_carries_the_ws_frame_request_id(self):
        sender = FakeSender()
        reporter = live_reporter(sender)
        err = classify_ws_frame(
            {"error": "nope", "error_code": "UNAUTHORIZED", "request_id": "rid-9"}
        )
        reporter.operation(OP_STREAM, TRANSPORT_WEBSOCKET).fail(err)
        reporter.flush()

        assert attrs_of(sender.records[0])["kugel.server_request_id"] == "rid-9"


# ---------------------------------------------------------------------------
# Operation scope: one operation per turn, connect is its own operation
# ---------------------------------------------------------------------------


def _restart_close():
    from websockets.exceptions import ConnectionClosedError
    from websockets.frames import Close

    return ConnectionClosedError(Close(1012, "service restart"), None)


def _audio(context_id=None):
    frame = {"audio": "AAAA", "idx": 0, "sr": 24000, "samples": 1}
    if context_id is not None:
        frame["context_id"] = context_id
    return frame


class TestOperationScope:
    async def test_a_recovered_retry_does_not_leak_into_the_next_turn(
        self, monkeypatch
    ):
        from kugelaudio import streaming
        from kugelaudio.streaming import StreamingSession

        from .diagnostics_support import ScriptedWebSocket

        monkeypatch.setattr(streaming, "_restart_delay_s", lambda err: 0.0)
        sender = FakeSender()
        reporter = live_reporter(sender)
        session = StreamingSession(
            api_key="k", tts_url="https://api.kugelaudio.com", diagnostics=reporter
        )
        first = ScriptedWebSocket([_restart_close()])
        second = ScriptedWebSocket(
            [
                # Turn 1, replayed after the rolling-deploy close: completes.
                _audio(),
                {"final": True},
                {"session_closed": True},
                # Turn 2: fails on its own.
                {"error": "boom", "error_code": "INTERNAL_ERROR"},
            ]
        )
        session._ws = first
        session._is_started = True

        async def reconnect() -> None:
            session._ws = second
            session._is_started = True

        monkeypatch.setattr(session, "_connect_socket", reconnect)

        turn_one = [c async for c in session.send("hallo", flush=True)]
        assert len(turn_one) == 1
        # The recovery reconnect belonged to turn 1: no separate event.
        assert reporter.counters() == {"success": 1, "failure": 0, "cancelled": 0}

        await asyncio.sleep(0.2)
        with pytest.raises(Exception):
            async for _ in session.send("weiter", flush=True):
                pass

        reporter.flush(timeout=5.0)
        (event,) = sender.records and [attrs_of(r) for r in sender.records]
        assert event["kugel.event"] == EVENT_REQUEST_FAILED  # not retry_exhausted
        assert event["kugel.retry_count"] == "0"
        assert event["kugel.audio_chunks"] == "0"
        # elapsed_ms runs from the start of turn 2, not from the session.
        assert int(event["kugel.elapsed_ms"]) < 200

    async def test_session_connect_is_its_own_operation(self):
        from kugelaudio.streaming import StreamingSession

        from .diagnostics_support import ScriptedWebSocket

        sender = FakeSender()
        reporter = live_reporter(sender)
        session = StreamingSession(
            api_key="k", tts_url="https://api.kugelaudio.com", diagnostics=reporter
        )
        ws = ScriptedWebSocket([{"error": "nope", "error_code": "INTERNAL_ERROR"}])

        with patch("websockets.connect", AsyncMock(return_value=ws)):
            await session.connect()
        assert reporter.counters()["success"] == 1  # the connect operation

        with pytest.raises(Exception):
            async for _ in session.send("hallo", flush=True):
                pass

        reporter.flush(timeout=5.0)
        (event,) = [attrs_of(r) for r in sender.records]
        assert event["kugel.failure_stage"] == STAGE_AWAITING_FIRST_AUDIO
        assert event["kugel.event"] == EVENT_REQUEST_FAILED

    async def test_cancelling_one_context_never_touches_another(self):
        from kugelaudio.streaming import MultiContextSession

        from .diagnostics_support import BLOCK, ScriptedWebSocket

        sender = FakeSender()
        reporter = live_reporter(sender)
        session = MultiContextSession(
            api_key="k", tts_url="https://api.kugelaudio.com", diagnostics=reporter
        )
        session._ws = ScriptedWebSocket(
            [
                _audio("a"),
                BLOCK,  # context a's turn stays open
                _audio("b"),
                {"context_closed": True, "context_id": "a"},
                {"error": "boom", "error_code": "INTERNAL_ERROR", "context_id": "b"},
            ]
        )
        session._is_started = True
        session._contexts = {"a", "b"}

        assert len([c async for c in session.send("a", "Hallo")]) == 1

        b_turn = session.send("b", "Guten Tag", flush=True)
        await b_turn.__anext__()  # b has one chunk in flight

        async for _ in session.close_context("a", immediate=True):
            pass
        assert reporter.counters() == {"success": 0, "failure": 0, "cancelled": 1}

        with pytest.raises(Exception):
            await b_turn.__anext__()

        reporter.flush(timeout=5.0)
        (event,) = [attrs_of(r) for r in sender.records]
        assert event["kugel.operation"] == "multi_context"
        assert event["kugel.event"] == EVENT_REQUEST_FAILED
        assert event["kugel.audio_chunks"] == "1"
        assert reporter.counters() == {"success": 0, "failure": 1, "cancelled": 1}

    async def test_a_final_frame_ends_the_context_turn(self):
        from kugelaudio.streaming import MultiContextSession

        from .diagnostics_support import ScriptedWebSocket

        reporter = live_reporter(FakeSender())
        session = MultiContextSession(
            api_key="k", tts_url="https://api.kugelaudio.com", diagnostics=reporter
        )
        session._ws = ScriptedWebSocket(
            [_audio("a"), {"final": True, "context_id": "a"}]
        )
        session._is_started = True
        session._contexts = {"a"}

        first = session._turn("a")
        async for _ in session.send("a", "Hallo", flush=True):
            pass

        assert first.settled
        assert reporter.counters()["success"] == 1
        assert session._turn("a") is not first


    async def test_a_context_error_frame_is_request_failed_with_its_ids(self):
        from kugelaudio.streaming import MultiContextSession

        from .diagnostics_support import ScriptedWebSocket

        sender = FakeSender()
        reporter = live_reporter(sender)
        session = MultiContextSession(
            api_key="k", tts_url="https://api.kugelaudio.com", diagnostics=reporter
        )
        session._ws = ScriptedWebSocket(
            [
                _audio("a"),
                {
                    "error": "model unavailable",
                    "error_code": "MODEL_UNAVAILABLE",
                    "request_id": "ws-rid-ctx",
                    "context_id": "a",
                },
            ]
        )
        session._is_started = True
        session._contexts = {"a"}

        with pytest.raises(Exception):
            async for _ in session.send("a", "Hallo", flush=True):
                pass

        reporter.flush(timeout=5.0)
        (event,) = [attrs_of(r) for r in sender.records]
        assert event["kugel.event"] == EVENT_REQUEST_FAILED
        assert event["kugel.error_code"] == "MODEL_UNAVAILABLE"
        assert event["kugel.server_request_id"] == "ws-rid-ctx"
        assert event["kugel.audio_chunks"] == "1"

    async def test_a_context_drop_without_an_error_frame_stays_interrupted(self):
        from websockets.exceptions import ConnectionClosedError

        from kugelaudio.streaming import MultiContextSession

        from .diagnostics_support import ScriptedWebSocket

        sender = FakeSender()
        reporter = live_reporter(sender)
        session = MultiContextSession(
            api_key="k", tts_url="https://api.kugelaudio.com", diagnostics=reporter
        )
        session._ws = ScriptedWebSocket([_audio("a"), ConnectionClosedError(None, None)])
        session._is_started = True
        session._contexts = {"a"}

        with pytest.raises(Exception):
            async for _ in session.send("a", "Hallo", flush=True):
                pass

        reporter.flush(timeout=5.0)
        (event,) = [attrs_of(r) for r in sender.records]
        assert event["kugel.event"] == EVENT_STREAM_INTERRUPTED
        assert event["kugel.ws_close_code"] == "1006"

    async def test_a_one_shot_stream_error_frame_is_request_failed(self):
        sender = FakeSender()
        client = KugelAudio(api_key="k")
        client._diagnostics = live_reporter(sender)
        ws = MagicMock()
        ws.send = AsyncMock()
        ws.recv = AsyncMock(
            return_value=json.dumps(
                {"error": "slow down", "error_code": "RATE_LIMITED", "request_id": "rid-1s"}
            )
        )
        with patch.object(
            client.tts, "_get_ws_connection", AsyncMock(return_value=ws)
        ), patch.object(client.tts, "_close_ws_connection", AsyncMock()):
            with pytest.raises(Exception):
                async for _ in client.tts.stream_async("hallo"):
                    pass

        client._diagnostics.flush(timeout=5.0)
        (event,) = [attrs_of(r) for r in sender.records]
        assert event["kugel.event"] == EVENT_REQUEST_FAILED
        assert event["kugel.error_code"] == "RATE_LIMITED"
        assert event["kugel.server_request_id"] == "rid-1s"

    async def test_a_recovered_stream_turn_is_back_to_awaiting_audio(
        self, monkeypatch
    ):
        from kugelaudio import streaming
        from kugelaudio.streaming import StreamingSession

        from .diagnostics_support import ScriptedWebSocket

        monkeypatch.setattr(streaming, "_restart_delay_s", lambda err: 0.0)
        sender = FakeSender()
        reporter = live_reporter(sender)
        session = StreamingSession(
            api_key="k", tts_url="https://api.kugelaudio.com", diagnostics=reporter
        )
        second = ScriptedWebSocket([{"error": "boom", "error_code": "INTERNAL_ERROR"}])
        session._ws = ScriptedWebSocket([_restart_close()])
        session._is_started = True

        async def reconnect() -> None:
            session._ws = second
            session._is_started = True

        monkeypatch.setattr(session, "_connect_socket", reconnect)
        with pytest.raises(Exception):
            async for _ in session.send("hallo", flush=True):
                pass

        reporter.flush(timeout=5.0)
        (event,) = [attrs_of(r) for r in sender.records]
        assert event["kugel.event"] == "retry_exhausted"
        assert event["kugel.failure_stage"] == STAGE_AWAITING_FIRST_AUDIO

    async def test_a_recovered_context_turn_is_back_to_awaiting_audio(
        self, monkeypatch
    ):
        from kugelaudio import streaming
        from kugelaudio.streaming import MultiContextSession

        from .diagnostics_support import ScriptedWebSocket

        monkeypatch.setattr(streaming, "_restart_delay_s", lambda err: 0.0)
        sender = FakeSender()
        reporter = live_reporter(sender)
        session = MultiContextSession(
            api_key="k", tts_url="https://api.kugelaudio.com", diagnostics=reporter
        )
        second = ScriptedWebSocket(
            [
                {"context_created": True, "context_id": "a"},
                {"error": "boom", "error_code": "INTERNAL_ERROR", "context_id": "a"},
            ]
        )
        session._ws = ScriptedWebSocket([_restart_close()])
        session._is_started = True
        session._contexts = {"a"}
        session._context_voices = {"a": None}

        async def reconnect() -> None:
            session._ws = second

        monkeypatch.setattr(session, "_connect_socket", reconnect)
        with pytest.raises(Exception):
            async for _ in session.send("a", "Hallo", flush=True):
                pass

        reporter.flush(timeout=5.0)
        (event,) = [attrs_of(r) for r in sender.records]
        assert event["kugel.event"] == "retry_exhausted"
        assert event["kugel.failure_stage"] == STAGE_AWAITING_FIRST_AUDIO

    async def test_flush_or_close_after_the_turn_ended_is_not_another_success(self):
        from kugelaudio.streaming import MultiContextSession

        from .diagnostics_support import ScriptedWebSocket

        reporter = live_reporter(FakeSender())
        session = MultiContextSession(
            api_key="k", tts_url="https://api.kugelaudio.com", diagnostics=reporter
        )
        session._ws = ScriptedWebSocket(
            [
                _audio("a"),
                {"final": True, "context_id": "a"},  # the turn ends here
                {"final": True, "context_id": "a"},  # flush with nothing left
                {"final": True, "context_id": "a"},
                {"context_closed": True, "context_id": "a"},  # per-turn close
            ]
        )
        session._is_started = True
        session._contexts = {"a"}

        async for _ in session.send("a", "Hallo", flush=True):
            pass
        async for _ in session.flush("a"):
            pass
        async for _ in session.close_context("a"):
            pass

        assert reporter.counters() == {"success": 1, "failure": 0, "cancelled": 0}


# ---------------------------------------------------------------------------
# WS handshake rejection: request id from the rejection response header
# ---------------------------------------------------------------------------


def _rejection(status: int, request_id: str):
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response

    headers = Headers({"x-request-id": request_id})
    return InvalidStatus(Response(status, "Rejected", headers, b""))


class TestHandshakeRequestId:
    def test_rejection_header_lands_on_the_typed_error(self):
        from kugelaudio.exceptions import (
            AuthenticationError,
            RateLimitError,
            classify_ws_handshake_error,
        )

        err = classify_ws_handshake_error(_rejection(429, "rid-429"))
        assert isinstance(err, RateLimitError)
        assert err.request_id == "rid-429"

        auth = classify_ws_handshake_error(_rejection(403, "rid-403"))
        assert isinstance(auth, AuthenticationError)
        assert auth.request_id == "rid-403"

    async def test_rejected_session_connect_reports_the_server_request_id(self):
        from kugelaudio.exceptions import AuthenticationError
        from kugelaudio.streaming import StreamingSession

        sender = FakeSender()
        reporter = live_reporter(sender)
        session = StreamingSession(
            api_key="k", tts_url="https://api.kugelaudio.com", diagnostics=reporter
        )
        with patch(
            "websockets.connect",
            AsyncMock(side_effect=_rejection(401, "rid-hs-1")),
        ):
            with pytest.raises(AuthenticationError) as exc_info:
                await session.connect()
        assert exc_info.value.request_id == "rid-hs-1"

        reporter.flush(timeout=5.0)
        (event,) = [attrs_of(r) for r in sender.records]
        assert event["kugel.event"] == EVENT_CONNECTION_FAILED
        assert event["kugel.server_request_id"] == "rid-hs-1"
        assert event["kugel.http_status"] == "401"
        assert event["kugel.error_code"] == "UNAUTHORIZED"


# ---------------------------------------------------------------------------
# kugel.integration is a per-client label, never process-global
# ---------------------------------------------------------------------------


class TestPluginLabelIsPerClient:
    def test_importing_the_plugins_does_not_relabel_other_clients(self):
        import kugelaudio.livekit  # noqa: F401
        import kugelaudio.pipecat  # noqa: F401

        assert KugelAudio(api_key="k")._diagnostics.integration == "none"

    def test_each_client_stamps_its_own_integration(self):
        plain_sender, plugin_sender = FakeSender(), FakeSender()
        plain = live_reporter(plain_sender)
        plugin = live_reporter(plugin_sender, integration="livekit")
        for reporter in (plain, plugin):
            reporter.operation(OP_STREAM, TRANSPORT_WEBSOCKET).fail(RuntimeError("x"))
            reporter.flush(timeout=5.0)

        assert attrs_of(plain_sender.records[0])["kugel.integration"] == "none"
        assert attrs_of(plugin_sender.records[0])["kugel.integration"] == "livekit"
        assert (
            KugelAudio(api_key="k", _integration="pipecat")._diagnostics.integration
            == "pipecat"
        )

    def test_an_unknown_integration_is_rejected(self):
        with pytest.raises(ValueError):
            live_reporter(FakeSender(), integration="crewai")


def test_reading_the_close_code_emits_no_deprecation_warning():
    """``ConnectionClosed.code`` warns on websockets >= 13.1; telemetry must not."""
    import warnings

    from websockets.exceptions import ConnectionClosedError
    from websockets.frames import Close

    from kugelaudio.exceptions import ConnectionError as KugelAudioConnectionError

    sender = FakeSender()
    reporter = live_reporter(sender)
    for closed, expected in (
        (ConnectionClosedError(None, None), "1006"),
        (ConnectionClosedError(Close(4029, "slow down"), None), "4029"),
    ):
        typed = KugelAudioConnectionError("dropped")
        typed.__cause__ = closed
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reporter.operation(OP_STREAM, TRANSPORT_WEBSOCKET).fail(typed)
        assert caught == []
        reporter.flush(timeout=5.0)
        assert attrs_of(sender.records[-1])["kugel.ws_close_code"] == expected
