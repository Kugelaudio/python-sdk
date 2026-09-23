"""Tests for the centralized error classification helpers and messages."""

from __future__ import annotations


import pytest

from kugelaudio.exceptions import (
    AuthenticationError,
    ConnectionError as KugelAudioConnectionError,
    KugelAudioError,
    NotFoundError,
    RateLimitError,
    ValidationError,
    classify_http_response,
    classify_ws_close,
    classify_ws_frame,
)


class _FakeResponse:
    """Minimal httpx.Response stand-in for the classifier."""

    def __init__(
        self,
        status_code: int,
        body=None,
        headers: dict | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


class TestClassifyHttpResponse:


    def test_unknown_status_falls_back_to_base(self):
        resp = _FakeResponse(418, body={"error": "teapot"})
        err = classify_http_response(resp)
        assert type(err) is KugelAudioError
        assert err.status_code == 418


    def test_404_without_error_code_still_builds_not_found(self):
        resp = _FakeResponse(404, body={"detail": "nope"})
        err = classify_http_response(resp)
        assert isinstance(err, NotFoundError)
        assert "nope" in err.message
        assert err.error_code == "NOT_FOUND"

    def test_error_code_wins_over_status_mismatch(self):
        # Server returns 500 but error_code says UNAUTHORIZED (hypothetical).
        resp = _FakeResponse(500, body={"error": "x", "error_code": "UNAUTHORIZED"})
        err = classify_http_response(resp)
        assert isinstance(err, AuthenticationError)

    def test_fastapi_detail_field_is_read(self):
        resp = _FakeResponse(400, body={"detail": "field required"})
        err = classify_http_response(resp)
        assert isinstance(err, ValidationError)
        assert "field required" in err.message
        # Canonical error_code must be filled in even when the server body
        # omitted it. Guards against `setdefault` being a no-op when the key
        # already exists with value None.
        assert err.error_code == "VALIDATION_ERROR"
        assert err.status_code == 400

    def test_retry_after_header_when_body_missing_it(self):
        resp = _FakeResponse(
            429, body={"error": "slow"}, headers={"Retry-After": "12"}
        )
        err = classify_http_response(resp)
        assert isinstance(err, RateLimitError)
        assert err.retry_after == 12

    def test_request_id_header_is_captured(self):
        resp = _FakeResponse(
            401,
            body={"error": "bad", "error_code": "UNAUTHORIZED"},
            headers={"x-request-id": "req_abc"},
        )
        err = classify_http_response(resp)
        assert err.request_id == "req_abc"
        assert "req_abc" in str(err)

    def test_unparseable_body_falls_back_to_text(self):
        resp = _FakeResponse(500, body=None, text="upstream exploded")
        err = classify_http_response(resp)
        assert type(err) is KugelAudioError
        assert "upstream exploded" in err.message


class TestClassifyWsHandshakeError:
    def test_invalid_status_v14_style(self):
        """websockets >= 14 raises InvalidStatus with exc.response.status_code."""
        from kugelaudio.exceptions import classify_ws_handshake_error

        class _FakeResponse:
            status_code = 401

        class _FakeExc(Exception):
            response = _FakeResponse()

        err = classify_ws_handshake_error(_FakeExc("rejected"))
        assert isinstance(err, AuthenticationError)
        assert err.status_code == 401
        assert err.error_code == "UNAUTHORIZED"

    def test_invalid_status_code_legacy_style(self):
        """websockets < 14 raises InvalidStatusCode with exc.status_code."""
        from kugelaudio.exceptions import classify_ws_handshake_error

        class _FakeExc(Exception):
            status_code = 429

        err = classify_ws_handshake_error(_FakeExc("rate-limited"))
        assert isinstance(err, RateLimitError)
        assert err.status_code == 429

    def test_missing_status_returns_none(self):
        """Unknown exception shape — caller falls back to generic wrapping."""
        from kugelaudio.exceptions import classify_ws_handshake_error

        assert classify_ws_handshake_error(RuntimeError("boom")) is None


class TestClassifyWsFrame:


    def test_frame_without_error_code_is_generic(self):
        err = classify_ws_frame({"error": "something broke"})
        # No error_code and no status → base KugelAudioError
        assert type(err) is KugelAudioError
        assert "something broke" in err.message

    def test_frame_with_retry_after(self):
        err = classify_ws_frame(
            {"error": "slow", "error_code": "RATE_LIMITED", "retry_after": 3}
        )
        assert isinstance(err, RateLimitError)
        assert err.retry_after == 3

    def test_ingress_frame_code_classifies_missing_voice_as_validation(self):
        err = classify_ws_frame(
            {
                "error": "voice_id is required",
                "error_code": "MISSING_VOICE_ID",
                "code": 400,
            }
        )
        assert isinstance(err, ValidationError)
        assert err.status_code == 400
        assert err.error_code == "MISSING_VOICE_ID"

    def test_ingress_frame_code_classifies_context_cap_as_rate_limit(self):
        err = classify_ws_frame(
            {
                "error": "Too many concurrent contexts",
                "error_code": "TOO_MANY_CONTEXTS",
                "code": 429,
            }
        )
        assert isinstance(err, RateLimitError)
        assert err.status_code == 429
        assert err.error_code == "TOO_MANY_CONTEXTS"

    def test_frame_populates_canonical_status_code(self):
        # WS frames carry no HTTP status. The classifier must still set the
        # canonical status on the built exception so callers can rely on
        # err.status_code regardless of transport.
        assert (
            classify_ws_frame({"error": "x", "error_code": "UNAUTHORIZED"}).status_code
            == 401
        )
        assert (
            classify_ws_frame(
                {"error": "x", "error_code": "INSUFFICIENT_CREDITS"}
            ).status_code
            == 402
        )
        assert (
            classify_ws_frame({"error": "x", "error_code": "RATE_LIMITED"}).status_code
            == 429
        )
        assert (
            classify_ws_frame(
                {"error": "x", "error_code": "VALIDATION_ERROR"}
            ).status_code
            == 400
        )
        assert (
            classify_ws_frame(
                {"error": "x", "error_code": "MODEL_UNAVAILABLE"}
            ).status_code
            == 503
        )
        assert (
            classify_ws_frame({"error": "x", "error_code": "NOT_FOUND"}).status_code
            == 404
        )


class TestClassifyWsClose:


    def test_unknown_close_code_is_connection_error(self):
        err = classify_ws_close(1011, "server error")
        assert isinstance(err, KugelAudioConnectionError)
        assert "server error" in err.message

    def test_no_code_and_no_reason(self):
        err = classify_ws_close(None)
        assert isinstance(err, KugelAudioConnectionError)
        assert "no reason given" in err.message


class TestClientConstructorMessage:
    """The missing-API-key message is the first thing new users see — lock it in."""

    def test_missing_api_key_message(self):
        from kugelaudio import KugelAudio

        with pytest.raises(ValidationError) as excinfo:
            KugelAudio(api_key="")
        msg = str(excinfo.value)
        # The SDK never reads an env var, so the message must not suggest one.
        assert "KUGELAUDIO_API_KEY" not in msg
        assert "api_key" in msg
        assert "https://app.kugelaudio.com/settings/api-keys" in msg
