"""Tests for kugelaudio client."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from kugelaudio import KugelAudio
from kugelaudio.client import TTSResource
from kugelaudio.exceptions import (
    AuthenticationError,
    ConnectionError as KugelAudioConnectionError,
    InsufficientCreditsError,
    NotFoundError,
    RateLimitError,
    ValidationError,
)


class TestKugelAudioInit:


    def test_custom_urls(self):
        client = KugelAudio(
            api_key="test_key",
            api_url="https://custom-api.example.com/",
            tts_url="https://custom-tts.example.com/",
        )

        assert client._api_url == "https://custom-api.example.com"
        assert client._tts_url == "https://custom-tts.example.com"


class TestKugelAudioRegion:
    """Test multi-region support."""


    def test_explicit_region_eu(self):
        client = KugelAudio(api_key="ka_test123", region="eu")
        assert client._api_url == "https://api.eu.kugelaudio.com"


    def test_key_prefix_eu_strips_prefix(self):
        client = KugelAudio(api_key="eu-ka_test123")
        assert client._api_url == "https://api.eu.kugelaudio.com"
        assert client._api_key == "ka_test123"

    def test_explicit_region_overrides_key_prefix(self):
        client = KugelAudio(api_key="us-ka_test123", region="global")
        assert client._api_url == "https://api.kugelaudio.com"
        assert client._api_key == "ka_test123"

    def test_api_url_overrides_region(self):
        client = KugelAudio(
            api_key="us-ka_test123",
            region="global",
            api_url="https://custom.example.com",
        )
        assert client._api_url == "https://custom.example.com"
        assert client._api_key == "ka_test123"

    def test_invalid_region_raises(self):
        with pytest.raises(ValidationError):
            KugelAudio(api_key="ka_test123", region="mars")


    def test_tts_url_defaults_to_region_url(self):
        client = KugelAudio(api_key="us-ka_test123")
        assert client._tts_url == "https://api.kugelaudio.com"


class TestASRResource:
    def test_transcribe_uses_public_multipart_contract(self):
        client = KugelAudio(api_key="test_key")
        client._http_client.request = MagicMock(
            return_value=httpx.Response(
                200,
                json={
                    "text": "Ich heiße Müller.",
                    "transcript": "Ich heiße Müller.",
                    "language": "de",
                    "duration_s": 1.5,
                    "model": "luchs-1",
                    "model_revision": "7278e1e70fe206f11671096ffdd38061171dd6e5",
                    "word_alternatives": [],
                },
            )
        )

        result = client.asr.transcribe(b"RIFFdata", language="de")

        assert result.text == "Ich heiße Müller."
        assert result.model_revision == "7278e1e70fe206f11671096ffdd38061171dd6e5"
        request = client._http_client.request.call_args.kwargs
        assert request["url"].endswith("/v1/audio/transcriptions")
        assert ("model", (None, "luchs-1", "text/plain")) in request["files"]

    def test_transcribe_rejects_unknown_model_before_network(self):
        client = KugelAudio(api_key="test_key")
        client._http_client.request = MagicMock()

        with pytest.raises(ValidationError):
            client.asr.transcribe(b"RIFF", model="whisper-1")

        client._http_client.request.assert_not_called()


class TestKugelAudioErrorPropagation:
    @pytest.mark.parametrize(
        (
            "status_code",
            "body",
            "headers",
            "expected_type",
            "expected_error_code",
        ),
        [
            (
                401,
                {"error": "bad key", "error_code": "UNAUTHORIZED"},
                {},
                AuthenticationError,
                "UNAUTHORIZED",
            ),
            (
                402,
                {"error": "no credits", "error_code": "INSUFFICIENT_CREDITS"},
                {},
                InsufficientCreditsError,
                "INSUFFICIENT_CREDITS",
            ),
            (
                429,
                {"error": "slow down", "error_code": "RATE_LIMITED"},
                {"Retry-After": "9"},
                RateLimitError,
                "RATE_LIMITED",
            ),
            (
                400,
                {"detail": "field required"},
                {},
                ValidationError,
                "VALIDATION_ERROR",
            ),
            (
                404,
                {"error": "voice missing", "error_code": "NOT_FOUND"},
                {},
                NotFoundError,
                "NOT_FOUND",
            ),
            (
                503,
                {"error": "model cold", "error_code": "MODEL_UNAVAILABLE"},
                {},
                KugelAudioConnectionError,
                "MODEL_UNAVAILABLE",
            ),
        ],
    )
    def test_public_http_methods_raise_typed_sdk_errors(
        self, status_code, body, headers, expected_type, expected_error_code
    ):
        client = KugelAudio(api_key="test_key")
        client._http_client.request = MagicMock(
            return_value=httpx.Response(status_code, json=body, headers=headers)
        )

        with pytest.raises(expected_type) as exc_info:
            client.models.list()

        err = exc_info.value
        assert err.status_code == status_code
        assert err.error_code == expected_error_code
        if status_code == 429:
            assert err.retry_after == 9

    @pytest.mark.asyncio
    async def test_pooled_ws_handshake_raises_typed_rate_limit_error(self):
        import kugelaudio.client as client_module
        import websockets as websockets_module

        class FakeHandshakeError(Exception):
            status_code = 429

        client = KugelAudio(api_key="test_key", keepalive_ping_interval=None)

        with (
            patch.object(
                websockets_module,
                "connect",
                AsyncMock(side_effect=FakeHandshakeError("rate limited")),
            ),
            patch.object(
                client_module,
                "ws_handshake_error_types",
                return_value=(FakeHandshakeError,),
            ),
        ):
            with pytest.raises(RateLimitError) as exc_info:
                await client.tts._get_ws_connection("kugel-1-turbo")

        assert exc_info.value.status_code == 429
        assert exc_info.value.error_code == "RATE_LIMITED"

    @pytest.mark.asyncio
    async def test_pooled_ws_send_close_raises_typed_rate_limit_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import websockets as websockets_module
        import websockets.exceptions as websockets_exceptions

        monkeypatch.delattr(websockets_module, "exceptions", raising=False)

        class FakeConnectionClosed(Exception):
            code = 4029
            reason = "Rate limit exceeded"

        class FakeWebSocket:
            async def send(self, _data):
                raise FakeConnectionClosed()

        client = KugelAudio(api_key="test_key", keepalive_ping_interval=None)

        with (
            patch.object(
                client.tts,
                "_get_ws_connection",
                AsyncMock(return_value=FakeWebSocket()),
            ),
            patch.object(
                websockets_exceptions,
                "ConnectionClosed",
                FakeConnectionClosed,
            ),
        ):
            with pytest.raises(RateLimitError) as exc_info:
                async for _ in client.tts.stream_async("hello"):
                    pass

        assert exc_info.value.status_code == 429
        assert exc_info.value.error_code == "RATE_LIMITED"


class TestKeepalivePing:
    """Unit tests for the WebSocket keepalive ping mechanism."""


    def _make_mock_ws(self):
        """Create a mock WebSocket with open=True and closed=False."""
        mock_ws = MagicMock()
        mock_ws.ping = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws.open = True
        mock_ws.closed = False
        return mock_ws

    @pytest.mark.asyncio
    async def test_keepalive_task_started_after_connect(self):
        """_keepalive_task is created when a WS connection is opened."""
        import websockets as _ws_module

        client = KugelAudio(api_key="test_key", keepalive_ping_interval=20.0)
        tts: TTSResource = client.tts
        mock_ws = self._make_mock_ws()

        mock_connect = AsyncMock(return_value=mock_ws)
        with patch.object(_ws_module, "connect", mock_connect):
            await tts._get_ws_connection("kugel-1-turbo")

        assert tts._keepalive_task is not None
        assert isinstance(tts._keepalive_task, asyncio.Task)
        assert not tts._keepalive_task.done()

        tts._cancel_keepalive()

    @pytest.mark.asyncio
    async def test_keepalive_task_not_started_when_disabled(self):
        """No keepalive task is created when interval is None."""
        import websockets as _ws_module

        client = KugelAudio(api_key="test_key", keepalive_ping_interval=None)
        tts: TTSResource = client.tts
        mock_ws = self._make_mock_ws()

        mock_connect = AsyncMock(return_value=mock_ws)
        with patch.object(_ws_module, "connect", mock_connect):
            await tts._get_ws_connection("kugel-1-turbo")

        assert tts._keepalive_task is None or tts._keepalive_task.done()

    @pytest.mark.asyncio
    async def test_cancel_keepalive_cancels_task(self):
        """_cancel_keepalive() stops the background task."""
        import websockets as _ws_module

        client = KugelAudio(api_key="test_key", keepalive_ping_interval=20.0)
        tts: TTSResource = client.tts
        mock_ws = self._make_mock_ws()

        mock_connect = AsyncMock(return_value=mock_ws)
        with patch.object(_ws_module, "connect", mock_connect):
            await tts._get_ws_connection("kugel-1-turbo")

        task = tts._keepalive_task
        assert task is not None

        tts._cancel_keepalive()

        assert tts._keepalive_task is None
        await asyncio.sleep(0)
        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_keepalive_sends_ping(self):
        """The keepalive coroutine calls ws.ping() after the configured interval."""
        client = KugelAudio(api_key="test_key", keepalive_ping_interval=0.05)
        tts: TTSResource = client.tts

        mock_ws = AsyncMock()
        mock_ws.ping = AsyncMock()
        # Pretend the ws is still the current connection
        tts._ws_connection = mock_ws

        task = asyncio.create_task(tts._start_keepalive(mock_ws))
        # Wait slightly longer than the ping interval
        await asyncio.sleep(0.12)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert mock_ws.ping.call_count >= 1

    @pytest.mark.asyncio
    async def test_keepalive_stops_when_connection_replaced(self):
        """The keepalive loop exits if the stored connection changes."""
        client = KugelAudio(api_key="test_key", keepalive_ping_interval=0.05)
        tts: TTSResource = client.tts

        original_ws = AsyncMock()
        original_ws.ping = AsyncMock()
        tts._ws_connection = original_ws

        task = asyncio.create_task(tts._start_keepalive(original_ws))

        # Replace the connection before the first ping fires
        new_ws = AsyncMock()
        tts._ws_connection = new_ws

        await asyncio.sleep(0.15)
        # Task should have exited cleanly (not cancelled, just returned)
        assert task.done()
        # Original ws should have received no pings (loop saw ws mismatch)
        assert original_ws.ping.call_count == 0


def test_installed_version_matches_module_version():
    """pyproject reads its dynamic version from __init__.py; the two must never drift."""
    from importlib import metadata

    import kugelaudio
    from kugelaudio._sdk_metadata import sdk_headers

    assert metadata.version("kugelaudio") == kugelaudio.__version__
    assert sdk_headers()["X-KugelAudio-SDK-Version"] == kugelaudio.__version__


class _RecordingWebSocket:
    """Pooled WS stand-in: records sent frames, answers with a final frame."""

    def __init__(self) -> None:
        self.sent: list = []

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def recv(self) -> str:
        return json.dumps({"final": True})


class TestProjectIdWire:
    """ENG-560: project_id reaches the wire so dictionaries apply at synthesis."""

    def _client_with_ws(self):
        client = KugelAudio(api_key="test_key", keepalive_ping_interval=None)
        ws = _RecordingWebSocket()
        return client, ws, patch.object(
            client.tts, "_get_ws_connection", AsyncMock(return_value=ws)
        )

    @pytest.mark.asyncio
    async def test_generate_async_sends_project_id_with_dictionary_ids(self):
        client, ws, patched = self._client_with_ws()
        with patched:
            await client.tts.generate_async(
                "hi", language="en", project_id=42, dictionary_ids=[7]
            )
        assert ws.sent[0]["project_id"] == 42
        assert ws.sent[0]["dictionary_ids"] == [7]

    @pytest.mark.asyncio
    async def test_stream_async_omits_project_id_when_unset(self):
        client, ws, patched = self._client_with_ws()
        with patched:
            async for _ in client.tts.stream_async("hi", language="en"):
                pass
        assert "project_id" not in ws.sent[0]

    def test_sync_generate_sends_project_id(self):
        client, ws, patched = self._client_with_ws()
        with patched:
            client.tts.generate("hi", language="en", project_id=42)
        assert ws.sent[0]["project_id"] == 42

    def test_streaming_session_config_carries_project_id(self):
        client = KugelAudio(api_key="test_key", keepalive_ping_interval=None)
        session = client.tts.streaming_session(project_id=42, dictionary_ids=[7])
        assert session._config.to_dict()["project_id"] == 42
        unset = client.tts.streaming_session()
        assert "project_id" not in unset._config.to_dict()

    def test_multi_context_session_carries_project_id(self):
        client = KugelAudio(api_key="test_key", keepalive_ping_interval=None)
        assert client.tts.multi_context_session(project_id=42)._project_id == 42
        assert client.tts.multi_context_session()._project_id is None
