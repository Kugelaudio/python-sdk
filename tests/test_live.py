"""Contract tests for the official OpenAI Live binding."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest
from kugelaudio.exceptions import ValidationError
from kugelaudio.live import (
    KUGEL_LIVE_MODEL,
    KugelLiveSessionConfig,
    create_live_client,
)
from openai import AsyncOpenAI


def test_native_live_model_name() -> None:
    assert KUGEL_LIVE_MODEL == "fluid-1"


@dataclass(slots=True)
class _ConnectionCapture:
    url: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    sent: list[str] = field(default_factory=list)
    closed: bool = False


class _FakeWebSocket:
    def __init__(self, capture: _ConnectionCapture) -> None:
        self._capture = capture

    async def send(self, data: str) -> None:
        self._capture.sent.append(data)

    async def close(self, *, code: int = 1000, reason: str = "") -> None:
        del code, reason
        self._capture.closed = True


@pytest.mark.asyncio
async def test_create_live_client_uses_explicit_base_url_and_cleans_region_key() -> (
    None
):
    local = create_live_client(
        "eu-ka_test",
        base_url="http://127.0.0.1:8080/v1/",
    )

    assert isinstance(local, AsyncOpenAI)
    assert str(local.base_url) == "http://127.0.0.1:8080/v1/"
    assert local.api_key == "ka_test"
    await local.close()


def test_create_live_client_rejects_empty_key() -> None:
    with pytest.raises(ValidationError, match="API key is missing"):
        create_live_client("", base_url="http://localhost:8080/v1")


def test_create_live_client_rejects_empty_explicit_base_url() -> None:
    with pytest.raises(ValidationError, match="base_url must not be empty"):
        create_live_client("ka_test", base_url=" ")


@pytest.mark.asyncio
async def test_official_client_connects_and_returns_client_delegation_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # RISK: this private connector path is version-sensitive; a failing contract test
    # is intentional if an OpenAI SDK upgrade changes how Live connections are opened.
    import openai.lib._websocket as websocket_module

    capture = _ConnectionCapture()

    async def fake_connect(
        url: str,
        *,
        user_agent_header: str,
        additional_headers: Mapping[str, str],
    ) -> _FakeWebSocket:
        assert user_agent_header
        capture.url = url
        capture.headers = additional_headers
        return _FakeWebSocket(capture)

    monkeypatch.setattr(websocket_module, "_WebSocketConnect", fake_connect)
    client = create_live_client("eu-ka_test", base_url="https://s2s.example.test/v1")
    session: KugelLiveSessionConfig = {
        "model": KUGEL_LIVE_MODEL,
        "delegation": {"type": "client"},
        "store": False,
        "kugel": {
            "overlap_control": "suspend_and_classify",
        },
    }
    manager = client.live.connect(max_retries=0)
    manager.send({"type": "session.start", "session": session})

    async with manager as connection:
        await connection.session.thinking.append(
            delegation_id="delegation_1", content="Looking up the order."
        )
        await connection.session.commentary.append(
            delegation_id="delegation_1", content="The order has shipped."
        )

    assert capture.url == "wss://s2s.example.test/v1/live/sessions"
    assert capture.headers["Authorization"] == "Bearer ka_test"
    assert capture.closed is True
    assert json.loads(capture.sent[0]) == {
        "type": "session.start",
        "session": session,
    }
    assert [json.loads(payload) for payload in capture.sent[1:]] == [
        {
            "type": "session.thinking.append",
            "delegation_id": "delegation_1",
            "content": "Looking up the order.",
        },
        {
            "type": "session.commentary.append",
            "delegation_id": "delegation_1",
            "content": "The order has shipped.",
        },
    ]
    await client.close()
