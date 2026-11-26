"""Fluid 1 as a LiveKit realtime model: options and the session.start it sends.

The wire test runs the real plugin against a local WebSocket server, so the URL,
the credential and the start event are the ones a call to KugelAudio carries.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("livekit.plugins.openai.realtime")

from aiohttp import WSMsgType, web
from livekit.agents import llm

from kugelaudio.client import DEFAULT_API_URL, EU_API_URL
from kugelaudio.livekit import RealtimeModel
from kugelaudio.livekit.realtime import DEFAULT_BACK_OFFICE_MODEL


@pytest.fixture(autouse=True)
def _no_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KUGELAUDIO_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_defaults_point_at_the_default_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KUGELAUDIO_API_KEY", "k-123")
    model = RealtimeModel()
    assert model.model == "fluid-1"
    assert model._opts.base_url == f"{DEFAULT_API_URL}/v1"
    assert model._opts.api_key == "k-123"
    assert model._opts.delegation == "responses"
    assert model._opts.responses == {"model": DEFAULT_BACK_OFFICE_MODEL}


def test_eu_prefix_selects_the_eu_endpoint_and_is_stripped() -> None:
    model = RealtimeModel(api_key="eu-k-123")
    assert model._opts.base_url == f"{EU_API_URL}/v1"
    assert model._opts.api_key == "k-123"


def test_base_url_overrides_region() -> None:
    model = RealtimeModel(api_key="eu-k", base_url="http://localhost:8080/")
    assert model._opts.base_url == "http://localhost:8080/v1"


def test_missing_key_raises() -> None:
    with pytest.raises(ValueError, match="KUGELAUDIO_API_KEY"):
        RealtimeModel()


@pytest.mark.parametrize("key", ["eu-", "us-"])
def test_prefix_only_key_raises(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # The plugin would fall back to this key and send it to KugelAudio.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-for-kugelaudio")
    with pytest.raises(ValueError, match="region prefix"):
        RealtimeModel(api_key=key)


def test_unknown_region_raises() -> None:
    with pytest.raises(ValueError, match="Invalid region"):
        RealtimeModel(api_key="k", region="mars")


@pytest.mark.parametrize("voice", [973, "973"])
def test_numeric_voice_is_sent_as_text(voice: int | str) -> None:
    assert RealtimeModel(api_key="k", voice=voice)._opts.voice == "973"


@pytest.mark.parametrize("voice", ["marin", "", True])
def test_non_numeric_voice_raises(voice: str | bool) -> None:
    with pytest.raises(ValueError, match="numeric"):
        RealtimeModel(api_key="k", voice=voice)


def test_named_back_office_model_is_kept() -> None:
    model = RealtimeModel(api_key="k", responses_options={"model": "gpt-6.1-terra"})
    assert model._opts.responses == {"model": "gpt-6.1-terra"}


def test_supported_back_office_settings_are_kept() -> None:
    settings = {
        "model": "m",
        "instructions": "Be brief.",
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "reasoning": {"effort": "low"},
    }
    model = RealtimeModel(api_key="k", responses_options=settings)
    assert model._opts.responses == settings


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({"text": {"verbosity": "low"}}, "does not support text"),
        (
            {"service_tier": "flex", "max_output_tokens": 64},
            "max_output_tokens, service_tier",
        ),
        (
            {"tool_choice": {"type": "function", "function": {"name": "f"}}},
            "tool_choice",
        ),
        ({"reasoning": {"effort": "low", "summary": "auto"}}, "reasoning"),
        ({"reasoning": {"effort": "max"}}, "reasoning"),
    ],
)
def test_unsupported_back_office_settings_raise(
    settings: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        RealtimeModel(api_key="k", responses_options=settings)  # type: ignore[arg-type]  # deliberately outside the supported subset


def test_client_delegation_sends_no_back_office() -> None:
    model = RealtimeModel(api_key="k", delegation="client")
    assert model._opts.delegation == "client"
    assert "model" not in model._opts.responses


def test_client_delegation_refuses_responses_options() -> None:
    with pytest.raises(ValueError, match="responses_options"):
        RealtimeModel(
            api_key="k", delegation="client", responses_options={"model": "x"}
        )


async def test_session_start_on_the_wire() -> None:
    seen: asyncio.Future[tuple[str, str, dict[str, object]]] = asyncio.Future()

    async def live(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if msg.type == WSMsgType.TEXT and not seen.done():
                seen.set_result(
                    (
                        request.path,
                        request.headers["Authorization"],
                        json.loads(msg.data),
                    )
                )
                break
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/v1/live/sessions", live)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]  # started above
    model = RealtimeModel(
        api_key="eu-k-123", base_url=f"http://127.0.0.1:{port}", voice=973
    )
    session = model.session()
    try:
        # RISK: `_update_session` is how AgentSession hands a realtime session its agent;
        # a rename in livekit-agents breaks this test, not customers.
        await session._update_session(
            instructions="Help the caller.", chat_ctx=llm.ChatContext.empty(), tools=[]
        )
        path, auth, event = await asyncio.wait_for(seen, 5)
    finally:
        await session.aclose()
        await model.aclose()
        await runner.cleanup()

    assert path == "/v1/live/sessions"
    assert auth == "Bearer k-123"
    assert event["type"] == "session.start"
    config = event["session"]
    assert isinstance(config, dict)
    assert config["model"] == "fluid-1"
    assert config["instructions"] == "Help the caller."
    assert config["audio"]["output"]["voice"] == "973"
    assert config["delegation"]["type"] == "responses"
    assert config["delegation"]["responses"]["model"] == DEFAULT_BACK_OFFICE_MODEL
