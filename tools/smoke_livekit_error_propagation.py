"""Smoke-check KugelAudio LiveKit error propagation.

This script does not call the live KugelAudio API. It loads the installed
LiveKit integration and feeds it ingress-shaped WebSocket failures so we can
verify users see LiveKit-native APIStatusError objects with the original
status/body.

Run from packages/public/python-sdk:
    uv run --extra livekit python tools/smoke_livekit_error_propagation.py
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
from livekit.agents import APIStatusError

from kugelaudio.livekit.tts import _Connection, _ContextData, _TTSOptions


def _options() -> _TTSOptions:
    return _TTSOptions(
        model="kugel-1-turbo",
        voice_id=None,
        sample_rate=24000,
        cfg_scale=2.0,
        max_new_tokens=2048,
        api_key="test-key",
        base_url="https://api.kugelaudio.com",
    )


async def _check_error_frame() -> None:
    payload = {
        "error": "Rate limit exceeded",
        "error_code": "RATE_LIMITED",
        "code": 429,
        "context_id": "ctx-livekit-smoke",
    }
    waiter = asyncio.get_running_loop().create_future()
    conn = _Connection(_options(), AsyncMock())
    conn._context_data[payload["context_id"]] = _ContextData(
        emitter=AsyncMock(),
        waiter=waiter,
    )
    conn._active_contexts.add(payload["context_id"])
    conn._ws = AsyncMock()
    conn._ws.closed = False
    conn._ws.receive = AsyncMock(
        side_effect=[
            SimpleNamespace(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps(payload),
            ),
            SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=""),
        ]
    )

    await conn._recv_loop_inner()

    err = waiter.exception()
    assert isinstance(err, APIStatusError), type(err)
    assert err.status_code == 429
    assert err.body == payload
    assert "Rate limit exceeded" in str(err)


async def _check_handshake_error() -> None:
    session = AsyncMock()
    session.ws_connect = AsyncMock(
        side_effect=aiohttp.WSServerHandshakeError(
            None,
            (),
            status=429,
            message="Too Many Requests",
        )
    )
    conn = _Connection(_options(), session)

    try:
        await conn.connect()
    except APIStatusError as err:
        assert err.status_code == 429
        assert err.body == {"error": "Too Many Requests", "code": 429}
    else:
        raise AssertionError("expected APIStatusError from rejected handshake")


async def main() -> None:
    await _check_error_frame()
    await _check_handshake_error()
    print("OK livekit error propagation smoke")


if __name__ == "__main__":
    asyncio.run(main())
