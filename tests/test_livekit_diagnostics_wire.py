"""LiveKit plugin diagnostics over a real socket, with the real sender.

The plugin's own aiohttp socket is rejected by the stub (401), so its
connect operation fails and ``aclose()`` flushes the report, or does not,
per the contract's enablement order.
"""

from __future__ import annotations

import pytest

pytest.importorskip("livekit.agents")

from .diagnostics_wire_support import (  # noqa: E402,F401  (fixtures)
    OPT_OUT_MATRIX,
    diagnostics_stub,
    hosted_probe,
)

pytestmark = pytest.mark.real_diagnostics_sender


@pytest.mark.parametrize(("env", "option", "hosted", "expect_posts"), OPT_OUT_MATRIX)
async def test_livekit_opt_out_matrix_on_the_wire(
    diagnostics_stub, hosted_probe, monkeypatch, env, option, hosted, expect_posts
):
    import aiohttp
    from livekit.agents import APIStatusError

    from kugelaudio.livekit import TTS

    if env is not None:
        monkeypatch.setenv("KUGELAUDIO_TELEMETRY", env)
    url = hosted_probe(diagnostics_stub) if hosted else diagnostics_stub.url()
    async with aiohttp.ClientSession() as session:
        tts = TTS(api_key="wire-key", base_url=url, http_session=session, telemetry=option)
        with pytest.raises(APIStatusError) as exc_info:
            await tts._ensure_connection(timeout=5.0)
        await tts.aclose()

    assert exc_info.value.status_code == 401
    assert exc_info.value.request_id == "rid-wire-401"
    assert bool(diagnostics_stub.posts) is expect_posts
    for post in diagnostics_stub.posts:
        events = [attrs["value"] for r in post.records for attrs in r["attributes"]
                  if attrs["key"] == "kugel.integration"]
        assert {e["stringValue"] for e in events} == {"livekit"}
