"""Live check: a token-streamed sentence is spoken as soon as it ends.

Streams one sentence token by token over ``/ws/tts/multi`` (20 ms apart,
the cadence of an LLM), then goes quiet, and times the last token to the
first audio frame. Each case is compared with the ``?`` control, which
ingress never held, so network and engine time cancel out: anything well
above the control is ingress holding a finished sentence (ENG-686; before
the fix a trailing ``.`` sat ~500 ms above it and a ``!`` token ~2 s).

Run against staging (see ``.claude/context/environments.md``)::

    KUGELAUDIO_BASE_URL=http://127.0.0.1:18000 KUGELAUDIO_API_KEY=... \\
        uv run pytest tests/test_streamed_first_audio_live.py -s

Skip conditions: no API key, or server not reachable.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time

import httpx
import pytest
import websockets

_BASE_URL = os.environ.get("KUGELAUDIO_BASE_URL", "http://localhost:8000")
_API_KEY = os.environ.get("KUGELAUDIO_API_KEY")
_REPEATS = int(os.environ.get("KUGELAUDIO_LIVE_REPEATS", "5"))
_TOKEN_GAP_S = 0.02
# Ingress may add at most this over the control before it counts as a hold.
_MAX_EXTRA_MS = 150.0

_GREETING = ["Guten", " Tag", ",", " wie", " kann", " ich", " Ihnen", " helfen"]
CASES: dict[str, list[str]] = {
    "question (control)": [*_GREETING, "?"],
    "period": [*_GREETING, "."],
    "exclamation token": [*_GREETING, "!"],
    "short first sentence": ["Ja", "."],
}

pytestmark = pytest.mark.skipif(not _API_KEY, reason="KUGELAUDIO_API_KEY not set")


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_API_KEY}"}


def _voice_id() -> int:
    try:
        response = httpx.get(
            f"{_BASE_URL}/v1/voices", params={"limit": 1}, headers=_headers()
        )
    except httpx.HTTPError as exc:
        pytest.skip(f"server not reachable at {_BASE_URL}: {exc}")
    response.raise_for_status()
    return int(response.json()["voices"][0]["id"])


async def _last_token_to_first_audio_ms(tokens: list[str], voice_id: int) -> float:
    url = _BASE_URL.replace("http", "ws", 1) + "/ws/tts/multi"
    async with websockets.connect(url, additional_headers=_headers()) as ws:
        await ws.send(
            json.dumps(
                {
                    "text": "",
                    "context_id": "ctx",
                    "voice_settings": {"voice_id": voice_id},
                }
            )
        )
        created = json.loads(await ws.recv())
        assert created.get("context_created"), created
        for token in tokens:
            await ws.send(json.dumps({"text": token, "context_id": "ctx"}))
            await asyncio.sleep(_TOKEN_GAP_S)
        last_sent = time.perf_counter()
        while True:
            frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            assert "error" not in frame, frame
            if "audio" in frame:
                waited_ms = (time.perf_counter() - last_sent) * 1000.0
                break
        await ws.send(json.dumps({"close_socket": True}))
    return waited_ms


async def _medians(voice_id: int) -> dict[str, float]:
    samples: dict[str, list[float]] = {name: [] for name in CASES}
    for _ in range(_REPEATS):
        # Interleaved, so a slow minute on the box hits every case alike.
        for name, tokens in CASES.items():
            samples[name].append(await _last_token_to_first_audio_ms(tokens, voice_id))
    for name, values in samples.items():
        print(
            f"{name:22s} median {statistics.median(values):6.0f} ms  {[round(v) for v in values]}"
        )
    return {name: statistics.median(values) for name, values in samples.items()}


def test_finished_sentence_is_not_held() -> None:
    medians = asyncio.run(_medians(_voice_id()))
    control = medians["question (control)"]

    held = {
        name: round(median - control)
        for name, median in medians.items()
        if median - control > _MAX_EXTRA_MS
    }
    assert not held, f"ms above the '?' control ({control:.0f} ms): {held}"
