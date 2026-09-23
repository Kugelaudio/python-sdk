"""Live verification: default LiveKit TTS path for kugel-2.5.

Usage (from packages/public/python-sdk):
    KUGELAUDIO_API_KEY=... uv run python tools/verify_livekit_default.py

Exits 0 when all checks pass; non-zero on failure.
"""

from __future__ import annotations

import asyncio
import os
import sys

import aiohttp

from kugelaudio.livekit import DEFAULT_MODEL, TTS

BASE_URL = os.environ.get("KUGELAUDIO_BASE_URL", "https://api.kugelaudio.com")
API_KEY = os.environ.get("KUGELAUDIO_API_KEY", "")
VOICE_ID = int(os.environ.get("KUGELAUDIO_VOICE_ID", "976"))
TEXT = "This LiveKit synthesis should produce clear audio."


async def _synthesize_pcm(tts: TTS) -> bytes:
    pcm = bytearray()
    async for event in tts.synthesize(TEXT):
        pcm.extend(bytes(event.frame.data))
    return bytes(pcm)


async def _run_case(name: str, session: aiohttp.ClientSession, **kwargs: object) -> None:
    tts = TTS(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=str(kwargs.get("model", "kugel-3")),
        voice_id=VOICE_ID,
        sample_rate=24000,
        language="en",
        word_timestamps=bool(kwargs.get("word_timestamps", False)),
        http_session=session,
    )
    try:
        pcm = await _synthesize_pcm(tts)
    finally:
        await tts.aclose()

    if not pcm:
        raise RuntimeError(f"{name}: no audio received")
    print(f"PASS {name}: {len(pcm)} bytes audio")


async def main() -> int:
    if not API_KEY:
        print("SKIP: set KUGELAUDIO_API_KEY to run live verification", file=sys.stderr)
        return 2

    assert DEFAULT_MODEL == "kugel-3", f"unexpected DEFAULT_MODEL={DEFAULT_MODEL}"

    async with aiohttp.ClientSession() as session:
        # Customer default path (no word_timestamps kwarg)
        default = TTS(
            api_key=API_KEY,
            base_url=BASE_URL,
            model="kugel-3",
            voice_id=VOICE_ID,
            http_session=session,
        )
        assert default._opts.word_timestamps is False
        assert default.capabilities.aligned_transcript is False
        await default.aclose()

        await _run_case("default_kugel_3", session)
        await _run_case("explicit_word_timestamps_false", session, word_timestamps=False)

        # Opt-in alignment still available (may fail on some server stacks)
        try:
            await _run_case("opt_in_timestamps_kugel_3", session, word_timestamps=True)
            print("NOTE opt_in_timestamps_kugel_3 succeeded (server alignment OK)")
        except Exception as exc:
            print(f"NOTE opt_in_timestamps_kugel_3 failed as expected on some stacks: {exc!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
