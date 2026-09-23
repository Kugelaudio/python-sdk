"""Reproduce the pipecat 'no close frame received or sent' failure.

Hypothesis: an upstream proxy / load balancer drops the WebSocket after
~20-30s of idle. The SDK doesn't detect the dead WS, so the next send()
on the reused MultiContextSession raises a websockets ConnectionClosedError
("no close frame received or sent"), and the pipecat wrapper keeps trying
to reuse the dead session, so every subsequent turn fails the same way.

Usage:
    KUGELAUDIO_API_KEY=... uv run --project packages/public/python-sdk python \\
        packages/public/python-sdk/scripts/debug/repro_pipecat_ws_idle.py \\
        --voice-id 280 --idle-seconds 30 --turns 3

Outputs per turn:
    - WS state (alive / closed code / dead)
    - Whether send() raised, with exception type + message
    - Audio bytes received
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
import uuid
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("repro")


def _ws_state(ws: Any) -> str:
    """Render the WS object's lifecycle state in a way that survives across
    websockets versions. Returns 'none' / 'open' / 'closing' / 'closed:<code>'.
    """
    if ws is None:
        return "none"
    state = getattr(ws, "state", None)
    if state is not None:
        name = getattr(state, "name", str(state))
        close_code = getattr(ws, "close_code", None)
        if close_code is not None:
            return f"{name}:{close_code}"
        return str(name)
    return repr(ws)


async def _drain_one_turn(
    session: Any, text: str, voice_id: int
) -> tuple[bool, str, int]:
    """Send one TTS turn through the existing session.

    Returns (ok, error_str, audio_bytes). On exception, ok=False.
    """
    context_id = f"repro-{uuid.uuid4()}"
    audio_bytes = 0
    try:
        async for chunk in session.send(
            context_id,
            text,
            flush=True,
            chunk_complete_idle_timeout=0.0,
        ):
            audio_bytes += len(chunk.audio)
        return True, "", audio_bytes
    except Exception as e:  # noqa: BLE001 — we *want* to surface the type
        return False, f"{type(e).__name__}: {e}", audio_bytes


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-id", type=int, default=280)
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--model", type=str, default="kugel-1-turbo")
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=30.0,
        help="Seconds to sleep between turns (matches user's ~22s gap).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("KUGELAUDIO_TTS_URL", "https://api.kugelaudio.com"),
    )
    args = parser.parse_args()

    api_key = os.environ.get("KUGELAUDIO_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        logger.error(
            "KUGELAUDIO_API_KEY (or API_KEY) must be set in the environment"
        )
        return 2

    from kugelaudio import KugelAudio

    logger.info("Connecting MultiContextSession to %s", args.base_url)
    client = KugelAudio(api_key=api_key, tts_url=args.base_url)
    session = client.tts.multi_context_session(
        default_voice_id=args.voice_id,
        model_id=args.model,
        sample_rate=24000,
        language=args.language,
    )
    await session.connect()
    logger.info("Connected. Initial WS state: %s", _ws_state(session._ws))

    failures = 0
    for turn in range(1, args.turns + 1):
        text = f"This is turn number {turn}, testing websocket idle behaviour."
        t0 = time.perf_counter()
        ok, err, n_bytes = await _drain_one_turn(session, text, args.voice_id)
        dt_ms = (time.perf_counter() - t0) * 1000
        ws_state_after = _ws_state(session._ws)
        if ok:
            logger.info(
                "turn %d  OK  audio=%d bytes  elapsed=%.0fms  ws=%s",
                turn,
                n_bytes,
                dt_ms,
                ws_state_after,
            )
        else:
            failures += 1
            logger.error(
                "turn %d  FAIL  audio=%d bytes  elapsed=%.0fms  ws=%s  err=%s",
                turn,
                n_bytes,
                dt_ms,
                ws_state_after,
                err,
            )

        if turn < args.turns:
            logger.info(
                "idle %.1fs (mimics user think-time between voice turns)",
                args.idle_seconds,
            )
            await asyncio.sleep(args.idle_seconds)
            logger.info("post-idle WS state: %s", _ws_state(session._ws))

    try:
        await session.close()
    except Exception as e:  # noqa: BLE001 — close after a dead session is best-effort
        logger.warning("session.close() raised %s", e)
    client.close()
    logger.info("done. failures=%d / turns=%d", failures, args.turns)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
