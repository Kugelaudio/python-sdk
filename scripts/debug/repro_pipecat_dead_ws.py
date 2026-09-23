"""Force the dead-WS condition deterministically and assert the SDK fails.

Network-independent companion to repro_pipecat_ws_idle.py. Confirms two
distinct bugs in the SDK:

  Bug 1: MultiContextSession.send() does not detect that self._ws is dead
         before sending; it raises ConnectionClosedError /
         "no close frame received or sent" instead of reconnecting.

  Bug 2: KugelAudioTTSService (pipecat) holds onto the dead session in
         self._multi_session, so EVERY subsequent turn fails the same way.

We force the condition by closing the underlying TCP socket from outside
the SDK after a successful turn, then exercising both call sites.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("repro-dead")


async def _kill_underlying_socket(ws: Any) -> None:
    """Kill the TCP socket under the websocket, mimicking a proxy idle drop.

    This bypasses the WS close handshake — exactly what an LB / NAT timeout
    looks like to the client (the next send() raises ConnectionClosedError
    with 'no close frame received or sent').
    """
    transport = getattr(ws, "transport", None)
    if transport is not None:
        transport.abort()
        await asyncio.sleep(0.2)  # let the close propagate to the protocol
        return
    # Older asyncio impl: close the underlying socket
    sock = getattr(ws, "socket", None)
    if sock is not None:
        sock.close()
        await asyncio.sleep(0.2)
        return
    raise RuntimeError(
        f"don't know how to kill underlying transport for {type(ws).__name__}"
    )


async def repro_bug_1_sdk_session(api_key: str, base_url: str, voice_id: int) -> bool:
    """Returns True if bug reproduces (turn-2 send raises)."""
    from kugelaudio import KugelAudio

    logger.info("[bug 1] connecting MultiContextSession to %s", base_url)
    client = KugelAudio(api_key=api_key, tts_url=base_url)
    session = client.tts.multi_context_session(
        default_voice_id=voice_id,
        model_id="kugel-1-turbo",
        sample_rate=24000,
        language="en",
    )
    await session.connect()

    audio_bytes = 0
    async for chunk in session.send(
        "ctx-1", "First turn over a healthy socket.", flush=True,
        chunk_complete_idle_timeout=0.0,
    ):
        audio_bytes += len(chunk.audio)
    logger.info("[bug 1] turn 1 OK, audio=%d bytes", audio_bytes)

    logger.warning("[bug 1] killing TCP socket under the websocket")
    await _kill_underlying_socket(session._ws)

    try:
        async for chunk in session.send(
            "ctx-2", "Second turn over a dead socket.", flush=True,
            chunk_complete_idle_timeout=0.0,
        ):
            pass
    except Exception as e:  # noqa: BLE001
        logger.error("[bug 1] turn 2 raised %s: %s", type(e).__name__, e)
        client.close()
        return True
    client.close()
    logger.info("[bug 1] turn 2 unexpectedly succeeded — SDK self-healed")
    return False


async def repro_bug_2_pipecat_wrapper(
    api_key: str, base_url: str, voice_id: int
) -> tuple[bool, bool]:
    """Returns (turn2_failed, turn3_failed). Bug confirmed if both True."""
    from kugelaudio.pipecat import KugelAudioTTSService

    logger.info("[bug 2] constructing KugelAudioTTSService")
    tts = KugelAudioTTSService(
        api_key=api_key,
        model="kugel-1-turbo",
        voice_id=voice_id,
        sample_rate=24000,
        language="en",
        base_url=base_url,
        voice=str(voice_id),
    )

    async def _drain(text: str, ctx: str) -> tuple[bool, str]:
        from pipecat.frames.frames import ErrorFrame
        try:
            async for frame in tts.run_tts(text, ctx):
                if isinstance(frame, ErrorFrame):
                    return False, f"ErrorFrame: {frame.error}"
            return True, ""
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"

    ok1, err1 = await _drain("Pipecat turn 1.", "ctx-pc-1")
    logger.info("[bug 2] pipecat turn 1: ok=%s err=%s", ok1, err1)

    logger.warning("[bug 2] killing TCP socket under the wrapper's session")
    await _kill_underlying_socket(tts._multi_session._ws)

    ok2, err2 = await _drain("Pipecat turn 2 over dead socket.", "ctx-pc-2")
    logger.error("[bug 2] pipecat turn 2: ok=%s err=%s", ok2, err2)

    ok3, err3 = await _drain("Pipecat turn 3 — does wrapper reconnect?", "ctx-pc-3")
    logger.error("[bug 2] pipecat turn 3: ok=%s err=%s", ok3, err3)

    await tts.cleanup()
    return (not ok2), (not ok3)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-id", type=int, default=280)
    parser.add_argument(
        "--base-url", type=str,
        default=os.environ.get("KUGELAUDIO_TTS_URL", "https://api.kugelaudio.com"),
    )
    args = parser.parse_args()

    api_key = os.environ.get("KUGELAUDIO_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        logger.error("set KUGELAUDIO_API_KEY (or API_KEY)")
        return 2

    bug1 = await repro_bug_1_sdk_session(api_key, args.base_url, args.voice_id)
    bug2_t2, bug2_t3 = await repro_bug_2_pipecat_wrapper(
        api_key, args.base_url, args.voice_id
    )

    print()
    print("=" * 72)
    print(f"  bug 1 (SDK MultiContextSession.send raises on dead WS): {bug1}")
    print(f"  bug 2 (pipecat wrapper turn 2 fails on dead WS):       {bug2_t2}")
    print(f"  bug 2 (pipecat wrapper turn 3 ALSO fails — no reset):  {bug2_t3}")
    print("=" * 72)
    return 0 if (bug1 and bug2_t2 and bug2_t3) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
