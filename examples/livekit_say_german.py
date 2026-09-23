"""Say a fixed German handoff message through LiveKit using KugelAudio TTS.

Usage from ``packages/public/python-sdk``:

    export KUGELAUDIO_API_KEY="..."
    export KUGELAUDIO_VOICE_ID="1071"
    export LIVEKIT_URL="wss://your-livekit-server.com"
    export LIVEKIT_API_KEY="..."
    export LIVEKIT_API_SECRET="..."
    uv run --extra livekit python examples/livekit_say_german.py console
"""

from __future__ import annotations

import os

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli

from kugelaudio.livekit import TTS as KugelAudioTTS


MESSAGE = (
    "Perfekt. Ich habe alle Informationen gespeichert und weitergegeben. "
    "Wir werden Ihre Anfrage so schnell wie möglich bearbeiten. "
    "Kann ich sonst noch etwas für Sie tun?"
)
MODEL = "kugel-2.5"
SAMPLE_RATE = 24000


def required_int_env(name: str) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        raise RuntimeError(f"{name} must be set to the KugelAudio voice ID to use")

    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer voice ID, got {value!r}") from exc


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    session = AgentSession(
        tts=KugelAudioTTS(
            model=MODEL,
            voice_id=required_int_env("KUGELAUDIO_VOICE_ID"),
            language="de",
            sample_rate=SAMPLE_RATE,
            word_timestamps=False,
        )
    )

    await session.start(
        room=ctx.room,
        agent=Agent(instructions="Say the configured German handoff message once."),
    )

    speech = session.say(MESSAGE, allow_interruptions=False, add_to_chat_ctx=False)
    await speech.wait_for_playout()
    await session.drain()
    await session.aclose()
    ctx.shutdown("German handoff message spoken")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
