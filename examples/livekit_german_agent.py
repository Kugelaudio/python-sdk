"""A German voice agent on your own LiveKit, with KugelAudio as LLM, speech to text and voice.

Your LiveKit stays as it is. One key runs all three models in the EU, and each one is a single
argument of AgentSession, so you can adopt them one at a time. Every line of the conversation
reaches ``conversation_item_added``; store it there.

Usage from ``packages/public/python-sdk``:

    export KUGELAUDIO_API_KEY="..."
    export LIVEKIT_URL="wss://your-livekit-server.com"
    export LIVEKIT_API_KEY="..."
    export LIVEKIT_API_SECRET="..."
    uv run --extra livekit --with livekit-plugins-openai --with livekit-plugins-silero \
        python examples/livekit_german_agent.py dev
"""

from __future__ import annotations

import os

from livekit.agents import (
    Agent,
    AgentSession,
    ChatMessage,
    ConversationItemAddedEvent,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import openai, silero

from kugelaudio.livekit import STT as KugelAudioSTT
from kugelaudio.livekit import TTS as KugelAudioTTS

BASE_URL = os.environ.get("KUGELAUDIO_BASE_URL", "https://api.eu.kugelaudio.com")
GERMAN_VOICE = 1887
INSTRUCTIONS = """
Du bist Nora vom Kundenservice einer Sachversicherung und telefonierst auf Deutsch.
Antworte in ein bis zwei kurzen Sätzen, ohne Aufzählungen und ohne Markdown, und stelle höchstens eine Frage pro Antwort.
Wenn jemand einen Schaden meldet, frage nach Ort und Hergang, und rufe dann schaden_aufnehmen auf.
Nenne danach die Vorgangsnummer.
"""


class Nora(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)

    @function_tool()
    async def schaden_aufnehmen(self, context: RunContext, ort: str, hergang: str) -> str:
        """Einen Schaden aufnehmen.

        Args:
            ort: Wo der Schaden passiert ist.
            hergang: Was passiert ist, in einem Satz.
        """
        return "Aufgenommen. Vorgangsnummer SD-2026-4471."


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    tts = KugelAudioTTS(model="kugel-3.5", voice_id=GERMAN_VOICE, language="de", base_url=BASE_URL)
    tts.prewarm()

    session = AgentSession(
        # The end of a turn comes from Kugel Ear, so the agent does not need LiveKit's detector.
        stt=KugelAudioSTT(language="de", base_url=BASE_URL, boosted_phrases=["Berlin", "Selbstbeteiligung"]),
        turn_handling={"turn_detection": "stt"},
        # KugelAudio's LLM speaks the OpenAI protocol, so LiveKit's OpenAI plugin is its client.
        llm=openai.LLM(
            model="mistral-medium-3.5-128b",
            base_url=f"{BASE_URL}/v1",
            api_key=os.environ["KUGELAUDIO_API_KEY"],
        ),
        tts=tts,
        # Stays for barge-in: it notices the caller talking over the agent.
        vad=silero.VAD.load(),
    )

    @session.on("conversation_item_added")
    def on_item(event: ConversationItemAddedEvent) -> None:
        # Handoffs between agents come through here too; only a ChatMessage is a line of speech.
        if isinstance(event.item, ChatMessage):
            print(f"{event.item.role}: {event.item.text_content}", flush=True)

    await session.start(room=ctx.room, agent=Nora())
    await session.say("Guten Tag, hier ist Nora, die digitale Assistentin Ihrer Sachversicherung. Worum geht es?")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
