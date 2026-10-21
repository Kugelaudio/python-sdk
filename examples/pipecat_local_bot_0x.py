"""Minimal Pipecat 0.0.x voice bot using KugelAudio STT and TTS + local mic/speaker.

For Pipecat 1.x / 0.0.100+, use ``pipecat_local_bot.py`` instead.

Setup (Python 3.11+, once):
    cd packages/public/python-sdk
    uv pip install "pipecat-ai[local,openai]==0.0.62" -e ".[tools]"

Run:
    cd packages/public/python-sdk && uv run python examples/pipecat_local_bot_0x.py

Keys are loaded from examples/.env.local (next to this file).
"""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantContextAggregator,
    LLMUserContextAggregator,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kugelaudio.pipecat import KugelAudioSTTService, KugelAudioTTSService


async def main():
    here = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(here, ".env.local"), override=True)

    if os.environ.get("KUGELAUDIO_LOG", "").lower() in ("debug", "1", "true"):
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        )
        logging.getLogger("kugelaudio").setLevel(logging.DEBUG)
        logger.info("KugelAudio SDK debug logging enabled (KUGELAUDIO_LOG=debug)")

    kugelaudio_key = os.environ.get("KUGELAUDIO_API_KEY")
    kugelaudio_base_url = os.environ.get("KUGELAUDIO_BASE_URL")
    openai_key = os.environ.get("OPENAI_API_KEY")

    missing = [
        name
        for name, val in [
            ("KUGELAUDIO_API_KEY", kugelaudio_key),
            ("OPENAI_API_KEY", openai_key),
        ]
        if not val
    ]
    if missing:
        logger.error(
            f"Missing: {', '.join(missing)}  — set them in examples/.env.local"
        )
        sys.exit(1)

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        )
    )

    # KugelAudio's server decides when the user has finished and the service
    # emits the user started/stopped speaking frames; no local VAD.
    stt_kwargs: dict = {"api_key": kugelaudio_key, "language": "en"}
    if kugelaudio_base_url:
        stt_kwargs["base_url"] = kugelaudio_base_url
    stt = KugelAudioSTTService(**stt_kwargs)

    llm = OpenAILLMService(
        api_key=openai_key,
        model="gpt-4o-mini",
    )

    tts_kwargs: dict = {
        "api_key": kugelaudio_key,
        "model": "kugel-2-turbo",
        "voice_id": 280,
        "sample_rate": 24000,
        "language": "en",
    }
    if kugelaudio_base_url:
        tts_kwargs["base_url"] = kugelaudio_base_url
        logger.info(f"KugelAudio TTS via local ingress: {kugelaudio_base_url}")
    tts = KugelAudioTTSService(**tts_kwargs)
    tts.prewarm()

    context = OpenAILLMContext(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful voice assistant powered by KugelAudio. "
                    "Keep your answers short and conversational — one or two "
                    "sentences max. Be friendly and natural."
                ),
            }
        ]
    )
    user_aggregator = LLMUserContextAggregator(context)
    assistant_aggregator = LLMAssistantContextAggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    allow_interruptions = os.environ.get("ALLOW_INTERRUPTIONS", "1") not in (
        "0",
        "false",
        "False",
    )
    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=allow_interruptions),
    )

    await task.queue_frames(
        [TTSSpeakFrame("Hey! I'm your KugelAudio voice assistant. What's up?")]
    )

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
