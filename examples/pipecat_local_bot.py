"""Minimal Pipecat 1.x voice bot using KugelAudio TTS + local mic/speaker.

Uses headphones or low speaker volume recommended — local transport has
no echo cancellation, so speaker output can feed back into the mic.
Interruptions are enabled by default; set ``ALLOW_INTERRUPTIONS=0`` to disable
if speaker echo triggers false cutoffs (use headphones instead).

Setup (Python 3.11+, once):
    cd packages/public/python-sdk
    uv sync --extra pipecat --extra tools --python 3.12
    uv pip install "pipecat-ai[deepgram,local]"

Run:
    cd packages/public/python-sdk && uv run python examples/pipecat_local_bot.py

Keys are loaded from examples/.env.local (next to this file).
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kugelaudio.pipecat import KugelAudioTTSService


async def main():
    here = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(here, ".env.local"), override=True)

    kugelaudio_key = os.environ.get("KUGELAUDIO_API_KEY")
    kugelaudio_base_url = os.environ.get("KUGELAUDIO_BASE_URL")
    openai_key = os.environ.get("OPENAI_API_KEY")
    deepgram_key = os.environ.get("DEEPGRAM_API_KEY")

    missing = [
        name
        for name, val in [
            ("KUGELAUDIO_API_KEY", kugelaudio_key),
            ("OPENAI_API_KEY", openai_key),
            ("DEEPGRAM_API_KEY", deepgram_key),
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
            audio_out_sample_rate=24000,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=0.8,
                    start_secs=0.3,
                    stop_secs=1.0,
                    min_volume=0.5,
                )
            ),
        )
    )

    stt = DeepgramSTTService(api_key=deepgram_key)

    llm = OpenAILLMService(
        api_key=openai_key,
        settings=OpenAILLMService.Settings(
            model="gpt-4o-mini",
            system_instruction=(
                "You are a helpful voice assistant powered by KugelAudio. "
                "Keep your answers short and conversational — one or two "
                "sentences max. Be friendly and natural."
            ),
        ),
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

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
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
