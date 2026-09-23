"""Contract tests for the Pipecat semantic turn-stop strategy."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("pipecat.turns.user_stop")
pytest.importorskip("pydantic")

from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)

from kugelaudio.pipecat import KugelTurnStopStrategy
from kugelaudio.turn import TurnDetector, TurnProbabilities
from kugelaudio.turn._schema import TurnPolicy
from kugelaudio.turn.errors import TurnAudioError


@dataclass(slots=True)
class _Predictor:
    transcript: str = ""

    def predict_proba(
        self,
        audio: np.ndarray,
        *,
        transcript: str = "",
        sample_rate: int = 16_000,
    ) -> TurnProbabilities:
        self.transcript = transcript
        return TurnProbabilities(0.9, 0.05, 0.03, 0.02)


def _detector(predictor: _Predictor) -> TurnDetector:
    return TurnDetector(
        predictor,
        TurnPolicy.model_validate(
            {
                "schema_version": 1,
                "model": "onnx/recommended",
                "preset": "test",
                "score_after_silence_ms": 200,
                "end_turn_probability": "complete",
                "languages": {
                    language: {
                        "threshold": 0.6,
                        "action_delay_ms": 600,
                        "timeout_ms": 1000,
                    }
                    for language in ("de", "en", "es", "it", "nl")
                },
            }
        ),
        variant_path="onnx/recommended",
    )


@pytest.mark.asyncio
async def test_pipecat_strategy_consumes_frames_and_stops_turn() -> None:
    predictor = _Predictor()
    strategy = KugelTurnStopStrategy(
        _detector(predictor),
        language="en",
        vad_silence_ms=200,
        action_delay_ms=200,
        poll_interval_ms=5,
    )
    stopped = asyncio.Event()

    @strategy.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(strategy: object, params: object) -> None:
        stopped.set()

    await strategy.process_frame(
        InputAudioRawFrame(
            audio=np.ones(1600, dtype="<i2").tobytes(),
            sample_rate=16_000,
            num_channels=1,
        )
    )
    await strategy.process_frame(VADUserStartedSpeakingFrame())
    await strategy.process_frame(
        InterimTranscriptionFrame(
            text="hello", user_id="caller", timestamp="2026-07-14T00:00:00Z"
        )
    )
    await strategy.process_frame(VADUserStoppedSpeakingFrame())

    await asyncio.wait_for(stopped.wait(), timeout=1)

    assert predictor.transcript == "hello"
    await strategy.cleanup()


@pytest.mark.asyncio
async def test_pipecat_start_strategy_reset_preserves_active_vad_turn() -> None:
    """Pipecat resets stop strategies when its start strategy opens a turn."""
    predictor = _Predictor()
    strategy = KugelTurnStopStrategy(
        _detector(predictor),
        language="en",
        vad_silence_ms=200,
        action_delay_ms=200,
        poll_interval_ms=5,
    )
    stopped = asyncio.Event()

    @strategy.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(strategy: object, params: object) -> None:
        stopped.set()

    await strategy.process_frame(
        InputAudioRawFrame(
            audio=np.ones(1600, dtype="<i2").tobytes(),
            sample_rate=16_000,
            num_channels=1,
        )
    )
    await strategy.process_frame(VADUserStartedSpeakingFrame())

    # UserTurnController invokes this as soon as a transcription satisfies its
    # start strategy, before the matching VAD stop reaches the stop strategy.
    await strategy.reset()
    await strategy.process_frame(
        InterimTranscriptionFrame(
            text="hello", user_id="caller", timestamp="2026-07-14T00:00:00Z"
        )
    )
    await strategy.process_frame(VADUserStoppedSpeakingFrame())

    await asyncio.wait_for(stopped.wait(), timeout=1)

    assert predictor.transcript == "hello"
    await strategy.cleanup()


@pytest.mark.asyncio
async def test_pipecat_strategy_rejects_implicit_downmix() -> None:
    strategy = KugelTurnStopStrategy(_detector(_Predictor()), language="en")
    assert strategy.effective_timeout_ms == 1000

    with pytest.raises(TurnAudioError, match="mono"):
        await strategy.process_frame(
            InputAudioRawFrame(
                audio=np.ones(3200, dtype="<i2").tobytes(),
                sample_rate=16_000,
                num_channels=2,
            )
        )

    await strategy.cleanup()


