"""Tests for the framework-neutral asynchronous endpoint controller."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np
import pytest

pytest.importorskip("pydantic")

from kugelaudio.turn._detector import TurnDecision, TurnSession
from kugelaudio.turn._runtime import TurnProbabilities
from kugelaudio.turn._schema import LanguagePolicy
from kugelaudio.turn._stream import AsyncTurnEndpoint
from kugelaudio.turn.errors import TurnStateError


@dataclass(slots=True)
class _Predictor:
    transcripts: list[str] = field(default_factory=list)
    sample_counts: list[int] = field(default_factory=list)

    def predict_proba(
        self,
        audio: np.ndarray,
        *,
        transcript: str = "",
        sample_rate: int = 16_000,
    ) -> TurnProbabilities:
        self.transcripts.append(transcript)
        self.sample_counts.append(audio.size)
        return TurnProbabilities(0.9, 0.05, 0.03, 0.02)


def _turn(predictor: _Predictor) -> TurnSession:
    return TurnSession(
        predictor,
        LanguagePolicy(threshold=0.6, action_delay_ms=200, timeout_ms=1000),
        score_after_silence_ms=200,
    )


@pytest.mark.asyncio
async def test_endpoint_accumulates_preroll_transcripts_and_ends() -> None:
    predictor = _Predictor()
    ended = asyncio.Event()
    decisions: list[TurnDecision] = []
    observed: list[TurnDecision] = []

    async def on_end(decision: TurnDecision) -> None:
        decisions.append(decision)
        ended.set()

    endpoint = AsyncTurnEndpoint(
        _turn(predictor),
        on_end_turn=on_end,
        on_decision=observed.append,
        poll_interval_ms=5,
        pre_roll_ms=500,
    )
    endpoint.push_pcm16(np.ones(1600, dtype="<i2").tobytes())
    endpoint.speech_started()
    endpoint.update_final_transcript("hello")
    endpoint.update_interim_transcript("there")
    endpoint.speech_stopped(initial_silence_ms=200)

    await asyncio.wait_for(ended.wait(), timeout=1)

    assert decisions[0].end_turn is True
    assert predictor.transcripts == ["hello there"]
    assert predictor.sample_counts == [1600]
    assert [decision.reason.value for decision in observed] == [
        "model_complete"
    ]
    assert observed[0].inference_ms is not None
    assert endpoint.active is False


@pytest.mark.asyncio
async def test_speech_resume_cancels_pending_endpoint() -> None:
    predictor = _Predictor()
    ended = asyncio.Event()

    async def on_end(decision: TurnDecision) -> None:
        ended.set()

    endpoint = AsyncTurnEndpoint(
        _turn(predictor), on_end_turn=on_end, poll_interval_ms=5
    )
    endpoint.push_pcm16(np.ones(1600, dtype="<i2").tobytes())
    endpoint.speech_started()
    endpoint.update_interim_transcript("not done")
    endpoint.speech_stopped(initial_silence_ms=100)
    endpoint.speech_started()

    await asyncio.sleep(0.25)

    assert ended.is_set() is False
    assert endpoint.active is True
    await endpoint.aclose()


@pytest.mark.asyncio
async def test_empty_vad_lifecycle_closes_without_scoring() -> None:
    predictor = _Predictor()
    ended = asyncio.Event()

    async def on_end(decision: TurnDecision) -> None:
        ended.set()

    endpoint = AsyncTurnEndpoint(
        _turn(predictor), on_end_turn=on_end, poll_interval_ms=5
    )
    endpoint.speech_started()
    endpoint.speech_stopped(initial_silence_ms=200)

    assert endpoint.active is False
    assert predictor.sample_counts == []
    assert ended.is_set() is False

    endpoint.push_pcm16(np.ones(1600, dtype="<i2").tobytes())
    endpoint.speech_started()
    endpoint.speech_stopped(initial_silence_ms=200)

    await asyncio.wait_for(ended.wait(), timeout=1)
    assert predictor.sample_counts == [1600]


@pytest.mark.asyncio
async def test_endpoint_surfaces_background_failure_on_next_input() -> None:
    async def fail_commit(decision: TurnDecision) -> None:
        raise RuntimeError("commit failed")

    endpoint = AsyncTurnEndpoint(
        _turn(_Predictor()), on_end_turn=fail_commit, poll_interval_ms=5
    )
    endpoint.push_pcm16(np.ones(1600, dtype="<i2").tobytes())
    endpoint.speech_started()
    endpoint.speech_stopped(initial_silence_ms=1000)

    await asyncio.sleep(0.05)
    with pytest.raises(TurnStateError, match="endpoint task failed") as failure:
        endpoint.update_interim_transcript("next")

    assert isinstance(failure.value.__cause__, RuntimeError)
