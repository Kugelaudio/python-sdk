"""Contract tests for the LiveKit semantic turn bridge."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest

pytest.importorskip("livekit.agents")
pytest.importorskip("pydantic")

from livekit import rtc
from livekit.agents import (
    Agent,
    ModelSettings,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    stt,
)

from kugelaudio.livekit import KugelTurnBridge
from kugelaudio.turn import TurnDecision, TurnDetector, TurnProbabilities
from kugelaudio.turn._schema import TurnPolicy
from kugelaudio.turn.errors import TurnStateError


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


class _Session:
    def __init__(self) -> None:
        self.callbacks: dict[str, list[Callable[[object], None]]] = {}
        self.commits = 0

    def on(self, event: str, callback: Callable[[object], None]) -> None:
        self.callbacks.setdefault(event, []).append(callback)

    def off(self, event: str, callback: Callable[[object], None]) -> None:
        self.callbacks[event].remove(callback)

    def commit_user_turn(self) -> None:
        self.commits += 1

    def emit(self, event: str, value: object) -> None:
        for callback in self.callbacks[event]:
            callback(value)


class _Agent:
    def __init__(self, session: _Session) -> None:
        self.session = session


def _detector(predictor: _Predictor) -> TurnDetector:
    policy = TurnPolicy.model_validate(
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
    )
    return TurnDetector(predictor, policy, variant_path="onnx/recommended")


@pytest.mark.asyncio
async def test_livekit_bridge_tees_audio_transcript_and_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _Predictor()
    session = _Session()
    decisions: list[TurnDecision] = []
    bridge = KugelTurnBridge(
        _detector(predictor),
        language="en",
        on_decision=decisions.append,
        vad_silence_ms=200,
        action_delay_ms=200,
        poll_interval_ms=5,
    )
    assert bridge.effective_timeout_ms == 1000

    async def default_stt_node(
        agent: Agent,
        audio: AsyncIterable[rtc.AudioFrame],
        model_settings: ModelSettings,
    ) -> AsyncIterator[stt.SpeechEvent]:
        async for _ in audio:
            pass
        if False:
            yield cast(stt.SpeechEvent, object())

    monkeypatch.setattr(Agent.default, "stt_node", staticmethod(default_stt_node))

    async def audio() -> AsyncIterator[rtc.AudioFrame]:
        yield rtc.AudioFrame(
            data=np.ones(1600, dtype="<i2").tobytes(),
            sample_rate=16_000,
            num_channels=1,
            samples_per_channel=1600,
        )

    bridge.attach(session)
    session.emit(
        "user_state_changed",
        UserStateChangedEvent(old_state="listening", new_state="speaking"),
    )
    async for _ in bridge.stt_node(
        cast(Agent, _Agent(session)), audio(), cast(ModelSettings, object())
    ):
        pass
    session.emit(
        "user_input_transcribed",
        UserInputTranscribedEvent(transcript="hello", is_final=False),
    )
    session.emit(
        "user_state_changed",
        UserStateChangedEvent(old_state="speaking", new_state="listening"),
    )

    for _ in range(100):
        if session.commits:
            break
        await asyncio.sleep(0.01)

    assert session.commits == 1
    assert predictor.transcript == "hello"
    assert decisions[-1].end_turn is True
    assert decisions[-1].transcript == "hello"
    await bridge.aclose()


@dataclass(slots=True)
class _HoldPredictor:
    """Score every window as incomplete so the policy never ends the turn."""

    transcript: str = ""

    def predict_proba(
        self,
        audio: np.ndarray,
        *,
        transcript: str = "",
        sample_rate: int = 16_000,
    ) -> TurnProbabilities:
        self.transcript = transcript
        return TurnProbabilities(0.1, 0.8, 0.05, 0.05)


class _Activity:
    """The private LiveKit AgentActivity surface the bridge drives for barge-in."""

    def __init__(self) -> None:
        self._interruption_by_audio_activity_enabled = False
        self.interrupts: list[bool] = []
        self.resume_timeouts: list[float] = []

    def _interrupt_by_audio_activity(self) -> None:
        # Record the flag as the bridge left it: LiveKit no-ops when it is False.
        self.interrupts.append(self._interruption_by_audio_activity_enabled)
        self._interruption_by_audio_activity_enabled = False

    def _start_false_interruption_timer(self, timeout: float) -> None:
        self.resume_timeouts.append(timeout)


class _SpeakingSession(_Session):
    def __init__(self) -> None:
        super().__init__()
        self.agent_state = "speaking"
        self._activity = _Activity()


async def _drive_audio(bridge: KugelTurnBridge, session: _Session) -> None:
    async def audio() -> AsyncIterator[rtc.AudioFrame]:
        yield rtc.AudioFrame(
            data=np.ones(1600, dtype="<i2").tobytes(),
            sample_rate=16_000,
            num_channels=1,
            samples_per_channel=1600,
        )

    async for _ in bridge.stt_node(
        cast(Agent, _Agent(session)), audio(), cast(ModelSettings, object())
    ):
        pass


@pytest.fixture
def _patched_stt_node(monkeypatch: pytest.MonkeyPatch) -> None:
    async def default_stt_node(
        agent: Agent,
        audio: AsyncIterable[rtc.AudioFrame],
        model_settings: ModelSettings,
    ) -> AsyncIterator[stt.SpeechEvent]:
        async for _ in audio:
            pass
        if False:
            yield cast(stt.SpeechEvent, object())

    monkeypatch.setattr(Agent.default, "stt_node", staticmethod(default_stt_node))


@pytest.mark.asyncio
async def test_barge_in_pauses_agent_speech_after_overlap(
    _patched_stt_node: None,
) -> None:
    session = _SpeakingSession()
    bridge = KugelTurnBridge(
        _detector(_HoldPredictor()),
        language="en",
        vad_silence_ms=200,
        poll_interval_ms=5,
        barge_in_ms=30,
    )
    bridge.attach(session)
    session.emit(
        "user_state_changed",
        UserStateChangedEvent(old_state="listening", new_state="speaking"),
    )
    await asyncio.sleep(0.12)

    # The flag must be True at call time or LiveKit's manual-mode guard no-ops.
    assert session._activity.interrupts == [True]
    await bridge.aclose()


@pytest.mark.asyncio
async def test_barge_in_does_not_pause_before_the_overlap_threshold(
    _patched_stt_node: None,
) -> None:
    session = _SpeakingSession()
    bridge = KugelTurnBridge(
        _detector(_HoldPredictor()),
        language="en",
        vad_silence_ms=200,
        poll_interval_ms=5,
        barge_in_ms=500,
    )
    bridge.attach(session)
    session.emit(
        "user_state_changed",
        UserStateChangedEvent(old_state="listening", new_state="speaking"),
    )
    await asyncio.sleep(0.05)
    session.emit(
        "user_state_changed",
        UserStateChangedEvent(old_state="speaking", new_state="listening"),
    )
    await asyncio.sleep(0.05)

    assert session._activity.interrupts == []
    await bridge.aclose()


@pytest.mark.asyncio
async def test_barge_in_resumes_when_kugelturn_rejects_the_overlap(
    _patched_stt_node: None,
) -> None:
    session = _SpeakingSession()
    decisions: list[TurnDecision] = []
    bridge = KugelTurnBridge(
        _detector(_HoldPredictor()),
        language="en",
        on_decision=decisions.append,
        vad_silence_ms=200,
        poll_interval_ms=5,
        timeout_ms=10_000,
        barge_in_ms=30,
    )
    bridge.attach(session)
    session.emit(
        "user_state_changed",
        UserStateChangedEvent(old_state="listening", new_state="speaking"),
    )
    await _drive_audio(bridge, session)
    await asyncio.sleep(0.12)
    assert session._activity.interrupts == [True]

    session.emit(
        "user_state_changed",
        UserStateChangedEvent(old_state="speaking", new_state="listening"),
    )
    for _ in range(100):
        if session._activity.resume_timeouts:
            break
        await asyncio.sleep(0.01)

    # Zero delay runs LiveKit's own resume path immediately instead of waiting
    # out the false-interruption backstop.
    assert session._activity.resume_timeouts == [0]
    assert session.commits == 0
    assert decisions[-1].end_turn is False
    await bridge.aclose()


@pytest.mark.asyncio
async def test_barge_in_is_not_resumed_once_the_turn_commits(
    _patched_stt_node: None,
) -> None:
    session = _SpeakingSession()
    bridge = KugelTurnBridge(
        _detector(_Predictor()),
        language="en",
        vad_silence_ms=200,
        action_delay_ms=200,
        poll_interval_ms=5,
        barge_in_ms=30,
    )
    bridge.attach(session)
    session.emit(
        "user_state_changed",
        UserStateChangedEvent(old_state="listening", new_state="speaking"),
    )
    await _drive_audio(bridge, session)
    await asyncio.sleep(0.12)
    session.emit(
        "user_state_changed",
        UserStateChangedEvent(old_state="speaking", new_state="listening"),
    )
    for _ in range(100):
        if session.commits:
            break
        await asyncio.sleep(0.01)

    assert session.commits == 1
    assert session._activity.resume_timeouts == []
    await bridge.aclose()


def test_barge_in_requires_the_livekit_pause_hooks() -> None:
    bridge = KugelTurnBridge(
        _detector(_Predictor()),
        language="en",
        barge_in_ms=300,
    )
    with pytest.raises(TurnStateError, match="agent_state"):
        bridge.attach(_Session())


def test_barge_in_requires_the_private_activity_hooks() -> None:
    class _StaleActivity:
        pass

    class _StaleSession(_Session):
        def __init__(self) -> None:
            super().__init__()
            self.agent_state = "speaking"
            self._activity = _StaleActivity()

    bridge = KugelTurnBridge(
        _detector(_Predictor()),
        language="en",
        barge_in_ms=300,
    )
    with pytest.raises(TurnStateError, match="_interrupt_by_audio_activity"):
        bridge.attach(_StaleSession())


def test_barge_in_ms_must_be_positive() -> None:
    with pytest.raises(TurnStateError, match="barge_in_ms"):
        KugelTurnBridge(_detector(_Predictor()), language="en", barge_in_ms=0)
