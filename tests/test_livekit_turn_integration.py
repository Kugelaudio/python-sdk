"""Integration checks for barge-in against the real LiveKit AgentActivity.

``test_livekit_turn.py`` drives a stand-in activity, which proves the bridge's
own logic but not that LiveKit still honours the two private entry points the
bridge calls. These tests instantiate the real ``AgentActivity`` from the
installed livekit-agents and assert that a barge-in actually pauses the audio
output and that a KugelTurn hold actually resumes it.

Deliberately version-tolerant: livekit-agents 1.3 and 1.5 differ in the
``SpeechHandle`` constructor, in ``_paused_speech`` (a handle vs a dataclass),
and in whether ``_interrupt_by_audio_activity`` is gated behind an enable flag.
The assertions therefore target observable behaviour — the audio sink and the
agent state — not LiveKit's internal bookkeeping shape.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

pytest.importorskip("livekit.agents")
pytest.importorskip("pydantic")

from livekit import rtc
from livekit.agents import Agent, AgentSession
from livekit.agents.voice import io
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.speech_handle import SpeechHandle

from kugelaudio.livekit import KugelTurnBridge

from .test_livekit_turn import _HoldPredictor, _detector


class _RecordingAudioOutput(io.AudioOutput):
    """A pausable sink, matching LiveKit's own room output capabilities."""

    def __init__(self) -> None:
        super().__init__(
            label="test",
            next_in_chain=None,
            sample_rate=None,
            capabilities=io.AudioOutputCapabilities(pause=True),
        )
        self.events: list[str] = []

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)

    def flush(self) -> None:
        pass

    def clear_buffer(self) -> None:
        self.events.append("clear")

    def pause(self) -> None:
        self.events.append("pause")

    def resume(self) -> None:
        self.events.append("resume")


def _speech_handle() -> SpeechHandle:
    kwargs: dict[str, object] = {
        "speech_id": "test-speech",
        "allow_interruptions": True,
    }
    parameters = inspect.signature(SpeechHandle.__init__).parameters
    if "input_details" in parameters:  # livekit-agents 1.5+
        from livekit.agents.voice.speech_handle import DEFAULT_INPUT_DETAILS

        kwargs["input_details"] = DEFAULT_INPUT_DETAILS
    return SpeechHandle(**kwargs)  # type: ignore[arg-type]


def _activity_session() -> tuple[AgentSession, AgentActivity, _RecordingAudioOutput]:
    session = AgentSession(turn_detection="manual")
    audio_output = _RecordingAudioOutput()
    session.output.audio = audio_output
    activity = AgentActivity(Agent(instructions="test"), session)
    session._activity = activity
    activity._current_speech = _speech_handle()
    session._update_agent_state("speaking")
    # livekit-agents 1.5 suppresses interruption-by-audio-activity for the first
    # seconds of a session while AEC converges. Model a session past that window;
    # `test_barge_in_is_suppressed_during_aec_warmup` covers the window itself.
    if hasattr(session, "_aec_warmup_remaining"):
        session._aec_warmup_remaining = 0
    return session, activity, audio_output


def _bridge() -> KugelTurnBridge:
    return KugelTurnBridge(_detector(_HoldPredictor()), language="en", barge_in_ms=30)


@pytest.mark.asyncio
async def test_real_activity_pauses_on_barge_in() -> None:
    session, activity, audio_output = _activity_session()
    bridge = _bridge()
    bridge.attach(session)

    bridge._pause_agent_speech()

    assert audio_output.events == ["pause"]
    assert activity._paused_speech is not None
    assert session.agent_state == "listening"
    await bridge.aclose()


@pytest.mark.asyncio
async def test_real_activity_resumes_when_kugelturn_holds() -> None:
    session, activity, audio_output = _activity_session()
    bridge = _bridge()
    bridge.attach(session)
    bridge._pause_agent_speech()
    assert audio_output.events == ["pause"]

    bridge._resume_agent_speech()
    await asyncio.sleep(0.05)  # LiveKit's resume runs on the next loop iteration

    assert audio_output.events == ["pause", "resume"]
    assert activity._paused_speech is None
    assert session.agent_state == "speaking"
    await bridge.aclose()


@pytest.mark.asyncio
async def test_real_activity_resume_is_skipped_without_a_pause() -> None:
    session, _activity, audio_output = _activity_session()
    bridge = _bridge()
    bridge.attach(session)

    bridge._resume_agent_speech()
    await asyncio.sleep(0.05)

    assert audio_output.events == []
    await bridge.aclose()


@pytest.mark.asyncio
async def test_barge_in_is_suppressed_during_aec_warmup() -> None:
    """Pin LiveKit's echo guard: no barge-in until AEC has converged.

    On livekit-agents 1.5 this costs the first seconds of every session, and it
    is deliberate — without it the agent barges in on its own echo.
    """
    session, _activity, audio_output = _activity_session()
    if not hasattr(session, "_aec_warmup_remaining"):
        pytest.skip("livekit-agents predates the AEC warmup guard")
    session._aec_warmup_remaining = 3.0
    bridge = _bridge()
    bridge.attach(session)

    bridge._pause_agent_speech()

    assert audio_output.events == []
    await bridge.aclose()


@pytest.mark.asyncio
async def test_room_audio_output_can_pause() -> None:
    """The demo publishes through the room output; resume needs can_pause."""
    from livekit.agents.voice.room_io import _output

    participant_output = next(
        value
        for name, value in vars(_output).items()
        if name.endswith("AudioOutput") and isinstance(value, type)
    )
    source = inspect.getsource(participant_output.__init__)
    assert "AudioOutputCapabilities(pause=True)" in source
    assert _RecordingAudioOutput().can_pause is True
