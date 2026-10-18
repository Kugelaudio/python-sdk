"""Native LiveKit Agents bridge for KugelAudio semantic turn detection."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterable, AsyncIterator, Callable
from typing import Protocol, cast, runtime_checkable

from livekit import rtc
from livekit.agents import (
    Agent,
    ModelSettings,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    stt,
)

from kugelaudio.turn import (
    TurnDecision,
    TurnDecisionReason,
    TurnDetector,
    TurnLanguage,
    TurnOutcomeKind,
)
from kugelaudio.turn._stream import AsyncTurnEndpoint
from kugelaudio.turn.errors import TurnAudioError, TurnStateError

logger = logging.getLogger("kugelaudio.livekit")


@runtime_checkable
class _LiveKitSession(Protocol):
    def on(self, event: str, callback: Callable[[object], None]) -> object: ...

    def off(self, event: str, callback: Callable[[object], None]) -> object: ...

    def commit_user_turn(self) -> object: ...


class KugelTurnBridge:
    """Bind one LiveKit Agent session to a shared Kugel turn detector."""

    def __init__(
        self,
        detector: TurnDetector,
        *,
        language: TurnLanguage,
        on_decision: Callable[[TurnDecision], None] | None = None,
        vad_silence_ms: int = 200,
        action_delay_ms: int | None = None,
        timeout_ms: int | None = None,
        poll_interval_ms: int = 50,
        pre_roll_ms: int = 500,
        barge_in_ms: int | None = None,
    ) -> None:
        if vad_silence_ms < 0:
            raise TurnStateError(
                f"vad_silence_ms must be non-negative, got {vad_silence_ms}"
            )
        if barge_in_ms is not None and barge_in_ms <= 0:
            raise TurnStateError(
                f"barge_in_ms must be positive when set, got {barge_in_ms}"
            )
        self._session: _LiveKitSession | None = None
        self._vad_silence_ms = vad_silence_ms
        self._barge_in_ms = barge_in_ms
        self._on_decision = on_decision
        self._barge_in_task: asyncio.Task[None] | None = None
        self._paused_by_barge_in = False
        self._resampler: rtc.AudioResampler | None = None
        self._input_rate: int | None = None
        self._closed = False
        self._commit_futures: set[asyncio.Future[object]] = set()
        self._commit_failure: BaseException | None = None
        turn = detector.create_session(
            language,
            action_delay_ms=action_delay_ms,
            timeout_ms=timeout_ms,
        )
        self._effective_timeout_ms = turn.effective_timeout_ms
        self._endpoint = AsyncTurnEndpoint(
            turn,
            on_end_turn=self._commit_user_turn,
            on_decision=self._observe_decision,
            poll_interval_ms=poll_interval_ms,
            pre_roll_ms=pre_roll_ms,
        )

    @property
    def effective_timeout_ms(self) -> int:
        """Resolved timeout used by the underlying turn session."""
        return self._effective_timeout_ms

    def record_turn_outcome(
        self,
        kind: TurnOutcomeKind,
        *,
        speaker_pause_ms: int | None = None,
    ) -> None:
        """Record an app-confirmed false cutoff or intentional barge-in."""
        self._endpoint.record_outcome(kind, speaker_pause_ms=speaker_pause_ms)

    def attach(self, session: object) -> None:
        """Register transcript and VAD listeners on exactly one AgentSession."""
        if self._closed:
            raise TurnStateError("KugelTurnBridge is closed and cannot be reattached")
        if not isinstance(session, _LiveKitSession):
            raise TurnStateError(
                "session does not implement the LiveKit AgentSession API"
            )
        typed_session = cast(_LiveKitSession, session)
        if self._session is typed_session:
            return
        if self._session is not None:
            raise TurnStateError(
                "KugelTurnBridge is already attached to another session"
            )
        if self._barge_in_ms is not None:
            self._verify_barge_in_support(typed_session)
        self._session = typed_session
        typed_session.on(
            "user_input_transcribed",
            cast(Callable[[object], None], self._on_transcript),
        )
        typed_session.on(
            "user_state_changed", cast(Callable[[object], None], self._on_user_state)
        )

    async def stt_node(
        self,
        agent: Agent,
        audio: AsyncIterable[rtc.AudioFrame],
        model_settings: ModelSettings,
    ) -> AsyncIterator[stt.SpeechEvent]:
        """Tee raw user audio into KugelTurn while preserving LiveKit's STT node."""
        self.attach(agent.session)

        async def tee_audio() -> AsyncIterator[rtc.AudioFrame]:
            async for frame in audio:
                self._push_audio_frame(frame)
                yield frame

        async for event in Agent.default.stt_node(agent, tee_audio(), model_settings):
            yield event

    async def aclose(self) -> None:
        """Detach event listeners and cancel pending endpoint work."""
        session = self._session
        if session is not None:
            session.off(
                "user_input_transcribed",
                cast(Callable[[object], None], self._on_transcript),
            )
            session.off(
                "user_state_changed",
                cast(Callable[[object], None], self._on_user_state),
            )
        self._session = None
        self._cancel_barge_in()
        self._paused_by_barge_in = False
        await self._endpoint.aclose()
        try:
            if self._commit_futures:
                await asyncio.gather(*self._commit_futures)
        except Exception as exc:
            raise TurnStateError("LiveKit user-turn commit failed") from exc
        finally:
            self._closed = True
        self._raise_commit_failure()

    def _push_audio_frame(self, frame: rtc.AudioFrame) -> None:
        self._raise_commit_failure()
        if frame.num_channels != 1:
            raise TurnAudioError(
                f"LiveKit turn detection requires mono audio, got {frame.num_channels} channels"
            )
        if frame.sample_rate == 16_000:
            self._endpoint.push_pcm16(
                frame.data.cast("B").tobytes(), sample_rate=16_000
            )
            return
        if self._resampler is None:
            self._input_rate = frame.sample_rate
            self._resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=16_000,
                num_channels=1,
                quality=rtc.AudioResamplerQuality.HIGH,
            )
        elif frame.sample_rate != self._input_rate:
            raise TurnAudioError(
                "LiveKit input sample rate changed during a session: "
                f"{self._input_rate} -> {frame.sample_rate}"
            )
        for output in self._resampler.push(frame):
            self._endpoint.push_pcm16(
                output.data.cast("B").tobytes(), sample_rate=16_000
            )

    def _on_transcript(self, event: UserInputTranscribedEvent) -> None:
        self._raise_commit_failure()
        if event.is_final:
            self._endpoint.update_final_transcript(event.transcript)
        else:
            self._endpoint.update_interim_transcript(event.transcript)

    def _on_user_state(self, event: UserStateChangedEvent) -> None:
        self._raise_commit_failure()
        if event.new_state == "speaking":
            self._endpoint.speech_started()
            self._schedule_barge_in()
        elif event.new_state == "listening" and self._endpoint.active:
            self._cancel_barge_in()
            self._endpoint.speech_stopped(initial_silence_ms=self._vad_silence_ms)

    # region barge-in
    #
    # ``turn_detection="manual"`` disables every one of LiveKit's own
    # audio-activity interrupt paths (``AgentActivity.on_vad_inference_done``
    # and both transcript hooks return early, and
    # ``_resolve_interruption_detection`` yields ``None``), because LiveKit
    # assumes the application owns turn taking. KugelTurn owns end-of-turn but
    # still wants LiveKit's pause/resume behaviour for barge-in, so the bridge
    # drives ``AgentActivity`` directly. Those two entry points are private, so
    # ``_verify_barge_in_support`` pins them explicitly rather than letting a
    # LiveKit upgrade silently turn barge-in back off.

    _ACTIVITY_HOOKS = (
        "_interrupt_by_audio_activity",
        "_start_false_interruption_timer",
    )
    # LiveKit 1.5 added a gate that makes ``_interrupt_by_audio_activity`` a
    # no-op under manual turn detection; 1.3 has no such flag and needs no
    # opt-in. Treat it as optional so both API generations work.
    _ACTIVITY_ENABLE_FLAG = "_interruption_by_audio_activity_enabled"

    def _verify_barge_in_support(self, session: _LiveKitSession) -> None:
        """Fail loudly when the installed LiveKit lacks the pause hooks."""
        for attribute in ("agent_state", "_activity"):
            if not hasattr(session, attribute):
                raise TurnStateError(
                    f"barge_in_ms requires AgentSession.{attribute}; "
                    "the installed livekit-agents release does not expose it"
                )
        activity = getattr(session, "_activity", None)
        if activity is None:
            # The activity only exists once the agent has started; the hooks are
            # re-checked on the first barge-in.
            return
        self._verify_activity_hooks(activity)

    def _verify_activity_hooks(self, activity: object) -> None:
        missing = [name for name in self._ACTIVITY_HOOKS if not hasattr(activity, name)]
        if missing:
            raise TurnStateError(
                "barge_in_ms requires AgentActivity"
                f".{', .'.join(missing)}; the installed livekit-agents release "
                "does not expose them"
            )

    def _schedule_barge_in(self) -> None:
        if self._barge_in_ms is None or self._closed:
            return
        self._cancel_barge_in()
        task = asyncio.create_task(
            self._barge_in_after_delay(), name="kugelaudio-turn-barge-in"
        )
        task.add_done_callback(self._barge_in_done)
        self._barge_in_task = task

    def _cancel_barge_in(self) -> None:
        task = self._barge_in_task
        if task is not None and not task.done():
            task.cancel()
        self._barge_in_task = None

    async def _barge_in_after_delay(self) -> None:
        assert self._barge_in_ms is not None
        await asyncio.sleep(self._barge_in_ms / 1000)
        self._pause_agent_speech()

    def _pause_agent_speech(self) -> None:
        """Pause the agent mid-sentence once overlap outlasts ``barge_in_ms``."""
        session = self._session
        if session is None:
            return
        if getattr(session, "agent_state", None) != "speaking":
            return
        activity = getattr(session, "_activity", None)
        if activity is None:
            return
        self._verify_activity_hooks(activity)
        # LiveKit 1.5 disables this flag for manual turn detection, making the
        # call below a no-op without it. ``_interrupt_by_audio_activity``
        # restores it to the session default (still disabled) on the way out, so
        # LiveKit's own interrupt paths stay off and each barge-in re-arms it
        # deliberately.
        if hasattr(activity, self._ACTIVITY_ENABLE_FLAG):
            setattr(activity, self._ACTIVITY_ENABLE_FLAG, True)
        activity._interrupt_by_audio_activity()
        self._paused_by_barge_in = True
        logger.info("paused agent speech for barge-in")

    def _resume_agent_speech(self) -> None:
        """Resume a paused agent turn as soon as KugelTurn rejects the overlap."""
        if not self._paused_by_barge_in:
            return
        self._paused_by_barge_in = False
        session = self._session
        activity = getattr(session, "_activity", None) if session else None
        if activity is None:
            return
        self._verify_activity_hooks(activity)
        # Re-arm LiveKit's own false-interruption timer at zero delay: that runs
        # its resume path (agent-state restore, audio resume, the
        # `agent_false_interruption` event) instead of reimplementing it here.
        activity._start_false_interruption_timer(0)
        logger.info("resumed agent speech; KugelTurn rejected the overlap")

    def _barge_in_done(self, task: asyncio.Task[None]) -> None:
        if self._barge_in_task is task:
            self._barge_in_task = None
        if task.cancelled():
            return
        failure = task.exception()
        if failure is not None:
            self._commit_failure = failure
            logger.error(
                "barge-in pause failed",
                exc_info=(type(failure), failure, failure.__traceback__),
            )

    def _observe_decision(self, decision: TurnDecision) -> None:
        if decision.reason is TurnDecisionReason.MODEL_HOLD:
            self._resume_agent_speech()
        if self._on_decision is not None:
            self._on_decision(decision)

    # endregion

    async def _commit_user_turn(self, decision: TurnDecision) -> None:
        session = self._session
        if session is None:
            raise TurnStateError("LiveKit session detached before turn commit")
        # A committed turn is a confirmed interruption: LiveKit's own commit path
        # hard-interrupts the paused speech, so it must not be resumed.
        self._cancel_barge_in()
        self._paused_by_barge_in = False
        result = session.commit_user_turn()
        # LiveKit 1.3 returns None; newer releases return an awaitable transcript.
        # RISK: this compatibility branch is version-sensitive and must remain
        # covered against both supported LiveKit API generations.
        if inspect.isawaitable(result):
            future = asyncio.ensure_future(result)
            self._commit_futures.add(future)
            future.add_done_callback(self._commit_done)

    def _commit_done(self, future: asyncio.Future[object]) -> None:
        self._commit_futures.discard(future)
        if future.cancelled():
            return
        failure = future.exception()
        if failure is not None:
            self._commit_failure = failure
            logger.error(
                "LiveKit user-turn commit failed",
                exc_info=(type(failure), failure, failure.__traceback__),
            )

    def _raise_commit_failure(self) -> None:
        if self._commit_failure is not None:
            raise TurnStateError(
                "LiveKit user-turn commit failed"
            ) from self._commit_failure
