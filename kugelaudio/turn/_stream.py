"""Framework-neutral async streaming controller for turn endpointing."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
import logging
import time

from ._detector import (
    TurnDecision,
    TurnDecisionReason,
    TurnOutcomeKind,
    TurnSession,
)
from .errors import TurnAudioError, TurnStateError

logger = logging.getLogger("kugelaudio.turn")

EndTurnCallback = Callable[[TurnDecision], Awaitable[None]]
DecisionCallback = Callable[[TurnDecision], None]


class AsyncTurnEndpoint:
    """Own one streaming turn and run blocking inference outside the event loop."""

    def __init__(
        self,
        turn: TurnSession,
        *,
        on_end_turn: EndTurnCallback,
        on_decision: DecisionCallback | None = None,
        poll_interval_ms: int = 50,
        pre_roll_ms: int = 500,
    ) -> None:
        if poll_interval_ms <= 0:
            raise TurnStateError(
                f"poll_interval_ms must be positive, got {poll_interval_ms}"
            )
        if pre_roll_ms < 0:
            raise TurnStateError(f"pre_roll_ms must be non-negative, got {pre_roll_ms}")
        self._turn = turn
        self._on_end_turn = on_end_turn
        self._on_decision = on_decision
        self._poll_interval_s = poll_interval_ms / 1000
        self._pre_roll_bytes = pre_roll_ms * 16_000 * 2 // 1000
        self._pre_roll = bytearray()
        self._final_segments: list[str] = []
        self._interim = ""
        self._active = False
        self._silence_task: asyncio.Task[None] | None = None
        self._failure: BaseException | None = None
        self._last_notified_reason: TurnDecisionReason | None = None

    @property
    def active(self) -> bool:
        """Whether VAD has opened a user turn."""
        return self._active

    def push_pcm16(self, pcm: bytes, *, sample_rate: int = 16_000) -> None:
        """Feed mono PCM16, retaining a short pre-roll before speech starts."""
        self._raise_if_failed()
        if sample_rate != 16_000:
            raise TurnAudioError(
                f"stream adapter requires 16000 Hz PCM16, got {sample_rate}"
            )
        if not pcm or len(pcm) % 2:
            raise TurnAudioError("stream PCM16 must contain a whole number of samples")
        if self._active:
            self._turn.push_pcm16(pcm, sample_rate=sample_rate)
            return
        self._pre_roll.extend(pcm)
        if len(self._pre_roll) > self._pre_roll_bytes:
            del self._pre_roll[: len(self._pre_roll) - self._pre_roll_bytes]

    def update_interim_transcript(self, text: str) -> None:
        """Replace the current interim segment while preserving finalized segments."""
        self._raise_if_failed()
        if not self._active:
            return
        self._interim = text.strip()
        self._push_accumulated_transcript()

    def update_final_transcript(self, text: str) -> None:
        """Append one finalized STT segment to the current user turn."""
        self._raise_if_failed()
        if not self._active:
            return
        segment = text.strip()
        if segment:
            self._final_segments.append(segment)
        self._interim = ""
        self._push_accumulated_transcript()

    def speech_started(self) -> None:
        """Open a new turn or cancel the pending endpoint for resumed speech."""
        self._raise_if_failed()
        self._cancel_silence_task()
        if self._active:
            self._turn.speech_resumed()
            return
        self._active = True
        self._final_segments.clear()
        self._interim = ""
        if self._pre_roll:
            self._turn.push_pcm16(bytes(self._pre_roll), sample_rate=16_000)
            self._pre_roll.clear()

    def speech_stopped(self, *, initial_silence_ms: int = 0) -> None:
        """Start policy timing from the actual VAD silence onset."""
        self._raise_if_failed()
        if not self._active:
            raise TurnStateError("cannot stop speech before a user turn has started")
        if initial_silence_ms < 0:
            raise TurnStateError(
                f"initial_silence_ms must be non-negative, got {initial_silence_ms}"
            )
        self._cancel_silence_task()
        if not self._turn.has_audio:
            # Some VAD adapters can emit an initial speech lifecycle before their
            # first audio frame. Close that empty lifecycle without attempting a
            # model score; subsequent frames remain available as normal pre-roll.
            logger.debug("ignoring speech stop without audio evidence")
            self._turn.reset_turn()
            self._active = False
            self._final_segments.clear()
            self._interim = ""
            self._last_notified_reason = None
            return
        started_at = time.monotonic() - (initial_silence_ms / 1000)
        task = asyncio.create_task(
            self._monitor_silence(started_at), name="kugelaudio-turn-endpoint"
        )
        task.add_done_callback(self._silence_task_done)
        self._silence_task = task

    async def reset(self) -> None:
        """Cancel pending work and clear the complete per-turn state."""
        await self._cancel_and_wait()
        self._turn.reset_turn()
        self._active = False
        self._pre_roll.clear()
        self._final_segments.clear()
        self._interim = ""
        self._failure = None
        self._last_notified_reason = None

    async def aclose(self) -> None:
        """Stop the endpoint timer without accepting another turn."""
        await self._cancel_and_wait()
        self._active = False

    def record_outcome(
        self,
        kind: TurnOutcomeKind,
        *,
        speaker_pause_ms: int | None = None,
    ) -> None:
        """Forward explicit conversation feedback to the persistent session."""
        self._raise_if_failed()
        self._turn.record_outcome(kind, speaker_pause_ms=speaker_pause_ms)

    async def _monitor_silence(self, started_at: float) -> None:
        while self._active:
            silence_ms = max(0, int((time.monotonic() - started_at) * 1000))
            decision = await asyncio.to_thread(
                self._turn.observe_silence_and_reset,
                silence_ms,
            )
            if self._should_notify(decision):
                if self._on_decision is not None:
                    self._on_decision(decision)
                self._last_notified_reason = decision.reason
            if decision.end_turn:
                self._active = False
                self._final_segments.clear()
                self._interim = ""
                self._last_notified_reason = None
                await self._on_end_turn(decision)
                return
            await asyncio.sleep(self._poll_interval_s)

    def _should_notify(self, decision: TurnDecision) -> bool:
        return (
            decision.inference_ms is not None
            or decision.reason is not self._last_notified_reason
        )

    def _push_accumulated_transcript(self) -> None:
        parts = [*self._final_segments]
        if self._interim:
            parts.append(self._interim)
        self._turn.update_transcript(" ".join(parts))

    def _cancel_silence_task(self) -> None:
        if self._silence_task is not None and not self._silence_task.done():
            self._silence_task.cancel()

    async def _cancel_and_wait(self) -> None:
        task = self._silence_task
        self._cancel_silence_task()
        if task is not None:
            with suppress(asyncio.CancelledError):
                await task
        self._silence_task = None

    def _silence_task_done(self, task: asyncio.Task[None]) -> None:
        if self._silence_task is task:
            self._silence_task = None
        if task.cancelled():
            return
        failure = task.exception()
        if failure is not None:
            self._failure = failure
            logger.error(
                "turn endpoint task failed",
                exc_info=(type(failure), failure, failure.__traceback__),
            )

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise TurnStateError("turn endpoint task failed") from self._failure
