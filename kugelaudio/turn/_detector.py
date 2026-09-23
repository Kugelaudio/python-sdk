"""VAD-gated endpointing policy over the shared ONNX predictor."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ._bundle import DEFAULT_REPO_ID, DEFAULT_REVISION, download_bundle
from ._runtime import (
    ExecutionProvider,
    TurnMessage,
    TurnPredictor,
    TurnProbabilities,
    _validated_history,
)
from ._schema import LanguagePolicy, TurnPolicy, load_schema
from .errors import (
    TurnAudioError,
    TurnBundleError,
    TurnStateError,
    UnsupportedTurnLanguageError,
)

TurnLanguage = Literal["de", "en", "es", "it", "nl"]


class ProbabilityPredictor(Protocol):
    """Inference boundary used by independent conversation sessions."""

    def predict_proba(
        self,
        audio: NDArray[np.float32],
        *,
        transcript: str = "",
        history_messages: Sequence[TurnMessage] = (),
        sample_rate: int = 16000,
    ) -> TurnProbabilities: ...


@runtime_checkable
class StagedProbabilityPredictor(ProbabilityPredictor, Protocol):
    """Predictor that can reuse audio embeddings across temporal decisions."""

    def encode_audio(
        self,
        audio: NDArray[np.float32],
        *,
        sample_rate: int = 16000,
    ) -> NDArray[np.float32]: ...

    def predict_proba_from_audio(
        self,
        audio_embeds: NDArray[np.float32],
        *,
        transcript: str = "",
        history_messages: Sequence[TurnMessage] = (),
    ) -> TurnProbabilities: ...


class TurnDecisionReason(str, Enum):
    """Machine-readable reason for an endpoint-policy result."""

    INSUFFICIENT_SILENCE = "insufficient_silence"
    EVIDENCE_CHANGED = "evidence_changed"
    MODEL_ACTION_DELAY = "model_action_delay"
    MODEL_HOLD = "model_hold"
    MODEL_COMPLETE = "model_complete"
    TIMEOUT = "timeout"


class TurnOutcomeKind(str, Enum):
    """Explicit post-decision feedback supplied by the application."""

    LIKELY_FALSE_CUTOFF = "likely_false_cutoff"
    CONFIRMED_BARGE_IN = "confirmed_barge_in"


@dataclass(frozen=True, slots=True)
class TurnDecision:
    """One endpoint-policy result for the current silence span."""

    end_turn: bool
    reason: TurnDecisionReason
    silence_ms: int
    probabilities: TurnProbabilities | None = None
    inference_ms: float | None = None
    transcript: str = ""
    threshold: float | None = None


class TurnDetector:
    """One loaded model shared by independent per-conversation sessions."""

    def __init__(
        self,
        predictor: ProbabilityPredictor,
        policy: TurnPolicy,
        *,
        variant_path: str,
    ) -> None:
        if policy.model != variant_path:
            raise TurnBundleError(
                f"policy targets {policy.model!r}, but loaded variant is {variant_path!r}"
            )
        self.predictor = predictor
        self.policy = policy
        self.variant_path = variant_path

    @classmethod
    def from_pretrained(
        cls,
        *,
        repo_id: str = DEFAULT_REPO_ID,
        variant: str = "recommended",
        revision: str = DEFAULT_REVISION,
        preset: str | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        cpu_threads: int = 12,
        cpu_streams: int = 1,
        execution_provider: ExecutionProvider | None = None,
    ) -> TurnDetector:
        """Acquire the immutable model revision and load one shared runtime.

        ``execution_provider`` is an explicit portability override. Omit it to
        use the provider calibrated by the bundle policy; local macOS
        development may select ``"cpu"`` because OpenVINO is unavailable.
        """
        if execution_provider not in {None, "cpu", "openvino", "cuda"}:
            raise TurnBundleError(
                "execution_provider must be one of: cpu, openvino, cuda"
            )
        downloaded = download_bundle(
            repo_id=repo_id,
            variant=variant,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        policy = load_schema(downloaded.policy_path, TurnPolicy).select_preset(preset)
        predictor = TurnPredictor.from_verified_bundle(
            downloaded.bundle,
            cpu_threads=cpu_threads,
            cpu_streams=cpu_streams,
            execution_provider=execution_provider
            or (
                "openvino"
                if policy.runtime is not None
                and policy.runtime.startswith("onnxruntime-openvino")
                else "cpu"
            ),
        )
        return cls(predictor, policy, variant_path=downloaded.variant_path)

    def create_session(
        self,
        language: TurnLanguage,
        *,
        action_delay_ms: int | None = None,
        timeout_ms: int | None = None,
    ) -> TurnSession:
        """Create cheap conversation state while sharing the loaded model weights.

        ``action_delay_ms`` overrides the calibrated delay for applications that
        prefer committing a confident completion as soon as the first score is
        available. ``timeout_ms`` overrides the bundle's calibrated fallback.
        """
        try:
            language_policy = self.policy.languages[language]
        except KeyError as exc:
            raise UnsupportedTurnLanguageError(
                f"unsupported turn-detection language {language!r}; "
                f"available: {', '.join(sorted(self.policy.languages))}"
            ) from exc
        effective_action_delay_ms = (
            language_policy.action_delay_ms
            if action_delay_ms is None
            else action_delay_ms
        )
        effective_timeout_ms = (
            language_policy.timeout_ms if timeout_ms is None else timeout_ms
        )
        if effective_action_delay_ms < self.policy.score_after_silence_ms:
            raise TurnStateError(
                "action_delay_ms cannot precede score_after_silence_ms: "
                f"{effective_action_delay_ms} < {self.policy.score_after_silence_ms}"
            )
        if effective_timeout_ms <= 0:
            raise TurnStateError(
                f"timeout_ms must be positive, got {effective_timeout_ms}"
            )
        if effective_timeout_ms < effective_action_delay_ms:
            raise TurnStateError(
                "timeout_ms cannot precede action_delay_ms: "
                f"{effective_timeout_ms} < {effective_action_delay_ms}"
            )
        language_policy = language_policy.model_copy(
            update={
                "action_delay_ms": effective_action_delay_ms,
                "timeout_ms": effective_timeout_ms,
            }
        )
        return TurnSession(
            self.predictor,
            language_policy,
            score_after_silence_ms=self.policy.score_after_silence_ms,
            end_turn_probability=self.policy.end_turn_probability,
        )


class TurnSession:
    """Audio, transcript, and endpoint state for one live user turn."""

    def __init__(
        self,
        predictor: ProbabilityPredictor,
        policy: LanguagePolicy,
        *,
        score_after_silence_ms: int,
        end_turn_probability: str = "complete",
    ) -> None:
        self._predictor = predictor
        self._policy = policy
        self._score_after_silence_ms = score_after_silence_ms
        self._end_turn_probability = end_turn_probability
        self._sample_rate = 16000
        self._window_samples = 128000
        # eot-bench (KUG-1379) surfaced a customer-bundle case where the model
        # scored p(complete) high on a turn with almost no buffered audio yet
        # (leading dead air before the caller starts speaking). Below this
        # floor, a "complete" score isn't trustworthy enough to end the turn.
        self._min_complete_audio_samples = int(0.05 * self._sample_rate)
        self._audio = np.empty(0, dtype=np.float32)
        self._transcript = ""
        self._history_messages: tuple[TurnMessage, ...] = ()
        self._scored_history: tuple[TurnMessage, ...] = ()
        self._scored_transcript = ""
        self._last_silence_ms = 0
        self._probabilities: TurnProbabilities | None = None
        self._model_complete = False
        self._decision_threshold = policy.threshold
        self._next_temporal_score_index = 0
        self._interruption_risk = 0.0
        self._continuation_pause_ema_s = 0.3
        self._risk_before_turn_reset: float | None = None
        self._ended = False
        self._evidence_version = 0
        self._silence_version = 0
        self._audio_version = 0
        self._encoded_audio: NDArray[np.float32] | None = None
        self._encoded_audio_version = -1
        self._scored_audio_version = -1
        self._lock = RLock()

    @property
    def effective_timeout_ms(self) -> int:
        """Resolved timeout after applying SDK or bundle policy selection."""
        return self._policy.timeout_ms

    @property
    def has_audio(self) -> bool:
        """Whether the active turn contains audio evidence that can be scored."""
        with self._lock:
            return self._audio.size > 0

    def push_audio(
        self,
        audio: NDArray[np.float32],
        *,
        sample_rate: int = 16000,
    ) -> None:
        """Append explicitly normalized mono float32 PCM to the rolling window."""
        array = _validate_audio_chunk(audio, sample_rate=sample_rate)
        with self._lock:
            self._ensure_active()
            combined = np.concatenate((self._audio, array))
            self._audio = np.ascontiguousarray(combined[-self._window_samples :])
            self._audio_version += 1

    def push_pcm16(self, pcm: bytes, *, sample_rate: int = 16000) -> None:
        """Explicitly decode little-endian signed PCM16 into normalized float32."""
        if not pcm:
            raise TurnAudioError("PCM16 audio must contain at least one sample")
        if len(pcm) % 2:
            raise TurnAudioError(f"PCM16 byte length must be even, got {len(pcm)}")
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        self.push_audio(audio, sample_rate=sample_rate)

    def update_transcript(self, transcript: str) -> None:
        """Replace the live ASR transcript without invalidating an active score."""
        with self._lock:
            self._ensure_active()
            self._transcript = transcript

    def observe_silence(self, duration_ms: int) -> TurnDecision:
        """Evaluate one monotonically increasing VAD silence duration."""
        return self._observe_silence(duration_ms, reset_on_end=False)

    def update_history(self, history_messages: Sequence[TurnMessage]) -> None:
        """Replace authoritative prior dialogue; retain it until reset_conversation().

        Empty prior messages are omitted. Completed turns are never appended
        automatically: the application owns corrections and actual spoken replies.
        """
        history = _validated_history(history_messages)
        with self._lock:
            self._ensure_active()
            if history != self._history_messages:
                self._history_messages = history
                self._evidence_version += 1

    def observe_silence_and_reset(self, duration_ms: int) -> TurnDecision:
        """Evaluate silence and atomically reset when the policy ends the turn."""
        return self._observe_silence(duration_ms, reset_on_end=True)

    def _observe_silence(
        self,
        duration_ms: int,
        *,
        reset_on_end: bool,
    ) -> TurnDecision:
        with self._lock:
            self._ensure_active()
            if duration_ms < 0:
                raise TurnStateError(
                    f"duration_ms must be non-negative, got {duration_ms}"
                )
            if duration_ms < self._last_silence_ms:
                raise TurnStateError(
                    f"silence duration moved backwards: {duration_ms} < {self._last_silence_ms}; "
                    "call speech_resumed() when VAD detects speech"
                )
            self._last_silence_ms = duration_ms
            if duration_ms < self._score_after_silence_ms:
                return TurnDecision(
                    False,
                    TurnDecisionReason.INSUFFICIENT_SILENCE,
                    duration_ms,
                    transcript=self._transcript,
                    threshold=self._policy.threshold,
                )
            temporal = self._policy.temporal
            score_index = self._next_temporal_score_index
            temporal_score_due = (
                temporal is not None
                and score_index < len(temporal.score_points_ms)
                and duration_ms >= temporal.score_points_ms[score_index]
            )
            history_changed = self._scored_history != self._history_messages
            if (
                self._probabilities is not None
                and not temporal_score_due
                and not history_changed
            ):
                decision = self._evaluate_cached_decision(duration_ms)
                if reset_on_end and decision.end_turn:
                    self._reset_turn_locked()
                return decision
            if self._audio.size == 0:
                raise TurnAudioError(
                    "cannot score a turn before push_audio() or push_pcm16()"
                )
            audio = self._audio
            transcript = self._transcript
            history_messages = self._history_messages
            if temporal is not None and not temporal_score_due:
                score_index = max(0, score_index - 1)
            evidence_version = self._evidence_version
            silence_version = self._silence_version
            audio_version = self._audio_version
            staged_predictor = (
                self._predictor
                if isinstance(self._predictor, StagedProbabilityPredictor)
                else None
            )
            encoded_audio = (
                self._encoded_audio
                if self._encoded_audio_version == audio_version
                else None
            )
            reuse_probabilities = (
                staged_predictor is not None
                and self._probabilities is not None
                and self._scored_audio_version == audio_version
                and self._scored_transcript == transcript
                and self._scored_history == history_messages
            )
            cached_probabilities = self._probabilities

        if reuse_probabilities:
            if cached_probabilities is None:
                raise TurnStateError("cached temporal score disappeared")
            probabilities = cached_probabilities
            inference_ms = 0.0
        else:
            inference_started_at = time.perf_counter()
            # Published custom predictors need no new keyword for legacy calls.
            history_kwargs = (
                {"history_messages": history_messages} if history_messages else {}
            )
            if staged_predictor is not None:
                if encoded_audio is None:
                    encoded_audio = staged_predictor.encode_audio(
                        audio,
                        sample_rate=self._sample_rate,
                    )
                probabilities = staged_predictor.predict_proba_from_audio(
                    encoded_audio,
                    transcript=transcript,
                    **history_kwargs,
                )
            else:
                probabilities = self._predictor.predict_proba(
                    audio,
                    transcript=transcript,
                    sample_rate=self._sample_rate,
                    **history_kwargs,
                )
            inference_ms = (time.perf_counter() - inference_started_at) * 1000

        with self._lock:
            if (
                evidence_version != self._evidence_version
                or silence_version != self._silence_version
            ):
                return TurnDecision(
                    False,
                    TurnDecisionReason.EVIDENCE_CHANGED,
                    duration_ms,
                    inference_ms=inference_ms,
                    transcript=self._transcript,
                    threshold=self._policy.threshold,
                )
            self._ensure_active()
            if (
                staged_predictor is not None
                and encoded_audio is not None
                and audio_version == self._audio_version
            ):
                self._encoded_audio = encoded_audio
                self._encoded_audio_version = audio_version
            self._probabilities = probabilities
            self._scored_transcript = transcript
            self._scored_history = history_messages
            self._scored_audio_version = audio_version
            completion_score = _completion_score(
                probabilities,
                mode=self._end_turn_probability,
            )
            threshold = self._policy.threshold
            if temporal is not None:
                threshold_logit = (
                    temporal.threshold_logits[score_index]
                    + temporal.feedback_penalty[score_index] * self._interruption_risk
                    + temporal.pause_penalty[score_index] * self._pause_style()
                )
                threshold = _sigmoid(threshold_logit)
                if temporal_score_due:
                    self._next_temporal_score_index += 1
            self._decision_threshold = threshold
            self._model_complete = (
                completion_score >= threshold
                and self._audio.size >= self._min_complete_audio_samples
            )
            decision = self._evaluate_cached_decision(
                duration_ms,
                inference_ms=inference_ms,
            )
            if reset_on_end and decision.end_turn:
                self._reset_turn_locked()
            return decision

    def _evaluate_cached_decision(
        self,
        duration_ms: int,
        *,
        inference_ms: float | None = None,
    ) -> TurnDecision:
        """Evaluate timing for a score while ``self._lock`` is held."""
        if self._probabilities is None:
            raise TurnStateError(
                "cannot evaluate endpoint timing without a model score"
            )
        if self._model_complete:
            if duration_ms >= self._policy.action_delay_ms:
                self._ended = True
                return TurnDecision(
                    True,
                    TurnDecisionReason.MODEL_COMPLETE,
                    duration_ms,
                    self._probabilities,
                    inference_ms,
                    self._scored_transcript,
                    self._decision_threshold,
                )
            return TurnDecision(
                False,
                TurnDecisionReason.MODEL_ACTION_DELAY,
                duration_ms,
                self._probabilities,
                inference_ms,
                self._scored_transcript,
                self._decision_threshold,
            )
        if duration_ms >= self._policy.timeout_ms:
            self._ended = True
            return TurnDecision(
                True,
                TurnDecisionReason.TIMEOUT,
                duration_ms,
                self._probabilities,
                inference_ms,
                self._scored_transcript,
                self._decision_threshold,
            )
        return TurnDecision(
            False,
            TurnDecisionReason.MODEL_HOLD,
            duration_ms,
            self._probabilities,
            inference_ms,
            self._scored_transcript,
            self._decision_threshold,
        )

    def speech_resumed(self) -> None:
        """Cancel a pending endpoint while retaining the current turn context."""
        with self._lock:
            self._ensure_active()
            pause_ms = self._last_silence_ms
            temporal = self._policy.temporal
            if temporal is not None and pause_ms > 0:
                self._interruption_risk *= temporal.span_decay
                self._update_pause_ema(pause_ms / 1000)
            self._last_silence_ms = 0
            self._probabilities = None
            self._scored_transcript = ""
            self._model_complete = False
            self._decision_threshold = self._policy.threshold
            self._next_temporal_score_index = 0
            self._silence_version += 1

    def record_outcome(
        self,
        kind: TurnOutcomeKind,
        *,
        speaker_pause_ms: int | None = None,
    ) -> None:
        """Apply an application-confirmed outcome to conversation memory."""
        with self._lock:
            temporal = self._policy.temporal
            if temporal is None:
                raise TurnStateError("the active policy does not enable feedback")
            if kind is TurnOutcomeKind.LIKELY_FALSE_CUTOFF:
                if speaker_pause_ms is None or speaker_pause_ms <= 0:
                    raise TurnStateError(
                        "likely false cutoff feedback requires positive speaker_pause_ms"
                    )
                if self._risk_before_turn_reset is not None:
                    self._interruption_risk = (
                        self._risk_before_turn_reset * temporal.span_decay
                    )
                self._interruption_risk += temporal.false_cutoff_risk_increment
                self._update_pause_ema(speaker_pause_ms / 1000)
                self._risk_before_turn_reset = None
                return
            if kind is TurnOutcomeKind.CONFIRMED_BARGE_IN:
                if speaker_pause_ms is not None:
                    raise TurnStateError(
                        "confirmed barge-in feedback must not carry speaker_pause_ms"
                    )
                self._risk_before_turn_reset = None
                return
            raise TurnStateError(f"unsupported turn outcome {kind!r}")

    def reset_conversation(self) -> None:
        """Clear turn evidence and all speaker-adaptive endpoint memory."""
        with self._lock:
            self._reset_turn_locked()
            self._history_messages = ()
            self._interruption_risk = 0.0
            self._continuation_pause_ema_s = 0.3
            self._risk_before_turn_reset = None

    def _ensure_active(self) -> None:
        if self._ended:
            raise TurnStateError(
                "turn already ended; call reset_turn() before processing the next user turn"
            )

    def reset_turn(self) -> None:
        """Clear turn evidence while retaining speaker-adaptive conversation memory."""
        with self._lock:
            self._reset_turn_locked()

    def _reset_turn_locked(self) -> None:
        temporal = self._policy.temporal
        if temporal is not None:
            self._risk_before_turn_reset = self._interruption_risk
            self._interruption_risk *= temporal.turn_decay
        self._audio = np.empty(0, dtype=np.float32)
        self._transcript = ""
        self._scored_transcript = ""
        self._scored_history = ()
        self._last_silence_ms = 0
        self._probabilities = None
        self._model_complete = False
        self._decision_threshold = self._policy.threshold
        self._next_temporal_score_index = 0
        self._ended = False
        self._evidence_version += 1
        self._silence_version += 1
        self._audio_version += 1
        self._encoded_audio = None
        self._encoded_audio_version = -1
        self._scored_audio_version = -1

    def _update_pause_ema(self, pause_s: float) -> None:
        self._continuation_pause_ema_s = (
            0.7 * self._continuation_pause_ema_s + 0.3 * min(pause_s, 3.0)
        )

    def _pause_style(self) -> float:
        return min(
            2.0,
            max(0.0, (self._continuation_pause_ema_s - 0.3) / 1.0),
        )


def _validate_audio_chunk(
    audio: NDArray[np.float32], *, sample_rate: int
) -> NDArray[np.float32]:
    """Reject implicit resampling, downmixing, clipping, or dtype conversion."""
    array = np.asarray(audio)
    if array.dtype != np.float32:
        raise TurnAudioError(f"audio dtype must be float32, got {array.dtype}")
    if array.ndim != 1:
        raise TurnAudioError(
            f"audio must be mono with shape [samples], got {array.shape}"
        )
    if array.size == 0:
        raise TurnAudioError("audio must contain at least one sample")
    if sample_rate != 16000:
        raise TurnAudioError(
            f"audio sample_rate must be 16000, got {sample_rate}; resample explicitly"
        )
    if not np.isfinite(array).all():
        raise TurnAudioError("audio contains NaN or infinite samples")
    peak = float(np.max(np.abs(array)))
    if peak > 1.0:
        raise TurnAudioError(
            f"float32 audio must be normalized to [-1, 1], peak={peak}"
        )
    return np.ascontiguousarray(array)


def _completion_score(
    probabilities: TurnProbabilities,
    *,
    mode: str,
) -> float:
    if mode == "complete":
        return probabilities.complete
    if mode == "complete / (complete + incomplete)":
        denominator = probabilities.complete + probabilities.incomplete
        if denominator <= 0.0:
            raise TurnStateError(
                "complete and incomplete probabilities have zero total mass"
            )
        return probabilities.complete / denominator
    raise TurnStateError(f"unsupported end-turn probability mode {mode!r}")


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)
