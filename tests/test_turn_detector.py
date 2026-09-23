"""Unit tests for the VAD-gated customer endpoint policy."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("pydantic")

from kugelaudio.turn._detector import (
    TurnDecisionReason,
    TurnDetector,
    TurnLanguage,
    TurnOutcomeKind,
    TurnSession,
)
from kugelaudio.turn._runtime import TurnMessage, TurnPredictor, TurnProbabilities
from kugelaudio.turn._schema import LanguagePolicy, TurnPolicy
from kugelaudio.turn.errors import (
    TurnAudioError,
    TurnBundleError,
    TurnStateError,
    UnsupportedTurnLanguageError,
)


@dataclass(slots=True)
class _Predictor:
    probabilities: TurnProbabilities
    calls: int = 0
    expected_transcript: str = "still speaking"

    def predict_proba(
        self,
        audio: np.ndarray,
        *,
        transcript: str = "",
        sample_rate: int = 16000,
    ) -> TurnProbabilities:
        assert audio.dtype == np.float32
        assert transcript == self.expected_transcript
        assert sample_rate == 16000
        self.calls += 1
        return self.probabilities


def _policy() -> TurnPolicy:
    return TurnPolicy.model_validate(
        {
            "schema_version": 1,
            "model": "onnx/recommended",
            "preset": "conservative",
            "score_after_silence_ms": 200,
            "end_turn_probability": "complete",
            "languages": {
                language: {
                    "threshold": 0.6,
                    "action_delay_ms": 600,
                    "timeout_ms": 2000,
                }
                for language in ("de", "en", "es", "it", "nl")
            },
        }
    )


def _session(complete: float) -> tuple[TurnSession, _Predictor]:
    predictor = _Predictor(
        TurnProbabilities(
            complete=complete,
            incomplete=1.0 - complete,
            backchannel=0.0,
            wait=0.0,
        )
    )
    session = TurnSession(
        predictor,
        LanguagePolicy(threshold=0.6, action_delay_ms=600, timeout_ms=2000),
        score_after_silence_ms=200,
    )
    session.push_audio(np.ones(1600, dtype=np.float32) * 0.1)
    session.update_transcript("still speaking")
    return session, predictor


def test_from_pretrained_allows_explicit_cpu_portability_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy().model_copy(update={"runtime": "onnxruntime-openvino==1.24.1"})
    predictor = _Predictor(TurnProbabilities(0.8, 0.2, 0.0, 0.0))
    load_runtime = Mock(return_value=predictor)
    monkeypatch.setattr(
        "kugelaudio.turn._detector.download_bundle",
        Mock(
            return_value=SimpleNamespace(
                bundle=object(),
                policy_path=object(),
                variant_path="onnx/recommended",
            )
        ),
    )
    monkeypatch.setattr(
        "kugelaudio.turn._detector.load_schema", Mock(return_value=policy)
    )
    monkeypatch.setattr(TurnPredictor, "from_verified_bundle", load_runtime)

    detector = TurnDetector.from_pretrained(execution_provider="cpu")

    assert detector.predictor is predictor
    assert load_runtime.call_args.kwargs["execution_provider"] == "cpu"


def test_from_pretrained_allows_explicit_cuda_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy()
    predictor = _Predictor(TurnProbabilities(0.8, 0.2, 0.0, 0.0))
    load_runtime = Mock(return_value=predictor)
    monkeypatch.setattr(
        "kugelaudio.turn._detector.download_bundle",
        Mock(
            return_value=SimpleNamespace(
                bundle=object(),
                policy_path=object(),
                variant_path="onnx/recommended",
            )
        ),
    )
    monkeypatch.setattr(
        "kugelaudio.turn._detector.load_schema", Mock(return_value=policy)
    )
    monkeypatch.setattr(TurnPredictor, "from_verified_bundle", load_runtime)

    detector = TurnDetector.from_pretrained(execution_provider="cuda")

    assert detector.predictor is predictor
    assert load_runtime.call_args.kwargs["execution_provider"] == "cuda"


def test_from_pretrained_rejects_unknown_execution_provider() -> None:
    with pytest.raises(TurnBundleError, match="execution_provider"):
        TurnDetector.from_pretrained(execution_provider="metal")  # type: ignore[arg-type]


def test_complete_waits_for_action_delay_then_ends() -> None:
    session, predictor = _session(0.8)

    assert (
        session.observe_silence(100).reason is TurnDecisionReason.INSUFFICIENT_SILENCE
    )
    at_score = session.observe_silence(200)
    at_action = session.observe_silence(600)

    assert at_score.end_turn is False
    assert at_score.reason is TurnDecisionReason.MODEL_ACTION_DELAY
    assert at_score.inference_ms is not None
    assert at_score.transcript == "still speaking"
    assert at_score.threshold == 0.6
    assert at_action.end_turn is True
    assert at_action.reason is TurnDecisionReason.MODEL_COMPLETE
    assert at_action.inference_ms is None
    assert predictor.calls == 1


def test_complete_ignored_on_near_empty_audio_buffer() -> None:
    """eot-bench regression: a confident score on ~0.9s of near-empty leading
    audio must not end the turn (KUG-1379 failure-analysis finding)."""
    predictor = _Predictor(
        TurnProbabilities(complete=0.9, incomplete=0.1, backchannel=0.0, wait=0.0)
    )
    session = TurnSession(
        predictor,
        LanguagePolicy(threshold=0.6, action_delay_ms=600, timeout_ms=2000),
        score_after_silence_ms=200,
    )
    session.push_audio(
        np.ones(400, dtype=np.float32) * 0.1
    )  # 25ms, well under the floor
    session.update_transcript("still speaking")

    at_score = session.observe_silence(200)
    at_action = session.observe_silence(600)
    decision = session.observe_silence(2000)

    assert at_score.reason is TurnDecisionReason.MODEL_HOLD
    assert at_action.end_turn is False
    assert at_action.reason is TurnDecisionReason.MODEL_HOLD
    assert decision.end_turn is True
    assert decision.reason is TurnDecisionReason.TIMEOUT
    assert predictor.calls == 1


def test_session_action_delay_override_commits_first_confident_score() -> None:
    predictor = _Predictor(TurnProbabilities(0.8, 0.2, 0.0, 0.0))
    session = TurnDetector(
        predictor,
        _policy(),
        variant_path="onnx/recommended",
    ).create_session("en", action_delay_ms=200)
    session.push_audio(np.ones(1600, dtype=np.float32) * 0.1)
    session.update_transcript("still speaking")

    decision = session.observe_silence(200)

    assert decision.end_turn is True
    assert decision.reason is TurnDecisionReason.MODEL_COMPLETE
    assert decision.inference_ms is not None
    assert predictor.calls == 1


def test_conditional_complete_policy_renormalizes_binary_mass() -> None:
    predictor = _Predictor(TurnProbabilities(0.4, 0.1, 0.5, 0.0))
    raw = _policy().model_dump()
    raw["end_turn_probability"] = "complete / (complete + incomplete)"
    raw["languages"]["en"]["action_delay_ms"] = 200
    detector = TurnDetector(
        predictor,
        TurnPolicy.model_validate(raw),
        variant_path="onnx/recommended",
    )
    session = detector.create_session("en")
    session.push_audio(np.ones(1600, dtype=np.float32) * 0.1)
    session.update_transcript("still speaking")

    decision = session.observe_silence(200)

    assert decision.end_turn is True
    assert decision.reason is TurnDecisionReason.MODEL_COMPLETE


def test_policy_selects_named_temporal_preset() -> None:
    raw = _policy().model_dump()
    raw["schema_version"] = 2
    raw["presets"] = {
        "responsive-300ms": {
            "en": {
                "threshold": 0.7,
                "action_delay_ms": 200,
                "timeout_ms": 3500,
                "temporal": {
                    "score_points_ms": [200, 300, 400, 600],
                    "threshold_logits": [2.0, 1.6, 0.75, 0.45],
                    "feedback_penalty": [0.0, 0.0, 0.0, 0.0],
                    "pause_penalty": [0.0, 0.0, 0.0, 0.0],
                },
            }
        }
    }
    policy = TurnPolicy.model_validate(raw).select_preset("responsive-300ms")

    assert policy.preset == "responsive-300ms"
    assert tuple(policy.languages) == ("en",)
    assert policy.languages["en"].temporal is not None

    with pytest.raises(TurnBundleError, match="unknown turn policy preset"):
        policy.select_preset("missing")


@pytest.mark.parametrize("action_delay_ms", [199, 10001])
def test_session_rejects_action_delay_outside_policy_bounds(
    action_delay_ms: int,
) -> None:
    detector = TurnDetector(
        _Predictor(TurnProbabilities(0.8, 0.2, 0.0, 0.0)),
        _policy(),
        variant_path="onnx/recommended",
    )

    with pytest.raises(TurnStateError, match="action_delay_ms"):
        detector.create_session("en", action_delay_ms=action_delay_ms)


def test_session_uses_bundle_default_timeout() -> None:
    predictor = _Predictor(TurnProbabilities(0.2, 0.8, 0.0, 0.0))
    session = TurnDetector(
        predictor,
        _policy(),
        variant_path="onnx/recommended",
    ).create_session("en")
    session.push_audio(np.ones(1600, dtype=np.float32) * 0.1)
    session.update_transcript("still speaking")

    assert session.effective_timeout_ms == 2000
    assert session.observe_silence(200).reason is TurnDecisionReason.MODEL_HOLD
    assert session.observe_silence(1999).end_turn is False
    decision = session.observe_silence(2000)

    assert decision.end_turn is True
    assert decision.reason is TurnDecisionReason.TIMEOUT


def test_session_explicit_none_uses_calibrated_language_timeout() -> None:
    predictor = _Predictor(TurnProbabilities(0.2, 0.8, 0.0, 0.0))
    session = TurnDetector(
        predictor,
        _policy(),
        variant_path="onnx/recommended",
    ).create_session("en", timeout_ms=None)
    session.push_audio(np.ones(1600, dtype=np.float32) * 0.1)
    session.update_transcript("still speaking")

    assert session.effective_timeout_ms == 2000
    assert session.observe_silence(1999).end_turn is False
    decision = session.observe_silence(2000)

    assert decision.end_turn is True
    assert decision.reason is TurnDecisionReason.TIMEOUT


def test_temporal_policy_rescores_at_each_configured_duration() -> None:
    @dataclass(slots=True)
    class SequencePredictor:
        values: list[TurnProbabilities]
        calls: int = 0

        def predict_proba(
            self,
            audio: np.ndarray,
            *,
            transcript: str = "",
            sample_rate: int = 16000,
        ) -> TurnProbabilities:
            value = self.values[self.calls]
            self.calls += 1
            return value

    predictor = SequencePredictor(
        [
            TurnProbabilities(0.4, 0.6, 0.0, 0.0),
            TurnProbabilities(0.8, 0.2, 0.0, 0.0),
        ]
    )
    raw = _policy().model_dump()
    raw["schema_version"] = 2
    raw["languages"]["en"].update(
        {
            "action_delay_ms": 200,
            "temporal": {
                "score_points_ms": [200, 300, 400, 600],
                "threshold_logits": [0.0, 0.0, 0.0, 0.0],
                "feedback_penalty": [1.0, 1.0, 1.0, 1.0],
                "pause_penalty": [0.0, 0.0, 0.0, 0.0],
            },
        }
    )
    detector = TurnDetector(
        predictor,
        TurnPolicy.model_validate(raw),
        variant_path="onnx/recommended",
    )
    session = detector.create_session("en")
    session.push_audio(np.ones(1600, dtype=np.float32) * 0.1)

    assert session.observe_silence(200).reason is TurnDecisionReason.MODEL_HOLD
    decision = session.observe_silence(300)

    assert decision.end_turn is True
    assert decision.reason is TurnDecisionReason.MODEL_COMPLETE
    assert predictor.calls == 2


def test_temporal_policy_reuses_staged_audio_until_evidence_changes() -> None:
    @dataclass(slots=True)
    class StagedPredictor:
        audio_calls: int = 0
        fusion_calls: int = 0

        def predict_proba(
            self,
            audio: np.ndarray,
            *,
            transcript: str = "",
            sample_rate: int = 16000,
        ) -> TurnProbabilities:
            raise AssertionError("TurnSession should use staged inference")

        def encode_audio(
            self,
            audio: np.ndarray,
            *,
            sample_rate: int = 16000,
        ) -> np.ndarray:
            self.audio_calls += 1
            return np.asarray([[[float(audio.size)]]], dtype=np.float32)

        def predict_proba_from_audio(
            self,
            audio_embeds: np.ndarray,
            *,
            transcript: str = "",
        ) -> TurnProbabilities:
            self.fusion_calls += 1
            return TurnProbabilities(0.4, 0.6, 0.0, 0.0)

    predictor = StagedPredictor()
    raw = _policy().model_dump()
    raw["schema_version"] = 2
    raw["languages"]["en"].update(
        {
            "action_delay_ms": 200,
            "temporal": {
                "score_points_ms": [200, 300, 400, 600],
                "threshold_logits": [0.0, 0.0, 0.0, 0.0],
                "feedback_penalty": [0.0, 0.0, 0.0, 0.0],
                "pause_penalty": [0.0, 0.0, 0.0, 0.0],
            },
        }
    )
    session = TurnDetector(
        predictor,
        TurnPolicy.model_validate(raw),
        variant_path="onnx/recommended",
    ).create_session("en")
    session.push_audio(np.ones(1600, dtype=np.float32) * 0.1)
    session.update_transcript("first")

    session.observe_silence(200)
    cached = session.observe_silence(300)
    session.update_transcript("revised")
    session.observe_silence(400)
    session.push_audio(np.ones(160, dtype=np.float32) * 0.1)
    session.observe_silence(600)

    assert cached.inference_ms == 0.0
    assert predictor.audio_calls == 2
    assert predictor.fusion_calls == 3


def test_false_cutoff_feedback_raises_later_temporal_threshold() -> None:
    predictor = _Predictor(TurnProbabilities(0.6, 0.4, 0.0, 0.0))
    raw = _policy().model_dump()
    raw["languages"]["en"].update(
        {
            "action_delay_ms": 200,
            "temporal": {
                "score_points_ms": [200],
                "threshold_logits": [0.0],
                "feedback_penalty": [1.0],
                "pause_penalty": [0.0],
            },
        }
    )
    session = TurnDetector(
        predictor,
        TurnPolicy.model_validate(raw),
        variant_path="onnx/recommended",
    ).create_session("en")
    session.record_outcome(
        TurnOutcomeKind.LIKELY_FALSE_CUTOFF,
        speaker_pause_ms=350,
    )
    session.push_audio(np.ones(1600, dtype=np.float32) * 0.1)
    session.update_transcript("still speaking")

    decision = session.observe_silence(200)

    assert decision.end_turn is False
    assert decision.threshold == pytest.approx(0.7310586)


@pytest.mark.parametrize("timeout_ms", [0, 599])
def test_session_rejects_timeout_before_action_delay(timeout_ms: int) -> None:
    detector = TurnDetector(
        _Predictor(TurnProbabilities(0.8, 0.2, 0.0, 0.0)),
        _policy(),
        variant_path="onnx/recommended",
    )

    with pytest.raises(TurnStateError, match="timeout_ms"):
        detector.create_session("en", timeout_ms=timeout_ms)


def test_incomplete_uses_timeout_fallback() -> None:
    session, predictor = _session(0.2)

    assert session.observe_silence(200).reason is TurnDecisionReason.MODEL_HOLD
    decision = session.observe_silence(2000)

    assert decision.end_turn is True
    assert decision.reason is TurnDecisionReason.TIMEOUT
    assert predictor.calls == 1


def test_speech_resume_cancels_and_rescores() -> None:
    session, predictor = _session(0.8)

    session.observe_silence(200)
    session.speech_resumed()
    session.push_audio(np.ones(800, dtype=np.float32) * 0.1)
    session.observe_silence(200)

    assert predictor.calls == 2


def test_revised_transcript_is_used_on_the_next_silence_episode() -> None:
    session, predictor = _session(0.8)

    session.observe_silence(200)
    predictor.expected_transcript = "revised hypothesis"
    session.update_transcript("revised hypothesis")
    session.observe_silence(400)
    assert predictor.calls == 1

    session.speech_resumed()
    session.observe_silence(200)
    assert predictor.calls == 2


def test_transcript_revision_during_inference_keeps_snapshot_score() -> None:
    started = Event()
    release = Event()

    class BlockingPredictor(_Predictor):
        def predict_proba(
            self,
            audio: np.ndarray,
            *,
            transcript: str = "",
            sample_rate: int = 16000,
        ) -> TurnProbabilities:
            started.set()
            assert release.wait(timeout=2)
            return super().predict_proba(
                audio, transcript=transcript, sample_rate=sample_rate
            )

    predictor = BlockingPredictor(TurnProbabilities(0.8, 0.2, 0.0, 0.0))
    session = TurnSession(
        predictor,
        LanguagePolicy(threshold=0.6, action_delay_ms=600, timeout_ms=2000),
        score_after_silence_ms=200,
    )
    session.push_audio(np.ones(1600, dtype=np.float32) * 0.1)
    session.update_transcript("still speaking")

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(session.observe_silence, 200)
        assert started.wait(timeout=2)
        session.update_transcript("revised hypothesis")
        release.set()
        decision = result.result(timeout=2)

    assert decision.reason is TurnDecisionReason.MODEL_ACTION_DELAY
    assert decision.transcript == "still speaking"
    assert session.observe_silence(250).reason is TurnDecisionReason.MODEL_ACTION_DELAY
    assert predictor.calls == 1


def test_reset_is_required_after_end() -> None:
    session, _ = _session(0.8)
    session.observe_silence(600)

    with pytest.raises(TurnStateError, match="reset_turn"):
        session.observe_silence(700)
    with pytest.raises(TurnStateError, match="reset_turn"):
        session.push_audio(np.ones(10, dtype=np.float32))
    with pytest.raises(TurnStateError, match="reset_turn"):
        session.update_transcript("next turn")

    session.reset_turn()
    with pytest.raises(TurnAudioError, match="before push_audio"):
        session.observe_silence(200)


def test_atomic_endpoint_observation_resets_ended_turn() -> None:
    session, _ = _session(0.8)

    assert session.has_audio is True
    decision = session.observe_silence_and_reset(600)

    assert decision.end_turn is True
    assert session.has_audio is False
    session.push_audio(np.ones(10, dtype=np.float32))
    assert session.has_audio is True


def test_atomic_endpoint_observation_does_not_lock_during_inference() -> None:
    started = Event()
    release = Event()

    class BlockingPredictor(_Predictor):
        def predict_proba(
            self,
            audio: np.ndarray,
            *,
            transcript: str = "",
            sample_rate: int = 16000,
        ) -> TurnProbabilities:
            started.set()
            assert release.wait(timeout=2)
            return super().predict_proba(
                audio, transcript=transcript, sample_rate=sample_rate
            )

    predictor = BlockingPredictor(TurnProbabilities(0.8, 0.2, 0.0, 0.0))
    session = TurnSession(
        predictor,
        LanguagePolicy(threshold=0.6, action_delay_ms=600, timeout_ms=2000),
        score_after_silence_ms=200,
    )
    session.push_audio(np.ones(1600, dtype=np.float32) * 0.1)
    session.update_transcript("still speaking")

    with ThreadPoolExecutor(max_workers=2) as executor:
        observation = executor.submit(session.observe_silence_and_reset, 200)
        assert started.wait(timeout=2)
        update = executor.submit(session.update_transcript, "revised hypothesis")
        try:
            update.result(timeout=0.5)
        finally:
            release.set()
        decision = observation.result(timeout=2)

    assert decision.reason is TurnDecisionReason.MODEL_ACTION_DELAY
    assert decision.transcript == "still speaking"


def test_silence_must_be_monotonic_until_speech_resume() -> None:
    session, _ = _session(0.2)
    session.observe_silence(300)

    with pytest.raises(TurnStateError, match="moved backwards"):
        session.observe_silence(250)


def test_pcm16_decode_is_explicit() -> None:
    session, _ = _session(0.2)
    session.reset_turn()

    session.push_pcm16(np.asarray([0, 32767, -32768], dtype="<i2").tobytes())
    session.update_transcript("still speaking")

    assert session.observe_silence(200).reason is TurnDecisionReason.MODEL_HOLD
    with pytest.raises(TurnAudioError, match="even"):
        session.push_pcm16(b"x")


def test_detector_rejects_policy_variant_mismatch() -> None:
    predictor = _Predictor(TurnProbabilities(0.8, 0.2, 0.0, 0.0))

    with pytest.raises(TurnBundleError, match="policy targets"):
        TurnDetector(predictor, _policy(), variant_path="onnx/other")


def test_detector_creates_only_supported_language_sessions() -> None:
    predictor = _Predictor(TurnProbabilities(0.8, 0.2, 0.0, 0.0))
    detector = TurnDetector(predictor, _policy(), variant_path="onnx/recommended")

    assert isinstance(detector.create_session("en"), TurnSession)
    with pytest.raises(UnsupportedTurnLanguageError, match="unsupported"):
        detector.create_session(cast(TurnLanguage, "fr"))


def test_history_is_explicit_snapshotted_and_persists_only_within_conversation() -> None:
    from kugelaudio.turn import TurnMessage

    class ContextPredictor(_Predictor):
        def predict_proba(
            self, audio: np.ndarray, *, transcript: str = "", sample_rate: int = 16000,
            history_messages: Sequence[TurnMessage] = (),
        ) -> TurnProbabilities:
            self.seen = tuple(history_messages)
            return super().predict_proba(audio, transcript=transcript, sample_rate=sample_rate)

    predictor = ContextPredictor(TurnProbabilities(0.8, 0.2, 0.0, 0.0))
    session = TurnDetector(predictor, _policy(), variant_path="onnx/recommended").create_session("en")
    history = [TurnMessage("assistant", "Where to?")]
    session.update_history(history)
    history.clear()
    for _ in range(2):
        session.push_audio(np.ones(1600, dtype=np.float32) * 0.1)
        session.update_transcript("still speaking")
        assert session.observe_silence_and_reset(600).end_turn
        assert predictor.seen == (TurnMessage("assistant", "Where to?"),)
    session.reset_conversation()
    session.push_audio(np.ones(1600, dtype=np.float32) * 0.1)
    session.update_transcript("still speaking")
    session.observe_silence(200)
    assert predictor.seen == ()


def test_history_revision_rejects_in_flight_stale_decision() -> None:
    from kugelaudio.turn import TurnMessage

    started, release = Event(), Event()

    class BlockingPredictor(_Predictor):
        def predict_proba(
            self, audio: np.ndarray, *, transcript: str = "", sample_rate: int = 16000,
            history_messages: Sequence[TurnMessage] = (),
        ) -> TurnProbabilities:
            started.set()
            assert release.wait(5)
            return super().predict_proba(audio, transcript=transcript, sample_rate=sample_rate)

    predictor = BlockingPredictor(TurnProbabilities(0.8, 0.2, 0.0, 0.0))
    session = TurnDetector(predictor, _policy(), variant_path="onnx/recommended").create_session("en")
    session.push_audio(np.ones(1600, dtype=np.float32) * 0.1)
    session.update_transcript("still speaking")
    session.update_history([TurnMessage("assistant", "Where to?")])
    with ThreadPoolExecutor() as executor:
        future = executor.submit(session.observe_silence, 600)
        assert started.wait(5)
        session.update_history([TurnMessage("assistant", "When?")])
        release.set()
        decision = future.result(timeout=5)
    assert decision.reason is TurnDecisionReason.EVIDENCE_CHANGED
    assert not decision.end_turn


def test_history_revision_reuses_audio_and_rescores_without_advancing_threshold() -> None:
    from kugelaudio.turn import TurnMessage

    class StagedPredictor:
        audio_calls: int = 0
        histories: list[tuple[TurnMessage, ...]]

        def __init__(self) -> None:
            self.histories = []

        def predict_proba(
            self, audio: np.ndarray, *, transcript: str = "", sample_rate: int = 16000,
            history_messages: Sequence[TurnMessage] = (),
        ) -> TurnProbabilities:
            raise AssertionError("expected staged inference")

        def encode_audio(
            self, audio: np.ndarray, *, sample_rate: int = 16000,
        ) -> np.ndarray:
            self.audio_calls += 1
            return np.zeros((1, 16, 896), dtype=np.float32)

        def predict_proba_from_audio(
            self, audio_embeds: np.ndarray, *, transcript: str = "",
            history_messages: Sequence[TurnMessage] = (),
        ) -> TurnProbabilities:
            self.histories.append(tuple(history_messages))
            return TurnProbabilities(0.1, 0.9, 0.0, 0.0)

    predictor = StagedPredictor()
    raw = _policy().model_dump()
    raw["schema_version"] = 2
    raw["languages"]["en"]["temporal"] = {
        "score_points_ms": [200, 300], "threshold_logits": [2.0, 1.0],
        "feedback_penalty": [0.0, 0.0], "pause_penalty": [0.0, 0.0],
    }
    raw["languages"]["en"]["action_delay_ms"] = 200
    session = TurnDetector(predictor, TurnPolicy.model_validate(raw), variant_path="onnx/recommended").create_session("en")
    session.push_audio(np.ones(1600, dtype=np.float32) * 0.1)
    session.update_transcript("still speaking")
    first = session.observe_silence(200)
    session.update_history([TurnMessage("assistant", "Where to?")])
    changed = session.observe_silence(200)
    assert changed.threshold == first.threshold
    later = session.observe_silence(300)
    assert later.threshold < first.threshold
    assert len(predictor.histories) == 2  # Unchanged evidence reuses probabilities.
    session.update_history([])
    cleared = session.observe_silence(400)
    assert cleared.threshold == later.threshold  # Safe after the last score point.
    assert predictor.histories == [(), (TurnMessage("assistant", "Where to?"),), ()]
    assert predictor.audio_calls == 1
