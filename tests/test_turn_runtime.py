"""Unit tests for NumPy preprocessing and ONNX turn-model inference."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest

pytest.importorskip("pydantic")

from kugelaudio.turn._bundle import verify_bundle
from kugelaudio.turn._runtime import (
    TurnMessage,
    TurnPredictor,
    _execution_providers,
)
from kugelaudio.turn.errors import TurnAudioError, TurnBundleError


def test_pytorch_import_precedes_onnxruntime_initialization() -> None:
    source = textwrap.dedent(inspect.getsource(TurnPredictor.from_verified_bundle))
    imports = [
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name in {"torch", "onnxruntime"}
    ]

    assert imports == ["torch", "onnxruntime"]


def test_openvino_provider_applies_thread_limit() -> None:
    providers = _execution_providers("openvino", cpu_threads=4)
    assert providers[0][0] == "OpenVINOExecutionProvider"
    config = json.loads(providers[0][1]["load_config"])
    assert config == {
        "CPU": {
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
            "INFERENCE_NUM_THREADS": "4",
        }
    }


def test_openvino_provider_configures_streams_and_precision() -> None:
    providers = _execution_providers(
        "openvino",
        cpu_threads=12,
        cpu_streams=4,
        inference_precision="bf16",
    )
    config = json.loads(providers[0][1]["load_config"])
    assert config == {
        "CPU": {
            "INFERENCE_PRECISION_HINT": "bf16",
            "PERFORMANCE_HINT": "THROUGHPUT",
            "NUM_STREAMS": "4",
            "INFERENCE_NUM_THREADS": "12",
        }
    }


def test_cuda_provider_uses_device_zero_with_explicit_shape_fallback() -> None:
    providers = _execution_providers("cuda", cpu_threads=4)

    assert providers == [
        (
            "CUDAExecutionProvider",
            {
                "device_id": "0",
                "do_copy_in_default_stream": "1",
                "use_tf32": "0",
            },
        ),
        "CPUExecutionProvider",
    ]


@dataclass(slots=True)
class _Features:
    input_features: np.ndarray


class _FeatureExtractor:
    def __call__(
        self,
        raw_speech: list[np.ndarray],
        *,
        sampling_rate: int,
        return_tensors: str,
    ) -> _Features:
        assert len(raw_speech) == 1
        assert raw_speech[0].shape == (128000,)
        assert sampling_rate == 16000
        assert return_tensors == "np"
        return _Features(np.zeros((1, 80, 3000), dtype=np.float32))


class _Tokenizer:
    padding_side = "right"
    truncation_side = "right"

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert conversation == [{"role": "user", "content": "still speaking"}]
        assert tokenize is False
        assert add_generation_prompt is False
        return "formatted"

    def __call__(self, text: list[str], **kwargs: object) -> dict[str, np.ndarray]:
        assert text == ["formatted"]
        assert kwargs["return_tensors"] == "np"
        return {
            "input_ids": np.asarray([[1, 2]], dtype=np.int64),
            "attention_mask": np.asarray([[1, 1]], dtype=np.int64),
        }


class _AudioSession:
    def run(
        self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        assert output_names is None
        assert input_feed["input_features"].shape == (1, 80, 800)
        return [np.zeros((1, 16, 896), dtype=np.float32)]


class _FusionSession:
    def run(
        self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        assert output_names is None
        assert input_feed["input_ids"].dtype == np.int64
        assert input_feed["attention_mask"].shape == (1, 2)
        assert input_feed["audio_embeds"].shape == (1, 16, 896)
        return [np.asarray([[2.0, 1.0, 0.0, -1.0]], dtype=np.float32)]


def _verified_bundle(root: Path):
    contents = {
        "audio.onnx": b"audio",
        "config.json": json.dumps(
            {
                "audio_encoder_name": "openai/whisper-small",
                "lm_name": "Qwen/Qwen2.5-0.5B-Instruct",
                "sample_rate": 16000,
                "audio_window_s": 8.0,
                "encoder_frames_per_s": 50,
                "num_audio_tokens": 16,
                "audio_encoder_dim": 768,
                "whisper_input_frames": 800,
                "adapter_hidden_dim": 1024,
                "lm_hidden_dim": 896,
                "max_text_tokens": 128,
                "classes": ["complete", "incomplete", "backchannel", "wait"],
            }
        ).encode(),
        "feature_extractor/preprocessor_config.json": b"{}",
        "fusion.onnx": b"fusion",
        "tokenizer/tokenizer.json": b"{}",
        "tokenizer/tokenizer_config.json": b"{}",
    }
    files = []
    for relative, data in contents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files.append(
            {
                "path": relative,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "kugel-turn-onnx",
                "schema_version": 1,
                "source_checkpoint_sha256": "a" * 64,
                "audio_precision": "int4",
                "fusion_precision": "int8",
                "algorithm": "test",
                "input_feature_frames": 800,
                "files": files,
            }
        )
    )
    return verify_bundle(root)


def _predictor(tmp_path: Path) -> TurnPredictor:
    return TurnPredictor(
        _verified_bundle(tmp_path / "bundle"),
        _Tokenizer(),
        _FeatureExtractor(),
        _AudioSession(),
        _FusionSession(),
    )


def test_predict_proba_uses_numpy_graph_contract(tmp_path: Path) -> None:
    predictor = _predictor(tmp_path)

    probabilities = predictor.predict_proba(
        np.ones(16000, dtype=np.float32) * 0.1,
        transcript="still speaking",
    )

    assert probabilities.complete == pytest.approx(0.6439143)
    assert probabilities.incomplete == pytest.approx(0.2368828)
    assert sum(probabilities.as_dict().values()) == pytest.approx(1.0)


def test_staged_prediction_matches_composed_prediction(tmp_path: Path) -> None:
    predictor = _predictor(tmp_path)
    audio = np.ones(16000, dtype=np.float32) * 0.1

    audio_embeds = predictor.encode_audio(audio)
    staged = predictor.predict_proba_from_audio(
        audio_embeds,
        transcript="still speaking",
    )
    composed = predictor.predict_proba(audio, transcript="still speaking")

    assert staged == composed


@pytest.mark.parametrize("transcript", ["still speaking", ""])
def test_history_reaches_fusion_with_current_user_marker(
    tmp_path: Path, transcript: str
) -> None:
    from kugelaudio.turn import TurnMessage

    predictor = _predictor(tmp_path)
    predictor.config = predictor.config.model_copy(update={"context_messages": 3})

    class ContextTokenizer(_Tokenizer):
        def apply_chat_template(
            self, conversation: list[dict[str, str]], **kwargs: bool
        ) -> str:
            assert conversation == [
                {"role": "user", "content": "Book a trip"},
                {"role": "assistant", "content": "Where to?"},
                {"role": "user", "content": transcript},
            ]
            assert kwargs == {"tokenize": False, "add_generation_prompt": False}
            return "formatted"

    predictor.tokenizer = ContextTokenizer()
    history = [TurnMessage("user", "Book a trip"), TurnMessage("assistant", "Where to?")]
    audio = np.ones(16000, dtype=np.float32) * 0.1
    composed = predictor.predict_proba(audio, transcript=transcript, history_messages=history)
    staged = predictor.predict_proba_from_audio(
        predictor.encode_audio(audio), transcript=transcript, history_messages=history
    )
    assert staged == composed


def test_single_message_checkpoint_rejects_history(tmp_path: Path) -> None:
    from kugelaudio.turn import TurnMessage

    predictor = _predictor(tmp_path)
    with pytest.raises(ValueError, match="context_messages greater than 1"):
        predictor.predict_proba_from_audio(
            np.zeros((1, 16, 896), dtype=np.float32),
            history_messages=[TurnMessage("assistant", "Where to?")],
        )


def test_context_limit_keeps_recent_messages_and_open_current(tmp_path: Path) -> None:
    from kugelaudio.turn import TurnMessage

    predictor = _predictor(tmp_path)
    predictor.config = predictor.config.model_copy(
        update={"context_messages": 2, "open_user_message": True}
    )

    class ContextTokenizer(_Tokenizer):
        def apply_chat_template(
            self, conversation: list[dict[str, str]], **kwargs: bool
        ) -> str:
            assert conversation == [
                {"role": "assistant", "content": "Where to?"},
                {"role": "user", "content": "still speaking"},
            ]
            assert kwargs["continue_final_message"] is True
            return "formatted"

        def __call__(self, text: list[str], **kwargs: object) -> dict[str, np.ndarray]:
            assert kwargs["max_length"] == 128
            assert self.truncation_side == "left"
            return super().__call__(text, **kwargs)

    predictor.tokenizer = ContextTokenizer()
    predictor.tokenizer.truncation_side = "left"
    predictor.predict_proba_from_audio(
        np.zeros((1, 16, 896), dtype=np.float32), transcript="still speaking",
        history_messages=[
            TurnMessage("user", "old request"),
            TurnMessage("assistant", "  Where to?  "),
            TurnMessage("user", "   "),
        ],
    )


@pytest.mark.parametrize("history", ["text", {"role": "user"}, ["text"], None])
def test_invalid_history_is_rejected(tmp_path: Path, history: object) -> None:
    predictor = _predictor(tmp_path)
    with pytest.raises(TypeError, match="history_messages"):
        predictor.predict_proba_from_audio(
            np.zeros((1, 16, 896), dtype=np.float32),
            history_messages=cast(Sequence[TurnMessage], history),
        )


def test_turn_message_rejects_invalid_role_and_content() -> None:
    from kugelaudio.turn import TurnMessage

    with pytest.raises(ValueError, match="role"):
        TurnMessage(cast(Literal["user", "assistant"], "system"), "instructions")
    with pytest.raises(TypeError, match="content"):
        TurnMessage("user", cast(str, 12))


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "3"])
def test_context_message_count_requires_positive_integer(
    tmp_path: Path, count: object
) -> None:
    from kugelaudio.turn._schema import TurnModelConfig
    from pydantic import ValidationError

    config = _predictor(tmp_path).config.model_dump()
    config["context_messages"] = count
    with pytest.raises(ValidationError, match="context_messages"):
        TurnModelConfig.model_validate(config)


def test_no_history_empty_transcript_keeps_legacy_zero_text_tokens(tmp_path: Path) -> None:
    from kugelaudio.turn._runtime import _tokenize_transcript

    predictor = _predictor(tmp_path)
    ids, mask = _tokenize_transcript(
        predictor.tokenizer, transcript="   ", config=predictor.config,
    )
    assert ids.shape == mask.shape == (1, 0)


@pytest.mark.parametrize(
    "audio,sample_rate,match",
    [
        (np.ones(10, dtype=np.float64), 16000, "dtype must be float32"),
        (np.ones((2, 10), dtype=np.float32), 16000, "must be mono"),
        (np.asarray([], dtype=np.float32), 16000, "at least one"),
        (np.asarray([np.nan], dtype=np.float32), 16000, "NaN"),
        (np.asarray([1.1], dtype=np.float32), 16000, "normalized"),
        (np.ones(10, dtype=np.float32), 8000, "sample_rate must be 16000"),
    ],
)
def test_predict_proba_rejects_implicit_audio_coercion(
    tmp_path: Path,
    audio: np.ndarray,
    sample_rate: int,
    match: str,
) -> None:
    with pytest.raises(TurnAudioError, match=match):
        _predictor(tmp_path).predict_proba(audio, sample_rate=sample_rate)


def test_predict_proba_rejects_wrong_graph_shape(tmp_path: Path) -> None:
    class BadFusion(_FusionSession):
        def run(
            self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]
        ) -> list[np.ndarray]:
            return [np.zeros((1, 3), dtype=np.float32)]

    predictor = TurnPredictor(
        _verified_bundle(tmp_path / "bundle"),
        _Tokenizer(),
        _FeatureExtractor(),
        _AudioSession(),
        BadFusion(),
    )

    with pytest.raises(TurnBundleError, match=r"expected \(1, 4\)"):
        predictor.predict_proba(
            np.ones(10, dtype=np.float32), transcript="still speaking"
        )
