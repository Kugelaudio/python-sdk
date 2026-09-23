"""Strict schemas for externally supplied model and policy metadata."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .errors import TurnBundleError

_HEX_64_PATTERN = r"^[0-9a-f]{64}$"
_SchemaT = TypeVar("_SchemaT", bound=BaseModel)


def _validate_relative_path(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError("path must be safe and relative")
    return value


class BundleFile(BaseModel):
    """One checksummed file declared by an ONNX bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=_HEX_64_PATTERN)

    _safe_path = field_validator("path")(_validate_relative_path)


class BundleManifest(BaseModel):
    """Versioned integrity boundary for a self-contained ONNX bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["kugel-turn-onnx"]
    schema_version: Literal[1]
    source_checkpoint_sha256: str = Field(pattern=_HEX_64_PATTERN)
    audio_precision: str = Field(min_length=1)
    fusion_precision: str = Field(min_length=1)
    algorithm: str = Field(min_length=1)
    input_feature_frames: int = Field(gt=0)
    files: tuple[BundleFile, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_declared_files(self) -> "BundleManifest":
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("manifest contains duplicate file paths")
        required = {
            "audio.onnx",
            "config.json",
            "feature_extractor/preprocessor_config.json",
            "fusion.onnx",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        }
        missing = required - set(paths)
        if missing:
            raise ValueError(f"manifest omits required files: {sorted(missing)}")
        return self


class TurnModelConfig(BaseModel):
    """Architecture values consumed by the customer runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    audio_encoder_name: str = Field(min_length=1)
    lm_name: str = Field(min_length=1)
    sample_rate: int = Field(gt=0)
    audio_window_s: float = Field(gt=0)
    encoder_frames_per_s: int = Field(gt=0)
    num_audio_tokens: int = Field(gt=0)
    audio_token_pooling: Literal["uniform"] = "uniform"
    endpoint_pool_s: float = Field(default=1.6, gt=0)
    audio_encoder_dim: int = Field(gt=0)
    whisper_input_frames: int = Field(gt=0)
    adapter_hidden_dim: int = Field(gt=0)
    lm_hidden_dim: int = Field(gt=0)
    lm_num_layers: int | None = Field(default=None, gt=0)
    max_text_tokens: int = Field(gt=0)
    open_user_message: bool = False
    context_messages: int = Field(default=1, gt=0, strict=True)
    classes: tuple[str, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_runtime_contract(self) -> "TurnModelConfig":
        expected_classes = ("complete", "incomplete", "backchannel", "wait")
        if self.classes != expected_classes:
            raise ValueError(
                f"unsupported class order {self.classes}; expected {expected_classes}"
            )
        if self.whisper_input_frames % 2:
            raise ValueError("whisper_input_frames must be even")
        return self

    @property
    def window_samples(self) -> int:
        """Number of mono samples in the rolling audio window."""
        return int(round(self.sample_rate * self.audio_window_s))


class VariantMetadata(BaseModel):
    """One downloadable entry in the Hugging Face variant index."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    # KEEP-JUSTIFIED: schema-1/early schema-2 bundles predate per-variant
    # policies and intentionally use the repository-root policy.
    policy_path: str = "policy.json"
    status: str = Field(min_length=1)
    audio_precision: str | None = None
    fusion_precision: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    input_feature_frames: int | None = Field(default=None, gt=0)
    sha256: str | None = Field(default=None, pattern=_HEX_64_PATTERN)
    source_checkpoint_sha256: str | None = Field(default=None, pattern=_HEX_64_PATTERN)

    _safe_paths = field_validator("path", "policy_path")(_validate_relative_path)


class VariantIndex(BaseModel):
    """Versioned index used to select a customer-safe bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1, 2]
    recommended: str
    variants: tuple[VariantMetadata, ...] = Field(min_length=1)

    _safe_recommended = field_validator("recommended")(_validate_relative_path)

    @model_validator(mode="after")
    def validate_recommended_variant(self) -> "VariantIndex":
        paths = [variant.path for variant in self.variants]
        if len(paths) != len(set(paths)):
            raise ValueError("variant index contains duplicate paths")
        if self.recommended not in paths:
            raise ValueError("recommended variant is absent from variants")
        return self

    def resolve(self, variant: str) -> VariantMetadata:
        """Resolve ``recommended`` or an exact indexed path without fallback."""
        requested = self.recommended if variant == "recommended" else variant
        for item in self.variants:
            if item.path == requested:
                if item.path.startswith("float/"):
                    raise TurnBundleError(
                        f"variant {requested!r} is not a runtime bundle"
                    )
                return item
        raise TurnBundleError(
            f"unknown turn-detection variant {variant!r}; available ONNX variants: "
            + ", ".join(
                item.path
                for item in self.variants
                if not item.path.startswith("float/")
            )
        )


class TemporalPolicy(BaseModel):
    """Duration-specific thresholds with optional conversation adaptation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    score_points_ms: tuple[int, ...] = Field(min_length=1)
    threshold_logits: tuple[float, ...] = Field(min_length=1)
    feedback_penalty: tuple[float, ...] = Field(min_length=1)
    pause_penalty: tuple[float, ...] = Field(min_length=1)
    false_cutoff_risk_increment: float = Field(default=1.0, ge=0.0)
    span_decay: float = Field(default=1.0, ge=0.0, le=1.0)
    turn_decay: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_schedule(self) -> "TemporalPolicy":
        count = len(self.score_points_ms)
        if not (
            len(self.threshold_logits)
            == len(self.feedback_penalty)
            == len(self.pause_penalty)
            == count
        ):
            raise ValueError(
                "temporal score points, thresholds, and penalties must have "
                "equal lengths"
            )
        if any(point <= 0 for point in self.score_points_ms):
            raise ValueError("temporal score points must be positive")
        if any(
            not math.isfinite(value)
            for values in (
                self.threshold_logits,
                self.feedback_penalty,
                self.pause_penalty,
            )
            for value in values
        ):
            raise ValueError("temporal thresholds and penalties must be finite")
        if any(value < 0.0 for value in self.feedback_penalty):
            raise ValueError("temporal feedback penalties must be non-negative")
        if any(value < 0.0 for value in self.pause_penalty):
            raise ValueError("temporal pause penalties must be non-negative")
        if any(
            earlier >= later
            for earlier, later in zip(
                self.score_points_ms, self.score_points_ms[1:], strict=False
            )
        ):
            raise ValueError("temporal score points must be strictly increasing")
        return self


class LanguagePolicy(BaseModel):
    """One language's measured endpoint policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    threshold: float = Field(ge=0.0, le=1.0)
    action_delay_ms: int = Field(ge=0)
    timeout_ms: int = Field(gt=0)
    temporal: TemporalPolicy | None = None


class TurnPolicy(BaseModel):
    """Versioned policy paired with one exact model variant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1, 2]
    model: str
    preset: str = Field(min_length=1)
    score_after_silence_ms: int = Field(ge=0)
    runtime: str | None = None
    end_turn_probability: Literal["complete", "complete / (complete + incomplete)"]
    languages: dict[Literal["de", "en", "es", "it", "nl"], LanguagePolicy]
    presets: (
        dict[
            str,
            dict[Literal["de", "en", "es", "it", "nl"], LanguagePolicy],
        ]
        | None
    ) = None

    _safe_model = field_validator("model")(_validate_relative_path)

    @model_validator(mode="after")
    def validate_timing(self) -> "TurnPolicy":
        schedules = {self.preset: self.languages}
        if self.presets is not None:
            schedules.update(self.presets)
        for preset_name, languages in schedules.items():
            if not preset_name.strip():
                raise ValueError("policy preset names must not be empty")
            if not languages:
                raise ValueError(
                    f"policy preset {preset_name!r} must define at least one language"
                )
            for language, policy in languages.items():
                self._validate_language_timing(
                    preset_name=preset_name,
                    language=language,
                    policy=policy,
                )
        return self

    def select_preset(self, preset: str | None) -> "TurnPolicy":
        """Select one validated policy preset without mutating metadata."""
        if preset is None or preset == self.preset:
            return self
        if self.presets is None or preset not in self.presets:
            available = sorted({self.preset, *(self.presets or {})})
            raise TurnBundleError(
                f"unknown turn policy preset {preset!r}; available: "
                + ", ".join(available)
            )
        return self.model_copy(
            update={"preset": preset, "languages": self.presets[preset]}
        )

    def _validate_language_timing(
        self,
        *,
        preset_name: str,
        language: str,
        policy: LanguagePolicy,
    ) -> None:
        prefix = f"{preset_name}/{language}"
        if policy.action_delay_ms < self.score_after_silence_ms:
            raise ValueError(
                f"{prefix} action_delay_ms precedes score_after_silence_ms"
            )
        if policy.timeout_ms < policy.action_delay_ms:
            raise ValueError(f"{prefix} timeout_ms precedes action_delay_ms")
        if policy.temporal is None:
            return
        first_point = policy.temporal.score_points_ms[0]
        last_point = policy.temporal.score_points_ms[-1]
        if first_point != self.score_after_silence_ms:
            raise ValueError(
                f"{prefix} first temporal score must equal score_after_silence_ms"
            )
        if policy.action_delay_ms > first_point:
            raise ValueError(f"{prefix} action_delay_ms follows first temporal score")
        if policy.timeout_ms < last_point:
            raise ValueError(f"{prefix} timeout_ms precedes final temporal score")


def load_schema(path: Path, schema: type[_SchemaT]) -> _SchemaT:
    """Load and validate an external JSON file with a typed error boundary."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return schema.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise TurnBundleError(f"invalid {schema.__name__} at {path}: {exc}") from exc
