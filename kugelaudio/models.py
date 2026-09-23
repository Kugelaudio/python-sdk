"""Data models for KugelAudio SDK."""

from __future__ import annotations

import base64
import io
import logging
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Accepted classifier-free guidance band. Values outside [MIN, MAX] are
# clamped into the band (both client-side and by the server).
MIN_CFG_SCALE = 1.2
MAX_CFG_SCALE = 2.5
# The only accepted/returned public ASR model identifier. Ingress mints this
# name at the public boundary; there is no alias for any prior identifier.
PUBLIC_ASR_MODEL_ID = "luchs-1"


@dataclass(frozen=True)
class SpellingAlternative:
    spelling: str
    probability: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpellingAlternative":
        return cls(spelling=data["spelling"], probability=float(data["probability"]))


@dataclass(frozen=True)
class WordAlternatives:
    raw_word_index: int
    word: str
    alternatives: List[SpellingAlternative] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WordAlternatives":
        return cls(
            raw_word_index=int(data["raw_word_index"]),
            word=data["word"],
            alternatives=[
                SpellingAlternative.from_dict(item)
                for item in data.get("alternatives", [])
            ],
        )


@dataclass(frozen=True)
class TranscriptionResponse:
    """Final ASR response; ``text`` is the OpenAI-compatible transcript field."""

    text: str
    transcript: str
    language: str
    duration_s: float
    model: str
    word_alternatives: List[WordAlternatives] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)
    model_revision: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptionResponse":
        transcript = data.get("transcript") or data.get("text")
        if not isinstance(transcript, str):
            raise ValueError("transcription response is missing text")
        return cls(
            text=data.get("text", transcript),
            transcript=transcript,
            language=data["language"],
            duration_s=float(data["duration_s"]),
            model=data["model"],
            model_revision=data.get("model_revision"),
            word_alternatives=[
                WordAlternatives.from_dict(item)
                for item in data.get("word_alternatives", [])
            ],
            raw=dict(data),
        )


@dataclass(frozen=True)
class StreamingTranscriptionEvent:
    """Replaceable rolling ASR hypothesis from the public WebSocket."""

    partial_text: str
    is_final: bool
    type: str = "partial"
    turn_end_reason: Optional[str] = None
    turn_end_confidence: Optional[float] = None
    turn_end_inference_ms: Optional[float] = None
    word_alternatives: List[WordAlternatives] = field(default_factory=list)
    model: Optional[str] = None
    model_revision: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamingTranscriptionEvent":
        return cls(
            partial_text=str(data.get("partial_text", "")),
            is_final=bool(data.get("is_final", False)),
            type=str(data.get("type", "partial")),
            model=data.get("model"),
            model_revision=data.get("model_revision"),
            turn_end_reason=data.get("turn_end_reason"),
            turn_end_confidence=(
                float(data["turn_end_confidence"])
                if data.get("turn_end_confidence") is not None
                else None
            ),
            turn_end_inference_ms=(
                float(data["turn_end_inference_ms"])
                if data.get("turn_end_inference_ms") is not None
                else None
            ),
            word_alternatives=[
                WordAlternatives.from_dict(item)
                for item in data.get("word_alternatives", [])
            ],
        )


def clamp_cfg_scale(cfg_scale: float) -> float:
    """Clamp ``cfg_scale`` into [1.2, 2.5]."""
    return min(MAX_CFG_SCALE, max(MIN_CFG_SCALE, cfg_scale))


class VoiceCategory(str, Enum):
    """Voice category types.

    The API accepts and returns the seven categories from ``NARRATIVE_STORY``
    to ``INFORMATIVE_EDUCATIONAL``; it returns ``CLONED`` for a voice with no
    category set.
    """

    PREMADE = "premade"
    CLONED = "cloned"
    DESIGNED = "designed"
    CONVERSATIONAL = "conversational"
    NARRATIVE = "narrative"
    NARRATIVE_STORY = "narrative_story"
    CHARACTERS = "characters"
    CHARACTERS_ANIMATION = "characters_animation"
    SOCIAL_MEDIA = "social_media"
    ENTERTAINMENT_TV = "entertainment_tv"
    ADVERTISEMENT = "advertisement"
    INFORMATIVE_EDUCATIONAL = "informative_educational"

    @classmethod
    def _missing_(cls, value: object) -> "VoiceCategory":
        """Map a category this SDK version does not know to ``CLONED``.

        Raising would break ``voices.list()`` whenever the server adds a
        category, so the lossy mapping is logged instead.
        """
        logger.warning(
            "Unknown voice category %r mapped to %r; upgrade the kugelaudio SDK",
            value,
            cls.CLONED.value,
        )
        return cls.CLONED


class VoiceSex(str, Enum):
    """Voice sex types."""

    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class VoiceAge(str, Enum):
    """Voice age types."""

    YOUNG = "young"
    MIDDLE_AGED = "middle_aged"
    MIDDLE_AGE = "middle_age"  # Alternative spelling
    OLD = "old"


@dataclass
class Model:
    """TTS model information."""

    id: str
    name: str
    description: str
    parameters: str
    max_input_length: int = 5000
    sample_rate: int = 24000

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Model:
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            parameters=data.get("parameters", ""),
            max_input_length=data.get("max_input_length", 5000),
            sample_rate=data.get("sample_rate", 24000),
        )


@dataclass
class Voice:
    """Voice information."""

    id: int
    name: str
    description: Optional[str] = None
    category: Optional[VoiceCategory] = None
    sex: Optional[VoiceSex] = None
    age: Optional[VoiceAge] = None
    quality: str = "mid"
    supported_languages: List[str] = field(default_factory=list)
    sample_text: Optional[str] = None
    avatar_url: Optional[str] = None
    sample_url: Optional[str] = None
    is_public: bool = False
    verified: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Voice:
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            category=VoiceCategory(data["category"]) if data.get("category") else None,
            sex=VoiceSex(data["sex"]) if data.get("sex") else None,
            age=VoiceAge(data["age"]) if data.get("age") else None,
            quality=data.get("quality", "mid"),
            supported_languages=data.get("supported_languages") or [],
            sample_text=data.get("sample_text"),
            avatar_url=data.get("avatar_url"),
            sample_url=data.get("sample_url"),
            is_public=data.get("is_public", False),
            verified=data.get("verified", False),
        )


@dataclass
class VoiceListResponse:
    """Paginated response from the voices list endpoint."""

    voices: List[Voice]
    total: int
    limit: int
    offset: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceListResponse":
        return cls(
            voices=[Voice.from_dict(v) for v in data.get("voices", [])],
            total=data["total"],
            limit=data["limit"],
            offset=data["offset"],
        )


@dataclass
class VoiceDetail:
    """Extended voice information returned by voice management endpoints."""

    id: int
    name: str
    description: str = ""
    generative_voice_description: str = ""
    supported_languages: List[str] = field(default_factory=list)
    category: Optional[VoiceCategory] = None
    age: Optional[VoiceAge] = None
    sex: Optional[VoiceSex] = None
    quality: str = "mid"
    is_public: bool = False
    verified: bool = False
    pending_verification: bool = False
    sample_url: Optional[str] = None
    avatar_url: Optional[str] = None
    sample_text: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceDetail":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            generative_voice_description=data.get("generative_voice_description", ""),
            supported_languages=data.get("supported_languages") or [],
            category=VoiceCategory(data["category"]) if data.get("category") else None,
            age=VoiceAge(data["age"]) if data.get("age") else None,
            sex=VoiceSex(data["sex"]) if data.get("sex") else None,
            quality=data.get("quality", "mid"),
            is_public=data.get("is_public", False),
            verified=data.get("verified", False),
            pending_verification=data.get("pending_verification", False),
            sample_url=data.get("sample_url"),
            avatar_url=data.get("avatar_url"),
            sample_text=data.get("sample_text", ""),
        )


@dataclass
class GeneratedSample:
    """Stored preview sample returned by ``voices.generate_sample``.

    ``sample_url`` is a signed, time-limited URL; it is ``None`` when the
    server could not sign one.
    """

    sample_s3_path: str
    sample_url: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedSample":
        return cls(
            sample_s3_path=data["sample_s3_path"],
            sample_url=data.get("sample_url"),
        )


@dataclass
class VoiceReference:
    """Voice reference audio metadata."""

    id: int
    voice_id: int
    name: str = ""
    reference_text: str = ""
    s3_path: str = ""
    audio_url: Optional[str] = None
    is_generated: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceReference":
        return cls(
            id=data["id"],
            voice_id=data["voice_id"],
            name=data.get("name", ""),
            reference_text=data.get("reference_text", ""),
            s3_path=data.get("s3_path", ""),
            audio_url=data.get("audio_url"),
            is_generated=data.get("is_generated", False),
        )


@dataclass
class Dictionary:
    """A per-project pronunciation dictionary."""

    id: int
    project_id: int
    name: str
    description: Optional[str] = None
    language: Optional[str] = None
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Dictionary":
        return cls(
            id=data["id"],
            project_id=data["project_id"],
            name=data["name"],
            description=data.get("description"),
            language=data.get("language"),
            is_active=data.get("is_active", True),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass
class DictionaryEntry:
    """A single word → replacement / IPA mapping within a dictionary."""

    id: int
    dictionary_id: int
    word: str
    replacement: str
    ipa: Optional[str] = None
    case_sensitive: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DictionaryEntry":
        return cls(
            id=data["id"],
            dictionary_id=data["dictionary_id"],
            word=data["word"],
            replacement=data["replacement"],
            ipa=data.get("ipa"),
            case_sensitive=data.get("case_sensitive", False),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass
class DictionaryEntryList:
    """Paginated response from the entries list endpoint."""

    entries: List[DictionaryEntry]
    total: int
    limit: int
    offset: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DictionaryEntryList":
        return cls(
            entries=[DictionaryEntry.from_dict(e) for e in data.get("entries", [])],
            total=data["total"],
            limit=data["limit"],
            offset=data["offset"],
        )


@dataclass
class BulkReplaceResult:
    """Counts returned by ``dictionaries.entries.replace_all``."""

    upserted: int
    deleted: int
    total: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BulkReplaceResult":
        return cls(
            upserted=data["upserted"],
            deleted=data["deleted"],
            total=data["total"],
        )


@dataclass
class WordTimestamp:
    """A single word with its time boundaries within an audio chunk.

    Returned when ``word_timestamps=True`` is set on the request.
    """

    word: str
    start_ms: int
    end_ms: int
    char_start: int
    char_end: int
    score: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WordTimestamp":
        return cls(
            word=data["word"],
            start_ms=data["start_ms"],
            end_ms=data["end_ms"],
            char_start=data["char_start"],
            char_end=data["char_end"],
            score=data.get("score", 1.0),
        )

    @property
    def duration_ms(self) -> int:
        """Duration of this word in milliseconds."""
        return self.end_ms - self.start_ms


@dataclass
class SessionUsage:
    """Per-session usage reported in the ``session_closed`` frame.

    Lets you bill your own customers per conversation. ``cost_cents`` is the
    actual amount charged in **EUR cents**. When the charge could not be
    determined at session end (e.g. a transient billing error) ``cost_cents``
    is ``None`` and :attr:`cost_available` is ``False`` — never a misleading
    ``0``. ``audio_seconds`` is always reported.

    On ``/ws/tts/multi`` usage is reported per context (per conversation) on
    each ``context_closed`` frame, not aggregated across contexts.
    """

    audio_seconds: float
    cost_cents: Optional[float] = None
    currency: Optional[str] = None
    characters: Optional[int] = None
    model_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionUsage":
        return cls(
            audio_seconds=float(data.get("audio_seconds", 0.0) or 0.0),
            cost_cents=data.get("cost_cents"),
            currency=data.get("currency"),
            characters=data.get("characters"),
            model_id=data.get("model_id"),
        )

    @classmethod
    def from_session_payload(cls, data: Dict[str, Any]) -> Optional["SessionUsage"]:
        """Parse a ``session_closed`` / ``final`` frame into typed usage.

        Reads the nested ``usage`` block when present, otherwise falls back to
        the top-level ``total_audio_seconds`` for older servers. Returns
        ``None`` when no usage information is present.
        """
        usage = data.get("usage")
        if isinstance(usage, dict):
            return cls.from_dict(usage)
        if "total_audio_seconds" in data:
            return cls(
                audio_seconds=float(data.get("total_audio_seconds", 0.0) or 0.0),
            )
        return None

    @property
    def cost_available(self) -> bool:
        """True when an authoritative charge was returned for this session."""
        return self.cost_cents is not None


@dataclass
class GenerateRequest:
    """Request for TTS generation."""

    text: str
    model_id: str = "kugel-3"
    voice_id: Optional[int] = None
    cfg_scale: float = 2.0
    temperature: Optional[float] = None
    """Sampling variance in [0.0, 1.0]. ``None`` uses the server default (~0.5).

    ``0.0`` is most stable (near-greedy); ``1.0`` is most variance. Lower values
    produce more consistent reads across regenerations — useful for stable
    voiceovers, IVR prompts, and e-learning.
    """
    max_new_tokens: int = 2048
    sample_rate: int = 24000
    output_format: Optional[str] = None
    """Combined codec+rate token, e.g. ``'ulaw_8000'`` / ``'alaw_8000'`` / ``'pcm_8000'``.

    Opt-in telephony / fixed-format output. When set it is authoritative and must
    not contradict ``sample_rate``; absent ⇒ legacy PCM16 at ``sample_rate``.
    """
    normalize: bool = True
    """Enable text normalization (converts numbers, dates, etc. to spoken words).

    Default: True.  Set ``language`` when the text is not in the voice's
    primary language; the language is not detected from the text.
    """
    language: Optional[str] = None
    """ISO 639-1 language code for text normalization (e.g., 'de', 'en', 'fr').

    Supported languages: de, en, fr, es, it, pt, nl, pl, sv, da, no, fi, cs, hu, ro,
    el, uk, bg, tr, vi, ar, hi, zh, ja, ko, sk, sl, hr, sr, ru, he, fa, ur, bn, ta,
    yue, th, id, ms

    If not provided, the server uses the voice's primary language, falling back
    to English. The language is not detected from the text.
    """
    word_timestamps: bool = False
    """Request word-level timestamps for barge-in / alignment support.

    When enabled, the server sends a ``word_timestamps`` message after audio
    containing per-word start/end times in milliseconds.
    """
    speed: float = 1.0
    """Playback speed multiplier (0.8 = slower, 1.0 = normal, 1.2 = faster).

    Uses pitch-preserving time-stretching; applies to the whole
    request.  Wrap text in ``<prosody rate="slow|medium|fast|0.8-1.2">`` to
    override the rate for a span (the span rate wins inside the span).
    Range: [0.8, 1.2].
    """
    dictionary_ids: Optional[List[int]] = None
    """Per-request dictionary selection.

    ``None`` (default): all *active* dictionaries of the project apply,
    filtered by language. ``[]``: no dictionary applies to this request.
    A list of dictionary IDs: exactly those dictionaries apply — even
    inactive ones — bypassing the language filter.
    """
    project_id: Optional[int] = None
    """Project whose pronunciation dictionaries apply at synthesis.

    Required for any dictionary to apply and for a non-empty
    ``dictionary_ids``. ``None`` (default) sends no project.
    """

    def __post_init__(self) -> None:
        self.cfg_scale = clamp_cfg_scale(self.cfg_scale)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "text": self.text,
            "model_id": self.model_id,
            "cfg_scale": self.cfg_scale,
            "max_new_tokens": self.max_new_tokens,
            "sample_rate": self.sample_rate,
            "normalize": self.normalize,
            "word_timestamps": self.word_timestamps,
            "speed": self.speed,
        }
        if self.voice_id is not None:
            result["voice_id"] = self.voice_id
        if self.language is not None:
            result["language"] = self.language
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.output_format is not None:
            result["output_format"] = self.output_format
        if self.project_id is not None:
            result["project_id"] = self.project_id
        # [] is meaningful (explicit opt-out) and must be sent; only None
        # (use the project default) is omitted.
        if self.dictionary_ids is not None:
            result["dictionary_ids"] = self.dictionary_ids
        return result


@dataclass
class StreamConfig:
    """Configuration for streaming TTS sessions.

    The server accumulates LLM tokens internally and only starts generation
    at natural sentence boundaries.  Use ``chunk_length_schedule`` to tune
    how eagerly the server begins generating (smaller values = lower TTFA,
    less prosody context), or set ``auto_mode=True`` to start at the very
    first clean boundary — equivalent to ElevenLabs' ``auto_mode``.

    Example — optimise for low latency::

        config = StreamConfig(
            voice_id=123,
            auto_mode=True,
            chunk_length_schedule=[50, 100, 150, 250],
        )
    """

    voice_id: Optional[int] = None
    model_id: Optional[str] = None
    cfg_scale: float = 2.0
    temperature: Optional[float] = None
    """Sampling variance in [0.0, 1.0]. ``None`` uses the server default (~0.5).

    ``0.0`` is most stable (near-greedy); ``1.0`` is most variance.
    """
    max_new_tokens: int = 2048
    sample_rate: int = 24000
    output_format: Optional[str] = None
    """Combined codec+rate token, e.g. ``'ulaw_8000'`` / ``'alaw_8000'`` / ``'pcm_8000'``.

    Opt-in; authoritative when set (must not contradict ``sample_rate``).
    Set-once per session. Absent ⇒ legacy PCM16 at ``sample_rate``.
    """
    flush_timeout_ms: int = 500
    max_buffer_length: int = 1000
    normalize: bool = True
    """Enable text normalization (converts numbers, dates, etc. to spoken words).

    Default: True.  Set ``language`` when the text is not in the voice's
    primary language; the language is not detected from the text.
    """
    language: Optional[str] = None
    """ISO 639-1 language code for text normalization (e.g., 'de', 'en', 'fr')."""
    word_timestamps: bool = False
    """Request per-chunk word-level timestamps for barge-in / alignment support."""
    chunk_length_schedule: Optional[List[int]] = None
    """Minimum buffer sizes (chars) before each successive auto-chunk is emitted.

    Entry ``i`` applies to chunk ``i``; the last value is reused for all
    subsequent chunks.  When omitted the server uses its default schedule
    ``[5, 80, 150, 250]``.  Smaller values give lower TTFA; larger values
    improve prosody naturalness.

    Example schedules:

    * ``[50, 100, 150, 250]``  — low-latency
    * ``[120, 200, 300]``      — high-quality prosody
    """
    auto_mode: Optional[bool] = None
    """Start generating at the very first clean sentence boundary.

    When ``True`` the server ignores ``min_sentences_first_chunk`` and
    emits audio as soon as one complete sentence is in the buffer.
    Equivalent to ElevenLabs' ``auto_mode=true``.  Lower TTFA; slightly
    less prosody context on the first chunk.
    """
    speed: float = 1.0
    """Playback speed multiplier (0.8 = slower, 1.0 = normal, 1.2 = faster).

    Uses pitch-preserving time-stretching; applies to the whole
    request.  Wrap text in ``<prosody rate="slow|medium|fast|0.8-1.2">`` to
    override the rate for a span (the span rate wins inside the span).
    Range: [0.8, 1.2].
    """
    dictionary_ids: Optional[List[int]] = None
    """Per-request dictionary selection.

    ``None`` (default): all *active* dictionaries of the project apply,
    filtered by language. ``[]``: no dictionary applies to this session.
    A list of dictionary IDs: exactly those dictionaries apply — even
    inactive ones — bypassing the language filter.
    """
    project_id: Optional[int] = None
    """Project whose pronunciation dictionaries apply at synthesis.

    Required for any dictionary to apply and for a non-empty
    ``dictionary_ids``. ``None`` (default) sends no project.
    """

    def __post_init__(self) -> None:
        self.cfg_scale = clamp_cfg_scale(self.cfg_scale)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "cfg_scale": self.cfg_scale,
            "max_new_tokens": self.max_new_tokens,
            "sample_rate": self.sample_rate,
            "flush_timeout_ms": self.flush_timeout_ms,
            "max_buffer_length": self.max_buffer_length,
            "normalize": self.normalize,
            "word_timestamps": self.word_timestamps,
            "speed": self.speed,
        }
        if self.voice_id is not None:
            result["voice_id"] = self.voice_id
        if self.model_id is not None:
            result["model_id"] = self.model_id
        if self.output_format is not None:
            result["output_format"] = self.output_format
        if self.language is not None:
            result["language"] = self.language
        if self.chunk_length_schedule is not None:
            result["chunk_length_schedule"] = self.chunk_length_schedule
        if self.auto_mode is not None:
            result["auto_mode"] = self.auto_mode
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.project_id is not None:
            result["project_id"] = self.project_id
        # [] is meaningful (explicit opt-out) and must be sent; only None
        # (use the project default) is omitted.
        if self.dictionary_ids is not None:
            result["dictionary_ids"] = self.dictionary_ids
        return result


@dataclass
class AudioChunk:
    """Audio chunk from streaming TTS."""

    audio: bytes  # Raw PCM16 audio bytes
    index: int
    sample_rate: int
    samples: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AudioChunk:
        """Create from WebSocket message."""
        audio_b64 = data["audio"]
        audio_bytes = base64.b64decode(audio_b64)
        return cls(
            audio=audio_bytes,
            index=data["idx"],
            sample_rate=data["sr"],
            samples=data["samples"],
        )

    def to_float32(self) -> List[float]:
        """Convert PCM16 to float32 samples (-1.0 to 1.0)."""
        int16_samples = struct.unpack(f"<{len(self.audio) // 2}h", self.audio)
        return [s / 32768.0 for s in int16_samples]

    @property
    def duration_seconds(self) -> float:
        """Duration of this chunk in seconds."""
        return self.samples / self.sample_rate


@dataclass
class AudioResponse:
    """Complete audio response from TTS generation."""

    audio: bytes  # Raw PCM16 audio bytes
    sample_rate: int
    samples: int
    duration_ms: float
    generation_ms: float
    rtf: float  # Real-time factor
    word_timestamps: List["WordTimestamp"] = field(default_factory=list)
    """Per-word timing boundaries (populated when ``word_timestamps=True``)."""
    usage: Optional["SessionUsage"] = None
    """Per-request usage (audio time + amount charged); ``None`` if not reported."""

    @classmethod
    def from_chunks(
        cls,
        chunks: List[AudioChunk],
        final_stats: Dict[str, Any],
        word_timestamps: Optional[List["WordTimestamp"]] = None,
    ) -> "AudioResponse":
        """Create from collected chunks and final stats."""
        all_audio = b"".join(c.audio for c in chunks)
        sample_rate = chunks[0].sample_rate if chunks else 24000
        usage = SessionUsage.from_session_payload(final_stats)
        return cls(
            audio=all_audio,
            sample_rate=sample_rate,
            samples=final_stats.get("total_samples", len(all_audio) // 2),
            duration_ms=final_stats.get("dur_ms", 0.0),
            generation_ms=final_stats.get("gen_ms", 0.0),
            rtf=final_stats.get("rtf", 0.0),
            word_timestamps=word_timestamps or [],
            usage=usage,
        )

    def to_float32(self) -> List[float]:
        """Convert PCM16 to float32 samples (-1.0 to 1.0)."""
        int16_samples = struct.unpack(f"<{len(self.audio) // 2}h", self.audio)
        return [s / 32768.0 for s in int16_samples]

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return self.samples / self.sample_rate

    def save(self, path: str, format: str = "wav") -> None:
        """Save audio to file.

        Args:
            path: Output file path
            format: Output format ('wav' or 'raw')
        """
        if format == "wav":
            self._save_wav(path)
        elif format == "raw":
            with open(path, "wb") as f:
                f.write(self.audio)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _save_wav(self, path: str) -> None:
        """Save as WAV file."""
        with open(path, "wb") as f:
            # WAV header
            f.write(b"RIFF")
            data_size = len(self.audio)
            f.write(struct.pack("<I", 36 + data_size))  # File size - 8
            f.write(b"WAVEfmt ")
            f.write(
                struct.pack(
                    "<IHHIIHH",
                    16,  # Subchunk1Size (16 for PCM)
                    1,  # AudioFormat (1 for PCM)
                    1,  # NumChannels (mono)
                    self.sample_rate,  # SampleRate
                    self.sample_rate * 2,  # ByteRate
                    2,  # BlockAlign
                    16,  # BitsPerSample
                )
            )
            f.write(b"data")
            f.write(struct.pack("<I", data_size))
            f.write(self.audio)

    def to_wav_bytes(self) -> bytes:
        """Get WAV file as bytes."""
        buffer = io.BytesIO()
        # WAV header
        buffer.write(b"RIFF")
        data_size = len(self.audio)
        buffer.write(struct.pack("<I", 36 + data_size))
        buffer.write(b"WAVEfmt ")
        buffer.write(
            struct.pack(
                "<IHHIIHH", 16, 1, 1, self.sample_rate, self.sample_rate * 2, 2, 16
            )
        )
        buffer.write(b"data")
        buffer.write(struct.pack("<I", data_size))
        buffer.write(self.audio)
        return buffer.getvalue()
