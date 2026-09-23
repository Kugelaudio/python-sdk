"""Tests for kugelaudio models."""

import pytest
from kugelaudio.models import (
    AudioChunk,
    AudioResponse,
    GenerateRequest,
    StreamConfig,
    StreamingTranscriptionEvent,
    TranscriptionResponse,
    Voice,
    VoiceCategory,
)


def test_asr_model_identity_fields_preserve_positional_constructors() -> None:
    response = TranscriptionResponse("text", "text", "en", 1.0, "model", [], {})
    event = StreamingTranscriptionEvent(
        "text", True, "partial", "client_end_of_speech", 1.0, 2.0, []
    )

    assert response.word_alternatives == []
    assert response.model_revision is None
    assert event.turn_end_reason == "client_end_of_speech"
    assert event.model is None


class TestVoice:
    def test_from_dict_minimal(self):
        data = {
            "id": 123,
            "name": "Test Voice",
        }
        voice = Voice.from_dict(data)

        assert voice.id == 123
        assert voice.name == "Test Voice"
        assert voice.description is None
        assert voice.category is None
        assert voice.supported_languages == []

    def test_from_dict_full(self):
        data = {
            "id": 456,
            "name": "Full Voice",
            "description": "A test voice",
            "category": "premade",
            "sex": "female",
            "age": "young",
            "supported_languages": ["en", "de"],
            "sample_text": "Hello world",
            "is_public": True,
            "verified": True,
        }
        voice = Voice.from_dict(data)

        assert voice.id == 456
        assert voice.name == "Full Voice"
        assert voice.description == "A test voice"
        assert voice.category == VoiceCategory.PREMADE
        assert voice.supported_languages == ["en", "de"]
        assert voice.is_public is True
        assert voice.verified is True


class TestGenerateRequest:
    def test_to_dict_minimal(self):
        request = GenerateRequest(text="Hello world")
        data = request.to_dict()

        assert data["text"] == "Hello world"
        assert data["model_id"] == "kugel-3"
        assert data["cfg_scale"] == 2.0
        assert data["normalize"] is True
        assert "voice_id" not in data


    def test_dictionary_ids_empty_list_is_sent(self):
        """[] is the explicit opt-out and must reach the server."""
        data = GenerateRequest(text="hi", dictionary_ids=[]).to_dict()
        assert data["dictionary_ids"] == []

    def test_project_id_sent_only_when_set(self):
        assert "project_id" not in GenerateRequest(text="hi").to_dict()
        data = GenerateRequest(text="hi", project_id=42, dictionary_ids=[7]).to_dict()
        assert data["project_id"] == 42
        assert data["dictionary_ids"] == [7]


    @pytest.mark.parametrize(
        "cfg,expected",
        [(0.0, 1.2), (1.0, 1.2), (1.19, 1.2), (2.51, 2.5), (5.0, 2.5), (10.0, 2.5)],
    )
    def test_cfg_scale_out_of_band_clamped(self, cfg, expected):
        assert GenerateRequest(text="hi", cfg_scale=cfg).cfg_scale == expected


class TestAudioChunk:
    def test_duration_seconds(self):
        chunk = AudioChunk(
            audio=b"\x00" * 48000,  # 24000 samples at 16-bit
            index=0,
            sample_rate=24000,
            samples=24000,
        )
        assert chunk.duration_seconds == 1.0


class TestAudioResponse:
    def test_from_chunks(self):
        chunk1 = AudioChunk(audio=b"\x00\x10", index=0, sample_rate=24000, samples=1)
        chunk2 = AudioChunk(audio=b"\x00\x20", index=1, sample_rate=24000, samples=1)

        final_stats = {
            "total_samples": 2,
            "dur_ms": 83.33,
            "gen_ms": 50.0,
            "rtf": 0.6,
        }

        response = AudioResponse.from_chunks([chunk1, chunk2], final_stats)

        assert response.audio == b"\x00\x10\x00\x20"
        assert response.sample_rate == 24000
        assert response.samples == 2
        assert response.duration_ms == 83.33
        assert response.usage is None

    def test_from_chunks_parses_usage(self):
        chunk = AudioChunk(audio=b"\x00\x10", index=0, sample_rate=24000, samples=1)
        final_stats = {
            "total_samples": 1,
            "dur_ms": 5.0,
            "gen_ms": 3.0,
            "rtf": 0.6,
            "usage": {
                "audio_seconds": 5.4,
                "characters": 12,
                "cost_cents": 0.49,
                "currency": "eur",
                "model_id": "kugel-3",
            },
        }

        response = AudioResponse.from_chunks([chunk], final_stats)

        assert response.usage is not None
        assert response.usage.audio_seconds == 5.4
        assert response.usage.cost_cents == 0.49
        assert response.usage.cost_available is True

    def test_to_wav_bytes(self):
        response = AudioResponse(
            audio=b"\x00\x10\x00\x20",
            sample_rate=24000,
            samples=2,
            duration_ms=83.33,
            generation_ms=50.0,
            rtf=0.6,
        )

        wav_bytes = response.to_wav_bytes()

        # Check WAV header
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"
        assert wav_bytes[12:16] == b"fmt "


class TestStreamConfig:

    @pytest.mark.parametrize("cfg,expected", [(1.0, 1.2), (3.5, 2.5)])
    def test_cfg_scale_out_of_band_clamped(self, cfg, expected):
        assert StreamConfig(cfg_scale=cfg).cfg_scale == expected


    def test_to_dict_minimal(self):
        """Test to_dict with defaults — optional fields excluded."""
        config = StreamConfig()
        data = config.to_dict()

        assert data["cfg_scale"] == 2.0
        assert data["max_new_tokens"] == 2048
        assert data["sample_rate"] == 24000
        assert data["flush_timeout_ms"] == 500
        assert data["max_buffer_length"] == 1000
        assert data["normalize"] is True
        # Optional fields should not be present
        assert "voice_id" not in data
        assert "model_id" not in data
        assert "language" not in data
        assert "output_format" not in data


    def test_to_dict_dictionary_ids(self):
        assert "dictionary_ids" not in StreamConfig().to_dict()
        assert StreamConfig(dictionary_ids=[3]).to_dict()["dictionary_ids"] == [3]
        # [] is the explicit opt-out and must reach the server.
        assert StreamConfig(dictionary_ids=[]).to_dict()["dictionary_ids"] == []

    def test_to_dict_project_id(self):
        assert "project_id" not in StreamConfig().to_dict()
        assert StreamConfig(project_id=42).to_dict()["project_id"] == 42


# Mirrors public.voice_category (supabase/schemas/01_tables_views.sql) plus the
# "cloned" placeholder ingress returns for a null category.
_SERVER_CATEGORIES = [
    "narrative_story",
    "conversational",
    "characters_animation",
    "social_media",
    "entertainment_tv",
    "advertisement",
    "informative_educational",
    "cloned",
]


@pytest.mark.parametrize("value", _SERVER_CATEGORIES)
def test_every_server_category_round_trips(value: str) -> None:
    """A server category must never be rewritten to a different member."""
    assert VoiceCategory(value).value == value
    assert Voice.from_dict({"id": 1, "name": "v", "category": value}).category.value == value


def test_unknown_category_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="kugelaudio.models"):
        assert VoiceCategory("brand_new_category") is VoiceCategory.CLONED
    assert "brand_new_category" in caplog.text
