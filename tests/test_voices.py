"""Tests for the voices SDK surface against a stubbed ``client._request``."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import kugelaudio
from kugelaudio import GeneratedSample, KugelAudio, KugelAudioError, VoiceReference


def test_generate_sample_returns_server_shape():
    """The route returns ``{sample_s3_path, sample_url}``, not a voice row."""
    client = KugelAudio(api_key="test_key")
    payload = {
        "sample_s3_path": "voices/7/sample.wav",
        "sample_url": "https://cdn.example/voices/7/sample.wav?sig=abc",
    }
    with patch.object(client, "_request", return_value=payload) as m:
        result = client.voices.generate_sample(7)

    m.assert_called_once_with("POST", "/v1/voices/7/generate-sample")
    assert isinstance(result, GeneratedSample)
    assert result.sample_s3_path == "voices/7/sample.wav"
    assert result.sample_url == "https://cdn.example/voices/7/sample.wav?sig=abc"


def test_generate_sample_url_is_optional():
    """``sample_url`` is ``str | None`` on the server when signing fails."""
    client = KugelAudio(api_key="test_key")
    with patch.object(
        client, "_request", return_value={"sample_s3_path": "voices/7/sample.wav"}
    ):
        result = client.voices.generate_sample(7)

    assert result.sample_s3_path == "voices/7/sample.wav"
    assert result.sample_url is None


def test_generated_sample_is_exported():
    assert "GeneratedSample" in kugelaudio.__all__


def test_list_references_parses_bare_array():
    """GET /v1/voices/{id}/references returns a JSON array, not an object."""
    client = KugelAudio(api_key="test_key")
    payload = [
        {"id": 3, "voice_id": 7, "name": "ref.wav", "audio_url": "https://cdn/ref"},
        {"id": 4, "voice_id": 7, "is_generated": True},
    ]
    with patch.object(client, "_request", return_value=payload) as m:
        refs = client.voices.list_references(7)

    m.assert_called_once_with("GET", "/v1/voices/7/references")
    assert [r.id for r in refs] == [3, 4]
    assert all(isinstance(r, VoiceReference) for r in refs)
    assert refs[0].audio_url == "https://cdn/ref"
    assert refs[1].is_generated is True


def test_list_references_rejects_unexpected_shape():
    client = KugelAudio(api_key="test_key")
    with patch.object(client, "_request", return_value={"references": []}):
        with pytest.raises(KugelAudioError, match="JSON array"):
            client.voices.list_references(7)


def _multipart_fields(mock) -> dict:
    """Form fields of the single multipart request, keyed by part name."""
    (method, path), kwargs = mock.call_args
    return {"method": method, "path": path, **dict(kwargs["files"])}


def test_create_requires_and_sends_speaker_consent():
    client = KugelAudio(api_key="test_key")
    with patch.object(
        client, "_request_multipart", return_value={"id": 7, "name": "Anna"}
    ) as m:
        client.voices.create("Anna", "female", speaker_consent=True)

    fields = _multipart_fields(m)
    assert (fields["method"], fields["path"]) == ("POST", "/v1/voices")
    _, metadata, content_type = fields["metadata"]
    assert content_type == "application/json"
    assert json.loads(metadata)["speaker_consent"] is True

    with pytest.raises(TypeError, match="speaker_consent"):
        client.voices.create("Anna", "female")  # type: ignore[call-arg]


def test_create_sends_a_false_consent_for_the_server_to_refuse():
    client = KugelAudio(api_key="test_key")
    with patch.object(
        client, "_request_multipart", return_value={"id": 7, "name": "Anna"}
    ) as m:
        client.voices.create("Anna", "female", speaker_consent=False)

    _, metadata, _ = _multipart_fields(m)["metadata"]
    assert json.loads(metadata)["speaker_consent"] is False


def test_add_reference_requires_and_sends_speaker_consent(tmp_path):
    audio = tmp_path / "ref.wav"
    audio.write_bytes(b"\x01\x02\x03")
    client = KugelAudio(api_key="test_key")
    reference = {"id": 3, "voice_id": 7, "name": "ref.wav", "s3_path": "7/ref.wav"}
    with patch.object(client, "_request_multipart", return_value=reference) as m:
        result = client.voices.add_reference(
            7, audio, speaker_consent=True, reference_text="Hallo"
        )

    assert isinstance(result, VoiceReference)
    fields = _multipart_fields(m)
    assert (fields["method"], fields["path"]) == ("POST", "/v1/voices/7/references")
    assert fields["speaker_consent"] == (None, "true", "text/plain")
    assert fields["reference_text"] == (None, "Hallo", "text/plain")

    with pytest.raises(TypeError, match="speaker_consent"):
        client.voices.add_reference(7, audio)  # type: ignore[call-arg]
