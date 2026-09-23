"""
KugelAudio Python SDK - Official client for KugelAudio TTS API.

Example usage:
    from kugelaudio import KugelAudio

    client = KugelAudio(api_key="your_api_key")

    # List available models
    models = client.models.list()

    # List available voices
    voices = client.voices.list().voices

    # Generate audio (non-streaming)
    audio = client.tts.generate(
        text="Hello, world!",
        model_id="kugel-3",
        voice_id=123,
    )
    audio.save("output.wav")

    # Generate audio (streaming via WebSocket)
    for chunk in client.tts.stream(
        text="Hello, world!",
        model_id="kugel-3",
        voice_id=123,
    ):
        # Process audio chunk
        pass
"""

from kugelaudio.client import KugelAudio
from kugelaudio.exceptions import (
    AuthenticationError,
    InsufficientCreditsError,
    KugelAudioError,
    NotFoundError,
    RateLimitError,
    ServerRestartingError,
    ValidationError,
)
from kugelaudio.exceptions import (
    ConnectionError as KugelAudioConnectionError,
)
from kugelaudio.models import (
    AudioChunk,
    AudioResponse,
    BulkReplaceResult,
    Dictionary,
    DictionaryEntry,
    DictionaryEntryList,
    GeneratedSample,
    GenerateRequest,
    Model,
    SessionUsage,
    StreamConfig,
    StreamingTranscriptionEvent,
    TranscriptionResponse,
    Voice,
    VoiceDetail,
    VoiceListResponse,
    VoiceReference,
    WordAlternatives,
    WordTimestamp,
)
from kugelaudio.streaming import (
    MultiContextSession,
    StreamingSession,
    StreamingSessionSync,
)

__version__ = "2.1.0"
__all__ = [
    "AudioChunk",
    "AudioResponse",
    "AuthenticationError",
    "BulkReplaceResult",
    "Dictionary",
    "DictionaryEntry",
    "DictionaryEntryList",
    "GeneratedSample",
    "GenerateRequest",
    "InsufficientCreditsError",
    "KugelAudio",
    "KugelAudioConnectionError",
    "KugelAudioError",
    "Model",
    "MultiContextSession",
    "NotFoundError",
    "RateLimitError",
    "ServerRestartingError",
    "SessionUsage",
    "StreamConfig",
    "StreamingSession",
    "StreamingSessionSync",
    "StreamingTranscriptionEvent",
    "TranscriptionResponse",
    "ValidationError",
    "Voice",
    "VoiceDetail",
    "VoiceListResponse",
    "VoiceReference",
    "WordAlternatives",
    "WordTimestamp",
]
