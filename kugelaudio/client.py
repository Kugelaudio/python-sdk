"""Main client for KugelAudio SDK."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from urllib.parse import urljoin, urlparse

import httpx
from pathlib import Path

if TYPE_CHECKING:
    from kugelaudio.streaming import (
        MultiContextSession,
        StreamingSession,
        StreamingSessionSync,
    )
from kugelaudio.exceptions import (
    AuthenticationError,
    ConnectionError as KugelAudioConnectionError,
    InsufficientCreditsError,
    KugelAudioError,
    RateLimitError,
    ValidationError,
    classify_http_response,
    classify_ws_close,
    classify_ws_frame,
    classify_ws_handshake_error,
    ws_handshake_error_types,
)
from kugelaudio._diagnostics_config import DiagnosticsConfig
from kugelaudio._diagnostics import (
    OP_DICTIONARIES,
    OP_GENERATE,
    OP_MODELS,
    OP_STREAM,
    OP_TRANSCRIBE,
    OP_VOICES,
    STAGE_AWAITING_FIRST_AUDIO,
    STAGE_CONNECTING,
    STAGE_HANDSHAKE,
    STAGE_SENDING_REQUEST,
    TRANSPORT_HTTP,
    TRANSPORT_WEBSOCKET,
    INTEGRATION_NONE,
    Diagnostics,
    Operation,
)
from kugelaudio._sdk_metadata import sdk_headers, sdk_query_string
from kugelaudio.models import (
    AudioChunk,
    AudioResponse,
    GeneratedSample,
    GenerateRequest,
    Model,
    PUBLIC_ASR_MODEL_ID,
    TranscriptionResponse,
    StreamConfig,
    Voice,
    VoiceDetail,
    VoiceListResponse,
    VoiceReference,
    WordTimestamp,
    clamp_cfg_scale,
)

logger = logging.getLogger(__name__)

_language_warning_lock = threading.Lock()
_language_warning_logged = False


def _warn_if_no_language(language: Optional[str], normalize: bool = True) -> None:
    """Log a one-time warning when normalization is enabled without an explicit language."""
    global _language_warning_logged
    if language is None and normalize and not _language_warning_logged:
        with _language_warning_lock:
            if not _language_warning_logged:
                _language_warning_logged = True
                logger.warning(
                    "No 'language' set with normalization enabled: the server "
                    "normalizes in the voice's primary language (English if it has "
                    "none). Set language (e.g., language='de') when the text is in "
                    "another language."
                )


# Default API endpoints
# By default, TTS WebSocket connects to the same URL as the API
# (the backend proxies WebSocket requests to the TTS server)
DEFAULT_API_URL = "https://api.kugelaudio.com"
DEFAULT_TTS_URL = None  # If None, uses api_url

EU_API_URL = "https://api.eu.kugelaudio.com"
SUPPORTED_REGIONS = ("eu", "us", "global")


def _sdk_query_separator(url: str) -> str:
    return "&" if "?" in url else "?"


def _resolve_region_url(region: Optional[str]) -> str:
    """Resolve a supported region hint to an API URL.

    Only EU has a dedicated direct endpoint. All other supported hints use the
    canonical geo-routed API.
    """
    if region is None:
        return DEFAULT_API_URL
    if region not in SUPPORTED_REGIONS:
        raise ValidationError(
            f"Invalid region '{region}'. Must be one of: {', '.join(SUPPORTED_REGIONS)}"
        )
    if region == "eu":
        return EU_API_URL
    return DEFAULT_API_URL

# Prefixes that can be prepended to API keys to select a region
_REGION_PREFIXES = tuple(f"{region}-" for region in SUPPORTED_REGIONS)


#: REST path prefix -> ``kugel.operation`` value. Derived from the path so
#: the ~25 ``_request`` call sites stay untouched.
_HTTP_OPERATIONS = {
    "models": OP_MODELS,
    "voices": OP_VOICES,
    "dictionaries": OP_DICTIONARIES,
    "audio": OP_TRANSCRIBE,
}


def _operation_for_path(path: str) -> Optional[str]:
    """Map ``/v1/voices/12/references`` to ``voices``; unknown paths -> None."""
    trimmed = path.lstrip("/")
    if trimmed.startswith("v1/"):
        trimmed = trimmed[3:]
    return _HTTP_OPERATIONS.get(trimmed.split("/", 1)[0])


def _parse_api_key(api_key: str) -> tuple[str, Optional[str]]:
    """Strip a region prefix from an API key.

    Returns:
        (clean_key, detected_region) — detected_region is None when no prefix is present.
    """
    for prefix in _REGION_PREFIXES:
        if api_key.startswith(prefix):
            return api_key[len(prefix):], prefix.rstrip("-")
    return api_key, None


def _auth_headers(clean_key: str) -> Dict[str, str]:
    """The SDK's auth headers. The HTTP client and the diagnostics reporter
    must present the SAME credential; the reporter never builds its own."""
    return {"Authorization": f"Bearer {clean_key}", "X-API-Key": clean_key}


def _server_frame_error(op: Operation, data: Dict[str, Any]) -> KugelAudioError:
    """The typed error for a server error frame, marked on *op* so the event
    is ``request_failed`` (``retry_exhausted`` after a retry), not the
    ``stream_interrupted`` its receive stage would suggest."""
    err = classify_ws_frame(data)
    op.mark_server_error(err)
    return err


def _build_diagnostics(
    api_url: str, clean_key: str, telemetry: Optional[bool], integration: str
) -> Diagnostics:
    """The per-client reporter: ``<api_url>/v1/sdk-diagnostics``, this key's
    auth headers, and the integration (``none``/``livekit``/``pipecat``) that
    built the client. An ingress without the route answers 404 and the
    reporter switches itself off after three of them."""
    config = DiagnosticsConfig.resolve(
        api_url, telemetry, auth_headers=_auth_headers(clean_key)
    )
    return Diagnostics(config, integration=integration)


class ModelsResource:
    """Resource for listing available TTS models."""

    def __init__(self, client: KugelAudio):
        self._client = client

    def list(self) -> List[Model]:
        """List available TTS models.

        Returns:
            List of available models
        """
        response = self._client._request("GET", "/v1/models")
        return [Model.from_dict(m) for m in response.get("models", [])]


class ASRResource:
    """Speech-to-text uploads through the public KugelAudio ingress."""

    def __init__(self, client: KugelAudio):
        self._client = client

    def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: Optional[str] = None,
        model: str = PUBLIC_ASR_MODEL_ID,
    ) -> TranscriptionResponse:
        if not audio:
            raise ValidationError("ASR audio must not be empty")
        if model != PUBLIC_ASR_MODEL_ID:
            raise ValidationError(
                f"Unsupported ASR model {model!r}; use {PUBLIC_ASR_MODEL_ID!r}"
            )
        fields: List[Tuple[str, Tuple[Optional[str], Any, str]]] = [
            ("file", (filename, audio, content_type)),
            ("model", (None, model, "text/plain")),
        ]
        if language is not None:
            fields.append(("language", (None, language, "text/plain")))
        response = self._client._request_multipart(
            "POST", "/v1/audio/transcriptions", fields
        )
        return TranscriptionResponse.from_dict(response)


class VoicesResource:
    """Resource for managing voices."""

    def __init__(self, client: KugelAudio):
        self._client = client

    def list(
        self,
        language: Optional[str] = None,
        include_public: bool = True,
        limit: int = 20,
        offset: int = 0,
    ) -> VoiceListResponse:
        """List available voices.

        Args:
            language: Filter by language code (e.g., 'en', 'de')
            include_public: Include public voices
            limit: Maximum number of voices to return (max 100)
            offset: Number of voices to skip for pagination

        Returns:
            Paginated voice list response
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if language:
            params["language"] = language
        if include_public:
            params["include_public"] = "true"

        response = self._client._request("GET", "/v1/voices", params=params)
        return VoiceListResponse.from_dict(response)

    def get(self, voice_id: int) -> VoiceDetail:
        """Get a specific voice by ID.

        Args:
            voice_id: Voice ID

        Returns:
            Detailed voice information
        """
        response = self._client._request("GET", f"/v1/voices/{voice_id}")
        return VoiceDetail.from_dict(response)

    def create(
        self,
        name: str,
        sex: str,
        *,
        description: str = "",
        category: str = "conversational",
        age: str = "middle_age",
        quality: str = "mid",
        supported_languages: Optional[List[str]] = None,
        is_public: bool = False,
        sample_text: str = "",
        reference_files: Optional[List[Union[str, Path]]] = None,
    ) -> VoiceDetail:
        """Create a new voice.

        Args:
            name: Voice name
            sex: Voice sex ('male' or 'female')
            description: Human-readable description
            category: Voice category (e.g. 'conversational', 'narrative_story')
            age: Voice age ('young', 'middle_age', 'old')
            quality: Voice quality ('low', 'mid', 'high')
            supported_languages: List of language codes (default: ['en'])
            is_public: Whether the voice should be public
            sample_text: Text used for automatic sample generation
            reference_files: Optional list of reference audio file paths to upload

        Returns:
            Created voice details
        """
        metadata: Dict[str, Any] = {
            "name": name,
            "sex": sex,
            "description": description,
            "category": category,
            "age": age,
            "quality": quality,
            "supported_languages": supported_languages or ["en"],
            "is_public": is_public,
            "sample_text": sample_text,
        }

        files_list: List[Tuple[str, Tuple[Optional[str], Any, str]]] = [
            ("metadata", (None, json.dumps(metadata), "application/json")),
        ]
        opened_files: List[IO[bytes]] = []
        try:
            if reference_files:
                for fpath in reference_files:
                    p = Path(fpath)
                    fh = open(p, "rb")
                    opened_files.append(fh)
                    files_list.append(("files", (p.name, fh, "audio/wav")))

            response = self._client._request_multipart(
                "POST", "/v1/voices", files=files_list
            )
        finally:
            for fh in opened_files:
                fh.close()

        return VoiceDetail.from_dict(response)

    def update(
        self,
        voice_id: int,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        age: Optional[str] = None,
        sex: Optional[str] = None,
        quality: Optional[str] = None,
        supported_languages: Optional[List[str]] = None,
        is_public: Optional[bool] = None,
        sample_text: Optional[str] = None,
    ) -> VoiceDetail:
        """Update an existing voice.

        Only provided fields will be updated. Pass None to leave a field unchanged.

        Args:
            voice_id: Voice ID to update
            name: New voice name
            description: New description
            category: New category
            age: New age
            sex: New sex
            quality: New quality
            supported_languages: New list of language codes
            is_public: New public visibility
            sample_text: New sample text

        Returns:
            Updated voice details
        """
        payload: Dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if category is not None:
            payload["category"] = category
        if age is not None:
            payload["age"] = age
        if sex is not None:
            payload["sex"] = sex
        if quality is not None:
            payload["quality"] = quality
        if supported_languages is not None:
            payload["supported_languages"] = supported_languages
        if is_public is not None:
            payload["is_public"] = is_public
        if sample_text is not None:
            payload["sample_text"] = sample_text

        response = self._client._request(
            "PATCH", f"/v1/voices/{voice_id}", json_data=payload
        )
        return VoiceDetail.from_dict(response)

    def delete(self, voice_id: int) -> None:
        """Delete a voice.

        Args:
            voice_id: Voice ID to delete
        """
        self._client._request("DELETE", f"/v1/voices/{voice_id}")

    # -- Reference management --

    def list_references(self, voice_id: int) -> List[VoiceReference]:
        """List reference audio files for a voice.

        Args:
            voice_id: Voice ID

        Returns:
            List of voice references
        """
        path = f"/v1/voices/{voice_id}/references"
        # The route returns a bare JSON array, unlike the object-shaped
        # endpoints ``_request`` is annotated for.
        response: object = self._client._request("GET", path)
        if not isinstance(response, list):
            raise KugelAudioError(
                f"Expected a JSON array from GET {path}, "
                f"got {type(response).__name__}"
            )
        return [VoiceReference.from_dict(r) for r in response]

    def add_reference(
        self,
        voice_id: int,
        file: Union[str, Path],
        *,
        reference_text: str = "",
    ) -> VoiceReference:
        """Upload a reference audio file to a voice.

        Args:
            voice_id: Voice ID
            file: Path to the reference audio file
            reference_text: Transcript of the reference audio

        Returns:
            Created reference metadata
        """
        p = Path(file)
        fh = open(p, "rb")
        try:
            files_list: List[Tuple[str, Tuple[Optional[str], Any, str]]] = [
                ("file", (p.name, fh, "audio/wav")),
            ]
            if reference_text:
                files_list.append(
                    ("reference_text", (None, reference_text, "text/plain")),
                )

            response = self._client._request_multipart(
                "POST", f"/v1/voices/{voice_id}/references", files=files_list
            )
        finally:
            fh.close()

        return VoiceReference.from_dict(response)

    def delete_reference(self, voice_id: int, reference_id: int) -> None:
        """Delete a reference audio file from a voice.

        Args:
            voice_id: Voice ID
            reference_id: Reference ID to delete
        """
        self._client._request(
            "DELETE", f"/v1/voices/{voice_id}/references/{reference_id}"
        )

    # -- Publishing --

    def publish(self, voice_id: int) -> VoiceDetail:
        """Request publication of a voice.

        Sets the voice as public and marks it as pending verification.
        An admin must verify the voice before it appears in public listings.

        Args:
            voice_id: Voice ID

        Returns:
            Updated voice details
        """
        response = self._client._request(
            "POST", f"/v1/voices/{voice_id}/publish"
        )
        return VoiceDetail.from_dict(response)

    # -- Sample generation --

    def generate_sample(self, voice_id: int) -> GeneratedSample:
        """Trigger sample audio generation for a voice.

        Uses the voice's sample_text (or a default) to generate a preview sample.
        The voice needs at least one uploaded reference.

        Args:
            voice_id: Voice ID

        Returns:
            The stored sample's path and a signed ``sample_url``
        """
        response = self._client._request(
            "POST", f"/v1/voices/{voice_id}/generate-sample"
        )
        return GeneratedSample.from_dict(response)


class DictionaryEntriesResource:
    """Resource for managing entries within a single dictionary.

    Accessed via ``client.dictionaries.entries`` and parametrized with
    ``dictionary_id`` on each call.
    """

    def __init__(self, client: KugelAudio):
        self._client = client

    def list(
        self,
        dictionary_id: int,
        *,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        project_id: Optional[int] = None,
    ) -> "DictionaryEntryList":
        """List entries in a dictionary, optionally filtered by *search*.

        Args:
            dictionary_id: Dictionary ID.
            search: Case-insensitive substring filter on ``word``.
            limit: Page size, 1-500. Defaults to 100.
            offset: Pagination offset.
            project_id: Required only for master-key callers.
        """
        from kugelaudio.models import DictionaryEntryList

        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if search is not None:
            params["search"] = search
        if project_id is not None:
            params["project_id"] = project_id
        response = self._client._request(
            "GET",
            f"/v1/dictionaries/{dictionary_id}/entries",
            params=params,
        )
        return DictionaryEntryList.from_dict(response)

    def add(
        self,
        dictionary_id: int,
        *,
        word: str,
        replacement: str,
        ipa: Optional[str] = None,
        case_sensitive: bool = False,
        project_id: Optional[int] = None,
    ) -> "DictionaryEntry":
        """Add a single entry to a dictionary."""
        from kugelaudio.models import DictionaryEntry

        payload: Dict[str, Any] = {
            "word": word,
            "replacement": replacement,
            "case_sensitive": case_sensitive,
        }
        if ipa is not None:
            payload["ipa"] = ipa
        params = {"project_id": project_id} if project_id is not None else None
        response = self._client._request(
            "POST",
            f"/v1/dictionaries/{dictionary_id}/entries",
            params=params,
            json_data=payload,
        )
        return DictionaryEntry.from_dict(response)

    def update(
        self,
        dictionary_id: int,
        entry_id: int,
        *,
        word: Optional[str] = None,
        replacement: Optional[str] = None,
        ipa: Optional[str] = None,
        case_sensitive: Optional[bool] = None,
        project_id: Optional[int] = None,
    ) -> "DictionaryEntry":
        """Update an existing entry. Only provided fields are changed."""
        from kugelaudio.models import DictionaryEntry

        payload: Dict[str, Any] = {}
        if word is not None:
            payload["word"] = word
        if replacement is not None:
            payload["replacement"] = replacement
        if ipa is not None:
            payload["ipa"] = ipa
        if case_sensitive is not None:
            payload["case_sensitive"] = case_sensitive
        params = {"project_id": project_id} if project_id is not None else None
        response = self._client._request(
            "PATCH",
            f"/v1/dictionaries/{dictionary_id}/entries/{entry_id}",
            params=params,
            json_data=payload,
        )
        return DictionaryEntry.from_dict(response)

    def delete(
        self,
        dictionary_id: int,
        entry_id: int,
        *,
        project_id: Optional[int] = None,
    ) -> None:
        """Delete a single entry."""
        params = {"project_id": project_id} if project_id is not None else None
        self._client._request(
            "DELETE",
            f"/v1/dictionaries/{dictionary_id}/entries/{entry_id}",
            params=params,
        )

    def replace_all(
        self,
        dictionary_id: int,
        entries: List[Dict[str, Any]],
        *,
        project_id: Optional[int] = None,
    ) -> "BulkReplaceResult":
        """Atomically replace every entry in the dictionary with *entries*.

        Each item in *entries* must contain ``word`` and ``replacement``,
        optionally ``ipa`` and ``case_sensitive``. Entries currently in
        the dictionary whose ``word`` is not in the supplied list are
        deleted. Idempotent — useful for syncing from an external source.

        Returns counts of what changed (upserted, deleted, total).
        """
        from kugelaudio.models import BulkReplaceResult

        params = {"project_id": project_id} if project_id is not None else None
        response = self._client._request(
            "PUT",
            f"/v1/dictionaries/{dictionary_id}/entries",
            params=params,
            json_data={"entries": entries},
        )
        return BulkReplaceResult.from_dict(response)


class DictionariesResource:
    """Resource for managing per-project custom dictionaries.

    Customers use this to sync pronunciation/replacement dictionaries
    programmatically. The TTS-side cache is invalidated automatically
    after every mutation, so the next synthesis request picks up the
    change immediately.

    Example::

        # Create a dictionary and add a few words
        d = client.dictionaries.create(name="Brand names")
        client.dictionaries.entries.add(d.id, word="Postgres", replacement="post-gres")
        client.dictionaries.entries.add(d.id, word="K8s", replacement="kubernetes")

        # Sync from an external source — idempotent
        client.dictionaries.entries.replace_all(d.id, entries=[
            {"word": "Postgres", "replacement": "post-gres"},
            {"word": "Kubernetes", "replacement": "koo-ber-net-eez"},
        ])
    """

    def __init__(self, client: KugelAudio):
        self._client = client
        self.entries = DictionaryEntriesResource(client)

    def list(self, *, project_id: Optional[int] = None) -> List["Dictionary"]:
        """List every dictionary in the caller's project.

        Args:
            project_id: Required only for master-key callers.
        """
        from kugelaudio.models import Dictionary

        params = {"project_id": project_id} if project_id is not None else None
        response = self._client._request(
            "GET", "/v1/dictionaries", params=params
        )
        return [Dictionary.from_dict(d) for d in response.get("dictionaries", [])]

    def create(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        language: Optional[str] = None,
        project_id: Optional[int] = None,
    ) -> "Dictionary":
        """Create a new dictionary."""
        from kugelaudio.models import Dictionary

        payload: Dict[str, Any] = {"name": name}
        if description is not None:
            payload["description"] = description
        if language is not None:
            payload["language"] = language
        params = {"project_id": project_id} if project_id is not None else None
        response = self._client._request(
            "POST", "/v1/dictionaries", params=params, json_data=payload
        )
        return Dictionary.from_dict(response)

    def get(
        self,
        dictionary_id: int,
        *,
        project_id: Optional[int] = None,
    ) -> "Dictionary":
        """Fetch a single dictionary by ID."""
        from kugelaudio.models import Dictionary

        params = {"project_id": project_id} if project_id is not None else None
        response = self._client._request(
            "GET", f"/v1/dictionaries/{dictionary_id}", params=params
        )
        return Dictionary.from_dict(response)

    def update(
        self,
        dictionary_id: int,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        language: Optional[str] = None,
        is_active: Optional[bool] = None,
        project_id: Optional[int] = None,
    ) -> "Dictionary":
        """Update name / description / language / is_active.

        Only provided fields are changed.
        """
        from kugelaudio.models import Dictionary

        payload: Dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if language is not None:
            payload["language"] = language
        if is_active is not None:
            payload["is_active"] = is_active
        params = {"project_id": project_id} if project_id is not None else None
        response = self._client._request(
            "PATCH",
            f"/v1/dictionaries/{dictionary_id}",
            params=params,
            json_data=payload,
        )
        return Dictionary.from_dict(response)

    def delete(
        self,
        dictionary_id: int,
        *,
        project_id: Optional[int] = None,
    ) -> None:
        """Delete a dictionary (cascades to its entries)."""
        params = {"project_id": project_id} if project_id is not None else None
        self._client._request(
            "DELETE", f"/v1/dictionaries/{dictionary_id}", params=params
        )


class TTSResource:
    """Resource for text-to-speech generation."""

    def __init__(self, client: KugelAudio):
        self._client = client
        self._ws_connection: Optional[Any] = None
        self._ws_lock = asyncio.Lock()
        self._ws_url: Optional[str] = None
        self._keepalive_task: Optional[asyncio.Task[None]] = None

    def generate(
        self,
        text: str,
        model_id: str = "kugel-3",
        voice_id: Optional[int] = None,
        cfg_scale: float = 2.0,
        temperature: Optional[float] = None,
        max_new_tokens: int = 2048,
        sample_rate: int = 24000,
        output_format: Optional[str] = None,
        normalize: bool = True,
        language: Optional[str] = None,
        word_timestamps: bool = False,
        speed: float = 1.0,
        dictionary_ids: Optional[List[int]] = None,
        project_id: Optional[int] = None,
    ) -> AudioResponse:
        """Generate audio from text (non-streaming).

        This method collects all audio chunks internally and returns
        the complete audio response.

        Args:
            text: Text to synthesize
            model_id: Model to use (e.g. 'kugel-3')
            voice_id: Voice ID to use
            cfg_scale: Classifier-free guidance scale. Clamped to [1.2, 2.5].
            max_new_tokens: Maximum tokens to generate
            sample_rate: Output sample rate (24000)
            output_format: Optional combined codec+rate token (e.g. 'ulaw_8000',
                'alaw_8000', 'pcm_8000'). Opt-in; absent ⇒ PCM16 at sample_rate.
            normalize: Enable text normalization (default: True).
                Set ``language`` when the text is not in the voice's primary
                language; the language is not detected from the text.
            language: ISO 639-1 language code for normalization (e.g., 'de', 'en').
                Supported: de, en, fr, es, it, pt, nl, pl, sv, da, no, fi, cs, hu, ro,
                el, uk, bg, tr, vi, ar, hi, zh, ja, ko, sk, sl, hr, sr, ru, he, fa,
                ur, bn, ta, yue, th, id, ms
            word_timestamps: Request word-level timestamps. When enabled,
                ``AudioResponse.word_timestamps`` will be populated with
                per-word timing boundaries.
            speed: Playback speed multiplier (0.8–1.2, default 1.0).
                Uses pitch-preserving time-stretching.
            dictionary_ids: Per-request dictionary selection. Omit (None) to
                apply all active dictionaries of the project; pass [] to
                disable dictionaries for this request; pass a list of
                dictionary IDs to apply exactly those dictionaries (even
                inactive ones), bypassing the language filter.
            project_id: Project whose pronunciation dictionaries apply at
                synthesis. Required for any dictionary to apply and for a
                non-empty ``dictionary_ids``; omit (None) to send no project.

        Returns:
            Complete audio response (with ``word_timestamps`` if requested)
        """
        # Use sync wrapper for async streaming
        chunks: List[AudioChunk] = []
        all_stamps: List[WordTimestamp] = []
        final_stats: Dict[str, Any] = {}

        for item in self.stream(
            text=text,
            model_id=model_id,
            voice_id=voice_id,
            cfg_scale=cfg_scale,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            sample_rate=sample_rate,
            output_format=output_format,
            normalize=normalize,
            language=language,
            word_timestamps=word_timestamps,
            speed=speed,
            dictionary_ids=dictionary_ids,
            project_id=project_id,
            _operation=OP_GENERATE,
        ):
            if isinstance(item, AudioChunk):
                chunks.append(item)
            elif isinstance(item, list):
                all_stamps.extend(item)
            elif isinstance(item, dict) and item.get("final"):
                final_stats = item

        return AudioResponse.from_chunks(chunks, final_stats, all_stamps)

    def stream(
        self,
        text: str,
        model_id: str = "kugel-3",
        voice_id: Optional[int] = None,
        cfg_scale: float = 2.0,
        temperature: Optional[float] = None,
        max_new_tokens: int = 2048,
        sample_rate: int = 24000,
        output_format: Optional[str] = None,
        normalize: bool = True,
        language: Optional[str] = None,
        word_timestamps: bool = False,
        speed: float = 1.0,
        dictionary_ids: Optional[List[int]] = None,
        project_id: Optional[int] = None,
        *,
        _operation: str = OP_STREAM,
    ) -> Iterator[Union[AudioChunk, List[WordTimestamp], Dict[str, Any]]]:
        """Stream audio from text via WebSocket.

        Yields audio chunks as they are generated. The final message
        contains stats about the generation.

        Args:
            text: Text to synthesize
            model_id: Model to use (e.g. 'kugel-3')
            voice_id: Voice ID to use
            cfg_scale: Classifier-free guidance scale. Clamped to [1.2, 2.5].
            max_new_tokens: Maximum tokens to generate
            sample_rate: Output sample rate
            output_format: Optional combined codec+rate token (e.g. 'ulaw_8000',
                'alaw_8000', 'pcm_8000'). Opt-in; absent ⇒ PCM16 at sample_rate.
            normalize: Enable text normalization (default: True).
                Set ``language`` when the text is not in the voice's primary
                language; the language is not detected from the text.
            language: ISO 639-1 language code for normalization (e.g., 'de', 'en').
                Supported: de, en, fr, es, it, pt, nl, pl, sv, da, no, fi, cs, hu, ro,
                el, uk, bg, tr, vi, ar, hi, zh, ja, ko, sk, sl, hr, sr, ru, he, fa,
                ur, bn, ta, yue, th, id, ms
            word_timestamps: Request word-level timestamps. When enabled, the
                server sends a ``word_timestamps`` message after audio containing
                per-word timing. Yielded as ``list[WordTimestamp]``.
            speed: Playback speed multiplier (0.8–1.2, default 1.0).
                Uses pitch-preserving time-stretching.
            dictionary_ids: Per-request dictionary selection. Omit (None) to
                apply all active dictionaries of the project; pass [] to
                disable dictionaries for this request; pass a list of
                dictionary IDs to apply exactly those dictionaries (even
                inactive ones), bypassing the language filter.
            project_id: Project whose pronunciation dictionaries apply at
                synthesis. Required for any dictionary to apply and for a
                non-empty ``dictionary_ids``; omit (None) to send no project.

        Yields:
            AudioChunk for audio data, list[WordTimestamp] for word timestamps,
            dict for final stats
        """
        import queue
        import threading

        # Use a thread-safe queue for true streaming
        result_queue: queue.Queue = queue.Queue()
        exception_holder: List[Exception] = []
        done_sentinel = object()

        async def collect():
            try:
                async for item in self.stream_async(
                    text=text,
                    model_id=model_id,
                    voice_id=voice_id,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    sample_rate=sample_rate,
                    output_format=output_format,
                    normalize=normalize,
                    language=language,
                    word_timestamps=word_timestamps,
                    speed=speed,
                    dictionary_ids=dictionary_ids,
                    project_id=project_id,
                    _operation=_operation,
                ):
                    result_queue.put(item)
            except Exception as e:
                exception_holder.append(e)
            finally:
                # Close the WS connection before the loop is destroyed
                # so the next sync call can create a fresh one on its own loop.
                await self._close_ws_connection()
                result_queue.put(done_sentinel)

        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(collect())
            finally:
                loop.close()

        # Reset the asyncio lock so it binds to the new thread's loop
        self._ws_lock = asyncio.Lock()

        # Start async collection in background thread
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()

        # Yield items as they arrive (true streaming)
        while True:
            item = result_queue.get()
            if item is done_sentinel:
                break
            yield item

        # Check for exceptions after streaming completes
        if exception_holder:
            raise exception_holder[0]

    async def _get_ws_connection(
        self, model_id: str, op: Optional[Operation] = None
    ) -> Any:
        """Get or create a WebSocket connection for the given model.

        This implements connection pooling to avoid the ~220ms connect
        overhead on each request.

        *op* is the caller's diagnostics operation, threaded through so a
        handshake failure is attributed to the ``handshake`` stage of the
        operation that asked for the socket rather than reported twice.
        """
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets is required for streaming. Install with: pip install websockets"
            )

        # Build WebSocket URL
        ws_url = self._client._tts_url.replace("https://", "wss://").replace(
            "http://", "ws://"
        )
        ws_url = f"{ws_url}/ws/tts?api_key={self._client._api_key}"
        if model_id:
            ws_url += f"&model_id={model_id}"
        ws_url += f"{_sdk_query_separator(ws_url)}{sdk_query_string()}"

        async with self._ws_lock:
            # Check if we have a valid connection for this URL
            # websockets uses .state (OPEN = 1) or .closed property
            if self._ws_connection is not None and self._ws_url == ws_url:
                try:
                    # Check if connection is still open
                    is_open = (
                        (
                            hasattr(self._ws_connection, "open")
                            and self._ws_connection.open
                        )
                        or (
                            hasattr(self._ws_connection, "state")
                            and self._ws_connection.state.name == "OPEN"
                        )
                        or (
                            hasattr(self._ws_connection, "closed")
                            and not self._ws_connection.closed
                        )
                    )
                    if is_open:
                        return self._ws_connection
                except Exception:
                    pass

            # Close old connection if URL changed
            if self._ws_connection is not None:
                try:
                    await self._ws_connection.close()
                except Exception:
                    pass
                self._ws_connection = None

            # Create new connection
            self._cancel_keepalive()
            if op is not None:
                op.set_stage(STAGE_HANDSHAKE)
            try:
                self._ws_connection = await websockets.connect(
                    ws_url, compression=None
                )
            except ws_handshake_error_types(websockets) as e:
                typed = classify_ws_handshake_error(e)
                if typed is not None:
                    raise typed from e
                raise KugelAudioConnectionError(
                    f"KugelAudio WebSocket handshake failed: {e}."
                ) from e
            self._ws_url = ws_url
            if self._client._keepalive_ping_interval is not None:
                self._keepalive_task = asyncio.create_task(
                    self._start_keepalive(self._ws_connection)
                )
            return self._ws_connection

    async def _start_keepalive(self, ws: Any) -> None:
        """Send periodic pings on the pooled connection to prevent idle timeouts."""
        interval = self._client._keepalive_ping_interval
        if interval is None:
            return
        try:
            while True:
                await asyncio.sleep(interval)
                if self._ws_connection is not ws:
                    break
                try:
                    await ws.ping()
                    logger.debug("Keepalive ping sent on pooled WebSocket")
                except Exception as e:
                    logger.debug("Keepalive ping failed (connection may be closed): %s", e)
                    break
        except asyncio.CancelledError:
            pass

    def _cancel_keepalive(self) -> None:
        if self._keepalive_task is not None and not self._keepalive_task.done():
            self._keepalive_task.cancel()
        self._keepalive_task = None

    async def _close_ws_connection(self) -> None:
        """Close the pooled WebSocket connection."""
        async with self._ws_lock:
            self._cancel_keepalive()
            if self._ws_connection is not None:
                try:
                    await self._ws_connection.close()
                except Exception:
                    pass
                self._ws_connection = None
                self._ws_url = None

    async def connect_async(self, model: str = "kugel-3") -> None:
        """Pre-establish WebSocket connection for faster first request.

        Call this at application startup to eliminate cold start latency
        (~300-600ms) from your first TTS request.

        Args:
            model: Model to connect for (e.g. 'kugel-3').
                   The connection is model-specific due to routing.

        Example:
            client = KugelAudio(api_key="...")

            # Pre-connect at startup
            await client.tts.connect_async()

            # First request is now fast (~100ms instead of ~500ms)
            async for chunk in client.tts.stream_async("Hello"):
                ...
        """
        op = self._client._diagnostics.operation(OP_STREAM, TRANSPORT_WEBSOCKET)
        async with op.step(STAGE_CONNECTING):
            await self._get_ws_connection(model, op=op)
        op.succeed()
        logger.info("WebSocket connection pre-established for model: %s", model)

    def connect(self, model: str = "kugel-3") -> None:
        """Pre-establish WebSocket connection for faster first request (sync version).

        Call this at application startup to eliminate cold start latency
        (~300-600ms) from your first TTS request.

        Args:
            model: Model to connect for (e.g. 'kugel-3').
                   The connection is model-specific due to routing.

        Example:
            client = KugelAudio(api_key="...")

            # Pre-connect at startup
            client.tts.connect()

            # First request is now fast (~100ms instead of ~500ms)
            for chunk in client.tts.stream("Hello"):
                ...
        """
        import threading

        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.connect_async(model))
            finally:
                loop.close()

        thread = threading.Thread(target=run)
        thread.start()
        thread.join()

    def is_connected(self) -> bool:
        """Check if WebSocket connection is established and open.

        Returns:
            True if connected and ready for requests.
        """
        if self._ws_connection is None:
            return False
        try:
            is_open = (
                (hasattr(self._ws_connection, "open") and self._ws_connection.open)
                or (
                    hasattr(self._ws_connection, "state")
                    and self._ws_connection.state.name == "OPEN"
                )
                or (
                    hasattr(self._ws_connection, "closed")
                    and not self._ws_connection.closed
                )
            )
            return is_open
        except Exception:
            return False

    async def stream_async(
        self,
        text: str,
        model_id: str = "kugel-3",
        voice_id: Optional[int] = None,
        cfg_scale: float = 2.0,
        temperature: Optional[float] = None,
        max_new_tokens: int = 2048,
        sample_rate: int = 24000,
        output_format: Optional[str] = None,
        reuse_connection: bool = True,
        normalize: bool = True,
        language: Optional[str] = None,
        word_timestamps: bool = False,
        speed: float = 1.0,
        dictionary_ids: Optional[List[int]] = None,
        project_id: Optional[int] = None,
        *,
        _operation: str = OP_STREAM,
    ) -> AsyncIterator[Union[AudioChunk, List[WordTimestamp], Dict[str, Any]]]:
        """Stream audio asynchronously via WebSocket.

        Args:
            text: Text to synthesize
            model_id: Model to use (e.g. 'kugel-3')
            voice_id: Voice ID to use
            cfg_scale: Classifier-free guidance scale. Clamped to [1.2, 2.5].
            max_new_tokens: Maximum tokens to generate
            sample_rate: Output sample rate
            reuse_connection: If True (default), reuse WebSocket connection
                for faster TTFA (~175ms vs ~390ms). Set to False to always
                create a new connection.
            normalize: Enable text normalization (default: True).
                Set ``language`` when the text is not in the voice's primary
                language; the language is not detected from the text.
            language: ISO 639-1 language code for normalization (e.g., 'de', 'en').
                Supported: de, en, fr, es, it, pt, nl, pl, sv, da, no, fi, cs, hu, ro,
                el, uk, bg, tr, vi, ar, hi, zh, ja, ko, sk, sl, hr, sr, ru, he, fa,
                ur, bn, ta, yue, th, id, ms
            word_timestamps: Request word-level timestamps. When enabled, the
                server sends a ``word_timestamps`` message after audio containing
                per-word timing. Yielded as ``list[WordTimestamp]``.
            speed: Playback speed multiplier (0.8–1.2, default 1.0).
                Uses pitch-preserving time-stretching.
            dictionary_ids: Per-request dictionary selection. Omit (None) to
                apply all active dictionaries of the project; pass [] to
                disable dictionaries for this request; pass a list of
                dictionary IDs to apply exactly those dictionaries (even
                inactive ones), bypassing the language filter.
            project_id: Project whose pronunciation dictionaries apply at
                synthesis. Required for any dictionary to apply and for a
                non-empty ``dictionary_ids``; omit (None) to send no project.

        Yields:
            AudioChunk for audio data, list[WordTimestamp] for word timestamps,
            dict for final stats
        """
        # One diagnostics operation per public call. ``_operation`` only
        # relabels it (``generate`` reuses this engine); the id is minted
        # here, before we connect, and exactly one event is reported for a
        # failure no matter which of the ~10 raise sites below produced it.
        op = self._client._diagnostics.operation(_operation, TRANSPORT_WEBSOCKET)
        try:
            async for item in self._stream_frames(
                op,
                text=text,
                model_id=model_id,
                voice_id=voice_id,
                cfg_scale=cfg_scale,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                sample_rate=sample_rate,
                output_format=output_format,
                reuse_connection=reuse_connection,
                normalize=normalize,
                language=language,
                word_timestamps=word_timestamps,
                speed=speed,
                dictionary_ids=dictionary_ids,
                project_id=project_id,
            ):
                if isinstance(item, AudioChunk):
                    op.record_chunk(len(item.audio))
                yield item
        except BaseException as e:
            op.fail(e)
            raise
        else:
            op.succeed()

    async def _stream_frames(
        self,
        op: Operation,
        *,
        text: str,
        model_id: str = "kugel-3",
        voice_id: Optional[int] = None,
        cfg_scale: float = 2.0,
        temperature: Optional[float] = None,
        max_new_tokens: int = 2048,
        sample_rate: int = 24000,
        output_format: Optional[str] = None,
        reuse_connection: bool = True,
        normalize: bool = True,
        language: Optional[str] = None,
        word_timestamps: bool = False,
        speed: float = 1.0,
        dictionary_ids: Optional[List[int]] = None,
        project_id: Optional[int] = None,
    ) -> AsyncIterator[Union[AudioChunk, List[WordTimestamp], Dict[str, Any]]]:
        """The WebSocket engine behind :meth:`stream_async`.

        Kept separate so the diagnostics wrapper owns exactly one try/except
        around the whole operation instead of instrumenting every raise site.
        """
        try:
            # RISK: websockets 15 doesn't expose ``exceptions`` as a package
            # attribute; import the exception from its public submodule.
            import websockets
            from websockets.exceptions import ConnectionClosed
        except ImportError:
            raise ImportError(
                "websockets is required for streaming. Install with: pip install websockets"
            )

        cfg_scale = clamp_cfg_scale(cfg_scale)
        _warn_if_no_language(language, normalize)

        request_data = {
            "text": text,
            "model_id": model_id,
            "cfg_scale": cfg_scale,
            "max_new_tokens": max_new_tokens,
            "sample_rate": sample_rate,
            "normalize": normalize,
            "word_timestamps": word_timestamps,
            "speed": speed,
        }
        if voice_id is not None:
            request_data["voice_id"] = voice_id
        if output_format is not None:
            request_data["output_format"] = output_format
        if language is not None:
            request_data["language"] = language
        if temperature is not None:
            request_data["temperature"] = temperature
        if project_id is not None:
            request_data["project_id"] = project_id
        # [] is meaningful (explicit opt-out) and must be sent; only None
        # (use the project default) is omitted.
        if dictionary_ids is not None:
            request_data["dictionary_ids"] = dictionary_ids

        if reuse_connection:
            # Use connection pooling for faster TTFA
            op.set_stage(STAGE_CONNECTING)
            ws = await self._get_ws_connection(model_id, op=op)
            try:
                op.set_stage(STAGE_SENDING_REQUEST)
                await ws.send(json.dumps(request_data))
                op.set_stage(STAGE_AWAITING_FIRST_AUDIO)

                while True:
                    try:
                        msg = await ws.recv()
                        data = json.loads(msg)

                        if data.get("error"):
                            raise _server_frame_error(op, data)

                        if data.get("final"):
                            yield data
                            break

                        if data.get("audio"):
                            yield AudioChunk.from_dict(data)

                        if "word_timestamps" in data:
                            yield [
                                WordTimestamp.from_dict(w)
                                for w in data["word_timestamps"]
                            ]

                    except ConnectionClosed as e:
                        # Connection was closed, clear pool and retry once
                        await self._close_ws_connection()
                        raise classify_ws_close(
                            getattr(e, "code", None), getattr(e, "reason", None)
                        ) from e

            except ConnectionClosed as e:
                # Connection died outside the recv loop (for example during
                # send); clear the pool and surface the server close code.
                await self._close_ws_connection()
                raise classify_ws_close(
                    getattr(e, "code", None), getattr(e, "reason", None)
                ) from e

        else:
            # Original behavior: new connection per request
            ws_url = self._client._tts_url.replace("https://", "wss://").replace(
                "http://", "ws://"
            )
            ws_url = f"{ws_url}/ws/tts?api_key={self._client._api_key}"
            if model_id:
                ws_url += f"&model_id={model_id}"
            ws_url += f"{_sdk_query_separator(ws_url)}{sdk_query_string()}"

            op.set_stage(STAGE_HANDSHAKE)
            try:
                async with websockets.connect(ws_url) as ws:
                    op.set_stage(STAGE_SENDING_REQUEST)
                    await ws.send(json.dumps(request_data))
                    op.set_stage(STAGE_AWAITING_FIRST_AUDIO)

                    while True:
                        try:
                            msg = await ws.recv()
                            data = json.loads(msg)

                            if data.get("error"):
                                raise _server_frame_error(op, data)

                            if data.get("final"):
                                yield data
                                break

                            if data.get("audio"):
                                yield AudioChunk.from_dict(data)

                            if "word_timestamps" in data:
                                yield [
                                    WordTimestamp.from_dict(w)
                                    for w in data["word_timestamps"]
                                ]

                        except ConnectionClosed as e:
                            raise classify_ws_close(
                                getattr(e, "code", None),
                                getattr(e, "reason", None),
                            ) from e

            except ws_handshake_error_types(websockets) as e:
                typed = classify_ws_handshake_error(e)
                if typed is not None:
                    raise typed from e
                raise KugelAudioConnectionError(
                    f"KugelAudio WebSocket handshake failed: {e}."
                ) from e

    def streaming_session(
        self,
        voice_id: Optional[int] = None,
        model_id: Optional[str] = None,
        cfg_scale: float = 2.0,
        temperature: Optional[float] = None,
        max_new_tokens: int = 2048,
        sample_rate: int = 24000,
        flush_timeout_ms: int = 500,
        normalize: bool = True,
        language: Optional[str] = None,
        word_timestamps: bool = False,
        speed: float = 1.0,
        dictionary_ids: Optional[List[int]] = None,
        on_word_timestamps: Optional[Callable[[List[WordTimestamp]], None]] = None,
        project_id: Optional[int] = None,
    ) -> StreamingSession:
        """Create a streaming session for text-in/audio-out streaming.

        Use this when streaming text from an LLM and want audio as soon
        as complete sentences are available.

        Example:
            async with client.tts.streaming_session(voice_id=123) as session:
                async for token in llm_stream:
                    async for chunk in session.send(token):
                        play_audio(chunk)

                async for chunk in session.flush():
                    play_audio(chunk)

        Example with word timestamps:
            async with client.tts.streaming_session(
                voice_id=123, word_timestamps=True,
            ) as session:
                async for chunk in session.send("Hello world."):
                    play_audio(chunk)
                async for chunk in session.flush():
                    play_audio(chunk)

                for ts in session.last_word_timestamps:
                    print(f"{ts.word}: {ts.start_ms}-{ts.end_ms}ms")

        Args:
            voice_id: Voice ID to use
            model_id: Model to use (e.g. "kugel-3")
            cfg_scale: Classifier-free guidance scale. Clamped to [1.2, 2.5].
            max_new_tokens: Maximum tokens per generation
            sample_rate: Output sample rate
            flush_timeout_ms: Auto-flush timeout in milliseconds
            normalize: Enable text normalization
            language: ISO 639-1 language code for normalization
            word_timestamps: Request per-chunk word-level timestamps
            speed: Playback speed multiplier (0.8–1.2, default 1.0).
                Uses pitch-preserving time-stretching.
            dictionary_ids: Per-request dictionary selection. Omit (None) to
                apply all active dictionaries of the project; pass [] to
                disable dictionaries for this request; pass a list of
                dictionary IDs to apply exactly those dictionaries (even
                inactive ones), bypassing the language filter.
            project_id: Project whose pronunciation dictionaries apply at
                synthesis. Required for any dictionary to apply and for a
                non-empty ``dictionary_ids``; omit (None) to send no project.
            on_word_timestamps: Callback ``(stamps: list[WordTimestamp]) -> None``
                invoked each time the server sends word timestamps.

        Returns:
            StreamingSession for async use
        """
        from kugelaudio.models import StreamConfig
        from kugelaudio.streaming import StreamingSession

        config = StreamConfig(
            voice_id=voice_id,
            model_id=model_id,
            cfg_scale=cfg_scale,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            sample_rate=sample_rate,
            flush_timeout_ms=flush_timeout_ms,
            normalize=normalize,
            language=language,
            word_timestamps=word_timestamps,
            speed=speed,
            dictionary_ids=dictionary_ids,
            project_id=project_id,
        )

        return StreamingSession(
            api_key=self._client._api_key,
            tts_url=self._client._tts_url,
            config=config,
            on_word_timestamps=on_word_timestamps,
            diagnostics=self._client._diagnostics,
        )

    def streaming_session_sync(
        self,
        voice_id: Optional[int] = None,
        model_id: Optional[str] = None,
        cfg_scale: float = 2.0,
        temperature: Optional[float] = None,
        max_new_tokens: int = 2048,
        sample_rate: int = 24000,
        flush_timeout_ms: int = 500,
        normalize: bool = True,
        language: Optional[str] = None,
        word_timestamps: bool = False,
        speed: float = 1.0,
        dictionary_ids: Optional[List[int]] = None,
        on_word_timestamps: Optional[Callable[[List[WordTimestamp]], None]] = None,
        project_id: Optional[int] = None,
    ) -> StreamingSessionSync:
        """Create a synchronous streaming session.

        Example:
            with client.tts.streaming_session_sync(voice_id=123) as session:
                for token in llm_stream:
                    for chunk in session.send(token):
                        play_audio(chunk)

                for chunk in session.flush():
                    play_audio(chunk)

        Args:
            voice_id: Voice ID to use
            model_id: Model to use (e.g. "kugel-3")
            cfg_scale: Classifier-free guidance scale. Clamped to [1.2, 2.5].
            max_new_tokens: Maximum tokens per generation
            sample_rate: Output sample rate
            flush_timeout_ms: Auto-flush timeout in milliseconds
            normalize: Enable text normalization
            language: ISO 639-1 language code for normalization
            word_timestamps: Request per-chunk word-level timestamps
            speed: Playback speed multiplier (0.8–1.2, default 1.0).
                Uses pitch-preserving time-stretching.
            dictionary_ids: Per-request dictionary selection. Omit (None) to
                apply all active dictionaries of the project; pass [] to
                disable dictionaries for this request; pass a list of
                dictionary IDs to apply exactly those dictionaries (even
                inactive ones), bypassing the language filter.
            project_id: Project whose pronunciation dictionaries apply at
                synthesis. Required for any dictionary to apply and for a
                non-empty ``dictionary_ids``; omit (None) to send no project.
            on_word_timestamps: Callback ``(stamps: list[WordTimestamp]) -> None``
                invoked each time the server sends word timestamps.

        Returns:
            StreamingSessionSync for sync use
        """
        from kugelaudio.streaming import StreamingSessionSync

        async_session = self.streaming_session(
            voice_id=voice_id,
            model_id=model_id,
            cfg_scale=cfg_scale,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            sample_rate=sample_rate,
            flush_timeout_ms=flush_timeout_ms,
            normalize=normalize,
            language=language,
            word_timestamps=word_timestamps,
            speed=speed,
            dictionary_ids=dictionary_ids,
            project_id=project_id,
            on_word_timestamps=on_word_timestamps,
        )

        return StreamingSessionSync(async_session)

    def multi_context_session(
        self,
        default_voice_id: Optional[int] = None,
        model_id: Optional[str] = None,
        sample_rate: int = 24000,
        cfg_scale: float = 2.0,
        temperature: Optional[float] = None,
        max_new_tokens: int = 2048,
        normalize: bool = True,
        language: Optional[str] = None,
        dictionary_ids: Optional[List[int]] = None,
        inactivity_timeout: float = 20.0,
        word_timestamps: bool = False,
        on_word_timestamps: Optional[
            Callable[[str, List[WordTimestamp]], None]
        ] = None,
        project_id: Optional[int] = None,
    ) -> MultiContextSession:
        """Create a multi-context session for concurrent TTS streams.

        Allows managing up to 20 independent audio generation contexts
        over a single WebSocket connection. Each context has its own
        text buffer, voice settings, and generation queue.

        Use cases:
        - Multi-speaker conversations with different voices
        - Pre-buffering audio while another stream plays
        - Interleaved audio generation for dynamic conversations

        Args:
            default_voice_id: Default voice ID for new contexts
            model_id: Model to use (e.g. "kugel-3")
            sample_rate: Output sample rate (default 24000)
            cfg_scale: Classifier-free guidance scale. Clamped to [1.2, 2.5] (default 2.0).
            max_new_tokens: Maximum tokens to generate (default 2048)
            normalize: Enable text normalization (default True)
            language: ISO 639-1 language code for normalization (e.g., 'de', 'en').
                If not set, the server uses the voice's primary language,
                falling back to English.
            dictionary_ids: Per-request dictionary selection. Omit (None) to
                apply all active dictionaries of the project; pass [] to
                disable dictionaries for this request; pass a list of
                dictionary IDs to apply exactly those dictionaries (even
                inactive ones), bypassing the language filter.
            project_id: Project whose pronunciation dictionaries apply at
                synthesis. Required for any dictionary to apply and for a
                non-empty ``dictionary_ids``; omit (None) to send no project.
            inactivity_timeout: Seconds before context auto-closes (default 20.0)
            word_timestamps: Request per-chunk word-level timestamps.
            on_word_timestamps: Callback
                ``(context_id: str, stamps: list[WordTimestamp]) -> None``
                invoked each time the server sends word timestamps.

        Returns:
            MultiContextSession for async use

        Example:
            async with client.tts.multi_context_session(language="en") as session:
                # Create contexts with different voices
                await session.create_context("narrator", voice_id=123)
                await session.create_context("character", voice_id=456)

                # Send text to different speakers
                async for chunk in session.send("narrator", "The story begins."):
                    play_audio("narrator", chunk)

                async for chunk in session.send("character", "Hello!"):
                    play_audio("character", chunk)
        """
        from kugelaudio.streaming import MultiContextSession

        cfg_scale = clamp_cfg_scale(cfg_scale)

        return MultiContextSession(
            api_key=self._client._api_key,
            tts_url=self._client._tts_url,
            default_voice_id=default_voice_id,
            model_id=model_id,
            sample_rate=sample_rate,
            cfg_scale=cfg_scale,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            normalize=normalize,
            language=language,
            dictionary_ids=dictionary_ids,
            project_id=project_id,
            inactivity_timeout=inactivity_timeout,
            word_timestamps=word_timestamps,
            on_word_timestamps=on_word_timestamps,
            diagnostics=self._client._diagnostics,
        )

    async def generate_async(
        self,
        text: str,
        model_id: str = "kugel-3",
        voice_id: Optional[int] = None,
        cfg_scale: float = 2.0,
        temperature: Optional[float] = None,
        max_new_tokens: int = 2048,
        sample_rate: int = 24000,
        output_format: Optional[str] = None,
        normalize: bool = True,
        language: Optional[str] = None,
        word_timestamps: bool = False,
        speed: float = 1.0,
        dictionary_ids: Optional[List[int]] = None,
        project_id: Optional[int] = None,
    ) -> AudioResponse:
        """Generate audio asynchronously.

        Args:
            text: Text to synthesize
            model_id: Model to use (e.g. 'kugel-3')
            voice_id: Voice ID to use
            cfg_scale: Classifier-free guidance scale. Clamped to [1.2, 2.5].
            max_new_tokens: Maximum tokens to generate
            sample_rate: Output sample rate
            output_format: Optional combined codec+rate token (e.g. 'ulaw_8000',
                'alaw_8000', 'pcm_8000'). Opt-in; absent ⇒ PCM16 at sample_rate.
            normalize: Enable text normalization (default: True).
                Set ``language`` when the text is not in the voice's primary
                language; the language is not detected from the text.
            language: ISO 639-1 language code for normalization (e.g., 'de', 'en').
                Supported: de, en, fr, es, it, pt, nl, pl, sv, da, no, fi, cs, hu, ro,
                el, uk, bg, tr, vi, ar, hi, zh, ja, ko, sk, sl, hr, sr, ru, he, fa,
                ur, bn, ta, yue, th, id, ms
            word_timestamps: Request word-level timestamps. When enabled,
                ``AudioResponse.word_timestamps`` will be populated with
                per-word timing boundaries.
            speed: Playback speed multiplier (0.8–1.2, default 1.0).
                Uses pitch-preserving time-stretching.
            dictionary_ids: Per-request dictionary selection. Omit (None) to
                apply all active dictionaries of the project; pass [] to
                disable dictionaries for this request; pass a list of
                dictionary IDs to apply exactly those dictionaries (even
                inactive ones), bypassing the language filter.
            project_id: Project whose pronunciation dictionaries apply at
                synthesis. Required for any dictionary to apply and for a
                non-empty ``dictionary_ids``; omit (None) to send no project.

        Returns:
            Complete audio response (with ``word_timestamps`` if requested)
        """
        chunks: List[AudioChunk] = []
        all_stamps: List[WordTimestamp] = []
        final_stats: Dict[str, Any] = {}

        async for item in self.stream_async(
            text=text,
            model_id=model_id,
            voice_id=voice_id,
            cfg_scale=cfg_scale,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            sample_rate=sample_rate,
            output_format=output_format,
            normalize=normalize,
            language=language,
            word_timestamps=word_timestamps,
            speed=speed,
            dictionary_ids=dictionary_ids,
            project_id=project_id,
            _operation=OP_GENERATE,
        ):
            if isinstance(item, AudioChunk):
                chunks.append(item)
            elif isinstance(item, list):
                all_stamps.extend(item)
            elif isinstance(item, dict) and item.get("final"):
                final_stats = item

        return AudioResponse.from_chunks(chunks, final_stats, all_stamps)


class KugelAudio:
    """KugelAudio API client.

    Example:
        client = KugelAudio(api_key="your_api_key")

        # List models
        models = client.models.list()

        # List voices
        voices = client.voices.list().voices

        # Generate audio
        audio = client.tts.generate(
            text="Hello, world!",
            model_id="kugel-3",
        )
        audio.save("output.wav")
    """

    def __init__(
        self,
        api_key: str,
        api_url: Optional[str] = None,
        tts_url: Optional[str] = None,
        timeout: float = 60.0,
        keepalive_ping_interval: Optional[float] = 20.0,
        region: Optional[str] = None,
        telemetry: Optional[bool] = None,
        *,
        _integration: str = INTEGRATION_NONE,
    ):
        """Initialize KugelAudio client.

        Args:
            api_key: Your KugelAudio API key. Can be prefixed with ``eu-`` to
                select the direct EU endpoint (the prefix is stripped before
                authenticating).
            api_url: API base URL. When set, takes precedence over *region* and any
                key prefix. (default: ``https://api.kugelaudio.com``)
            tts_url: TTS server URL (default: same as api_url, the backend proxies WebSocket)
            timeout: Request timeout in seconds
            keepalive_ping_interval: Seconds between WebSocket ping frames sent on the
                pooled connection to prevent idle timeouts (default: 20.0). Set to None
                to disable keepalive pings.
            region: Deployment region. Use ``'eu'`` to select the direct EU
                endpoint. Takes precedence over an API-key prefix but not over
                an explicit *api_url*. If omitted, the client uses the default
                geo-routed API endpoint.
            telemetry: Opt in or out of anonymous client-error diagnostics.
                ``None`` (default) enables them only on the hosted
                ``*.kugelaudio.com`` API; a custom or on-premise *api_url* is
                off by default. The ``KUGELAUDIO_TELEMETRY`` environment
                variable (``1/true/on/yes`` or ``0/false/off/no``) overrides
                this argument. Reports carry error class, failure stage,
                timings and the server request id only — never text, audio,
                URLs, keys or exception messages.
            _integration: Internal. Set by the LiveKit/Pipecat plugins when
                they build a client, stamped as ``kugel.integration``.

        For fastest performance in async code, use the factory method:
            client = await KugelAudio.create(api_key="...")

        This pre-establishes the WebSocket connection so your first TTS request
        is fast (~100ms instead of ~600ms).
        """
        if not api_key:
            raise ValidationError(
                "KugelAudio API key is missing. Pass api_key=... to the "
                "client. Get a key at https://app.kugelaudio.com/settings/api-keys."
            )

        clean_key, detected_region = _parse_api_key(api_key)
        self._api_key = clean_key

        if api_url:
            self._api_url = api_url.rstrip("/")
        else:
            effective_region = region or detected_region
            self._api_url = _resolve_region_url(effective_region)

        # If tts_url not specified, use api_url (backend proxies to TTS server)
        self._tts_url = (tts_url or self._api_url).rstrip("/")
        self._keepalive_ping_interval = keepalive_ping_interval
        self._timeout = timeout

        self._diagnostics = _build_diagnostics(
            self._api_url, clean_key, telemetry, _integration
        )

        # Initialize resources
        self.models = ModelsResource(self)
        self.voices = VoicesResource(self)
        self.dictionaries = DictionariesResource(self)
        self.asr = ASRResource(self)
        self.tts = TTSResource(self)

        self._http_client = httpx.Client(
            timeout=timeout,
            headers={**_auth_headers(clean_key), **sdk_headers()},
        )

        # Note on auto_connect:
        # - For ASYNC usage: Use `await KugelAudio.create()` to get a pre-connected client
        # - For SYNC usage: Connection pooling doesn't work across calls because each
        #   sync `stream()` call creates its own event loop. The sync API is simpler
        #   but has ~500ms cold start per call. For best sync performance, use the
        #   async API with asyncio.run() or switch to `streaming_session_sync()` which
        #   maintains a persistent connection.

    @classmethod
    async def create(
        cls,
        api_key: str,
        api_url: Optional[str] = None,
        tts_url: Optional[str] = None,
        timeout: float = 60.0,
        model: str = "kugel-3",
        keepalive_ping_interval: Optional[float] = 20.0,
        region: Optional[str] = None,
        telemetry: Optional[bool] = None,
    ) -> "KugelAudio":
        """Async factory to create a pre-connected KugelAudio client.

        Use this in async code to get a client that's already connected
        and ready for fast TTS requests.

        Args:
            api_key: Your KugelAudio API key
            api_url: API base URL (default: https://api.kugelaudio.com)
            tts_url: TTS server URL (default: same as api_url)
            timeout: Request timeout in seconds
            model: Model to pre-connect for (e.g. 'kugel-3')
            keepalive_ping_interval: Seconds between WebSocket ping frames (default: 20.0).
            region: Deployment region. Use 'eu' for the direct EU endpoint.

        Returns:
            Pre-connected KugelAudio client

        Example:
            async def main():
                # Client is ready immediately - no cold start on first request
                client = await KugelAudio.create(api_key="...")

                # First request is fast (~100ms)
                async for chunk in client.tts.stream_async("Hello"):
                    ...
        """
        client = cls(
            api_key=api_key,
            api_url=api_url,
            tts_url=tts_url,
            timeout=timeout,
            keepalive_ping_interval=keepalive_ping_interval,
            region=region,
            telemetry=telemetry,
        )
        await client.connect_async(model=model)
        return client

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request to API.

        Args:
            method: HTTP method
            path: API path
            params: Query parameters
            json_data: JSON body

        Returns:
            Response data

        Raises:
            KugelAudioError: On API error
        """
        op = self._diagnostics.operation(
            _operation_for_path(path), TRANSPORT_HTTP
        )
        with op.step(STAGE_SENDING_REQUEST):
            result = self._request_inner(method, path, params, json_data)
        op.succeed()
        return result

    def _request_inner(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = urljoin(self._api_url + "/", path.lstrip("/"))

        try:
            response = self._http_client.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
            )

            if response.status_code >= 400:
                raise classify_http_response(response)

            if response.status_code == 204 or not response.content:
                return {}
            return response.json()

        except httpx.TimeoutException as e:
            raise KugelAudioConnectionError(
                f"Request to {method} {path} timed out after {self._timeout}s."
            ) from e
        except httpx.RequestError as e:
            raise KugelAudioConnectionError(
                f"Could not reach KugelAudio at {url}: {e}. "
                "Check network connectivity."
            ) from e

    def _request_multipart(
        self,
        method: str,
        path: str,
        files: List[Tuple[str, Tuple[Optional[str], Any, str]]],
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make multipart/form-data HTTP request to API.

        Args:
            method: HTTP method
            path: API path
            files: List of (field, (filename, file_obj, content_type)) tuples
            params: Query parameters

        Returns:
            Response data
        """
        op = self._diagnostics.operation(
            _operation_for_path(path), TRANSPORT_HTTP
        )
        with op.step(STAGE_SENDING_REQUEST):
            result = self._request_multipart_inner(method, path, files, params)
        op.succeed()
        return result

    def _request_multipart_inner(
        self,
        method: str,
        path: str,
        files: List[Tuple[str, Tuple[Optional[str], Any, str]]],
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = urljoin(self._api_url + "/", path.lstrip("/"))

        try:
            response = self._http_client.request(
                method=method,
                url=url,
                params=params,
                files=files,
            )

            if response.status_code >= 400:
                raise classify_http_response(response)

            return response.json()

        except httpx.TimeoutException as e:
            raise KugelAudioConnectionError(
                f"Request to {method} {path} timed out after {self._timeout}s."
            ) from e
        except httpx.RequestError as e:
            raise KugelAudioConnectionError(
                f"Could not reach KugelAudio at {url}: {e}. "
                "Check network connectivity."
            ) from e

    def close(self) -> None:
        """Close the client and release resources."""
        self._http_client.close()
        self._diagnostics.close()
        if self.tts._ws_connection is not None:
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop is not None and loop.is_running():
                    loop.create_task(self.tts._close_ws_connection())
                else:
                    _loop = asyncio.new_event_loop()
                    try:
                        _loop.run_until_complete(self.tts._close_ws_connection())
                    finally:
                        _loop.close()
            except Exception:
                pass

    async def aclose(self) -> None:
        """Close the client asynchronously."""
        self._http_client.close()
        if hasattr(self.tts, "_close_ws_connection"):
            await self.tts._close_ws_connection()
        # The reporter's bounded (<=1 s) wait for its delivery thread runs off
        # the event loop, so closing never stalls other coroutines.
        await asyncio.to_thread(self._diagnostics.close)

    def connect(self, model: str = "kugel-3") -> None:
        """Pre-establish WebSocket connection for faster first request.

        Call this at application startup to eliminate cold start latency
        (~300-600ms) from your first TTS request.

        Args:
            model: Model to connect for (e.g. 'kugel-3').
                   The connection is model-specific due to routing.

        Example:
            client = KugelAudio(api_key="...")

            # Pre-connect at startup
            client.connect()

            # First request is now fast (~100ms instead of ~500ms)
            for chunk in client.tts.stream("Hello"):
                ...
        """
        self.tts.connect(model)

    async def connect_async(self, model: str = "kugel-3") -> None:
        """Pre-establish WebSocket connection for faster first request (async).

        Call this at application startup to eliminate cold start latency
        (~300-600ms) from your first TTS request.

        Args:
            model: Model to connect for (e.g. 'kugel-3').
                   The connection is model-specific due to routing.

        Example:
            client = KugelAudio(api_key="...")

            # Pre-connect at startup
            await client.connect_async()

            # First request is now fast (~100ms instead of ~500ms)
            async for chunk in client.tts.stream_async("Hello"):
                ...
        """
        await self.tts.connect_async(model)

    def is_connected(self) -> bool:
        """Check if WebSocket connection is established and open.

        Returns:
            True if connected and ready for requests.
        """
        return self.tts.is_connected()

    def __enter__(self) -> KugelAudio:
        return self

    def __exit__(self, *args) -> None:
        self.close()
