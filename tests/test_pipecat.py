# Copyright 2024 KugelAudio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for kugelaudio.pipecat module.

These tests verify the Pipecat integration works correctly.
Requires: pip install kugelaudio[pipecat]
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

# Skip all tests if pipecat-ai is not installed
pytest.importorskip("pipecat")

from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame, ErrorFrame


class TestTTSInitialization:
    """Test TTS service initialization."""


    def test_init_with_env_var(self, monkeypatch):
        """Test initialization with environment variable."""
        from kugelaudio.pipecat import KugelAudioTTSService

        monkeypatch.setenv("KUGELAUDIO_API_KEY", "env-test-key")
        tts = KugelAudioTTSService(voice_id=1)
        assert tts._opts.api_key == "env-test-key"

    def test_init_without_api_key_raises(self, monkeypatch):
        """Test that missing API key raises ValueError."""
        from kugelaudio.pipecat import KugelAudioTTSService

        monkeypatch.delenv("KUGELAUDIO_API_KEY", raising=False)
        with pytest.raises((ValueError, TypeError)):
            KugelAudioTTSService()


    def test_init_strips_trailing_slash_from_base_url(self):
        """Test that trailing slash is stripped from base URL."""
        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(
            api_key="test-key", voice_id=1, base_url="https://custom.api.com/"
        )
        assert tts._opts.base_url == "https://custom.api.com"

    def test_init_with_unsupported_sample_rate_raises(self):
        """Test that unsupported sample rate raises ValueError."""
        from kugelaudio.pipecat import KugelAudioTTSService

        with pytest.raises(ValueError, match="Unsupported sample rate"):
            KugelAudioTTSService(api_key="test-key", voice_id=1, sample_rate=44100)


    def test_init_clamps_out_of_band_cfg_scale(self):
        """CFG scale outside [1.2, 2.5] is clamped before connecting."""
        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(api_key="test-key", voice_id=1, cfg_scale=3.5)
        assert tts._opts.cfg_scale == 2.5


class TestTTSRegion:
    """Test multi-region support."""


    def test_key_prefix_eu(self):
        """Test that 'eu-' key prefix auto-selects the direct EU endpoint."""
        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(api_key="eu-ka_test123", voice_id=1)
        assert tts._opts.base_url == "https://api.eu.kugelaudio.com"
        assert tts._opts.api_key == "ka_test123"

    def test_explicit_region_overrides_key_prefix(self):
        """Test that explicit region param takes priority over key prefix."""
        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(api_key="us-ka_test123", voice_id=1, region="global")
        assert tts._opts.base_url == "https://api.kugelaudio.com"
        assert tts._opts.api_key == "ka_test123"

    def test_base_url_overrides_region(self):
        """Test that explicit base_url overrides region entirely."""
        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(
            api_key="us-ka_test123", voice_id=1, region="global",
            base_url="https://custom.api.com",
        )
        assert tts._opts.base_url == "https://custom.api.com"
        assert tts._opts.api_key == "ka_test123"

    def test_invalid_region_raises(self):
        """Test that invalid region raises ValueError."""
        from kugelaudio.pipecat import KugelAudioTTSService

        with pytest.raises(ValueError, match="Invalid region"):
            KugelAudioTTSService(api_key="ka_test123", voice_id=1, region="mars")

    def test_env_var_with_prefix(self, monkeypatch):
        """Test that region prefix in env var is parsed correctly."""
        from kugelaudio.pipecat import KugelAudioTTSService

        monkeypatch.setenv("KUGELAUDIO_API_KEY", "us-ka_envkey")
        tts = KugelAudioTTSService(voice_id=1)
        assert tts._opts.base_url == "https://api.kugelaudio.com"
        assert tts._opts.api_key == "ka_envkey"

    def test_client_receives_clean_key(self):
        """Test that the internal KugelAudio client gets the stripped key."""
        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(api_key="us-ka_test123", voice_id=1)
        assert tts._client._api_key == "ka_test123"
        # Diagnostics: per-client integration label, same host as the audio.
        assert tts._client._diagnostics.integration == "pipecat"
        assert tts._client._api_url == tts._opts.base_url

    @pytest.mark.parametrize(
        ("env", "option", "enabled"),
        [(None, None, True), (None, False, False), ("1", False, True), ("0", True, False)],
    )
    def test_telemetry_option_follows_the_contract_precedence(
        self, monkeypatch, env, option, enabled
    ):
        """Env beats the option, the option beats the hosted default."""
        from kugelaudio.pipecat import KugelAudioTTSService

        monkeypatch.delenv("KUGELAUDIO_TELEMETRY", raising=False)
        if env is not None:
            monkeypatch.setenv("KUGELAUDIO_TELEMETRY", env)
        tts = KugelAudioTTSService(api_key="ka_k", voice_id=1, telemetry=option)
        assert tts._client._diagnostics.enabled is enabled


class TestTTSOptions:
    """Test TTS option updates."""

    @pytest.mark.asyncio
    async def test_set_voice(self):
        """Test updating voice via set_voice."""
        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(api_key="test-key", voice_id=1)
        await tts.set_voice("456")
        assert tts._opts.voice_id == 456

    @pytest.mark.asyncio
    async def test_set_voice_empty_string(self):
        """Test setting voice to empty string sets None."""
        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(api_key="test-key", voice_id=123)
        await tts.set_voice("")
        assert tts._opts.voice_id is None

    @pytest.mark.asyncio
    async def test_set_voice_closes_cached_session_before_return(self):
        """Voice changes must not return while the old persistent session is cached."""
        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(api_key="test-key", voice_id=1)
        old_session = AsyncMock()
        old_session.close = AsyncMock(return_value={"session_closed": True})
        tts._multi_session = old_session

        await tts.set_voice("456")

        assert tts._opts.voice_id == 456
        assert tts._multi_session is None
        old_session.close.assert_awaited_once()

    def test_init_without_voice_id_raises(self):
        """Test that missing voice_id raises TypeError."""
        from kugelaudio.pipecat import KugelAudioTTSService

        with pytest.raises(TypeError):
            KugelAudioTTSService(api_key="test-key")


class TestTTSConnectionManagement:
    """Test connection management via delegated KugelAudio client."""


    @pytest.mark.asyncio
    async def test_cleanup_delegates_to_pipecat(self):
        """The base service owns frame-processing tasks that must be stopped."""
        from pipecat.services.tts_service import TTSService

        from kugelaudio.pipecat import KugelAudioTTSService

        parent_cleanup = AsyncMock()
        with patch.object(TTSService, "cleanup", parent_cleanup):
            tts = KugelAudioTTSService(api_key="test-key", voice_id=1)
            await tts.cleanup()

        parent_cleanup.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_cleanup_cancels_a_slow_prewarm(self):
        """A connection handshake cannot outlive the service that started it."""
        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(api_key="test-key", voice_id=1)
        started = asyncio.Event()

        async def slow_connect():
            started.set()
            await asyncio.Event().wait()

        tts._get_multi_session = slow_connect
        tts.prewarm()
        prewarm = tts._prewarm_task
        assert prewarm is not None
        await started.wait()

        await tts.cleanup()

        assert prewarm.cancelled()
        assert tts._prewarm_task is None

    @pytest.mark.asyncio
    async def test_set_model_updates_opts(self):
        """Test that changing model updates the model in options."""
        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(api_key="test-key", voice_id=1, model="kugel-1-turbo")
        await tts.set_model("kugel-1")
        assert tts._opts.model == "kugel-1"


class TestTTSStreamDelegation:
    """Test that run_tts correctly delegates to the multi-context session."""

    class FakeMultiContextSession:
        def __init__(self, chunks=None, error=None):
            self.chunks = chunks or []
            self.error = error
            self.connected = False
            self.closed = False
            self.calls = []
            # Mirror the real MultiContextSession.is_alive contract — the
            # pipecat wrapper checks this before reusing a cached session.
            self.is_alive = True

        async def connect(self):
            self.connected = True
            self.is_alive = True

        async def close(self):
            self.closed = True
            self.is_alive = False
            return {"session_closed": True}

        async def send(
            self,
            context_id,
            text,
            flush=False,
            chunk_complete_idle_timeout=None,
        ):
            self.calls.append(
                {
                    "context_id": context_id,
                    "text": text,
                    "flush": flush,
                    "chunk_complete_idle_timeout": chunk_complete_idle_timeout,
                }
            )
            if self.error:
                raise self.error
            for chunk in self.chunks:
                yield chunk

    @pytest.mark.asyncio
    async def test_run_tts_waits_for_every_sentence_in_the_clause(self):
        """A server sentence boundary cannot truncate one Pipecat clause."""
        from kugelaudio.pipecat import KugelAudioTTSService
        from kugelaudio.models import AudioChunk

        tts = KugelAudioTTSService(
            api_key="test-key",
            model="kugel-1",
            voice_id=42,
            cfg_scale=2.4,
            max_new_tokens=1024,
            language="de",
            normalize=False,
        )

        mock_chunk = AudioChunk(audio=b"\x00\x00", index=0, sample_rate=24000, samples=1)
        fake_session = self.FakeMultiContextSession([mock_chunk])
        captured_kwargs = {}

        def mock_multi_context_session(**kwargs):
            captured_kwargs.update(kwargs)
            return fake_session

        tts._client.tts.multi_context_session = mock_multi_context_session

        frames = []
        async for frame in tts.run_tts("Test text", context_id="ctx-test"):
            frames.append(frame)

        assert fake_session.connected is True
        assert fake_session.calls == [
            {
                "context_id": "ctx-test",
                "text": "Test text",
                "flush": True,
                "chunk_complete_idle_timeout": None,
            }
        ]
        assert captured_kwargs == {
            "default_voice_id": 42,
            "model_id": "kugel-1",
            "sample_rate": 24000,
            "cfg_scale": 2.4,
            "max_new_tokens": 1024,
            "normalize": False,
            "language": "de",
            "word_timestamps": False,
            "on_word_timestamps": None,
        }

        # Verify we got the expected frame types
        frame_types = [type(f).__name__ for f in frames]
        assert "TTSStartedFrame" in frame_types
        assert "TTSAudioRawFrame" in frame_types
        assert "TTSStoppedFrame" in frame_types

    def test_word_timestamp_callback_enables_and_reaches_multi_context_session(self):
        """The optional Pipecat callback is passed through with context identity."""
        from kugelaudio.models import WordTimestamp
        from kugelaudio.pipecat import KugelAudioTTSService

        received: list[tuple[str, list[WordTimestamp]]] = []

        def on_word_timestamps(
            context_id: str, timestamps: list[WordTimestamp]
        ) -> None:
            received.append((context_id, timestamps))

        tts = KugelAudioTTSService(
            api_key="test-key",
            voice_id=42,
            on_word_timestamps=on_word_timestamps,
        )

        session = tts._create_multi_session()
        stamp = WordTimestamp(
            word="Hello",
            start_ms=0,
            end_ms=240,
            char_start=0,
            char_end=5,
        )
        assert session._word_timestamps is True
        assert session._on_word_timestamps is on_word_timestamps

        session._on_word_timestamps("ctx-test", [stamp])

        assert received == [("ctx-test", [stamp])]


    @pytest.mark.asyncio
    async def test_failed_multi_session_connect_is_not_retained(self):
        """Test failed prewarm/connect attempts do not poison the cached session."""
        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(api_key="test-key", voice_id=1)
        fake_session = self.FakeMultiContextSession()

        async def failing_connect():
            raise TimeoutError("connect failed")

        fake_session.connect = failing_connect
        tts._client.tts.multi_context_session = lambda **kwargs: fake_session

        frames = []
        async for frame in tts.run_tts("Hello", context_id="ctx-fail"):
            frames.append(frame)

        assert fake_session.closed is False
        assert tts._multi_session is None
        frame_types = [type(f).__name__ for f in frames]
        assert "ErrorFrame" in frame_types
        assert "TTSStoppedFrame" in frame_types

    @pytest.mark.asyncio
    async def test_run_tts_yields_audio_frames(self):
        """Test that audio chunks from the SDK become TTSAudioRawFrame."""
        from kugelaudio.pipecat import KugelAudioTTSService
        from kugelaudio.models import AudioChunk

        tts = KugelAudioTTSService(api_key="test-key", voice_id=1, language="en")

        audio_data = b"\x01\x02\x03\x04"
        mock_chunk = AudioChunk(audio=audio_data, index=0, sample_rate=24000, samples=2)
        tts._multi_session = self.FakeMultiContextSession([mock_chunk])

        audio_frames = []
        async for frame in tts.run_tts("Hello"):
            if isinstance(frame, TTSAudioRawFrame):
                audio_frames.append(frame)

        assert len(audio_frames) == 1
        assert audio_frames[0].audio == audio_data
        assert audio_frames[0].sample_rate == 24000
        assert audio_frames[0].num_channels == 1


    @pytest.mark.asyncio
    async def test_run_tts_error_frame_preserves_sdk_error_code(self):
        """Typed SDK errors should keep their code in Pipecat's ErrorFrame."""
        from kugelaudio.exceptions import RateLimitError
        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(api_key="test-key", voice_id=1, language="en")
        tts._multi_session = self.FakeMultiContextSession(
            error=RateLimitError("Rate limit exceeded")
        )

        frames = []
        async for frame in tts.run_tts("Hello"):
            frames.append(frame)

        error_frames = [frame for frame in frames if isinstance(frame, ErrorFrame)]
        assert len(error_frames) == 1
        assert "RATE_LIMITED" in error_frames[0].error
        assert "status=429" in error_frames[0].error
        assert "Rate limit exceeded" in error_frames[0].error


    @pytest.mark.asyncio
    async def test_run_tts_with_language_set(self):
        """Test that explicit language is passed through correctly."""
        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(api_key="test-key", voice_id=1, language="de")

        captured_kwargs = {}

        def mock_multi_context_session(**kwargs):
            captured_kwargs.update(kwargs)
            return self.FakeMultiContextSession()

        tts._client.tts.multi_context_session = mock_multi_context_session

        async for _ in tts.run_tts("Hallo"):
            pass

        assert captured_kwargs["language"] == "de"


class TestTTSReconnectOnDeadWS:
    """Verify the pipecat wrapper reconnects when the cached session's WS dies.

    Mirrors the production failure mode: an idle WS gets dropped by an
    upstream proxy / NAT between conversation turns. Without these checks,
    every subsequent run_tts() crashes with
    ``ConnectionClosedError("no close frame received or sent")``.
    """

    class _RawConnectionClosedError(Exception):
        """Stand-in for websockets.exceptions.ConnectionClosedError."""

        def __init__(self) -> None:
            super().__init__("no close frame received or sent")

    @pytest.mark.asyncio
    async def test_dead_session_is_replaced_in_get_multi_session(self):
        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(
            api_key="test-key", model="kugel-1-turbo", voice_id=42,
        )

        class _DeadSession:
            is_alive = False

            async def close(self) -> dict:
                return {"session_closed": True}

        dead = _DeadSession()
        tts._multi_session = dead

        called = {"connect": 0}

        class _FreshSession:
            is_alive = True

            async def connect(self) -> None:
                called["connect"] += 1

        tts._create_multi_session = lambda: _FreshSession()

        async with tts._multi_session_lock:
            new_session = await tts._get_multi_session()

        assert new_session is not dead
        assert called["connect"] == 1
        assert isinstance(new_session, _FreshSession)

    @pytest.mark.asyncio
    async def test_run_tts_retries_once_on_connection_error(self):
        """A dead-WS error before any audio is yielded should trigger one
        transparent reconnect+retry, so the conversation stays alive."""
        from kugelaudio.pipecat import KugelAudioTTSService
        from kugelaudio.exceptions import ConnectionError as KugelAudioConnectionError
        from kugelaudio.models import AudioChunk
        from pipecat.frames.frames import (
            ErrorFrame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame,
        )

        tts = KugelAudioTTSService(
            api_key="test-key", model="kugel-1-turbo", voice_id=42,
        )

        attempt_log: list[str] = []
        good_chunk = AudioChunk(
            audio=b"\x00\x01" * 100, sample_rate=24000, samples=100, index=0,
        )

        class _OneShotFailSession:
            is_alive = True
            connected = 0

            async def connect(self) -> None:
                self.connected += 1

            async def close(self) -> dict:
                return {"session_closed": True}

            async def send(self, ctx, text, flush=False, chunk_complete_idle_timeout=None):
                attempt_log.append(text)
                if len(attempt_log) == 1:
                    raise KugelAudioConnectionError("WS dropped")
                yield good_chunk
                return

        sessions: list[_OneShotFailSession] = []

        def _make() -> _OneShotFailSession:
            s = _OneShotFailSession()
            sessions.append(s)
            return s

        tts._create_multi_session = _make

        frames = [f async for f in tts.run_tts("Hello.", "ctx-1")]

        # Frame sequence: Started, then audio (after retry), then Stopped.
        # No ErrorFrame because the retry succeeded.
        kinds = [type(f).__name__ for f in frames]
        assert "TTSStartedFrame" in kinds
        assert "TTSAudioRawFrame" in kinds
        assert "TTSStoppedFrame" in kinds
        assert not any(isinstance(f, ErrorFrame) for f in frames)
        assert len(attempt_log) == 2  # one fail, one success
        # Second session was created (the first one was discarded as dead)
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_run_tts_retries_once_on_raw_connection_closed_error(self):
        """A bare websockets ConnectionClosedError should use the same
        reset+retry path as the SDK's typed connection error."""
        from kugelaudio.pipecat import KugelAudioTTSService
        from kugelaudio.models import AudioChunk
        from pipecat.frames.frames import (
            ErrorFrame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame,
        )

        tts = KugelAudioTTSService(
            api_key="test-key", model="kugel-1-turbo", voice_id=42,
        )

        attempt_log: list[str] = []
        good_chunk = AudioChunk(
            audio=b"\x00\x01" * 100, sample_rate=24000, samples=100, index=0,
        )

        class _RawCloseThenSuccessSession:
            is_alive = True

            async def connect(self) -> None:
                return None

            async def close(self) -> dict:
                return {"session_closed": True}

            async def send(
                self, ctx, text, flush=False, chunk_complete_idle_timeout=None
            ):
                attempt_log.append(text)
                if len(attempt_log) == 1:
                    raise TestTTSReconnectOnDeadWS._RawConnectionClosedError()
                yield good_chunk
                return

        sessions: list[_RawCloseThenSuccessSession] = []

        def _make() -> _RawCloseThenSuccessSession:
            session = _RawCloseThenSuccessSession()
            sessions.append(session)
            return session

        tts._create_multi_session = _make

        frames = [f async for f in tts.run_tts("Hello.", "ctx-1")]

        kinds = [type(f).__name__ for f in frames]
        assert "TTSStartedFrame" in kinds
        assert any(isinstance(f, TTSAudioRawFrame) for f in frames)
        assert "TTSStoppedFrame" in kinds
        assert not any(isinstance(f, ErrorFrame) for f in frames)
        assert len(attempt_log) == 2
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_run_tts_gives_up_after_one_retry(self):
        """If the retry also raises a connection error, surface it loudly
        as an ErrorFrame — don't loop forever."""
        from kugelaudio.pipecat import KugelAudioTTSService
        from kugelaudio.exceptions import ConnectionError as KugelAudioConnectionError
        from pipecat.frames.frames import ErrorFrame

        tts = KugelAudioTTSService(
            api_key="test-key", model="kugel-1-turbo", voice_id=42,
        )

        class _AlwaysFailSession:
            is_alive = True

            async def connect(self) -> None: pass
            async def close(self) -> dict: return {}

            async def send(self, *a, **kw):
                raise KugelAudioConnectionError("permanently dead")
                yield  # pragma: no cover — make this a generator

        tts._create_multi_session = lambda: _AlwaysFailSession()

        frames = [f async for f in tts.run_tts("Hello.", "ctx-1")]
        error_frames = [f for f in frames if isinstance(f, ErrorFrame)]
        assert len(error_frames) == 1
        assert "permanently dead" in error_frames[0].error

    @pytest.mark.asyncio
    async def test_run_tts_marks_reset_when_raw_close_becomes_error_frame(self):
        """If the raw close cannot be retried, the next turn must not reuse
        the dead cached WebSocket."""
        from kugelaudio.pipecat import KugelAudioTTSService
        from pipecat.frames.frames import ErrorFrame

        tts = KugelAudioTTSService(
            api_key="test-key", model="kugel-1-turbo", voice_id=42,
        )

        class _AlwaysRawCloseSession:
            is_alive = True

            async def connect(self) -> None:
                return None

            async def close(self) -> dict:
                return {}

            async def send(self, *a, **kw):
                raise TestTTSReconnectOnDeadWS._RawConnectionClosedError()
                yield  # pragma: no cover — make this a generator

        tts._create_multi_session = lambda: _AlwaysRawCloseSession()

        frames = [f async for f in tts.run_tts("Hello.", "ctx-1")]
        error_frames = [f for f in frames if isinstance(f, ErrorFrame)]
        assert len(error_frames) == 1
        assert "no close frame received or sent" in error_frames[0].error
        assert tts._multi_session_needs_reset is True


class TestPipecat1xTTSSettingsCompat:
    """Pipecat 1.0 introduced TTSSettings + AIService.start().validate_complete(),
    which logs ERROR for any field still set to the NOT_GIVEN sentinel. The
    wrapper must construct an initialized TTSSettings(model, voice, language)
    so the user's startup logs stay clean. See:
    https://github.com/pipecat-ai/pipecat/blob/v1.1.0/src/pipecat/services/settings.py
    """

    @pytest.mark.asyncio
    async def test_super_init_receives_settings_when_pipecat_has_TTSSettings(self):
        """If pipecat.services.settings.TTSSettings exists (1.x), we must
        forward an initialized settings object so model/voice/language are
        not NOT_GIVEN."""
        from kugelaudio.pipecat import tts as tts_mod
        from kugelaudio.pipecat import KugelAudioTTSService

        captured = {}

        # Stand in for pipecat 1.x's TTSSettings dataclass — captures the
        # init kwargs the wrapper passes.
        class _FakeTTSSettings:
            def __init__(self, *, model, voice, language):
                captured["model"] = model
                captured["voice"] = voice
                captured["language"] = language
                # Expose as attributes too: pipecat 1.x's base reads
                # settings.model (e.g. _sync_model_name_to_metrics) during init.
                self.model = model
                self.voice = voice
                self.language = language

        with patch.object(tts_mod, "_PipecatTTSSettings", _FakeTTSSettings):
            tts = KugelAudioTTSService(
                api_key="test", model="kugel-1-turbo",
                voice_id=42, language="en",
                voice="42",  # user passed it both ways — must be tolerated
            )

        assert captured == {"model": "kugel-1-turbo", "voice": "42", "language": "en"}

    @pytest.mark.asyncio
    async def test_no_settings_kwarg_on_pipecat_0x(self):
        """When TTSSettings is unavailable (0.x), we must NOT pass a settings
        kwarg — the 0.x signature would error on it."""
        from kugelaudio.pipecat import tts as tts_mod

        # Snapshot the kwargs the wrapper passes to TTSService.__init__
        captured: dict = {}
        original_super_init = tts_mod.TTSService.__init__

        def _spy(self, **kwargs):
            captured.update(kwargs)
            return original_super_init(self, **kwargs)

        with patch.object(tts_mod, "_PipecatTTSSettings", None), \
             patch.object(tts_mod.TTSService, "__init__", _spy):
            tts_mod.KugelAudioTTSService(
                api_key="test", model="kugel-1-turbo",
                voice_id=42, language="en",
            )

        assert "settings" not in captured


class TestTTSReusesSessionContext:
    """Legacy Pipecat 0.x context lifecycle (run_tts(text), no context_id).

    Each LLM turn gets its OWN server-side context — pre-created on the LLM
    start frame and closed on the end frame (driven from process_frame via
    _legacy_begin_turn / _legacy_end_turn) — so a turn's KV cache is freed at
    the turn boundary and never bleeds into the next utterance, matching the
    modern per-turn path. When run_tts is driven directly with no surrounding
    LLM frames (a raw TextFrame / static say()) there is no turn boundary to
    act on, so the lazily-minted context is simply reused until the next start
    frame or cleanup — it can never accumulate against the server's cap.
    """

    class _RecordingSession:
        is_alive = True

        def __init__(self) -> None:
            self.sent_contexts: list[str] = []
            self.created_contexts: list[str] = []
            self.closed_contexts: list[str] = []
            self.connected = 0
            from kugelaudio.models import AudioChunk

            self._chunk = AudioChunk(
                audio=b"\x00\x01" * 50, sample_rate=24000, samples=50, index=0,
            )

        async def connect(self) -> None:
            self.connected += 1

        async def close(self) -> dict:
            return {"session_closed": True}

        async def create_context(self, context_id) -> None:
            self.created_contexts.append(context_id)

        async def close_context(self, context_id, immediate=False):
            self.closed_contexts.append(context_id)
            return
            yield  # make this an async generator

        async def send(self, ctx, text, flush=False, chunk_complete_idle_timeout=None):
            self.sent_contexts.append(ctx)
            yield self._chunk

    @pytest.mark.asyncio
    async def test_turn_context_created_on_start_closed_on_end(self):
        """The normal legacy turn lifecycle: the LLM start frame pre-creates the
        turn's context (before any run_tts), every sentence of the turn shares
        it, and the end frame closes it — freeing the KV at the turn boundary."""
        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(api_key="test-key", voice_id=42)
        session = self._RecordingSession()
        tts._create_multi_session = lambda: session

        await tts._legacy_begin_turn()
        ctx = tts._legacy_turn_context_id
        assert ctx is not None and session.created_contexts == [ctx]

        # Two sentences in one turn share the one pre-created context.
        async for _ in tts.run_tts("First sentence."):
            pass
        async for _ in tts.run_tts("Second sentence."):
            pass
        assert session.sent_contexts == [ctx, ctx]
        assert session.closed_contexts == []  # not closed mid-turn

        await tts._legacy_end_turn()
        assert session.closed_contexts == [ctx]
        assert tts._legacy_turn_context_id is None

    @pytest.mark.asyncio
    async def test_end_frame_synthesizes_final_fragment_before_closing(
        self, monkeypatch
    ):
        """The legacy base synthesizes a turn's final buffered sentence WHILE
        handling LLMFullResponseEndFrame (inside super().process_frame). The
        wrapper must close the turn context AFTER super() runs, so that final
        fragment is rendered on the turn's own context — not on a fresh one that
        would outlive the turn boundary. Regression guard for the
        close-before-synthesize ordering bug (the manual-driving tests above
        can't catch it because they never route a real end frame through
        process_frame)."""
        from kugelaudio.pipecat import KugelAudioTTSService
        from kugelaudio.pipecat import tts as tts_module
        from pipecat.frames.frames import LLMFullResponseEndFrame
        from pipecat.processors.frame_processor import FrameDirection
        from pipecat.services.tts_service import TTSService

        # Force the legacy 0.x branch regardless of the installed pipecat
        # version so this guard runs identically across the compat matrix.
        monkeypatch.setattr(
            tts_module, "_base_run_tts_takes_context_id", lambda: False
        )

        tts = KugelAudioTTSService(api_key="test-key", voice_id=42)
        session = self._RecordingSession()
        tts._create_multi_session = lambda: session

        order: list[str] = []

        async def fake_base_process_frame(self, frame, direction):
            # Mirror the legacy base: the turn's trailing fragment is rendered
            # here, on the end frame — after the wrapper's pre-hook, before its
            # post-hook.
            order.append("super")
            if isinstance(frame, LLMFullResponseEndFrame):
                async for _ in self.run_tts("Trailing fragment without a period"):
                    pass

        monkeypatch.setattr(TTSService, "process_frame", fake_base_process_frame)

        orig_end = tts._legacy_end_turn

        async def recording_end():
            order.append("end")
            await orig_end()

        tts._legacy_end_turn = recording_end

        await tts._legacy_begin_turn()
        ctx = tts._legacy_turn_context_id
        assert ctx is not None

        await tts.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

        # Synthesis (super) ran before the context was closed (end)...
        assert order == ["super", "end"]
        # ...so the final fragment used the turn's own context, not a fresh one,
        # and that single context is the one closed at the turn boundary.
        assert session.sent_contexts == [ctx]
        assert session.closed_contexts == [ctx]
        assert tts._legacy_turn_context_id is None

    @pytest.mark.asyncio
    async def test_unframed_run_tts_reuses_one_context(self):
        """Fallback: run_tts driven directly with no surrounding LLM frames has
        no turn boundary to act on, so the lazily-minted context is reused for
        every call and never closed mid-conversation (so it can't accumulate)."""
        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(api_key="test-key", voice_id=42)
        session = self._RecordingSession()
        tts._create_multi_session = lambda: session

        for _ in range(4):
            async for _f in tts.run_tts("One sentence."):
                pass

        assert len(session.sent_contexts) == 4
        assert len(set(session.sent_contexts)) == 1  # one reused context
        assert session.connected == 1
        assert session.closed_contexts == []

    @pytest.mark.asyncio
    async def test_caller_supplied_context_id_used_verbatim(self):
        """Pipecat 1.x supplies a context_id; honour it and don't hijack it
        with a wrapper-owned legacy turn context."""
        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(api_key="test-key", voice_id=42)
        session = self._RecordingSession()
        tts._create_multi_session = lambda: session

        async for _ in tts.run_tts("Hi.", context_id="caller-ctx"):
            pass

        assert session.sent_contexts == ["caller-ctx"]
        # No legacy turn context was minted for a caller-managed turn.
        assert tts._legacy_turn_context_id is None

    @pytest.mark.asyncio
    async def test_ws_reset_uses_fresh_context(self):
        """After the cached WS is closed (reconnect), the next turn must use a
        fresh context — the old one died with the old connection."""
        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(api_key="test-key", voice_id=42)
        sessions: list[TestTTSReusesSessionContext._RecordingSession] = []

        def _make():
            s = self._RecordingSession()
            sessions.append(s)
            return s

        tts._create_multi_session = _make

        async for _ in tts.run_tts("Turn one."):
            pass
        first_ctx = tts._legacy_turn_context_id

        async with tts._multi_session_lock:
            await tts._close_multi_session()

        async for _ in tts.run_tts("Turn two."):
            pass
        second_ctx = tts._legacy_turn_context_id

        assert first_ctx is not None and second_ctx is not None
        assert first_ctx != second_ctx
        assert len(sessions) == 2
        assert sessions[0].sent_contexts == [first_ctx]
        assert sessions[1].sent_contexts == [second_ctx]

    @pytest.mark.asyncio
    async def test_interrupted_turn_closed_by_next_turn_start(self):
        """A barge-in cancels run_tts mid-stream; the interrupted turn's context
        is closed by the NEXT turn's start frame (the modern stale-close
        analogue), and the next turn uses a fresh context so no audio bleeds."""
        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(api_key="test-key", voice_id=42)
        session = self._RecordingSession()
        tts._create_multi_session = lambda: session

        # Turn one: the LLM start frame pre-creates the context, then a barge-in
        # cancels run_tts mid-stream.
        await tts._legacy_begin_turn()
        ctx_one = tts._legacy_turn_context_id
        assert ctx_one in session.created_contexts
        agen = tts.run_tts("Interrupted sentence.")
        await agen.__anext__()  # TTSStartedFrame
        await agen.__anext__()  # first audio frame
        await agen.aclose()     # barge-in cancellation
        # run_tts does not close the context; it stays open for the next turn.
        assert session.closed_contexts == []
        assert tts._legacy_turn_context_id == ctx_one

        # Turn two starts: the stale interrupted context is closed and a fresh
        # one is created and used.
        await tts._legacy_begin_turn()
        ctx_two = tts._legacy_turn_context_id
        assert session.closed_contexts == [ctx_one]
        assert ctx_two != ctx_one and ctx_two in session.created_contexts

        async for _ in tts.run_tts("Next sentence."):
            pass
        assert session.sent_contexts[-1] == ctx_two

    @pytest.mark.asyncio
    async def test_interrupted_caller_context_closed_immediately(self):
        """Pipecat 1.x / late 0.x: on_audio_context_interrupted must close the
        server-side context right away, not wait for the next turn."""
        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(api_key="test-key", voice_id=42)
        session = self._RecordingSession()
        tts._create_multi_session = lambda: session
        tts._multi_session = session

        ctx = "pipecat-turn-abc"
        tts._caller_contexts.add(ctx)
        tts._provisioned_contexts.add(ctx)

        await tts.on_audio_context_interrupted(ctx)

        assert session.closed_contexts == [ctx]
        assert ctx not in tts._caller_contexts
        assert ctx not in tts._provisioned_contexts

    @pytest.mark.asyncio
    async def test_interruption_before_audio_closes_preprovisioned_context(self):
        """Barge-in after LLM start but before run_tts must still drop the
        pre-provisioned server context (not wait for the next turn)."""
        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(api_key="test-key", voice_id=42)
        session = self._RecordingSession()
        tts._create_multi_session = lambda: session
        tts._multi_session = session

        ctx = "pipecat-turn-early-interrupt"
        tts._caller_contexts.add(ctx)
        tts._provisioned_contexts.add(ctx)

        await tts._close_remaining_caller_contexts()

        assert session.closed_contexts == [ctx]
        assert ctx not in tts._caller_contexts

    @pytest.mark.asyncio
    async def test_legacy_interruption_closes_turn_context_immediately(self):
        """Legacy 0.x: StartInterruptionFrame must drop the open turn context
        without waiting for the next LLMFullResponseStartFrame.

        Legacy-only. On the AudioContext surface (later 0.0.x / 1.x) the wrapper
        takes the caller-context branch instead, the base _handle_interruption
        creates an audio-context task (needs a task manager this unit harness
        doesn't initialize), and StartInterruptionFrame was renamed to
        InterruptionFrame — so this scenario only applies when run_tts(text) is
        the base surface.
        """
        from kugelaudio.pipecat.tts import _base_run_tts_takes_context_id

        if _base_run_tts_takes_context_id():
            pytest.skip("legacy run_tts(text) surface only")

        from kugelaudio.pipecat import KugelAudioTTSService

        tts = KugelAudioTTSService(api_key="test-key", voice_id=42)
        session = self._RecordingSession()
        tts._create_multi_session = lambda: session
        tts._multi_session = session

        await tts._legacy_begin_turn()
        ctx = tts._legacy_turn_context_id
        assert ctx is not None

        from pipecat.frames.frames import StartInterruptionFrame
        from pipecat.processors.frame_processor import FrameDirection

        await tts._handle_interruption(
            StartInterruptionFrame(), FrameDirection.DOWNSTREAM
        )

        assert session.closed_contexts == [ctx]
        assert tts._legacy_turn_context_id is None
