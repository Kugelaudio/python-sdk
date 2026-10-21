"""Tests for KugelAudioSTTService, the Pipecat streaming speech-to-text service.

Requires: pip install kugelaudio[pipecat]. ``client.asr.stream`` is a scripted
fake; nothing touches the network.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Any

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (  # noqa: E402
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection  # noqa: E402
from pipecat.transcriptions.language import Language  # noqa: E402

from kugelaudio.exceptions import AuthenticationError  # noqa: E402
from kugelaudio.exceptions import ConnectionError as KugelAudioConnectionError  # noqa: E402
from kugelaudio.models import StreamingTranscriptionEvent  # noqa: E402
from kugelaudio.pipecat.stt import (  # noqa: E402
    _PROPOSES_TURNS,
    _TurnStartFrame as Start,
    _TurnStopFrame as Stop,
)

RATE = 16000
CHUNK = b"\x01\x00" * (RATE // 50)  # 20 ms
TURN_FRAMES = (
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    Start,
    Stop,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
# Before 1.11 the service interrupts the bot itself; from 1.11 the aggregator
# does it when it accepts the proposed turn start.
INTERRUPTS_ITSELF = not _PROPOSES_TURNS


def partial(text: str) -> StreamingTranscriptionEvent:
    return StreamingTranscriptionEvent(partial_text=text, is_final=False)


def final(text: str) -> StreamingTranscriptionEvent:
    return StreamingTranscriptionEvent(partial_text=text, is_final=True)


def alternatives(text: str) -> StreamingTranscriptionEvent:
    return StreamingTranscriptionEvent(
        partial_text=text, is_final=False, type="alternatives", revises_final=True
    )


class FakeServer:
    """One scripted stream: events after the n-th chunk, then after the input ends."""

    def __init__(
        self,
        after_chunk: dict[int, list[StreamingTranscriptionEvent]] | None = None,
        on_end: Iterable[StreamingTranscriptionEvent] = (),
        error: Exception | None = None,
    ) -> None:
        self.after_chunk = after_chunk or {}
        self.on_end = list(on_end)
        self.error = error
        self.chunks: list[bytes] = []
        self.kwargs: dict[str, Any] = {}
        self.input_ended = False
        self.closed = False

    async def run(self, audio: AsyncIterator[bytes]) -> AsyncIterator[StreamingTranscriptionEvent]:
        try:
            if self.error is not None:
                raise self.error
            async for chunk in audio:
                self.chunks.append(chunk)
                for event in self.after_chunk.get(len(self.chunks), []):
                    yield event
            self.input_ended = True
            for event in self.on_end:
                yield event
        finally:
            self.closed = True


class FakeASR:
    def __init__(self, servers: list[FakeServer]) -> None:
        self.servers = servers
        self.opened = 0

    def stream(self, audio: AsyncIterator[bytes], **kwargs: Any) -> AsyncIterator[StreamingTranscriptionEvent]:
        server = self.servers[self.opened]
        self.opened += 1
        server.kwargs = kwargs
        return server.run(audio)


class FakeClient:
    def __init__(self, *servers: FakeServer) -> None:
        self.asr = FakeASR(list(servers))
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class Harness:
    """The service with its client faked and every pushed frame recorded."""

    def __init__(self, *servers: FakeServer, **options: Any) -> None:
        from kugelaudio.pipecat import KugelAudioSTTService

        self.stt = KugelAudioSTTService(api_key="ka_test", **options)
        self.client = FakeClient(*servers)
        self.stt._client = self.client  # type: ignore[assignment]  # fake client
        self.stt._sample_rate = RATE
        self.pushed: list[tuple[Frame, FrameDirection]] = []
        self.interruptions = 0

        async def record(frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
            self.pushed.append((frame, direction))

        async def interrupt() -> None:
            self.interruptions += 1

        self.stt.push_frame = record  # type: ignore[method-assign]  # capture frames
        if hasattr(self.stt, "broadcast_interruption"):
            self.stt.broadcast_interruption = interrupt  # type: ignore[method-assign]  # count them

    async def send(self, count: int = 1) -> None:
        for _ in range(count):
            async for _frame in self.stt.run_stt(CHUNK):
                pass
            await settle()

    def down(self) -> list[Frame]:
        return [
            frame
            for frame, direction in self.pushed
            if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TURN_FRAMES)
        ]

    def errors(self) -> list[ErrorFrame]:
        return [
            frame
            for frame, direction in self.pushed
            if direction == FrameDirection.UPSTREAM and isinstance(frame, ErrorFrame)
        ]


async def settle(rounds: int = 20) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0)


async def until(condition: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "condition not reached"
        await asyncio.sleep(0.005)


def kinds(frames: list[Frame]) -> list[type]:
    return [type(frame) for frame in frames]


def test_lazy_export() -> None:
    import kugelaudio.pipecat as package
    from kugelaudio.pipecat.stt import KugelAudioSTTService

    assert package.KugelAudioSTTService is KugelAudioSTTService


def test_requires_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from kugelaudio.pipecat import KugelAudioSTTService

    monkeypatch.delenv("KUGELAUDIO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="KUGELAUDIO_API_KEY"):
        KugelAudioSTTService()


def test_region_prefixed_key_selects_the_eu_endpoint() -> None:
    from kugelaudio.client import EU_API_URL
    from kugelaudio.pipecat import KugelAudioSTTService

    stt = KugelAudioSTTService(api_key="eu-ka_test")
    assert stt._client._api_key == "ka_test"
    assert stt._client._api_url == EU_API_URL


def test_base_url_overrides_the_region() -> None:
    from kugelaudio.pipecat import KugelAudioSTTService

    stt = KugelAudioSTTService(api_key="ka_test", region="eu", base_url="http://localhost:9/")
    assert stt._client._api_url == "http://localhost:9"


async def test_short_turn_with_only_a_final() -> None:
    h = Harness(FakeServer(after_chunk={2: [final("Guten Tag.")]}), language="de")
    h.stt._open_stream()
    await h.send(2)
    await until(lambda: len(h.down()) == 3)

    down = h.down()
    assert kinds(down) == [Start, TranscriptionFrame, Stop]
    assert down[1].text == "Guten Tag."
    assert down[1].language == Language.DE
    assert isinstance(down[1].timestamp, str) and down[1].timestamp
    own_interruption = INTERRUPTS_ITSELF and hasattr(type(h.stt), "broadcast_interruption")
    assert h.interruptions == (1 if own_interruption else 0)
    await h.stt._close_stream(graceful=True)


async def test_partials_then_final() -> None:
    h = Harness(
        FakeServer(after_chunk={1: [partial("Guten")], 2: [partial("Guten Tag")], 3: [final("Guten Tag.")]})
    )
    h.stt._open_stream()
    await h.send(3)
    await until(lambda: len(h.down()) == 5)

    down = h.down()
    assert kinds(down) == [
        Start,
        InterimTranscriptionFrame,
        InterimTranscriptionFrame,
        TranscriptionFrame,
        Stop,
    ]
    assert [f.text for f in down[1:4]] == ["Guten", "Guten Tag", "Guten Tag."]
    assert down[3].language is None  # no single configured language: unknown
    await h.stt._close_stream(graceful=True)


async def test_two_turns_each_start_and_stop() -> None:
    h = Harness(FakeServer(after_chunk={1: [final("Eins.")], 3: [partial("Zwei"), final("Zwei.")]}))
    h.stt._open_stream()
    await h.send(3)
    await until(lambda: len(h.down()) == 7)

    assert kinds(h.down()) == [
        Start,
        TranscriptionFrame,
        Stop,
        Start,
        InterimTranscriptionFrame,
        TranscriptionFrame,
        Stop,
    ]
    await h.stt._close_stream(graceful=True)


async def test_alternatives_frames_are_ignored() -> None:
    h = Harness(FakeServer(after_chunk={1: [final("Hallo.")], 2: [alternatives("Hallo!")]}))
    h.stt._open_stream()
    await h.send(3)
    await until(lambda: len(h.down()) == 3)
    await settle()

    assert kinds(h.down()) == [Start, TranscriptionFrame, Stop]
    await h.stt._close_stream(graceful=True)


async def test_empty_final_without_a_turn_is_ignored() -> None:
    h = Harness(FakeServer(after_chunk={1: [partial("  "), final("")]}))
    h.stt._open_stream()
    await h.send(2)
    await settle()

    assert h.down() == []
    assert h.interruptions == 0
    await h.stt._close_stream(graceful=True)


async def test_end_frame_ends_the_input_and_closes() -> None:
    from pipecat.tests.utils import run_test

    from kugelaudio.pipecat import KugelAudioSTTService

    server = FakeServer()
    client = FakeClient(server)
    stt = KugelAudioSTTService(api_key="ka_test")
    stt._client = client  # type: ignore[assignment]  # fake client
    frames = [InputAudioRawFrame(audio=CHUNK, sample_rate=RATE, num_channels=1) for _ in range(3)]

    await run_test(stt, frames_to_send=frames, expected_down_frames=None)

    assert server.chunks == [CHUNK] * 3
    assert server.input_ended
    assert server.closed
    assert client.asr.opened == 1


async def test_turn_flows_through_a_real_pipeline() -> None:
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.processors.frame_processor import FrameProcessor
    from pipecat.tests.utils import SleepFrame, run_test

    from kugelaudio.pipecat import KugelAudioSTTService

    class Recorder(FrameProcessor):
        def __init__(self) -> None:
            super().__init__()
            self.seen: list[Frame] = []

        async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
            await super().process_frame(frame, direction)
            if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TURN_FRAMES):
                self.seen.append(frame)
            await self.push_frame(frame, direction)

    server = FakeServer(after_chunk={2: [partial("Hallo"), final("Hallo Welt.")]})
    stt = KugelAudioSTTService(api_key="ka_test")
    stt._client = FakeClient(server)  # type: ignore[assignment]  # fake client
    recorder = Recorder()
    audio = InputAudioRawFrame(audio=CHUNK, sample_rate=RATE, num_channels=1)
    frames: list[Frame] = [audio, audio, SleepFrame(sleep=0.2), audio]

    await run_test(Pipeline([stt, recorder]), frames_to_send=frames, expected_down_frames=None)

    expected = [Start, InterimTranscriptionFrame, TranscriptionFrame, Stop]
    if _PROPOSES_TURNS:
        # The proposed stop is a control frame: it stays behind the final.
        assert kinds(recorder.seen) == expected
    else:
        # UserStoppedSpeakingFrame is a system frame and may overtake the
        # queued transcripts; the aggregator waits for the text.
        assert sorted(kinds(recorder.seen), key=str) == sorted(expected, key=str)
    assert server.input_ended and server.closed


async def test_user_aggregator_closes_the_turn_on_its_recommended_strategies() -> None:
    from kugelaudio.pipecat.stt import _recommends_turn_strategies

    if not (_PROPOSES_TURNS and _recommends_turn_strategies()):
        pytest.skip("needs Pipecat >= 1.11 (proposed turns, recommended strategies)")
    from pipecat.frames.frames import LLMContextFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
    from pipecat.processors.frame_processor import FrameProcessor
    from pipecat.tests.utils import SleepFrame, run_test

    from kugelaudio.pipecat import KugelAudioSTTService

    class Recorder(FrameProcessor):
        def __init__(self) -> None:
            super().__init__()
            self.seen: list[Frame] = []

        async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
            await super().process_frame(frame, direction)
            if direction == FrameDirection.DOWNSTREAM:
                self.seen.append(frame)
            await self.push_frame(frame, direction)

    server = FakeServer(after_chunk={2: [final("Hallo Welt.")]})
    stt = KugelAudioSTTService(api_key="ka_test")
    stt._client = FakeClient(server)  # type: ignore[assignment]  # fake client
    context = LLMContext()
    pair = LLMContextAggregatorPair(context)  # no strategies: the STT recommends them
    recorder = Recorder()
    audio = InputAudioRawFrame(audio=CHUNK, sample_rate=RATE, num_channels=1)
    frames: list[Frame] = [audio, audio, SleepFrame(sleep=0.3)]

    await run_test(Pipeline([stt, pair.user(), recorder]), frames_to_send=frames, expected_down_frames=None)

    seen = kinds(recorder.seen)
    assert UserStartedSpeakingFrame in seen and UserStoppedSpeakingFrame in seen
    assert LLMContextFrame in seen
    assert any("Hallo Welt." in str(message) for message in context.get_messages())


async def test_final_sent_after_the_input_ends_still_arrives() -> None:
    h = Harness(FakeServer(on_end=[final("Tschüss.")]))
    h.stt._open_stream()
    await h.send(1)

    await h.stt._close_stream(graceful=True)

    assert kinds(h.down()) == [Start, TranscriptionFrame, Stop]
    assert h.interruptions == 0  # the pipeline is ending: nothing to interrupt


async def test_auth_error_is_fatal() -> None:
    h = Harness(FakeServer(error=AuthenticationError()), FakeServer())
    h.stt._open_stream()
    await until(lambda: len(h.errors()) == 1)
    await settle()

    error = h.errors()[0]
    assert "rejected the API key" in error.error
    if hasattr(h.stt, "is_usable"):  # Pipecat >= 1.8: a permanent error
        assert not h.stt.is_usable
    else:
        assert error.fatal
    assert h.client.asr.opened == 1  # no reconnect
    await h.send(1)
    assert h.stt._queue is not None and h.stt._queue.empty()  # audio no longer queued
    await h.stt._close_stream(graceful=True)


async def test_connection_error_reconnects(monkeypatch: pytest.MonkeyPatch) -> None:
    import kugelaudio.pipecat.stt as stt_module

    monkeypatch.setattr(stt_module, "_RECONNECT_DELAYS_S", (0.0, 0.0, 0.0))
    second = FakeServer(after_chunk={1: [final("Wieder da.")]})
    h = Harness(FakeServer(error=KugelAudioConnectionError("dropped")), second)
    h.stt._open_stream()
    await until(lambda: h.client.asr.opened == 2)
    await h.send(1)
    await until(lambda: len(h.down()) == 3)

    errors = h.errors()
    assert len(errors) == 1 and "dropped" in errors[0].error
    assert not errors[0].fatal
    assert kinds(h.down())[1] is TranscriptionFrame
    await h.stt._close_stream(graceful=True)


async def test_unset_options_are_not_sent() -> None:
    server = FakeServer()
    h = Harness(server)
    h.stt._open_stream()
    await until(lambda: h.client.asr.opened == 1)

    assert server.kwargs == {"sample_rate": RATE}
    await h.stt._close_stream(graceful=True)


async def test_set_options_are_sent() -> None:
    server = FakeServer()
    options: dict[str, Any] = {
        "model": "test-model",
        "language": ["de", "en"],
        "turn_detection": "kugelturn",
        "turn_detection_level": "conservative",
        "boosted_phrases": ["KugelAudio"],
        "project_id": 7,
        "dictionary_ids": [],
    }
    h = Harness(server, sample_rate=8000, **options)
    h.stt._open_stream()
    await until(lambda: h.client.asr.opened == 1)

    assert server.kwargs == {"sample_rate": 8000, **options}
    await h.stt._close_stream(graceful=True)


def _language_delta(language: object) -> object:
    """The settings delta this Pipecat's STTUpdateSettingsFrame carries."""
    from kugelaudio.pipecat.stt import _PipecatSTTSettings

    if _PipecatSTTSettings is None:  # Pipecat < 0.0.104: a plain mapping
        return {"language": language}
    return _PipecatSTTSettings(language=language)


def _model_delta(model: str) -> object:
    from kugelaudio.pipecat.stt import _PipecatSTTSettings

    if _PipecatSTTSettings is None:
        return {"model": model}
    return _PipecatSTTSettings(model=model)


async def test_settings_update_reconnects_with_the_new_language() -> None:
    from pipecat.transcriptions.language import Language

    first = FakeServer(after_chunk={1: [partial("Hallo")]})
    second = FakeServer(after_chunk={1: [final("Bonjour.")]})
    h = Harness(first, second, language="de", model="test-model")
    h.stt._open_stream()
    await until(lambda: h.client.asr.opened == 1)
    await h.send(1)
    await until(lambda: len(h.down()) == 2)  # the German turn started

    await h.stt._update_settings(_language_delta(Language.FR))
    await until(lambda: h.client.asr.opened == 2)
    assert first.closed
    assert second.kwargs == {"sample_rate": RATE, "model": "test-model", "language": "fr"}
    # The interrupted German turn was closed before the new stream opened.
    assert kinds(h.down())[-1] is not TranscriptionFrame

    await h.send(1)
    await until(lambda: any(isinstance(f, TranscriptionFrame) for f in h.down()))
    final_frame = next(f for f in h.down() if isinstance(f, TranscriptionFrame))
    assert final_frame.text == "Bonjour." and final_frame.language == Language.FR
    await h.stt._close_stream(graceful=True)


async def test_model_update_keeps_a_language_list() -> None:
    first, second = FakeServer(), FakeServer()
    h = Harness(first, second, language=["de", "fr"], model="test-model")
    h.stt._open_stream()
    await until(lambda: h.client.asr.opened == 1)

    await h.stt._update_settings(_model_delta("other-model"))
    await until(lambda: h.client.asr.opened == 2)
    assert second.kwargs == {
        "sample_rate": RATE,
        "model": "other-model",
        "language": ["de", "fr"],
    }
    await h.stt._close_stream(graceful=True)


async def test_stereo_audio_is_refused() -> None:
    from pipecat.frames.frames import InputAudioRawFrame

    server = FakeServer()
    h = Harness(server)
    h.stt._open_stream()
    await until(lambda: h.client.asr.opened == 1)

    stereo = InputAudioRawFrame(audio=CHUNK * 2, sample_rate=RATE, num_channels=2)
    await h.stt.process_audio_frame(stereo, FrameDirection.DOWNSTREAM)
    await h.stt.process_audio_frame(stereo, FrameDirection.DOWNSTREAM)
    await settle()

    errors = h.errors()
    assert len(errors) == 1 and "mono" in errors[0].error
    if hasattr(h.stt, "is_usable"):  # Pipecat >= 1.8: a permanent error
        assert not h.stt.is_usable
    else:
        assert errors[0].fatal
    assert server.chunks == []
    await h.stt._close_stream(graceful=True)


def test_recommends_external_turn_strategies_when_supported() -> None:
    from kugelaudio.pipecat.stt import _recommends_turn_strategies

    if not _recommends_turn_strategies():
        pytest.skip("this Pipecat has no user_turn_strategies on STT metadata")
    from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies

    stt = Harness().stt
    assert isinstance(stt.service_metadata_frame().user_turn_strategies, ExternalUserTurnStrategies)
