"""Contract tests for the LiveKit speech-to-text plugin, with no network.

The WebSocket is replaced at ``kugelaudio.asr._run_stream_async`` (so the
real ``ASRResource.stream`` still builds the config frame) and HTTP at
``KugelAudio._request_multipart``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

import pytest

pytest.importorskip("livekit.agents")

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    stt,
)

import kugelaudio.asr as asr_module
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
from kugelaudio.livekit import STT, SpeechStream
from kugelaudio.livekit.stt import _api_error
from kugelaudio.models import StreamingTranscriptionEvent

E = stt.SpeechEventType
NO_RETRY = APIConnectOptions(max_retry=0, retry_interval=0.0, timeout=1.0)
SAMPLES = 1600  # 0.1 s at 16 kHz


def _partial(text: str) -> dict[str, object]:
    return {"type": "partial", "partial_text": text, "is_final": False}


def _final(text: str) -> dict[str, object]:
    return {"type": "partial", "partial_text": text, "is_final": True}


def _revision(text: str) -> dict[str, object]:
    return {
        "type": "alternatives",
        "partial_text": text,
        "is_final": False,
        "revises_final": True,
        "language": "de",
    }


@dataclass
class _Wire:
    """Records what the plugin handed the WebSocket layer."""

    config: dict[str, object] | None = None
    chunks: list[bytes] = field(default_factory=list)
    input_ended: bool = False
    calls: int = 0


Script = Callable[
    [AsyncIterator[bytes], _Wire], AsyncIterator[StreamingTranscriptionEvent]
]


def _after_input(
    frames: list[dict[str, object]], *, error: BaseException | None = None
) -> Script:
    """Read the whole input (until end_of_speech), then answer *frames*."""

    async def script(
        audio: AsyncIterator[bytes], wire: _Wire
    ) -> AsyncIterator[StreamingTranscriptionEvent]:
        async for chunk in audio:
            wire.chunks.append(chunk)
        wire.input_ended = True
        if error is not None:
            raise error
        for frame in frames:
            yield StreamingTranscriptionEvent.from_dict(frame)

    return script


def _install(monkeypatch: pytest.MonkeyPatch, script: Script) -> _Wire:
    wire = _Wire()

    async def fake_run_stream_async(
        url: str, config: dict[str, object], max_bytes: int, audio: AsyncIterator[bytes]
    ) -> AsyncIterator[StreamingTranscriptionEvent]:
        wire.config = config
        wire.calls += 1
        async for event in script(audio, wire):
            yield event

    monkeypatch.setattr(asr_module, "_run_stream_async", fake_run_stream_async)
    return wire


def _frame(
    samples: int = SAMPLES, channels: int = 1, sample_rate: int = 16000
) -> rtc.AudioFrame:
    data = bytes(range(256)) * ((samples * channels * 2) // 256 + 1)
    return rtc.AudioFrame(
        data=data[: samples * channels * 2],
        sample_rate=sample_rate,
        num_channels=channels,
        samples_per_channel=samples,
    )


async def _drain(stream: SpeechStream) -> list[stt.SpeechEvent]:
    return [event async for event in stream]


def _shape(events: list[stt.SpeechEvent]) -> list[tuple[E, str]]:
    """(type, text) of every event but usage."""
    return [
        (e.type, e.alternatives[0].text if e.alternatives else "")
        for e in events
        if e.type != E.RECOGNITION_USAGE
    ]


async def _run_turns(
    monkeypatch: pytest.MonkeyPatch,
    frames: list[dict[str, object]],
    **stt_kwargs: object,
) -> tuple[list[stt.SpeechEvent], _Wire]:
    wire = _install(monkeypatch, _after_input(frames))
    plugin = STT(api_key="test-key", **stt_kwargs)  # type: ignore[arg-type]  # test passthrough
    stream = plugin.stream(conn_options=NO_RETRY)
    stream.push_frame(_frame())
    stream.end_input()
    events = await _drain(stream)
    await stream.aclose()
    await plugin.aclose()
    return events, wire


async def test_short_turn_with_only_a_final(monkeypatch: pytest.MonkeyPatch) -> None:
    events, _ = await _run_turns(monkeypatch, [_final("Hallo.")], language="de")

    assert _shape(events) == [
        (E.START_OF_SPEECH, ""),
        (E.FINAL_TRANSCRIPT, "Hallo."),
        (E.END_OF_SPEECH, ""),
    ]
    final = next(e for e in events if e.type == E.FINAL_TRANSCRIPT)
    assert final.alternatives[0].language == "de"


async def test_partials_arrive_before_the_input_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def script(
        audio: AsyncIterator[bytes], wire: _Wire
    ) -> AsyncIterator[StreamingTranscriptionEvent]:
        wire.chunks.append(await anext(audio))
        yield StreamingTranscriptionEvent.from_dict(_partial("I think"))
        yield StreamingTranscriptionEvent.from_dict(_partial("I think we should"))
        async for chunk in audio:
            wire.chunks.append(chunk)
        wire.input_ended = True
        yield StreamingTranscriptionEvent.from_dict(_final("I think we should go."))

    _install(monkeypatch, script)
    plugin = STT(api_key="test-key")
    stream = plugin.stream(conn_options=NO_RETRY)
    stream.push_frame(_frame())
    first = [await anext(stream), await anext(stream), await anext(stream)]
    assert _shape(first) == [
        (E.START_OF_SPEECH, ""),
        (E.INTERIM_TRANSCRIPT, "I think"),
        (E.INTERIM_TRANSCRIPT, "I think we should"),
    ]
    stream.end_input()
    rest = await _drain(stream)
    await plugin.aclose()

    assert _shape(rest) == [
        (E.FINAL_TRANSCRIPT, "I think we should go."),
        (E.END_OF_SPEECH, ""),
    ]


async def test_two_turns_on_one_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    events, wire = await _run_turns(
        monkeypatch,
        [_partial("one"), _final("One."), _final("Two.")],
        language="en",
    )

    assert _shape(events) == [
        (E.START_OF_SPEECH, ""),
        (E.INTERIM_TRANSCRIPT, "one"),
        (E.FINAL_TRANSCRIPT, "One."),
        (E.END_OF_SPEECH, ""),
        (E.START_OF_SPEECH, ""),
        (E.FINAL_TRANSCRIPT, "Two."),
        (E.END_OF_SPEECH, ""),
    ]
    assert wire.calls == 1


async def test_alternatives_and_revisions_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, _ = await _run_turns(
        monkeypatch,
        [
            _final("Guten Tag."),
            {"type": "alternatives", "partial_text": "Guten Tag", "is_final": False},
            _revision("Guten Tag!"),
        ],
    )

    assert _shape(events) == [
        (E.START_OF_SPEECH, ""),
        (E.FINAL_TRANSCRIPT, "Guten Tag."),
        (E.END_OF_SPEECH, ""),
    ]


async def test_empty_final_with_nothing_started_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, _ = await _run_turns(
        monkeypatch, [_partial(""), _final(""), _final("Yes.")]
    )

    assert _shape(events) == [
        (E.START_OF_SPEECH, ""),
        (E.FINAL_TRANSCRIPT, "Yes."),
        (E.END_OF_SPEECH, ""),
    ]


async def test_empty_final_after_a_start_still_ends_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, _ = await _run_turns(monkeypatch, [_partial("uh"), _final("")])

    assert _shape(events) == [
        (E.START_OF_SPEECH, ""),
        (E.INTERIM_TRANSCRIPT, "uh"),
        (E.FINAL_TRANSCRIPT, ""),
        (E.END_OF_SPEECH, ""),
    ]


async def test_usage_is_the_audio_sent_per_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, _ = await _run_turns(monkeypatch, [_final("One."), _final("Two.")])

    usage = [e for e in events if e.type == E.RECOGNITION_USAGE]
    assert len(usage) == 1  # all audio was sent before the first final
    assert usage[0].recognition_usage is not None
    assert usage[0].recognition_usage.audio_duration == pytest.approx(SAMPLES / 16000)
    # It follows the first turn's final and end of speech.
    assert events.index(usage[0]) == 3


@pytest.mark.parametrize(
    ("language", "expected"),
    [("de", "de"), (["en"], "en"), ("auto", ""), (["de", "en"], ""), (None, "")],
)
async def test_transcript_language_is_never_guessed(
    monkeypatch: pytest.MonkeyPatch, language: object, expected: str
) -> None:
    events, _ = await _run_turns(monkeypatch, [_final("x")], language=language)

    final = next(e for e in events if e.type == E.FINAL_TRANSCRIPT)
    assert final.alternatives[0].language == expected


async def test_end_input_ends_the_audio_and_sends_every_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = _install(monkeypatch, _after_input([]))
    plugin = STT(api_key="test-key")
    stream = plugin.stream(conn_options=NO_RETRY)
    frames = [_frame(), _frame(320)]
    for frame in frames:
        stream.push_frame(frame)
    stream.flush()  # ignored: the server ends turns itself
    stream.push_frame(_frame())
    frames.append(_frame())
    stream.end_input()
    events = await _drain(stream)
    await plugin.aclose()

    assert wire.input_ended
    assert wire.calls == 1
    assert b"".join(wire.chunks) == b"".join(f.data.tobytes() for f in frames)
    usage = [e for e in events if e.type == E.RECOGNITION_USAGE]
    assert [
        u.recognition_usage.audio_duration for u in usage if u.recognition_usage
    ] == [pytest.approx((2 * SAMPLES + 320) / 16000)]


async def test_server_closing_before_the_input_ended_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def script(
        audio: AsyncIterator[bytes], wire: _Wire
    ) -> AsyncIterator[StreamingTranscriptionEvent]:
        wire.chunks.append(await anext(audio))
        yield StreamingTranscriptionEvent.from_dict(_final("Bye."))

    _install(monkeypatch, script)
    plugin = STT(api_key="test-key")
    stream = plugin.stream(conn_options=NO_RETRY)
    stream.push_frame(_frame())

    with pytest.raises(APIConnectionError, match="before the input ended"):
        await _drain(stream)
    await plugin.aclose()


@pytest.mark.parametrize("sample_rate", [16000, 48000])
async def test_stereo_input_is_refused_before_resampling(
    monkeypatch: pytest.MonkeyPatch, sample_rate: int
) -> None:
    # At 48 kHz the base class resamples, and its mono resampler would read
    # interleaved stereo as one channel; the frame must be refused first.
    _install(monkeypatch, _after_input([]))
    plugin = STT(api_key="test-key")
    stream = plugin.stream(conn_options=NO_RETRY)

    with pytest.raises(ValueError, match="mono"):
        stream.push_frame(_frame(channels=2, sample_rate=sample_rate))
    await stream.aclose()
    await plugin.aclose()


async def test_recognize_refuses_stereo() -> None:
    plugin = STT(api_key="test-key")
    with pytest.raises(ValueError, match="mono"):
        await plugin.recognize([_frame(channels=2)], conn_options=NO_RETRY)
    await plugin.aclose()


@pytest.mark.parametrize(
    ("error", "kind", "retryable"),
    [
        (AuthenticationError(), APIStatusError, False),
        (InsufficientCreditsError(), APIStatusError, False),
        (ValidationError("bad option"), APIStatusError, False),
        (NotFoundError(), APIStatusError, False),
        (RateLimitError(retry_after=2), APIStatusError, True),
        (KugelAudioConnectionError("dropped"), APIConnectionError, True),
        (ServerRestartingError(), APIConnectionError, True),
        (KugelAudioError("model unavailable", status_code=503), APIStatusError, True),
    ],
)
def test_error_mapping(
    error: KugelAudioError, kind: type[APIError], retryable: bool
) -> None:
    mapped = _api_error(error)

    assert isinstance(mapped, kind)
    assert mapped.retryable is retryable


@pytest.mark.parametrize(
    ("error", "kind", "retryable"),
    [
        (AuthenticationError(), APIStatusError, False),
        (RateLimitError(), APIStatusError, True),
        (KugelAudioConnectionError("dropped"), APIConnectionError, True),
        (ConnectionRefusedError(61, "refused"), APIConnectionError, True),
    ],
)
async def test_stream_errors_reach_livekit_typed(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
    kind: type[APIError],
    retryable: bool,
) -> None:
    _install(monkeypatch, _after_input([], error=error))
    plugin = STT(api_key="test-key")
    stream = plugin.stream(conn_options=NO_RETRY)
    stream.push_frame(_frame())
    stream.end_input()

    with pytest.raises(kind) as raised:
        await _drain(stream)
    assert raised.value.retryable is retryable
    await plugin.aclose()


async def test_unset_options_are_not_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    _, wire = await _run_turns(monkeypatch, [])

    assert wire.config == {"type": "config", "sample_rate": 16000}


async def test_set_options_are_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    _, wire = await _run_turns(
        monkeypatch,
        [],
        model="test-model",
        language=["de", "en"],
        turn_detection="kugelturn",
        turn_detection_level="conservative",
        boosted_phrases=["KugelAudio"],
        project_id=7,
        dictionary_ids=[],
    )

    assert wire.config == {
        "type": "config",
        "sample_rate": 16000,
        "model": "test-model",
        "language": ["de", "en"],
        "turn_detection": "kugelturn",
        "turn_detection_level": "conservative",
        "boosted_phrases": ["KugelAudio"],
        "project_id": 7,
        "dictionary_ids": [],
    }


async def test_stream_language_overrides_the_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = _install(monkeypatch, _after_input([_final("Ciao.")]))
    plugin = STT(api_key="test-key", language="de")
    stream = plugin.stream(language="it", conn_options=NO_RETRY)
    stream.push_frame(_frame())
    stream.end_input()
    events = await _drain(stream)
    await plugin.aclose()

    assert wire.config is not None and wire.config["language"] == "it"
    final = next(e for e in events if e.type == E.FINAL_TRANSCRIPT)
    assert final.alternatives[0].language == "it"


def _capture_multipart(
    monkeypatch: pytest.MonkeyPatch, plugin: STT
) -> list[list[tuple[str, tuple[str | None, object, str]]]]:
    sent: list[list[tuple[str, tuple[str | None, object, str]]]] = []

    def fake_request_multipart(
        method: str, path: str, fields: list[tuple[str, tuple[str | None, object, str]]]
    ) -> dict[str, object]:
        sent.append(fields)
        return {
            "text": "Hello there.",
            "language": "en",
            "duration_s": 0.2,
            "model": "test-model",
        }

    monkeypatch.setattr(plugin._client, "_request_multipart", fake_request_multipart)
    return sent


async def test_recognize_one_shot(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = STT(api_key="test-key")
    sent = _capture_multipart(monkeypatch, plugin)

    event = await plugin.recognize([_frame(), _frame()], conn_options=NO_RETRY)
    await plugin.aclose()

    assert event.type == E.FINAL_TRANSCRIPT
    assert event.alternatives[0].text == "Hello there."
    assert event.alternatives[0].language == "en"
    assert [name for name, _ in sent[0]] == ["file"]  # unset options are not sent
    wav = sent[0][0][1][1]
    assert isinstance(wav, bytes) and wav[:4] == b"RIFF"


async def test_recognize_sends_set_options(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = STT(
        api_key="test-key",
        model="test-model",
        language=["de"],
        boosted_phrases=["Kugel"],
        project_id=3,
        dictionary_ids=[5],
    )
    sent = _capture_multipart(monkeypatch, plugin)

    await plugin.recognize([_frame()], conn_options=NO_RETRY)
    await plugin.aclose()

    values = {name: value[1] for name, value in sent[0][1:]}
    assert values == {
        "model": "test-model",
        "language": "de",
        "boosted_phrases": "Kugel",
        "project_id": "3",
        "dictionary_ids": "5",
    }


async def test_recognize_refuses_a_language_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = STT(api_key="test-key", language=["de", "en"])
    _capture_multipart(monkeypatch, plugin)

    with pytest.raises(ValueError, match="one language"):
        await plugin.recognize([_frame()], conn_options=NO_RETRY)
    await plugin.aclose()


async def test_recognize_maps_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = STT(api_key="test-key")

    def refuse(*args: object, **kwargs: object) -> dict[str, object]:
        raise AuthenticationError()

    monkeypatch.setattr(plugin._client, "_request_multipart", refuse)

    with pytest.raises(APIStatusError) as raised:
        await plugin.recognize([_frame()], conn_options=NO_RETRY)
    assert raised.value.retryable is False
    await plugin.aclose()


def test_key_region_and_base_url_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KUGELAUDIO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="KUGELAUDIO_API_KEY"):
        STT()
    with pytest.raises(ValueError, match="region"):
        STT(api_key="k", region="mars")

    eu = STT(api_key="eu-secret")
    assert eu._client._api_key == "secret"
    assert eu._client._api_url == "https://api.eu.kugelaudio.com"

    monkeypatch.setenv("KUGELAUDIO_API_KEY", "env-key")
    custom = STT(base_url="http://localhost:8000/")
    assert custom._client._api_key == "env-key"
    assert custom._client._api_url == "http://localhost:8000"


def test_capabilities_and_identity() -> None:
    plugin = STT(api_key="test-key", model="test-model")

    assert plugin.capabilities.streaming is True
    assert plugin.capabilities.interim_results is True
    assert plugin.provider == "KugelAudio"
    assert plugin.model == "test-model"
    assert STT(api_key="test-key").model == "unknown"


async def test_aclose_closes_open_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _after_input([]))
    plugin = STT(api_key="test-key")
    stream = plugin.stream(conn_options=NO_RETRY)
    stream.push_frame(_frame())

    await plugin.aclose()

    with pytest.raises(RuntimeError, match="closed|input ended"):
        stream.push_frame(_frame())
