"""Tests for the LiveKit enhancement FrameProcessor (kugelaudio.livekit.enhance).

Requires: pip install kugelaudio[livekit]. The enhancement stream is a
scripted fake; nothing touches the network. Output frames are also fed
through LiveKit's own AudioProcessingModule, which the agents input path runs
after the processor: it raises on an empty frame and aborts the process on
one that is not a multiple of 10 ms.
"""

from __future__ import annotations

import pytest

pytest.importorskip("livekit.agents")

from livekit import rtc  # noqa: E402

from kugelaudio._enhance_live import LiveEnhancerState  # noqa: E402
from kugelaudio.exceptions import InsufficientCreditsError, ValidationError  # noqa: E402

from .test_enhance_realtime import FakeClient, FakeServer, settle  # noqa: E402


def make_processor(*servers: FakeServer, **kwargs: object):  # noqa: ANN201 - livekit type
    from kugelaudio.livekit import enhancement

    return enhancement(client=FakeClient(*servers), **kwargs)  # type: ignore[arg-type]  # fake client


def frame(rate: int, ms: int = 50, value: int = 1, channels: int = 1) -> rtc.AudioFrame:
    samples = rate * ms // 1000
    data = value.to_bytes(2, "little", signed=True) * samples * channels
    return rtc.AudioFrame(data, rate, channels, samples)


async def run_frames(processor, rate: int, count: int) -> list[rtc.AudioFrame]:  # noqa: ANN001
    agc = rtc.AudioProcessingModule(auto_gain_control=True)
    out = []
    for _ in range(count):
        result = processor._process(frame(rate))
        agc.process_stream(rtc.AudioFrame(bytes(result.data.cast("B")), result.sample_rate, 1, result.samples_per_channel))
        out.append(result)
        await settle()
    return out


def test_lazy_exports() -> None:
    import kugelaudio.livekit as package
    from kugelaudio.livekit.enhance import EnhanceFrameProcessor, enhancement

    assert package.enhancement is enhancement
    assert package.EnhanceFrameProcessor is EnhanceFrameProcessor


def test_requires_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from kugelaudio.livekit import enhancement

    monkeypatch.delenv("KUGELAUDIO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="KUGELAUDIO_API_KEY"):
        enhancement()


def test_is_a_frame_processor_for_audio_input_options() -> None:
    from livekit.agents.voice import room_io

    processor = make_processor()
    assert isinstance(processor, rtc.FrameProcessor)
    options = room_io.AudioInputOptions(noise_cancellation=processor)
    assert options.noise_cancellation is processor


@pytest.mark.parametrize("rate", [24000, 48000, 16000])
async def test_frames_keep_their_size_and_enhanced_audio_arrives(rate: int) -> None:
    server = FakeServer()
    processor = make_processor(server)

    out = await run_frames(processor, rate, 30)

    assert all(f.samples_per_channel == rate // 20 and f.sample_rate == rate for f in out)
    assert bytes(out[0].data.cast("B")) == bytes(frame(rate).data.cast("B"))  # warming
    assert processor.enhancer.state is LiveEnhancerState.LIVE
    assert max(max(f.data) for f in out[-5:]) > 1  # the fake's enhanced value (16)
    processor._close()
    await settle()
    assert server.closed


async def test_fatal_error_returns_the_original_audio() -> None:
    errors: list[Exception] = []
    processor = make_processor(
        FakeServer(connect_error=InsufficientCreditsError()), on_error=errors.append
    )
    out = await run_frames(processor, 24000, 5)
    assert all(max(f.data) == 1 and min(f.data) == 1 for f in out)
    assert processor.enhancer.state is LiveEnhancerState.DISABLED
    assert len(errors) == 1


async def test_stereo_input_disables_loudly() -> None:
    errors: list[Exception] = []
    processor = make_processor(FakeServer(), on_error=errors.append)
    stereo = frame(24000, channels=2)
    assert processor._process(stereo) is stereo
    assert processor._process(stereo) is stereo
    assert processor.enhancer.state is LiveEnhancerState.DISABLED
    assert len(errors) == 1 and isinstance(errors[0], ValidationError)


async def test_processor_is_reusable_after_close() -> None:
    first, second = FakeServer(), FakeServer()
    processor = make_processor(first, second)
    await run_frames(processor, 24000, 3)
    processor._close()
    await settle()
    assert first.closed

    await run_frames(processor, 24000, 3)
    assert processor.enhancer.state is LiveEnhancerState.LIVE
    processor._close()
    await settle()
    assert second.closed


async def test_disabling_closes_the_stream() -> None:
    server = FakeServer()
    processor = make_processor(server)
    await run_frames(processor, 24000, 3)
    processor.enabled = False
    await settle()
    assert not processor.enabled
    assert server.closed
    assert processor.enhancer.state is LiveEnhancerState.IDLE


async def test_prewarms_at_construction_and_on_attach() -> None:
    processor = make_processor(FakeServer())
    client = processor.enhancer._client
    await settle()
    assert client.enhance.prewarms == 1

    processor._on_stream_info_updated(
        room_name="room", participant_identity="caller", publication_sid="PA_1"
    )
    await settle()
    assert client.enhance.prewarms == 2
    processor._close()


def test_construction_outside_a_loop_defers_prewarm_to_attach() -> None:
    processor = make_processor(FakeServer())
    assert processor.enhancer._client.enhance.prewarms == 0
