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
    assert client.enhance.connects == 1

    processor._on_stream_info_updated(
        room_name="room", participant_identity="caller", publication_sid="PA_1"
    )
    await settle()
    assert client.enhance.connects == 2
    processor._close()


def test_construction_outside_a_loop_defers_prewarm_to_attach() -> None:
    processor = make_processor(FakeServer())
    assert processor.enhancer._client.enhance.connects == 0


# ------------------------------------ in a real rtc.AudioStream, real socket
#
# The processor runs inside LiveKit's own AudioStream (a local track fed by an
# AudioSource), against the local session server of test_enhance_session.py,
# which counts the connections it accepts.

import asyncio  # noqa: E402

from kugelaudio import KugelAudio  # noqa: E402

from .test_enhance_session import SessionServer, env, wait_for  # noqa: E402, F401 - env is a fixture

ENHANCED = 1000  # the sample value the server answers with


async def through_track(
    processor: object, rate: int, count: int, *, close_processor: bool = True
) -> list[rtc.AudioFrame]:
    """Feed ``count`` 10 ms frames at real-time pace through a local track and
    return what LiveKit's AudioStream hands on after the processor."""
    source = rtc.AudioSource(rate, 1)
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    stream = rtc.AudioStream(
        track,
        sample_rate=rate,
        num_channels=1,
        noise_cancellation=processor,  # type: ignore[arg-type]
        auto_close_noise_cancellation=close_processor,
    )

    async def feed() -> None:
        for _ in range(count):
            await source.capture_frame(frame(rate, ms=10))
            await asyncio.sleep(0.01)

    async def read() -> list[rtc.AudioFrame]:
        out: list[rtc.AudioFrame] = []
        async for event in stream:
            out.append(event.frame)
            if len(out) == count:
                break
        return out

    feeder = asyncio.create_task(feed())
    try:
        out = await asyncio.wait_for(read(), 10)
    finally:
        await feeder
        await stream.aclose()
        await source.aclose()
    return out


def assert_input_sized(frames: list[rtc.AudioFrame], rate: int, count: int) -> None:
    assert len(frames) == count
    assert {(f.sample_rate, f.num_channels, f.samples_per_channel) for f in frames} == {
        (rate, 1, rate // 100)
    }


def enhanced(frames: list[rtc.AudioFrame]) -> bool:
    return any(max(f.data) > ENHANCED // 2 for f in frames)


def session_processor(server: SessionServer, client: KugelAudio):  # noqa: ANN201 - livekit type
    from kugelaudio.livekit import enhancement

    server.reply_sample = ENHANCED
    return enhancement(client=client)


async def test_prewarm_opens_the_session_socket_before_the_first_frame(env) -> None:
    server, client = env
    processor = session_processor(server, client)
    await wait_for(lambda: server.connections == 1)
    assert server.configs == []
    out = await through_track(processor, 16000, 60)
    assert_input_sized(out, 16000, 60)
    assert enhanced(out)
    assert server.connections == 1 and processor.enhancer.failures == 0
    await processor.enhancer.aclose()


async def test_two_tracks_and_a_rate_change_share_one_socket(env) -> None:
    server, client = env
    processor = session_processor(server, client)
    first = await through_track(processor, 16000, 60)
    second = await through_track(processor, 48000, 60)
    assert_input_sized(first, 16000, 60)
    assert_input_sized(second, 48000, 60)
    assert enhanced(first) and enhanced(second)
    assert server.rates() == [16000, 48000]
    assert server.connections == 1 and processor.enhancer.failures == 0
    await processor.enhancer.aclose()


async def test_a_rate_change_mid_track_sends_a_new_config_on_the_socket(env) -> None:
    server, client = env
    processor = session_processor(server, client)
    await through_track(processor, 16000, 40, close_processor=False)
    out = await through_track(processor, 24000, 60)
    assert_input_sized(out, 24000, 60)
    assert enhanced(out)
    assert server.rates() == [16000, 24000] and server.connections == 1
    await processor.enhancer.aclose()


@pytest.mark.parametrize(
    ("server_rule", "connections"),
    [("restart_1012", 2), ("idle_close", 2), ("no_session_support", 2)],
)
async def test_a_socket_the_server_closed_is_replaced_without_losing_audio(
    env, server_rule: str, connections: int
) -> None:
    server, client = env
    if server_rule == "restart_1012":
        server.close_after_audio = 1012
    elif server_rule == "idle_close":
        server.session_idle_close_s = 0.2
    else:
        server.supports_sessions = False
    processor = session_processor(server, client)
    await wait_for(lambda: server.connections == 1)
    if server_rule == "idle_close":
        await asyncio.sleep(0.4)  # the prewarmed socket is closed by the server
    first = await through_track(processor, 16000, 60)
    second = await through_track(processor, 16000, 60)
    # Every frame comes back, at its size: pass-through while reconnecting.
    assert_input_sized(first, 16000, 60)
    assert_input_sized(second, 16000, 60)
    assert enhanced(second)
    assert len(server.configs) == 2
    assert server.connections == connections
    assert processor.enhancer.failures == 0
    await processor.enhancer.aclose()
