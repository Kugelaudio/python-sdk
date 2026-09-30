"""Tests for KugelAudioEnhanceFilter, the Pipecat input audio filter.

Requires: pip install kugelaudio[pipecat]. The enhancement stream is a
scripted fake; nothing touches the network.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import FilterEnableFrame  # noqa: E402

from kugelaudio._enhance_live import LiveEnhancerState  # noqa: E402
from kugelaudio.exceptions import AuthenticationError  # noqa: E402

from .test_enhance_realtime import FakeClient, FakeServer, settle  # noqa: E402

RATE = 16000
FRAME = b"\x01\x00" * (RATE // 50)  # 20 ms


def make_filter(*servers: FakeServer, **kwargs: object):  # noqa: ANN201 - pipecat type
    from kugelaudio.pipecat import KugelAudioEnhanceFilter

    return KugelAudioEnhanceFilter(client=FakeClient(*servers), **kwargs)  # type: ignore[arg-type]  # fake client


async def run_frames(audio_filter, count: int) -> list[bytes]:  # noqa: ANN001 - pipecat type
    out = []
    for _ in range(count):
        out.append(await audio_filter.filter(FRAME))
        await settle()
    return out


def test_lazy_export() -> None:
    import kugelaudio.pipecat as package
    from kugelaudio.pipecat.enhance import KugelAudioEnhanceFilter

    assert package.KugelAudioEnhanceFilter is KugelAudioEnhanceFilter


def test_requires_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from kugelaudio.pipecat import KugelAudioEnhanceFilter

    monkeypatch.delenv("KUGELAUDIO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="KUGELAUDIO_API_KEY"):
        KugelAudioEnhanceFilter()


def test_rejects_client_and_api_key_together() -> None:
    with pytest.raises(ValueError, match="not both"):
        make_filter(api_key="ka_test")


def test_builds_a_client_from_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from kugelaudio.pipecat import KugelAudioEnhanceFilter

    monkeypatch.setenv("KUGELAUDIO_API_KEY", "ka_env")
    audio_filter = KugelAudioEnhanceFilter()
    assert audio_filter._client._api_key == "ka_env"


def test_plugs_into_transport_params() -> None:
    from pipecat.transports.base_transport import TransportParams

    audio_filter = make_filter()
    assert TransportParams(audio_in_filter=audio_filter).audio_in_filter is audio_filter


async def test_filter_before_start_raises() -> None:
    with pytest.raises(RuntimeError, match="before start"):
        await make_filter().filter(FRAME)


async def test_every_frame_keeps_its_size_and_enhanced_audio_arrives() -> None:
    audio_filter = make_filter(FakeServer())
    await audio_filter.start(RATE)

    out = await run_frames(audio_filter, 40)

    assert all(len(chunk) == len(FRAME) for chunk in out)
    assert out[0] == FRAME  # warming: the original passes through
    assert audio_filter.enhancer.state is LiveEnhancerState.LIVE
    tail = b"".join(out[-5:])
    samples = [int.from_bytes(tail[i : i + 2], "little", signed=True) for i in range(0, len(tail), 2)]
    assert max(samples) > 1  # the fake's enhanced value (16), resampled to 16 kHz
    await audio_filter.stop()


async def test_fatal_error_returns_the_original_audio() -> None:
    errors: list[Exception] = []
    audio_filter = make_filter(
        FakeServer(connect_error=AuthenticationError()), on_error=errors.append
    )
    await audio_filter.start(RATE)

    out = await run_frames(audio_filter, 5)

    assert out == [FRAME] * 5
    assert audio_filter.enhancer.state is LiveEnhancerState.DISABLED
    assert len(errors) == 1
    await audio_filter.stop()


async def test_disable_drains_then_passes_through_and_enable_reopens() -> None:
    first, second = FakeServer(), FakeServer()
    audio_filter = make_filter(first, second)
    await audio_filter.start(RATE)
    await run_frames(audio_filter, 20)

    await audio_filter.process_frame(FilterEnableFrame(enable=False))
    await settle()
    drained = await audio_filter.filter(FRAME)
    assert len(drained) >= len(FRAME)  # the delayed audio, then this frame
    assert drained.endswith(FRAME)
    assert await audio_filter.filter(FRAME) == FRAME
    assert first.closed

    await audio_filter.process_frame(FilterEnableFrame(enable=True))
    await run_frames(audio_filter, 3)
    assert audio_filter.enhancer.state is LiveEnhancerState.LIVE
    await audio_filter.stop()
    assert second.closed


async def test_start_prewarms_the_connection() -> None:
    audio_filter = make_filter(FakeServer())
    client = audio_filter._client
    assert client.enhance.connects == 0
    await audio_filter.start(RATE)
    await settle()
    assert client.enhance.connects == 1
    await audio_filter.stop()


# ------------------------------------------ in a real pipeline, real socket
#
# The filter runs as TransportParams(audio_in_filter=...) of an input
# transport inside a real Pipecat pipeline (PipelineTask + PipelineRunner),
# against the local session server of test_enhance_session.py, which counts
# the connections it accepts. The input sample rate is fixed per pipeline run
# (the transport calls filter.start(sample_rate) once), so a rate change is
# covered on LiveKit and in test_enhance_session.py.

import asyncio  # noqa: E402
from typing import Any  # noqa: E402

from kugelaudio import KugelAudio  # noqa: E402

from .test_enhance_session import SessionServer, env, wait_for  # noqa: E402, F401 - env is a fixture

ENHANCED = 1000  # the sample value the server answers with


class PipelineRun:
    """A running pipeline: an input transport with the filter, then a sink."""

    def __init__(self, audio_filter: Any, rate: int) -> None:
        from pipecat.frames.frames import InputAudioRawFrame
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.processors.frame_processor import FrameProcessor
        from pipecat.transports.base_input import BaseInputTransport
        from pipecat.transports.base_transport import TransportParams

        class Microphone(BaseInputTransport):
            async def start(self, frame: Any) -> None:
                await super().start(frame)
                # Pipecat >= 0.0.7x starts the audio task once the transport
                # says it is ready; 0.0.62 starts it in start().
                ready = getattr(self, "set_transport_ready", None)
                if ready is not None:
                    await ready(frame)

        class Sink(FrameProcessor):
            def __init__(self) -> None:
                super().__init__()
                self.audio: list[bytes] = []

            async def process_frame(self, frame: Any, direction: Any) -> None:
                await super().process_frame(frame, direction)
                if isinstance(frame, InputAudioRawFrame):
                    self.audio.append(frame.audio)
                await self.push_frame(frame, direction)

        self.rate = rate
        self.frame_type = InputAudioRawFrame
        self.microphone = Microphone(
            TransportParams(
                audio_in_enabled=True, audio_in_sample_rate=rate, audio_in_filter=audio_filter
            )
        )
        self.sink = Sink()
        self.task = PipelineTask(
            Pipeline([self.microphone, self.sink]),
            params=PipelineParams(audio_in_sample_rate=rate),
            idle_timeout_secs=None,
        )
        self.runner = PipelineRunner(handle_sigint=False)
        self.running = asyncio.create_task(self.runner.run(self.task))

    async def speak(self, count: int, frame_ms: int = 20) -> list[bytes]:
        """Push ``count`` frames at real-time pace; the frames the sink got."""
        chunk = b"\x01\x00" * (self.rate * frame_ms // 1000)
        before = len(self.sink.audio)
        for _ in range(count):
            await self.microphone.push_audio_frame(
                self.frame_type(audio=chunk, sample_rate=self.rate, num_channels=1)
            )
            await asyncio.sleep(frame_ms / 1000)
        await wait_for(lambda: len(self.sink.audio) >= before + count, timeout=5)
        return self.sink.audio[before:]

    async def end(self) -> None:
        from pipecat.frames.frames import EndFrame

        await self.task.queue_frame(EndFrame())
        await asyncio.wait_for(self.running, 10)


def enhanced(chunks: list[bytes]) -> bool:
    return any(
        int.from_bytes(c[i : i + 2], "little", signed=True) > ENHANCED // 2
        for c in chunks
        for i in range(0, len(c), 2)
    )


def session_filter(server: SessionServer, client: KugelAudio):  # noqa: ANN201 - pipecat type
    from kugelaudio.pipecat import KugelAudioEnhanceFilter

    server.reply_sample = ENHANCED
    return KugelAudioEnhanceFilter(client=client)


@pytest.mark.parametrize(
    ("server_rule", "connections"),
    [
        ("session", 1),
        ("restart_1012", 2),
        # Closed while prewarmed, and again during the muted gap.
        ("idle_close", 3),
        ("no_session_support", 2),
    ],
)
async def test_a_pipeline_streams_two_utterances_over_the_session(
    env, server_rule: str, connections: int
) -> None:
    server, client = env
    if server_rule == "restart_1012":
        server.close_after_audio = 1012
    elif server_rule == "idle_close":
        server.session_idle_close_s = 0.2
    elif server_rule == "no_session_support":
        server.supports_sessions = False
    audio_filter = session_filter(server, client)
    run = PipelineRun(audio_filter, RATE)
    # start() prewarms: the socket is open before the first frame.
    await wait_for(lambda: server.connections == 1)
    assert server.configs == []
    enhancer = audio_filter.enhancer
    assert enhancer is not None
    enhancer._idle_close_s = 0.2  # a muted gap this long ends the stream
    if server_rule == "idle_close":
        await asyncio.sleep(0.4)  # the prewarmed socket is closed by the server

    first = await run.speak(40)
    await asyncio.sleep(0.5)  # muted: no frames, the stream ends
    second = await run.speak(40)
    await run.end()

    # Every frame comes back, at its size: pass-through while reconnecting.
    assert [len(c) for c in first + second] == [len(FRAME)] * 80
    assert enhanced(first) and enhanced(second)
    assert len(server.configs) == 2
    assert server.connections == connections
    assert enhancer.failures == 0
