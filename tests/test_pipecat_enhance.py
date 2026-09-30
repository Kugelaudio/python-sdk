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
    assert client.enhance.prewarms == 0
    await audio_filter.start(RATE)
    await settle()
    assert client.enhance.prewarms == 1
    await audio_filter.stop()
