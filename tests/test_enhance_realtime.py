"""Tests for kugelaudio._enhance_live: the framework-free real-time enhancer.

``client.enhance.stream`` is replaced by scripted fakes; nothing touches the
network. The file name avoids the substring "live" on purpose: the
python-sdk CI unit deselects ``-k "not live"``, which matches module names.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest

from kugelaudio._enhance_live import (
    LiveEnhancer,
    LiveEnhancerState,
    LiveSegment,
    PlayoutBuffer,
)
from kugelaudio.exceptions import (
    AuthenticationError,
    InsufficientCreditsError,
    RateLimitError,
    ValidationError,
)
from kugelaudio.exceptions import ConnectionError as KugelAudioConnectionError

RATE = 16000
FRAME = b"\x01\x00" * (RATE // 50)  # 20 ms of original audio, sample value 1
ENHANCED_SAMPLE = b"\x10\x00"  # every enhanced sample the fake emits is 16
ENHANCED_FRAME = ENHANCED_SAMPLE * 4800  # 200 ms at 24 kHz


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@dataclass
class FakeServer:
    """Stands in for client.enhance.stream: 200 ms out per 200 ms in."""

    connect_error: Exception | None = None
    hang_before_ready: bool = False
    fail_after_s: float | None = None
    fail_error: Exception | None = None
    silent: bool = False
    end_after_frames: int | None = None
    received: bytearray = field(default_factory=bytearray)
    closed: bool = False

    async def run(self, audio: AsyncIterator[bytes], rate: int) -> AsyncIterator[bytes]:
        try:
            if self.connect_error is not None:
                raise self.connect_error
            if self.hang_before_ready:
                await asyncio.Event().wait()
            frame_bytes = 2 * rate * 200 // 1000
            pending = 0
            frames = 0
            async for chunk in audio:
                self.received += chunk
                pending += len(chunk)
                if (
                    self.fail_after_s is not None
                    and len(self.received) / 2 / rate >= self.fail_after_s
                ):
                    raise self.fail_error or KugelAudioConnectionError("dropped")
                while pending >= frame_bytes and not self.silent:
                    pending -= frame_bytes
                    frames += 1
                    yield ENHANCED_FRAME
                    if self.end_after_frames == frames:
                        return
        finally:
            self.closed = True


class FakeEnhance:
    def __init__(self, servers: list[FakeServer]) -> None:
        self.servers = servers
        self.calls: list[dict[str, object]] = []
        self.prewarms = 0
        self.prewarm_error: Exception | None = None
        self.prewarm_hangs = False

    async def prewarm(self) -> None:
        self.prewarms += 1
        if self.prewarm_hangs:
            await asyncio.Event().wait()
        if self.prewarm_error is not None:
            raise self.prewarm_error

    def stream(
        self,
        audio: AsyncIterator[bytes],
        *,
        model: str,
        sample_rate: int | None = None,
        speaker: object | None = None,
    ) -> AsyncIterator[bytes]:
        self.calls.append({"model": model, "sample_rate": sample_rate, "speaker": speaker})
        server = self.servers[len(self.calls) - 1]
        assert sample_rate is not None
        return server.run(audio, sample_rate)


class FakeClient:
    def __init__(self, *servers: FakeServer) -> None:
        self.enhance = FakeEnhance(list(servers))


async def settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


def seconds(segments: list[LiveSegment]) -> float:
    return sum(len(s.pcm) / 2 / s.sample_rate for s in segments)


def make(
    *servers: FakeServer, clock: Clock | None = None, **kwargs: object
) -> tuple[LiveEnhancer, FakeClient]:
    client = FakeClient(*servers)
    enhancer = LiveEnhancer(client, clock=clock or Clock(), **kwargs)  # type: ignore[arg-type]  # fake client
    return enhancer, client


async def push_frames(enhancer: LiveEnhancer, count: int) -> None:
    for _ in range(count):
        enhancer.push(FRAME, RATE)
        await settle()


# ---------------------------------------------------------------- happy path


async def test_passes_original_through_until_ready_then_returns_enhanced() -> None:
    enhancer, client = make(FakeServer())

    enhancer.push(FRAME, RATE)
    assert enhancer.state is LiveEnhancerState.WARMING
    assert enhancer.pull() == [LiveSegment(FRAME, RATE, enhanced=False)]

    await settle()
    assert enhancer.state is LiveEnhancerState.LIVE
    assert client.enhance.calls == [{"model": "clarity-1", "sample_rate": RATE, "speaker": None}]

    await push_frames(enhancer, 12)  # 240 ms in: one 200 ms burst out
    assert enhancer.pull() == [LiveSegment(ENHANCED_FRAME, 24000, enhanced=True)]
    await enhancer.aclose()


async def test_output_timeline_matches_input_across_a_failure() -> None:
    server = FakeServer(fail_after_s=0.3)
    enhancer, _ = make(server)
    enhancer.push(FRAME, RATE)
    await settle()

    await push_frames(enhancer, 20)  # fails at 300 ms, after one 200 ms burst
    await push_frames(enhancer, 5)  # reconnect backoff: pass-through

    output = enhancer.pull()
    assert enhancer.state is LiveEnhancerState.RECONNECTING
    assert seconds(output) == pytest.approx(26 * 0.02)
    kinds = [s.enhanced for s in output]
    assert kinds[:2] == [False, True]  # warm-up frame, then the enhanced frame
    assert not any(kinds[2:])  # unanswered input came back as original, in order


# ------------------------------------------------------------------ failures


@pytest.mark.parametrize(
    "error",
    [
        AuthenticationError(),
        InsufficientCreditsError(),
        ValidationError("speaker sample too short"),
    ],
    ids=["auth", "credits", "validation"],
)
async def test_fatal_error_disables_for_the_session(
    error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    clock = Clock()
    errors: list[Exception] = []
    enhancer, client = make(
        FakeServer(connect_error=error), clock=clock, on_error=errors.append
    )
    enhancer.push(FRAME, RATE)
    await settle()

    clock.now = 3600.0
    await push_frames(enhancer, 3)

    assert enhancer.state is LiveEnhancerState.DISABLED
    assert enhancer.fatal_error is error
    assert errors == [error]
    assert len(client.enhance.calls) == 1
    assert all(not s.enhanced for s in enhancer.pull())
    assert [r.levelname for r in caplog.records if r.name == "kugelaudio.enhance"] == ["ERROR"]


async def test_validation_error_raised_by_stream_call_disables() -> None:
    enhancer, client = make()

    def reject(*args: object, **kwargs: object) -> AsyncIterator[bytes]:
        raise ValidationError('model must be a non-empty string, e.g. model="clarity-1".')

    client.enhance.stream = reject  # type: ignore[method-assign]  # scripted failure
    enhancer.push(FRAME, RATE)
    await settle()
    assert enhancer.state is LiveEnhancerState.DISABLED


async def test_unexpected_error_disables_with_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    enhancer, _ = make(FakeServer(connect_error=RuntimeError("bug")))
    enhancer.push(FRAME, RATE)
    await settle()
    assert enhancer.state is LiveEnhancerState.DISABLED
    record = next(r for r in caplog.records if r.levelname == "ERROR")
    assert record.exc_info is not None


async def test_rate_limit_honours_retry_after(caplog: pytest.LogCaptureFixture) -> None:
    clock = Clock()
    enhancer, client = make(
        FakeServer(connect_error=RateLimitError(retry_after=3)), FakeServer(), clock=clock
    )
    enhancer.push(FRAME, RATE)
    await settle()
    assert enhancer.state is LiveEnhancerState.RECONNECTING
    assert enhancer.failures == 1

    clock.now = 2.9
    await push_frames(enhancer, 1)
    assert len(client.enhance.calls) == 1

    clock.now = 3.1
    await push_frames(enhancer, 1)
    assert len(client.enhance.calls) == 2
    assert enhancer.state is LiveEnhancerState.LIVE
    assert any(r.levelname == "WARNING" and "retrying in 3.0s" in r.getMessage() for r in caplog.records)
    await enhancer.aclose()


async def test_backoff_doubles_and_resets_after_enhanced_audio() -> None:
    clock = Clock()
    down = KugelAudioConnectionError("unreachable")
    enhancer, client = make(
        FakeServer(connect_error=down),
        FakeServer(connect_error=down),
        FakeServer(),
        clock=clock,
    )
    enhancer.push(FRAME, RATE)
    await settle()  # failure 1: 0.5 s
    clock.now = 0.6
    await push_frames(enhancer, 1)  # failure 2: 1.0 s
    clock.now = 1.5
    await push_frames(enhancer, 1)
    assert len(client.enhance.calls) == 2
    clock.now = 1.7
    await push_frames(enhancer, 1)
    assert len(client.enhance.calls) == 3
    await push_frames(enhancer, 12)
    assert any(s.enhanced for s in enhancer.pull())
    assert enhancer._attempt == 0
    await enhancer.aclose()


async def test_backlog_overflow_falls_back_and_keeps_every_sample(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = FakeServer(silent=True)
    enhancer, _ = make(server, max_backlog_s=0.5)
    enhancer.push(FRAME, RATE)
    await settle()
    await push_frames(enhancer, 30)

    assert enhancer.state is LiveEnhancerState.RECONNECTING
    assert enhancer.failures == 1
    assert seconds(enhancer.pull()) == pytest.approx(31 * 0.02)
    assert server.closed
    assert any("fell 0.5s behind" in r.getMessage() for r in caplog.records)


async def test_ready_timeout_reconnects() -> None:
    clock = Clock()
    server = FakeServer(hang_before_ready=True)
    enhancer, _ = make(server, clock=clock, ready_timeout_s=5.0)
    enhancer.push(FRAME, RATE)
    await settle()
    assert enhancer.state is LiveEnhancerState.WARMING

    clock.now = 5.1
    await push_frames(enhancer, 1)
    assert enhancer.state is LiveEnhancerState.RECONNECTING
    assert server.closed


async def test_stream_ending_early_is_a_transient_failure() -> None:
    enhancer, _ = make(FakeServer(end_after_frames=1))
    enhancer.push(FRAME, RATE)
    await settle()
    await push_frames(enhancer, 13)
    assert enhancer.state is LiveEnhancerState.RECONNECTING
    assert enhancer.failures == 1


async def test_broken_on_error_callback_is_logged_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def explode(error: Exception) -> None:
        raise RuntimeError("callback bug")

    enhancer, _ = make(FakeServer(connect_error=AuthenticationError()), on_error=explode)
    enhancer.push(FRAME, RATE)
    await settle()
    assert enhancer.state is LiveEnhancerState.DISABLED
    assert any("on_error callback raised" in r.getMessage() for r in caplog.records)


# ----------------------------------------------------------------- lifecycle


async def test_sample_rate_change_reopens_the_stream() -> None:
    first, second = FakeServer(), FakeServer()
    enhancer, client = make(first, second)
    enhancer.push(FRAME, RATE)
    await settle()
    enhancer.push(b"\x01\x00" * 960, 48000)
    await settle()
    assert [c["sample_rate"] for c in client.enhance.calls] == [RATE, 48000]
    assert first.closed
    assert enhancer.failures == 0
    await enhancer.aclose()


async def test_rotates_before_the_one_hour_limit() -> None:
    clock = Clock()
    enhancer, client = make(FakeServer(), FakeServer(), clock=clock)
    enhancer.push(FRAME, RATE)
    await settle()
    clock.now = 3541.0
    await push_frames(enhancer, 1)
    assert len(client.enhance.calls) == 2
    assert enhancer.failures == 0
    await enhancer.aclose()


async def test_idle_stream_is_closed_without_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = FakeServer()
    enhancer = LiveEnhancer(FakeClient(server), idle_close_s=0.05)  # type: ignore[arg-type]  # fake client
    enhancer.push(FRAME, RATE)
    await settle()
    assert enhancer.state is LiveEnhancerState.LIVE
    await asyncio.sleep(0.15)
    await settle()
    assert enhancer.state is LiveEnhancerState.IDLE
    assert server.closed
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


async def test_pause_returns_unanswered_input_and_close_drops_it() -> None:
    enhancer, _ = make(FakeServer(), FakeServer())
    enhancer.push(FRAME, RATE)
    await settle()
    await push_frames(enhancer, 3)
    enhancer.pull()

    enhancer.pause()
    assert seconds(enhancer.pull()) == pytest.approx(0.06)
    assert enhancer.state is LiveEnhancerState.IDLE

    enhancer.push(FRAME, RATE)
    await settle()
    await push_frames(enhancer, 3)
    server_task = enhancer._stream.task if enhancer._stream else None
    await enhancer.aclose()
    assert enhancer.pull() == []
    assert server_task is not None and server_task.done()


async def test_rejects_odd_length_pcm() -> None:
    enhancer, _ = make(FakeServer())
    with pytest.raises(ValueError, match="2-byte samples"):
        enhancer.push(b"\x00", RATE)


def test_rejects_non_positive_limits() -> None:
    with pytest.raises(ValueError, match="max_backlog_s"):
        LiveEnhancer(FakeClient(), max_backlog_s=0)  # type: ignore[arg-type]  # fake client


# ------------------------------------------------------------- playout buffer


def test_playout_returns_exact_sizes_and_counts_underruns() -> None:
    buffer = PlayoutBuffer(RATE, max_delay_s=1.0)
    assert buffer.read(640) == bytes(640)
    assert buffer.underrun_s == pytest.approx(0.02)
    buffer.write(b"\x05\x00" * 500)
    out = buffer.read(640)
    assert out == b"\x05\x00" * 320
    assert len(buffer) == 360


def test_playout_drops_oldest_past_max_delay(caplog: pytest.LogCaptureFixture) -> None:
    buffer = PlayoutBuffer(RATE, max_delay_s=0.1)
    buffer.write(b"\x01\x00" * 1600 + b"\x02\x00" * 800)
    assert len(buffer) == 3200
    assert buffer.drain().startswith(b"\x01\x00" * 800 + b"\x02\x00")
    assert buffer.dropped_s == pytest.approx(0.05)
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_playout_trims_excess_delay_only_in_quiet_windows() -> None:
    buffer = PlayoutBuffer(RATE, max_delay_s=5.0)
    loud = b"\x00\x40" * 160  # 10 ms, sample 16384
    quiet = bytes(320)
    buffer.write((loud + quiet) * 50)  # 1 s buffered, half of it quiet
    for _ in range(250):  # 5 s of 20 ms reads, refilled to keep the level
        buffer.read(640)
        buffer.write(loud + quiet)
    assert buffer.trimmed_s > 0
    kept = buffer.drain()
    assert kept.count(loud) == 50  # speech is never trimmed


async def test_enhancer_through_playout_keeps_frame_size() -> None:
    enhancer, _ = make(FakeServer())
    buffer = PlayoutBuffer(RATE, max_delay_s=2.5)
    for _ in range(40):
        enhancer.push(FRAME, RATE)
        await settle()
        for segment in enhancer.pull():
            if segment.sample_rate == RATE:
                buffer.write(segment.pcm)
            else:
                buffer.write(_to_16k(segment.pcm))
        assert len(buffer.read(len(FRAME))) == len(FRAME)
    await enhancer.aclose()


def _to_16k(pcm: bytes) -> bytes:
    """24 kHz -> 16 kHz by keeping 2 of every 3 samples; enough for sizing."""
    samples = [pcm[i : i + 2] for i in range(0, len(pcm), 2)]
    return b"".join(s for i, s in enumerate(samples) if i % 3 != 2)


@dataclass
class SlowCloseServer(FakeServer):
    """A server whose socket takes a while to close once its stream is cancelled."""

    close_delay_s: float = 0.05
    cleanup_done: bool = False

    async def run(self, audio: AsyncIterator[bytes], rate: int) -> AsyncIterator[bytes]:
        try:
            async for out in super().run(audio, rate):
                yield out
        finally:
            await asyncio.shield(asyncio.sleep(self.close_delay_s))
            self.cleanup_done = True


async def test_aclose_waits_for_replaced_streams_still_closing() -> None:
    first = SlowCloseServer()
    enhancer, client = make(first, FakeServer())
    enhancer.push(FRAME, RATE)
    await settle()
    # A new sample rate replaces the first stream; its cleanup is still running.
    enhancer.push(b"\x01\x00" * 960, 48000)
    await settle()
    assert len(client.enhance.calls) == 2
    await enhancer.aclose()
    assert first.cleanup_done


# ------------------------------------------------------------------ prewarm


async def test_prewarm_warms_in_the_background_and_audio_still_passes_through() -> None:
    enhancer, client = make(FakeServer())
    client.enhance.prewarm_hangs = True

    enhancer.prewarm()
    enhancer.prewarm()  # a no-op while the first is still running
    enhancer.push(FRAME, RATE)
    assert enhancer.pull() == [LiveSegment(FRAME, RATE, enhanced=False)]
    await settle()
    assert client.enhance.prewarms == 1
    assert enhancer.state is LiveEnhancerState.LIVE

    await asyncio.wait_for(enhancer.aclose(), 1)  # cancels the hanging prewarm


async def test_prewarm_again_after_the_first_finished() -> None:
    enhancer, client = make()
    enhancer.prewarm()
    await settle()
    enhancer.prewarm()
    await settle()
    assert client.enhance.prewarms == 2


def test_prewarm_outside_an_event_loop_is_skipped() -> None:
    enhancer, client = make()
    enhancer.prewarm()
    assert client.enhance.prewarms == 0


async def test_prewarm_bug_is_logged_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    enhancer, client = make()
    client.enhance.prewarm_error = RuntimeError("boom")
    with caplog.at_level(logging.WARNING, logger="kugelaudio.enhance"):
        enhancer.prewarm()
        await settle()
    assert "Prewarming the enhancement connection failed" in caplog.text
    assert enhancer.state is LiveEnhancerState.IDLE
