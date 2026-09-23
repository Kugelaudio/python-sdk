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

"""Regression tests for the LiveKit TTS plugin (KUG-1620) and the
session-wide generation options (``speed``, ``temperature``) that share
its wire rules.

Two independent defects are covered here:

1. ``speed`` was not wired into the plugin at all.  It must ride as a
   TOP-LEVEL key on the config (first) frame of a context — the ingress
   ``/ws/tts/multi`` route parses top-level ``speed`` via ``StreamUpdate``
   and *silently discards* a ``speed`` nested inside ``voice_settings``.
   Out-of-range values are rejected by the server, so the SDK raises
   instead of clamping.

2. ``prewarm()`` held the shared connection lock for its full 10 s budget,
   so a synthesis call passing its own 2.5 s ``conn_options.timeout`` waited
   up to 10 s for the lock and only then started its own 2.5 s connect —
   12.5 s against a dead API.  ``_ensure_connection(timeout)`` must bound
   lock acquisition *plus* connect by ``timeout``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any, AsyncIterator

import pytest

pytest.importorskip("livekit.agents")

import aiohttp
from aiohttp import web
from livekit.agents import APITimeoutError

from kugelaudio.livekit.tts import (
    TTS,
    _Connection,
    _SynthesizeContent,
    _TTSOptions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _opts(**overrides: Any) -> _TTSOptions:
    base: dict[str, Any] = dict(
        model="kugel-3",
        voice_id=None,
        sample_rate=24000,
        cfg_scale=2.0,
        max_new_tokens=2048,
        api_key="k",
        base_url="https://x",
    )
    base.update(overrides)
    return _TTSOptions(**base)


class _CapturingWS:
    """Minimal stand-in for aiohttp's ClientWebSocketResponse."""

    closed = False

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_str(self, payload: str) -> None:
        self.sent.append(payload)


async def _run_send_loop(
    conn: _Connection, items: list[_SynthesizeContent]
) -> list[dict[str, Any]]:
    """Push *items* through the real ``_send_loop`` and return the frames."""
    ws = _CapturingWS()
    conn._ws = ws  # type: ignore[assignment]
    for item in items:
        conn.send_content(item)
    conn._input_queue.put_nowait(None)  # sentinel stops the loop
    await conn._send_loop()
    return [json.loads(raw) for raw in ws.sent]


@contextlib.asynccontextmanager
async def _black_hole_server() -> AsyncIterator[int]:
    """A TCP server that accepts connections and never answers.

    The WS upgrade handshake therefore hangs forever, which is what an
    unreachable-but-routable API looks like to aiohttp.
    """
    writers: list[asyncio.StreamWriter] = []

    async def handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        writers.append(writer)
        try:
            await reader.read()
        finally:
            writer.close()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield port
    finally:
        for writer in writers:
            writer.close()
        # Let the closed transports detach from the server before it goes
        # away, otherwise their __del__ fires against a torn-down server.
        await asyncio.sleep(0.05)
        server.close()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(server.wait_closed(), timeout=2.0)


@contextlib.asynccontextmanager
async def _working_ws_server() -> AsyncIterator[tuple[int, list[int]]]:
    """A real WS server on /ws/tts/multi that accepts and then idles.

    Yields ``(port, accepted)`` where ``accepted`` is a one-element list
    counting completed upgrades.
    """
    accepted = [0]

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        accepted[0] += 1
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/ws/tts/multi", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    try:
        yield port, accepted
    finally:
        await runner.cleanup()


@contextlib.asynccontextmanager
async def _tts_against(port: int) -> AsyncIterator[TTS]:
    session = aiohttp.ClientSession()
    tts = TTS(
        api_key="test-key",
        base_url=f"http://127.0.0.1:{port}",
        http_session=session,
    )
    try:
        yield tts
    finally:
        with contextlib.suppress(Exception):
            await tts.aclose()
        await session.close()


@contextlib.contextmanager
def _tracked_tasks():
    """Capture tasks spawned inside the block and cancel them on exit."""
    before = asyncio.all_tasks()
    spawned: list[asyncio.Task] = []
    try:
        yield spawned
    finally:
        spawned.extend(asyncio.all_tasks() - before)


async def _cancel_all(tasks: list[asyncio.Task]) -> None:
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Fix 1 — speed on the wire
# ---------------------------------------------------------------------------


class TestSpeedWireFormat:
    """``speed`` must be a top-level config-frame key, never nested."""

    async def test_speed_is_top_level_on_the_config_frame(self) -> None:
        conn = _Connection(_opts(speed=1.15, voice_id=7), object())  # type: ignore[arg-type]
        frames = await _run_send_loop(
            conn, [_SynthesizeContent("ctx1", "Hallo", flush=True)]
        )

        assert len(frames) == 2
        config = frames[1]
        assert config["speed"] == 1.15
        assert "speed" not in config.get("voice_settings", {}), (
            "speed nested in voice_settings is silently discarded by "
            "/ws/tts/multi; it must be a top-level StreamUpdate field"
        )


    async def test_speed_only_on_the_config_frame(self) -> None:
        """Follow-up text frames for the same context carry no speed."""
        conn = _Connection(_opts(speed=0.9), object())  # type: ignore[arg-type]
        frames = await _run_send_loop(
            conn,
            [
                _SynthesizeContent("ctx1", "erste"),
                _SynthesizeContent("ctx1", "zweite"),
            ],
        )

        assert len(frames) == 2
        assert frames[0]["speed"] == 0.9
        assert "speed" not in frames[1]
        assert "model_id" not in frames[1]


class TestVoicePreparationFrame:
    async def test_selected_voice_is_prepared_before_first_text(self) -> None:
        conn = _Connection(_opts(voice_id=7), object())  # type: ignore[arg-type]
        frames = await _run_send_loop(
            conn,
            [_SynthesizeContent("first", "Hello"), _SynthesizeContent("second", "Hi")],
        )
        assert frames[0] == {"text": "", "voice_id": 7, "model_id": "kugel-3"}
        assert [frame["context_id"] for frame in frames[1:]] == ["first", "second"]
        for frame in frames[1:]:
            # Older servers ignore preparation; actual requests remain complete.
            assert frame["voice_settings"]["voice_id"] == 7
            assert frame["model_id"] == "kugel-3"

    async def test_connection_without_selected_voice_sends_no_preparation(self) -> None:
        conn = _Connection(_opts(), object())  # type: ignore[arg-type]
        assert await _run_send_loop(conn, []) == []

    async def test_idle_connection_prepares_without_waiting_for_text(self) -> None:
        conn = _Connection(_opts(voice_id=7), object())  # type: ignore[arg-type]
        assert await _run_send_loop(conn, []) == [
            {"text": "", "voice_id": 7, "model_id": "kugel-3"}
        ]

    async def test_connection_closed_before_writer_starts_does_not_prepare(self) -> None:
        conn = _Connection(_opts(voice_id=7), object())  # type: ignore[arg-type]
        conn._closed = True
        assert await _run_send_loop(conn, []) == []


class TestSpeedValidation:
    """Out-of-range speed raises — the server rejects, it does not clamp."""

    @pytest.mark.parametrize("speed", [0.79, 0.5, 1.21, 2.0, 0.0])
    def test_init_rejects_out_of_range(self, speed: float) -> None:
        with pytest.raises(ValueError, match="speed"):
            TTS(api_key="test-key", speed=speed)

    @pytest.mark.parametrize("speed", [0.8, 1.0, 1.2, None])
    def test_init_accepts_in_range(self, speed: float | None) -> None:
        tts = TTS(api_key="test-key", speed=speed)
        assert tts._opts.speed == speed


    @pytest.mark.parametrize("speed", [0.79, 1.21])
    def test_update_options_rejects_out_of_range(self, speed: float) -> None:
        tts = TTS(api_key="test-key")
        with pytest.raises(ValueError, match="speed"):
            tts.update_options(speed=speed)
        assert tts._opts.speed is None, "rejected value must not be applied"


# ---------------------------------------------------------------------------
# temperature: same session-wide, top-level, reject-don't-clamp rules as speed
# ---------------------------------------------------------------------------


class TestTemperatureWireFormat:
    """``temperature`` rides top-level on the config frame, never nested."""

    async def test_temperature_is_top_level_on_the_config_frame(self) -> None:
        conn = _Connection(_opts(temperature=0.3, voice_id=7), object())  # type: ignore[arg-type]
        frames = await _run_send_loop(
            conn, [_SynthesizeContent("ctx1", "Hallo", flush=True)]
        )

        config = frames[1]
        assert config["temperature"] == 0.3
        assert "temperature" not in config.get("voice_settings", {}), (
            "temperature nested in voice_settings is silently discarded by "
            "/ws/tts/multi; it must be a top-level StreamUpdate field"
        )

    async def test_temperature_absent_when_not_set(self) -> None:
        conn = _Connection(_opts(), object())  # type: ignore[arg-type]
        frames = await _run_send_loop(
            conn, [_SynthesizeContent("ctx1", "Hallo", flush=True)]
        )

        assert "temperature" not in frames[0]

    async def test_temperature_only_on_the_config_frame(self) -> None:
        conn = _Connection(_opts(temperature=0.0), object())  # type: ignore[arg-type]
        frames = await _run_send_loop(
            conn,
            [
                _SynthesizeContent("ctx1", "erste"),
                _SynthesizeContent("ctx1", "zweite"),
            ],
        )

        # 0.0 is a real value (most stable), not "unset"; it must be sent.
        assert frames[0]["temperature"] == 0.0
        assert "temperature" not in frames[1]


class TestTemperatureValidation:
    @pytest.mark.parametrize("temperature", [-0.01, -1.0, 1.01, 2.0])
    def test_init_rejects_out_of_range(self, temperature: float) -> None:
        with pytest.raises(ValueError, match="temperature"):
            TTS(api_key="test-key", temperature=temperature)

    @pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0, None])
    def test_init_accepts_in_range(self, temperature: float | None) -> None:
        tts = TTS(api_key="test-key", temperature=temperature)
        assert tts._opts.temperature == temperature

    def test_init_defaults_to_none(self) -> None:
        assert TTS(api_key="test-key")._opts.temperature is None

    @pytest.mark.parametrize("temperature", [-0.01, 1.01])
    def test_update_options_rejects_out_of_range(self, temperature: float) -> None:
        tts = TTS(api_key="test-key")
        with pytest.raises(ValueError, match="temperature"):
            tts.update_options(temperature=temperature)
        assert tts._opts.temperature is None, "rejected value must not be applied"

    def test_update_options_applies_and_invalidates_connection(self) -> None:
        from unittest.mock import MagicMock

        tts = TTS(api_key="test-key")
        conn = MagicMock(spec=_Connection)
        conn.is_current = True
        conn._closed = False
        tts._current_connection = conn

        tts.update_options(temperature=0.4)

        assert tts._opts.temperature == 0.4
        conn.mark_non_current.assert_called_once()
        assert tts._current_connection is None


# ---------------------------------------------------------------------------
# Fix 2 — prewarm must not monopolise the connection lock
# ---------------------------------------------------------------------------


class TestPrewarmLockContention:
    async def test_close_cancels_pending_and_queued_acquisitions(self) -> None:
        async with _black_hole_server() as port:
            async with _tts_against(port) as tts:
                pending = asyncio.create_task(tts._ensure_connection(5.0))
                queued = asyncio.create_task(tts._ensure_connection(5.0))
                try:
                    await asyncio.sleep(0.05)
                    await tts.aclose()
                    await asyncio.sleep(0)
                    assert pending.done(), "aclose left the handshake running"
                    assert queued.done(), "aclose left a queued acquisition running"
                finally:
                    pending.cancel()
                    queued.cancel()
                    await asyncio.gather(pending, queued, return_exceptions=True)

    async def test_ensure_connection_bounds_lock_wait_plus_connect(self) -> None:
        """THE regression: a 2.5 s caller must not wait behind a 10 s prewarm.

        Before the fix ``_ensure_connection`` acquired ``_connection_lock``
        unconditionally, so the caller's own 2.5 s budget only started once
        prewarm's connect gave up.  Against this black-hole server the old
        code never returns at all (aiohttp's ``sock_connect`` budget never
        fires — the TCP connect succeeds and the handshake read is
        unbounded), so the outer guard below trips instead.
        """
        async with _black_hole_server() as port:
            async with _tts_against(port) as tts:
                with _tracked_tasks() as spawned:
                    tts.prewarm(timeout=10.0)
                    # Let the prewarm task take the lock and block on connect.
                    await asyncio.sleep(0.05)

                    loop = asyncio.get_running_loop()
                    started = loop.time()
                    try:
                        with pytest.raises(APITimeoutError):
                            await asyncio.wait_for(
                                tts._ensure_connection(2.5), timeout=8.0
                            )
                        elapsed = loop.time() - started
                    finally:
                        await _cancel_all(spawned)

                assert elapsed >= 2.0, (
                    f"gave up too early ({elapsed:.2f}s); the caller's own "
                    "2.5 s budget must be honoured"
                )
                assert elapsed < 5.0, (
                    f"took {elapsed:.2f}s — the caller queued behind "
                    "prewarm's lock instead of bounding its own budget"
                )

    async def test_caller_budget_is_not_serialised_after_prewarm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same bug, scaled down and made deterministic.

        ``connect`` here always burns exactly its whole budget and fails,
        which is the 10 s / 2.5 s customer scenario at 1/10th scale: the
        caller must fail at ~0.25 s, not at ~1.25 s.
        """

        async def slow_connect(self: _Connection, timeout: float = 10.0) -> None:
            await asyncio.sleep(timeout)
            raise APITimeoutError()

        monkeypatch.setattr(_Connection, "connect", slow_connect)

        tts = TTS(api_key="test-key", http_session=aiohttp.ClientSession())
        try:
            with _tracked_tasks() as spawned:
                tts.prewarm(timeout=1.0)
                await asyncio.sleep(0.02)

                loop = asyncio.get_running_loop()
                started = loop.time()
                try:
                    with pytest.raises(APITimeoutError):
                        await tts._ensure_connection(0.25)
                    elapsed = loop.time() - started
                finally:
                    await _cancel_all(spawned)

            assert 0.2 <= elapsed < 0.7, (
                f"caller took {elapsed:.2f}s; expected ~0.25s, meaning it "
                "waited out prewarm's 1.0 s lock hold"
            )
        finally:
            await tts._session.close()  # type: ignore[union-attr]

    async def test_timed_out_waiter_leaves_the_lock_usable(self) -> None:
        """A waiter that gives up must not strand the lock or a successor."""
        async with _working_ws_server() as (port, accepted):
            async with _tts_against(port) as tts:
                await tts._connection_lock.acquire()
                try:
                    # Outer guard: before the fix this blocks forever on the
                    # unbounded ``async with self._connection_lock``.
                    with pytest.raises(APITimeoutError):
                        await asyncio.wait_for(
                            tts._ensure_connection(0.2), timeout=3.0
                        )
                finally:
                    tts._connection_lock.release()

                conn = await asyncio.wait_for(
                    tts._ensure_connection(5.0), timeout=5.0
                )

                assert conn is tts._current_connection
                assert accepted[0] == 1
                assert not tts._connection_lock.locked()

    async def test_acquire_that_won_the_cancel_race_is_released(self) -> None:
        """Deterministic cover for the acquired-then-cancelled edge case.

        ``asyncio.wait_for``/``asyncio.wait`` can time out in the same loop
        iteration in which ``Lock.acquire`` was granted.  Cancelling a
        *completed* acquire is a no-op, so the lock would stay held forever
        unless the timeout path hands it back explicitly.
        """
        tts = TTS(api_key="test-key")
        acquire_task = asyncio.ensure_future(tts._connection_lock.acquire())
        await acquire_task
        assert tts._connection_lock.locked()

        await tts._release_stranded_acquire(acquire_task)

        assert not tts._connection_lock.locked()

    # NB: deliberately not named "concurrent_*" — the CI python-sdk job runs
    # pytest with -k "not concurrent", which would silently deselect it.
    # (That job does not run this file at all today: it syncs without the
    # `livekit` extra, and its -k "not live" substring-matches "livekit".)
    async def test_parallel_callers_open_exactly_one_socket(self) -> None:
        async with _working_ws_server() as (port, accepted):
            async with _tts_against(port) as tts:
                conns = await asyncio.gather(
                    *[tts._ensure_connection(5.0) for _ in range(8)]
                )

                assert accepted[0] == 1
                assert len({id(c) for c in conns}) == 1
                assert conns[0] is tts._current_connection
