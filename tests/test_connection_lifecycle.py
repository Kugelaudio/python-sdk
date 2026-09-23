"""Loopback handshake ownership checks for the standalone Python sessions."""
from __future__ import annotations

import asyncio
import pytest
from kugelaudio.streaming import StreamingSession, MultiContextSession


@pytest.mark.asyncio
@pytest.mark.parametrize("session_type", [StreamingSession, MultiContextSession])
async def test_close_cancels_pending_upgrade(
    session_type: type[StreamingSession] | type[MultiContextSession],
) -> None:
    accepted = asyncio.Event()
    disconnected = asyncio.Event()
    writers: list[asyncio.StreamWriter] = []

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writers.append(writer)
        await reader.readuntil(b"\r\n\r\n")
        accepted.set()
        await reader.read()
        disconnected.set()
        writer.close()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    session = session_type(api_key="test-key", tts_url=f"http://127.0.0.1:{port}")
    connecting = asyncio.create_task(session.connect())
    try:
        await asyncio.wait_for(accepted.wait(), 2)
        await session.close()
        await asyncio.sleep(0)
        assert connecting.done(), "session.close left the handshake running"
        await asyncio.wait_for(disconnected.wait(), 1)
    finally:
        connecting.cancel()
        await asyncio.gather(connecting, return_exceptions=True)
        for writer in writers:
            writer.close()
            await writer.wait_closed()
        server.close()
        await server.wait_closed()
