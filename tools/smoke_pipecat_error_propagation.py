"""Smoke-check KugelAudio Pipecat error propagation.

This script does not call the live KugelAudio API. It loads the installed
Pipecat integration and injects a typed SDK error into the MultiContextSession
path. Pipecat users should receive an ErrorFrame containing the KugelAudio
error_code and status.

Run from packages/public/python-sdk, for example:
    uv run --extra pipecat python tools/smoke_pipecat_error_propagation.py
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pipecat

from kugelaudio.exceptions import RateLimitError
from kugelaudio.models import AudioChunk
from kugelaudio.pipecat import KugelAudioTTSService


class _FailingMultiContextSession:
    is_alive = True

    async def connect(self) -> None:
        return None

    async def close(self) -> dict[str, Any]:
        return {"session_closed": True}

    async def send(
        self,
        _context_id: str,
        _text: str,
        flush: bool = False,
        chunk_complete_idle_timeout: float | None = None,
    ) -> AsyncIterator[AudioChunk]:
        raise RateLimitError("Rate limit exceeded")
        yield  # pragma: no cover - make this an async generator


async def main() -> None:
    service = KugelAudioTTSService(
        api_key="test-key",
        voice_id=1,
        language="en",
    )
    service._multi_session = _FailingMultiContextSession()

    frames = [frame async for frame in service.run_tts("Hello from smoke.")]
    error_frames = [frame for frame in frames if type(frame).__name__ == "ErrorFrame"]

    assert len(error_frames) == 1, [type(frame).__name__ for frame in frames]
    error_text = str(getattr(error_frames[0], "error", ""))
    assert "RATE_LIMITED" in error_text, error_text
    assert "status=429" in error_text, error_text
    assert "Rate limit exceeded" in error_text, error_text

    await service.cleanup()
    print(f"OK pipecat error propagation smoke: pipecat {pipecat.__version__}")


if __name__ == "__main__":
    asyncio.run(main())
