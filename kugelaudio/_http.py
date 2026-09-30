"""The client's shared keep-alive HTTP connections.

The sync ``httpx.Client`` lives for the whole client. The async one is created
lazily, because an ``httpx.AsyncClient``'s sockets belong to the event loop
that opened them: :class:`SharedAsyncHttp` keeps one for the running loop and
replaces it when the caller moves to another loop (``asyncio.run`` twice).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import httpx

# Idle pooled connections stay open this long (httpx's default is 5 s), so a
# prewarmed connection is still there for a request that comes a little later.
# Kept under the 60 s idle timeout of the load balancers in front of the API.
KEEPALIVE_EXPIRY_S = 50.0
HTTP_LIMITS = httpx.Limits(
    max_connections=100, max_keepalive_connections=20, keepalive_expiry=KEEPALIVE_EXPIRY_S
)


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


class SharedAsyncHttp:
    """One lazily created keep-alive ``httpx.AsyncClient`` for the running loop."""

    def __init__(self, factory: Callable[[], httpx.AsyncClient]) -> None:
        self._factory = factory
        self._http: httpx.AsyncClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def get(self) -> httpx.AsyncClient:
        """The client for the running loop, created on first use.

        Raises:
            RuntimeError: called outside a running event loop.
        """
        loop = asyncio.get_running_loop()
        if self._http is None or self._loop is not loop:
            # RISK: two threads driving one KugelAudio client from two live
            # loops at once replace each other's client; every request still
            # finishes on the client it started with, only reuse is lost.
            self.close()
            self._http = self._factory()
            self._loop = loop
        return self._http

    async def aclose(self) -> None:
        """Close the client; awaited when it belongs to the running loop."""
        detached = self._detach()
        if detached is None:
            return
        http, loop = detached
        if loop is asyncio.get_running_loop():
            await http.aclose()
        else:
            _close_on(http, loop)

    def close(self) -> None:
        """Close the client without awaiting (sync ``KugelAudio.close``)."""
        detached = self._detach()
        if detached is None:
            return
        http, loop = detached
        if loop is _running_loop():
            # RISK: fire-and-forget; KugelAudio.aclose() awaits the close instead.
            loop.create_task(http.aclose())
        else:
            _close_on(http, loop)

    def _detach(self) -> tuple[httpx.AsyncClient, asyncio.AbstractEventLoop] | None:
        """Forget the client; ``None`` when there was none."""
        http, loop = self._http, self._loop
        self._http = self._loop = None
        if http is None or loop is None:
            return None
        return http, loop


def _close_on(http: httpx.AsyncClient, loop: asyncio.AbstractEventLoop) -> None:
    """Close ``http`` on its own loop when that loop is still running elsewhere.

    A stopped or closed loop cannot run ``aclose()``; the client is dropped and
    its sockets are released with their transports when garbage collected.
    """
    if loop.is_running() and not loop.is_closed():
        asyncio.run_coroutine_threadsafe(http.aclose(), loop)
