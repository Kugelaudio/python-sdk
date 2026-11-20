"""The credential of a WebSocket upgrade: an ``Authorization: Bearer`` header.

The key never goes into the URL, where proxies and access logs would record it.
"""

from __future__ import annotations

from typing import Any

from websockets.uri import WebSocketURI, parse_uri


def _websockets_major() -> int:
    from websockets import version

    return int(version.version.split(".")[0])


def bearer_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def async_ws_auth_kwargs(url: str, api_key: str) -> dict[str, Any]:
    """Carry the bearer header only when a handshake targets the original origin."""
    original = parse_uri(url)
    headers = bearer_headers(api_key)

    def same_origin(target: WebSocketURI) -> bool:
        return (target.secure, target.host, target.port) == (
            original.secure, original.host, original.port
        )

    if _websockets_major() >= 14:
        from websockets.asyncio.client import ClientConnection

        class OriginBoundConnection(ClientConnection):
            async def handshake(
                self, additional_headers: Any = None, *args: Any, **kwargs: Any
            ) -> None:
                if not same_origin(self.protocol.uri):
                    additional_headers = None
                await super().handshake(additional_headers, *args, **kwargs)

        return {"additional_headers": headers, "create_connection": OriginBoundConnection}

    from websockets.legacy.client import WebSocketClientProtocol

    class OriginBoundProtocol(WebSocketClientProtocol):
        async def handshake(
            self, wsuri: WebSocketURI, *, extra_headers: Any = None, **kwargs: Any
        ) -> None:
            if not same_origin(wsuri):
                # Legacy websockets reads the instance attribute as well as the argument.
                self.extra_headers = extra_headers = None
            await super().handshake(wsuri, extra_headers=extra_headers, **kwargs)

    return {"extra_headers": headers, "create_protocol": OriginBoundProtocol}
