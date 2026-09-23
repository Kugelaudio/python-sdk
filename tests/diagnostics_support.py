"""Shared helpers for the diagnostics test modules.

Every diagnostics test injects a fake sender; the autouse fixture in
``conftest.py`` keeps the default one off the wire.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from kugelaudio._diagnostics_config import DIAGNOSTICS_PATH, DiagnosticsConfig
from kugelaudio._diagnostics import (
    Diagnostics,
)

Call = Tuple[str, Dict[str, str], bytes]

TEST_API_URL = "https://api.kugelaudio.com"
TEST_DIAGNOSTICS_URL = TEST_API_URL + DIAGNOSTICS_PATH
#: What the client would really pass through: its own auth headers, verbatim.
TEST_AUTH_HEADERS = {
    "Authorization": "Bearer test-key",
    "X-API-Key": "test-key",
}


class FakeSender:
    """Records ``(url, headers, payload)`` and answers a status code.

    ``fail=True`` raises instead, standing in for a transport failure (which
    the reporter retries once); ``status``/``statuses`` stand in for an HTTP
    response. ``delay_s`` makes every call block, like a slow network.
    """

    def __init__(
        self,
        fail: bool = False,
        status: int = 202,
        statuses: Optional[List[int]] = None,
        delay_s: float = 0.0,
    ) -> None:
        self.calls: List[Call] = []
        self.threads: List[threading.Thread] = []
        self.fail = fail
        self.status = status
        self.statuses = list(statuses) if statuses is not None else None
        self.delay_s = delay_s

    def __call__(self, url: str, headers: Dict[str, str], payload: bytes) -> int:
        self.calls.append((url, dict(headers), payload))
        self.threads.append(threading.current_thread())
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail:
            raise RuntimeError("sender exploded")
        if self.statuses:
            return self.statuses.pop(0)
        return self.status

    @property
    def batches(self) -> List[List[Dict[str, Any]]]:
        out: List[List[Dict[str, Any]]] = []
        for _url, _headers, payload in self.calls:
            body = json.loads(payload.decode("utf-8"))
            batch: List[Dict[str, Any]] = []
            for resource in body["resourceLogs"]:
                for scope in resource["scopeLogs"]:
                    batch.extend(scope["logRecords"])
            out.append(batch)
        return out

    @property
    def records(self) -> List[Dict[str, Any]]:
        return [record for batch in self.batches for record in batch]

    @property
    def raw(self) -> str:
        return "".join(payload.decode("utf-8") for _u, _h, payload in self.calls)


class ScriptedSender(FakeSender):
    """A sender whose script mixes status codes with transport failures."""

    def __init__(self, script: List[Any]) -> None:
        super().__init__()
        self.script = list(script)

    def __call__(self, url: str, headers: Dict[str, str], payload: bytes) -> int:
        self.calls.append((url, dict(headers), payload))
        item = self.script.pop(0) if self.script else 202
        if isinstance(item, BaseException):
            raise item
        return int(item)


def attrs_of(record: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for item in record["attributes"]:
        value = item["value"]
        flat[item["key"]] = value.get("stringValue", value.get("intValue"))
    return flat


def events_of(sender: FakeSender) -> List[Dict[str, Any]]:
    """Flattened attributes of every non-``sdk_stats`` record sent."""
    return [
        attrs_of(r)
        for r in sender.records
        if attrs_of(r)["kugel.event"] != "sdk_stats"
    ]


def live_config(**kw: Any) -> DiagnosticsConfig:
    """A config with a resolved target URL, so the reporter is not inert."""
    kw.setdefault("enabled", True)
    kw.setdefault("url", TEST_DIAGNOSTICS_URL)
    kw.setdefault("auth_headers", TEST_AUTH_HEADERS)
    kw.setdefault("endpoint_kind", "hosted")
    return DiagnosticsConfig(**kw)


def live_reporter(sender: FakeSender, **kw: Any) -> Diagnostics:
    """Records stay queued until ``flush()``/``close()`` hands them over."""
    return Diagnostics(
        live_config(), sender, autostart=False, version="9.9.9", **kw
    )


class FakeResponse:
    """Minimal httpx.Response stand-in."""

    def __init__(
        self,
        status_code: int,
        body: Any = None,
        headers: Optional[Dict[str, str]] = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = text
        self.content = text.encode()

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no body")
        return self._body


BLOCK = object()


class ScriptedWebSocket:
    """A ``websockets`` connection stand-in fed from a frame script.

    Each ``recv()`` pops the next item: a dict is returned as JSON, an
    exception is raised, and :data:`BLOCK` waits forever (so the session's
    own ``wait_for`` timeout fires). Sent frames are recorded as dicts.
    """

    def __init__(self, script: List[Any]) -> None:
        self.script = list(script)
        self.sent: List[Dict[str, Any]] = []
        self.close_code: Optional[int] = None

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def recv(self) -> str:
        item = self.script.pop(0) if self.script else BLOCK
        if item is BLOCK:
            await asyncio.Event().wait()
        if isinstance(item, BaseException):
            raise item
        return json.dumps(item)

    async def close(self) -> None:
        return None
