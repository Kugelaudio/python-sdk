"""A real HTTP stub for diagnostics wire tests.

``ThreadingHTTPServer`` on port 0, speaking HTTP/1.1 (``websockets`` rejects
HTTP/1.0 handshake responses). Every API call and every WebSocket upgrade is
rejected with 401 plus an ``x-request-id``, so each SDK call fails and queues
exactly one diagnostics event. ``POST .../v1/sdk-diagnostics`` is recorded
and answered according to the path prefix, which lets one server serve many
scenarios at once::

    http://127.0.0.1:<port>/<mode>/<anything>/v1/sdk-diagnostics

``<mode>`` is ``accept`` (202), ``404``, ``500``, ``413`` or ``hang`` (never
answers until the server shuts down).
"""

from __future__ import annotations

import json
import socket
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Iterator, List

import pytest

DIAGNOSTICS_SUFFIX = "/v1/sdk-diagnostics"
REJECT_REQUEST_ID = "rid-wire-401"
HOSTED_PROBE_HOST = "probe.kugelaudio.com"


@dataclass
class Post:
    path: str
    headers: Dict[str, str]
    body: bytes

    @property
    def records(self) -> List[Dict[str, Any]]:
        payload = json.loads(self.body)
        return [
            record
            for resource in payload["resourceLogs"]
            for scope in resource["scopeLogs"]
            for record in scope["logRecords"]
        ]


@dataclass
class DiagnosticsStub:
    port: int
    posts: List[Post] = field(default_factory=list)
    api_bodies: List[bytes] = field(default_factory=list)
    release: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def url(self, prefix: str = "accept/x", host: str = "127.0.0.1") -> str:
        return f"http://{host}:{self.port}/{prefix}"

    def posts_under(self, prefix: str) -> List[Post]:
        with self.lock:
            return [p for p in self.posts if p.path.startswith(f"/{prefix}/")]


def _handler(stub: DiagnosticsStub) -> type:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            return None

        def _reject(self) -> None:
            body = json.dumps(
                {"error": "bad key", "error_code": "UNAUTHORIZED"}
            ).encode()
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("x-request-id", REJECT_REQUEST_ID)
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length else b""

        def do_GET(self) -> None:
            self._reject()

        def do_POST(self) -> None:
            body = self._body()
            if not self.path.endswith(DIAGNOSTICS_SUFFIX):
                with stub.lock:
                    stub.api_bodies.append(body)
                self._reject()
                return
            with stub.lock:
                stub.posts.append(Post(self.path, dict(self.headers.items()), body))
            mode = self.path.split("/")[1]
            if mode == "hang":
                stub.release.wait(60)
                self.close_connection = True
                return
            status = {"accept": 202, "404": 404, "500": 500, "413": 413}[mode]
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

    return Handler


@pytest.fixture
def diagnostics_stub() -> Iterator[DiagnosticsStub]:
    stub = DiagnosticsStub(port=0)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(stub))
    server.daemon_threads = True
    stub.port = server.server_address[1]
    # A short poll interval keeps shutdown() (fixture teardown) fast.
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    try:
        yield stub
    finally:
        stub.release.set()
        server.shutdown()
        server.server_close()
        thread.join(5)


@pytest.fixture
def hosted_probe(monkeypatch: pytest.MonkeyPatch) -> Callable[[DiagnosticsStub], str]:
    """Resolve ``probe.kugelaudio.com`` to 127.0.0.1 for this test.

    Returns a function giving the stub's hosted-looking base URL, so the
    reporter's default (on for ``*.kugelaudio.com``) is exercised for real.
    """
    real = socket.getaddrinfo

    def resolve(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host == HOSTED_PROBE_HOST:
            host = "127.0.0.1"
        return real(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.delenv("KUGELAUDIO_TELEMETRY", raising=False)

    def url(stub: DiagnosticsStub, prefix: str = "accept/x") -> str:
        return stub.url(prefix, host=HOSTED_PROBE_HOST)

    return url


#: The opt-out matrix: (env KUGELAUDIO_TELEMETRY, telemetry option, hosted
#: URL?, POSTs expected). Contract "Enablement": env beats the option, the
#: option beats the hosted default.
OPT_OUT_MATRIX = [
    pytest.param(None, None, True, True, id="hosted-default-on"),
    pytest.param(None, False, True, False, id="option-false"),
    pytest.param("0", True, True, False, id="env0-beats-option-true"),
    pytest.param("0", None, True, False, id="env0-beats-hosted-default"),
    pytest.param("1", False, True, True, id="env1-beats-option-false"),
    pytest.param(None, None, False, False, id="custom-url-default-off"),
]
