"""Client-error diagnostics for the KugelAudio Python SDK.

A dependency-free OTLP/HTTP-JSON log reporter plus the ``Operation`` seam
that public SDK entry points wrap themselves in. Implements the normative
contract ``services/ingress/docs/sdk-diagnostics-contract.md`` (v3), shared
with the JS and Java SDKs: attribute names, enablement order, delivery bounds
and the injectable-sender shape are fixed there. Do not diverge without
changing all three SDKs and that document.

Enablement and the target URL live in :mod:`kugelaudio._diagnostics_config`.
Delivery goes to ``<the client's own API base URL>/v1/sdk-diagnostics``,
authenticated with the client's existing auth headers. There is no endpoint
override: the auth headers must never be sent anywhere else. Tests inject a
sender instead.

Design in one paragraph: every unit of work mints an :class:`Operation`
carrying ``operation_id``, start time, transport, chunk/byte counters and a
retry count. On failure the operation reports **exactly one** event and the
exception continues unchanged. A caller-initiated cancellation is not a
fault: it emits no record and only bumps a counter reported in ``sdk_stats``.
Events are encoded against a strict attribute allowlist, queued in a bounded
queue and delivered from a daemon thread. All network I/O happens on that
thread, so telemetry never blocks a synthesis call or ``close()``, never
delays process exit, and never surfaces an error to the caller.

Nothing here is exported from ``kugelaudio/__init__.py``: the only public
surface is the ``telemetry=`` constructor option on
:class:`~kugelaudio.KugelAudio` and the ``KUGELAUDIO_TELEMETRY`` env var.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import weakref
from collections import deque
from types import TracebackType
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Type

from kugelaudio._diagnostics_config import DiagnosticsConfig
from kugelaudio._sdk_metadata import SDK_NAME, sdk_version

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Contract constants
# --------------------------------------------------------------------------

#: An ingress that predates the diagnostics route answers 404. Three in a row
#: and the reporter switches itself off for the rest of the process.
NOT_FOUND_STATUS = 404
MAX_CONSECUTIVE_NOT_FOUND = 3
#: Ingress rejects an oversized batch with 413. Resending the same bytes can
#: only be rejected again, so it is never retried.
PAYLOAD_TOO_LARGE_STATUS = 413

QUEUE_CAPACITY = 64
#: At most this many records per POST, on every path including close().
BATCH_SIZE = 8
FLUSH_INTERVAL_S = 5.0
REQUEST_TIMEOUT_S = 3.0
MAX_SEND_ATTEMPTS = 2  # initial try + at most 1 retry
CLOSE_DEADLINE_S = 1.0

# ``kugel.event`` values
EVENT_CONNECTION_FAILED = "connection_failed"
EVENT_REQUEST_FAILED = "request_failed"
EVENT_STREAM_INTERRUPTED = "stream_interrupted"
EVENT_RETRY_EXHAUSTED = "retry_exhausted"
EVENT_SDK_STATS = "sdk_stats"

# ``kugel.operation`` values
OP_GENERATE = "generate"
OP_STREAM = "stream"
OP_STREAM_SESSION = "stream_session"
OP_MULTI_CONTEXT = "multi_context"
OP_TRANSCRIBE = "transcribe"
OP_VOICES = "voices"
OP_DICTIONARIES = "dictionaries"
OP_MODELS = "models"

# ``kugel.failure_stage`` values
STAGE_CONNECTING = "connecting"
STAGE_HANDSHAKE = "handshake"
STAGE_SENDING_REQUEST = "sending_request"
STAGE_AWAITING_FIRST_AUDIO = "awaiting_first_audio"
STAGE_RECEIVING_AUDIO = "receiving_audio"
STAGE_FINALIZING = "finalizing"

#: The only ``kugel.outcome`` value. A cancellation emits no record at all,
#: and a recovered retry is folded into the operation's ``retry_count``.
OUTCOME_FAILED = "failed"

TRANSPORT_HTTP = "http"
TRANSPORT_WEBSOCKET = "websocket"

# ``kugel.integration`` values: set per client, never process-global.
INTEGRATION_NONE = "none"
INTEGRATION_LIVEKIT = "livekit"
INTEGRATION_PIPECAT = "pipecat"
INTEGRATIONS = frozenset({INTEGRATION_NONE, INTEGRATION_LIVEKIT, INTEGRATION_PIPECAT})

_STRING = "string"
_INT = "int"

#: The EXACT attribute allowlist from the contract. Any key not in this table
#: is dropped by :func:`encode_attributes`: that is the only thing standing
#: between a caller's text and the ingestion endpoint.
ATTRIBUTE_TYPES: Dict[str, str] = {
    "kugel.event": _STRING,
    "kugel.event_id": _STRING,
    "kugel.operation_id": _STRING,
    "kugel.operation": _STRING,
    "kugel.sdk.name": _STRING,
    "kugel.sdk.version": _STRING,
    "kugel.runtime": _STRING,
    "kugel.integration": _STRING,
    "kugel.transport": _STRING,
    "kugel.failure_stage": _STRING,
    "kugel.error_type": _STRING,
    "kugel.error_code": _STRING,
    "kugel.http_status": _INT,
    "kugel.ws_close_code": _INT,
    "kugel.server_request_id": _STRING,
    "kugel.elapsed_ms": _INT,
    "kugel.audio_chunks": _INT,
    "kugel.audio_bytes": _INT,
    "kugel.retry_count": _INT,
    "kugel.outcome": _STRING,
    "kugel.endpoint_kind": _STRING,
    "kugel.success_count": _INT,
    "kugel.failure_count": _INT,
    "kugel.cancelled_count": _INT,
}

REQUIRED_ATTRIBUTES = (
    "kugel.event",
    "kugel.event_id",
    "kugel.operation_id",
    "kugel.sdk.name",
    "kugel.sdk.version",
)

RESOURCE_SERVICE_NAME = "kugelaudio-sdk"
SCOPE_NAME = "kugelaudio.diagnostics"
SCOPE_VERSION = "1"

SEVERITY_ERROR = (17, "ERROR")
SEVERITY_INFO = (9, "INFO")

#: ``Callable[[url, headers, payload], int]``: the injectable transport seam.
#: A sender raises only for transport-level failures (DNS, timeout, reset);
#: every HTTP response, 2xx or not, comes back as its status code, because
#: the reporter must SEE a 404 or 413 to apply the contract's rules.
Sender = Callable[[str, Dict[str, str], bytes], int]

Record = Dict[str, Any]


def runtime_string() -> str:
    """``python/3.12.4``: major.minor.patch only, never a build string."""
    major, minor, micro = sys.version_info[:3]
    return f"python/{major}.{minor}.{micro}"


def new_id() -> str:
    """32 lowercase hex characters."""
    return uuid.uuid4().hex


# --------------------------------------------------------------------------
# OTLP/HTTP JSON encoding
# --------------------------------------------------------------------------


def encode_attributes(attributes: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Encode an attribute mapping, dropping every key outside the allowlist."""
    encoded: List[Dict[str, Any]] = []
    for key, value in attributes.items():
        kind = ATTRIBUTE_TYPES.get(key)
        if kind is None or value is None:
            continue
        if kind is _INT:
            try:
                encoded.append({"key": key, "value": {"intValue": str(int(value))}})
            except (TypeError, ValueError):
                continue
        else:
            encoded.append({"key": key, "value": {"stringValue": str(value)}})
    return encoded


def build_log_record(
    event: str, attributes: Mapping[str, Any], time_unix_nano: Optional[int] = None
) -> Record:
    """One OTLP log record. ``sdk_stats`` is INFO, every failure is ERROR."""
    if time_unix_nano is None:
        time_unix_nano = time.time_ns()
    number, text = SEVERITY_INFO if event == EVENT_SDK_STATS else SEVERITY_ERROR
    return {
        "timeUnixNano": str(time_unix_nano),
        "severityNumber": number,
        "severityText": text,
        "body": {"stringValue": event},
        "attributes": encode_attributes(attributes),
    }


def build_payload(records: List[Record], version: str) -> bytes:
    body = {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "service.name",
                            "value": {"stringValue": RESOURCE_SERVICE_NAME},
                        },
                        {
                            "key": "service.version",
                            "value": {"stringValue": version},
                        },
                        {
                            "key": "telemetry.sdk.language",
                            "value": {"stringValue": SDK_NAME},
                        },
                    ]
                },
                "scopeLogs": [
                    {
                        "scope": {"name": SCOPE_NAME, "version": SCOPE_VERSION},
                        "logRecords": records,
                    }
                ],
            }
        ]
    }
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def urllib_sender(url: str, headers: Dict[str, str], payload: bytes) -> int:
    """Default transport: stdlib only, so the SDK gains no dependency.

    Returns the HTTP status code. ``urllib`` raises ``HTTPError`` on a non-2xx,
    but an error response is *data* for the reporter (the 404 and 413 rules),
    not a transport failure, so it is converted back into its code. Genuine
    transport failures (``URLError``, timeout) still raise and are retried
    once.
    """
    request = urllib.request.Request(
        url, data=payload, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            response.read()
            return int(response.status)
    except urllib.error.HTTPError as exc:
        # KEEP-JUSTIFIED: an HTTPError IS the server's response. The reporter
        # needs its status; propagating it would retry a decided answer.
        try:
            exc.read()
        finally:
            exc.close()
        return int(exc.code)


# --------------------------------------------------------------------------
# Reporter
# --------------------------------------------------------------------------


class Diagnostics:
    """Bounded, off-path OTLP reporter. One per client.

    Args:
        config: resolved enablement, target URL and auth headers. ``None``
            means disabled.
        sender: injectable transport returning the HTTP status code. Defaults
            to :func:`urllib_sender`.
        integration: ``kugel.integration`` for every record of this client:
            ``none``, or ``livekit`` / ``pipecat`` when that plugin built the
            client.
        autostart: start the delivery thread on the first event. Tests set
            ``False`` so records stay queued until :meth:`flush`.
        version: SDK version override, for tests.
    """

    def __init__(
        self,
        config: Optional[DiagnosticsConfig] = None,
        sender: Optional[Sender] = None,
        *,
        integration: str = INTEGRATION_NONE,
        autostart: bool = True,
        version: Optional[str] = None,
    ) -> None:
        if integration not in INTEGRATIONS:
            raise ValueError(
                f"integration must be one of {sorted(INTEGRATIONS)}, got {integration!r}"
            )
        self._config = config if config is not None else DiagnosticsConfig.disabled()
        self._sender: Sender = sender if sender is not None else urllib_sender
        self._version = version if version is not None else sdk_version()
        self._integration = integration
        self._records: Deque[Record] = deque(maxlen=QUEUE_CAPACITY)
        # One condition guards the queue and signals both directions: the
        # worker waits on it for work, flush()/close() wait on it for "drained".
        self._changed = threading.Condition(threading.Lock())
        self._worker: Optional[threading.Thread] = None
        self._autostart = autostart
        self._closed = False
        self._flush_requested = False
        self._in_flight = False
        self._deadline: Optional[float] = None
        self._success_count = 0
        self._failure_count = 0
        self._cancelled_count = 0
        self._consecutive_not_found = 0
        self._route_missing = False
        #: Set in a forked child: inherited records belong to the parent.
        self._forked = False
        with _REGISTRY_LOCK:
            _ALL_REPORTERS.add(self)

    # -- introspection ----------------------------------------------------

    @property
    def config(self) -> DiagnosticsConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def integration(self) -> str:
        return self._integration

    @property
    def route_missing(self) -> bool:
        """True once the server answered 404 three times in a row."""
        return self._route_missing

    @property
    def inert(self) -> bool:
        """True when no network call may be made, for any reason."""
        return self._config.inert or self._route_missing or self._forked

    @property
    def url(self) -> str:
        return self._config.url

    def pending(self) -> int:
        with self._changed:
            return len(self._records)

    def counters(self) -> Dict[str, int]:
        return {
            "success": self._success_count,
            "failure": self._failure_count,
            "cancelled": self._cancelled_count,
        }

    # -- operation seam ---------------------------------------------------

    def operation(
        self, operation: Optional[str], transport: str, stage: Optional[str] = None
    ) -> "Operation":
        """Mint an operation. The id exists before we connect."""
        return Operation(self, operation, transport, stage)

    # -- event intake -----------------------------------------------------

    def record(
        self,
        event: str,
        attributes: Optional[Mapping[str, Any]] = None,
        time_unix_nano: Optional[int] = None,
    ) -> Record:
        """Build the OTLP record this reporter would queue for *event*.

        Fills the required attributes and this client's identity (runtime,
        integration, endpoint kind) unless *attributes* already carries them.
        Pure: queues nothing and touches no network.
        """
        attrs: Dict[str, Any] = dict(attributes or {})
        attrs["kugel.event"] = event
        attrs.setdefault("kugel.event_id", new_id())
        attrs.setdefault("kugel.operation_id", new_id())
        attrs.setdefault("kugel.sdk.name", SDK_NAME)
        attrs.setdefault("kugel.sdk.version", self._version)
        attrs.setdefault("kugel.runtime", runtime_string())
        attrs.setdefault("kugel.integration", self._integration)
        attrs.setdefault("kugel.endpoint_kind", self._config.endpoint_kind)
        return build_log_record(event, attrs, time_unix_nano)

    def report(self, event: str, attributes: Optional[Mapping[str, Any]] = None) -> None:
        """Queue one event. Never raises, never blocks on the network."""
        try:
            if self.inert or self._closed:
                return
            record = self.record(event, attributes)
            with self._changed:
                # ``deque(maxlen=...)`` drops the OLDEST record on overflow.
                self._records.append(record)
                if self._deadline is None:
                    self._deadline = time.monotonic() + FLUSH_INTERVAL_S
                with _REGISTRY_LOCK:
                    _EXIT_PENDING.add(self)
                self._changed.notify_all()
            if self._autostart:
                self._ensure_worker()
        except Exception:  # pragma: no cover - defensive
            # KEEP-JUSTIFIED: a telemetry failure must never surface to the
            # caller, and the contract caps its logging at DEBUG.
            logger.debug("diagnostics: dropping event %s", event, exc_info=True)

    def note_success(self) -> None:
        self._success_count += 1

    def note_failure(self) -> None:
        self._failure_count += 1

    def note_cancelled(self) -> None:
        """Count a caller-initiated cancellation: never a record, never a failure."""
        self._cancelled_count += 1

    # -- delivery (all network I/O happens on the worker thread) ----------

    def _ensure_worker(self) -> None:
        with self._changed:
            if self._worker is not None and self._worker.is_alive():
                return
            # Started under the lock so two racing callers cannot both see
            # "not alive" and start a second worker.
            worker = threading.Thread(
                target=self._run,
                name="kugelaudio-diagnostics",
                daemon=True,  # must never delay process exit
            )
            self._worker = worker
            worker.start()

    def _next_batch(self) -> Optional[List[Record]]:
        """Block until a batch is due. ``None`` tells the worker to exit."""
        with self._changed:
            while True:
                if self._route_missing:
                    self._records.clear()
                    self._deadline = None
                    self._changed.notify_all()
                    return None
                if not self._records:
                    self._flush_requested = False
                    self._deadline = None
                    with _REGISTRY_LOCK:
                        _EXIT_PENDING.discard(self)
                    self._changed.notify_all()  # wakes flush()/close() waiters
                    if self._closed:
                        return None
                    self._changed.wait()
                    continue
                now = time.monotonic()
                due = (
                    len(self._records) >= BATCH_SIZE
                    or self._flush_requested
                    or self._closed
                    or (self._deadline is not None and now >= self._deadline)
                )
                if due:
                    size = min(BATCH_SIZE, len(self._records))
                    batch = [self._records.popleft() for _ in range(size)]
                    self._deadline = now + FLUSH_INTERVAL_S if self._records else None
                    self._in_flight = True
                    return batch
                wait_s = (
                    FLUSH_INTERVAL_S if self._deadline is None else self._deadline - now
                )
                self._changed.wait(wait_s)

    def _run(self) -> None:
        while True:
            batch = self._next_batch()
            if batch is None:
                return
            try:
                self._deliver_batch(batch)
            except Exception:  # pragma: no cover - defensive
                # KEEP-JUSTIFIED: the delivery thread must never raise into
                # the interpreter; the batch is dropped and the loop goes on.
                logger.debug("diagnostics: delivery failed", exc_info=True)
            finally:
                with self._changed:
                    self._in_flight = False
                    self._changed.notify_all()

    def _note_not_found(self) -> None:
        """Three consecutive 404 batches switch this reporter off.

        Per instance: one process may hold clients pointed at different hosts,
        and only the stale one should go quiet.
        """
        self._consecutive_not_found += 1
        if self._consecutive_not_found >= MAX_CONSECUTIVE_NOT_FOUND:
            with self._changed:
                self._route_missing = True
                self._records.clear()
                self._deadline = None
                self._changed.notify_all()

    def _deliver_batch(self, batch: List[Record]) -> None:
        """POST one batch (<= ``BATCH_SIZE`` records). Worker thread only."""
        payload = build_payload(batch, self._version)
        # The client's own auth headers verbatim, plus the mandatory content
        # type, set last so no caller header can displace it.
        headers = dict(self._config.auth_headers)
        headers["Content-Type"] = "application/json"
        for _ in range(MAX_SEND_ATTEMPTS):
            try:
                status = self._sender(self.url, dict(headers), payload)
            except Exception:
                # KEEP-JUSTIFIED: the injected sender owns the network; its
                # failure is ours to absorb. A throw is not a response, so it
                # neither bumps nor resets the 404 streak.
                logger.debug("diagnostics: delivery attempt failed", exc_info=True)
                continue
            if not isinstance(status, int):
                logger.debug("diagnostics: sender returned %r, not a status", status)
                return
            if 200 <= status < 300:
                self._consecutive_not_found = 0
                return
            if status == NOT_FOUND_STATUS:
                self._note_not_found()
                return
            self._consecutive_not_found = 0
            if status == PAYLOAD_TOO_LARGE_STATUS:
                return
        # After the one allowed retry the batch is dropped, silently.

    def flush(self, timeout: float = CLOSE_DEADLINE_S) -> None:
        """Hand everything queued to the worker and wait at most *timeout*.

        Never sends on the caller's thread. Swallows everything.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        if self._request_flush():
            self._wait_drained(deadline)

    def _request_flush(self) -> bool:
        """Ask the worker to send everything now. True if there is anything
        to wait for."""
        if self.inert:
            return False
        try:
            with self._changed:
                if not self._records and not self._in_flight:
                    return False
                self._flush_requested = True
                self._changed.notify_all()
            self._ensure_worker()
            return True
        except Exception:  # pragma: no cover - defensive
            # KEEP-JUSTIFIED: runs from close() and at interpreter exit (where
            # a thread may no longer start); telemetry must stay silent.
            logger.debug("diagnostics: flush failed", exc_info=True)
            return False

    def _wait_drained(self, deadline: float) -> None:
        """Wait until nothing is queued or in flight, or until *deadline*."""
        try:
            with self._changed:
                while self._records or self._in_flight:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return
                    self._changed.wait(remaining)
        except Exception:  # pragma: no cover - defensive
            # KEEP-JUSTIFIED: runs from close() and at interpreter exit;
            # telemetry must never raise or print there.
            logger.debug("diagnostics: waiting for delivery failed", exc_info=True)

    def _after_fork_in_child(self) -> None:
        """A forked child must never re-send the parent's records, nor block
        on a lock copied while another parent thread held it."""
        self._changed = threading.Condition(threading.Lock())
        self._records.clear()
        self._deadline = None
        self._in_flight = False
        self._flush_requested = False
        self._worker = None
        self._forked = True

    def close(self, timeout: float = CLOSE_DEADLINE_S) -> None:
        """Queue ``sdk_stats``, hand the flush to the worker, wait <= *timeout*.

        No network I/O on the caller's thread. Records the worker has not
        sent by the deadline keep going out from the daemon thread.
        """
        if self._closed:
            return
        try:
            self._emit_stats()
        finally:
            with self._changed:
                self._closed = True
                self._changed.notify_all()
            with _REGISTRY_LOCK:
                _EXIT_PENDING.discard(self)
            self.flush(timeout)

    def _emit_stats(self) -> None:
        if (
            self._success_count == 0
            and self._failure_count == 0
            and self._cancelled_count == 0
        ):
            return
        self.report(
            EVENT_SDK_STATS,
            {
                "kugel.success_count": self._success_count,
                "kugel.failure_count": self._failure_count,
                # The ONLY place a cancellation is reported.
                "kugel.cancelled_count": self._cancelled_count,
            },
        )


# --------------------------------------------------------------------------
# Operation seam
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Process lifecycle: exit flush and fork safety
# --------------------------------------------------------------------------

#: Guards the two registries below. Lock order: a reporter's own condition,
#: then this lock; never the reverse.
_REGISTRY_LOCK = threading.Lock()
#: Every live reporter (weak), so a forked child can disable inherited ones.
_ALL_REPORTERS: "weakref.WeakSet[Diagnostics]" = weakref.WeakSet()
#: Reporters with records queued (weak): joined on the first queued record,
#: left when the queue drains or on close().
_EXIT_PENDING: "weakref.WeakSet[Diagnostics]" = weakref.WeakSet()


def _flush_all_at_exit() -> None:
    """Contract "Exit flush": one hook for the whole process.

    Errors the caller never caught, on clients never closed: every reporter
    with records still queued is flushed in parallel (each has its own
    worker) against ONE 1 s deadline, so N clients never add N seconds.
    Nothing queued means no delay at all.
    """
    try:
        with _REGISTRY_LOCK:
            pending = [r for r in _EXIT_PENDING if not r._closed]
            _EXIT_PENDING.clear()
        if not pending:
            return
        deadline = time.monotonic() + CLOSE_DEADLINE_S
        waiting = [reporter for reporter in pending if reporter._request_flush()]
        for reporter in waiting:
            reporter._wait_drained(deadline)
    except Exception:  # pragma: no cover - defensive
        # KEEP-JUSTIFIED: an exception here would print "Error in
        # atexit._run_exitfuncs"; telemetry must stay silent at exit.
        logger.debug("diagnostics: exit flush failed", exc_info=True)


def _disable_inherited_reporters() -> None:
    """``after_in_child`` of ``os.fork``: inherited reporters go quiet."""
    global _REGISTRY_LOCK, _EXIT_PENDING
    _REGISTRY_LOCK = threading.Lock()
    _EXIT_PENDING = weakref.WeakSet()
    for reporter in list(_ALL_REPORTERS):
        reporter._after_fork_in_child()


atexit.register(_flush_all_at_exit)
if hasattr(os, "register_at_fork"):  # POSIX only
    os.register_at_fork(after_in_child=_disable_inherited_reporters)


def _close_code_from(exc: Optional[BaseException]) -> Optional[int]:
    """Walk the ``__cause__`` chain for a WebSocket close code.

    Every WS error path in the SDK raises its typed error ``from`` the
    original ``ConnectionClosed``, so the close code is one hop away.
    Deprecated ``code`` properties (``websockets`` >= 13.1, aiohttp's
    handshake error) are never touched: reading them warns, and telemetry
    must not print anything.
    """
    # RISK: reads no close code if a WS raise site drops the ``from e``
    # chaining; ``kugel.ws_close_code`` silently going missing is the symptom.
    seen = 0
    current = exc
    while current is not None and seen < 4:
        code = _own_close_code(current)
        if code is not None:
            return code
        current = current.__cause__
        seen += 1
    return None


def _own_close_code(exc: BaseException) -> Optional[int]:
    attributes = getattr(exc, "__dict__", {})
    if "rcvd" in attributes:
        # websockets ConnectionClosed: the received close frame, or none
        # at all for an abnormal closure (1006).
        received = attributes["rcvd"]
        code = getattr(received, "code", 1006) if received is not None else 1006
    else:
        code = attributes.get("code")
    if isinstance(code, int) and 1000 <= code <= 4999:
        return code
    return None


class Operation:
    """One unit of work that independently succeeds or fails.

    A one-shot call, an HTTP request, one session turn, or one session
    connect. Its id, elapsed time, audio counters and retry count are scoped
    to it. Reports **at most one** event: the first ``succeed()``, ``fail()``
    or ``cancel()`` settles it and every later call is a no-op.
    """

    __slots__ = (
        "_diagnostics",
        "operation",
        "transport",
        "operation_id",
        "_started",
        "stage",
        "audio_chunks",
        "audio_bytes",
        "retry_count",
        "ws_close_code",
        "server_error",
        "settled",
    )

    def __init__(
        self,
        diagnostics: Diagnostics,
        operation: Optional[str],
        transport: str,
        stage: Optional[str] = None,
    ) -> None:
        self._diagnostics = diagnostics
        self.operation = operation
        self.transport = transport
        self.operation_id = new_id()
        self._started = time.monotonic()
        if stage is None:
            stage = (
                STAGE_SENDING_REQUEST if transport == TRANSPORT_HTTP else STAGE_CONNECTING
            )
        self.stage = stage
        self.audio_chunks = 0
        self.audio_bytes = 0
        self.retry_count = 0
        #: Set by a transport that sees the close code but raises an error
        #: without a ``code`` in its cause chain (the LiveKit aiohttp socket).
        self.ws_close_code: Optional[int] = None
        #: The server's own error for this operation, when a transport only
        #: learns about it through a wrapper (the LiveKit plugin's error
        #: frames). ``fail()`` reports it instead of the wrapper.
        self.server_error: Optional[BaseException] = None
        self.settled = False

    # -- recording --------------------------------------------------------

    def set_stage(self, stage: str) -> "Operation":
        self.stage = stage
        return self

    def expect_audio(self) -> "Operation":
        """Enter the receive phase: awaiting first audio, or receiving more."""
        self.stage = (
            STAGE_RECEIVING_AUDIO if self.audio_chunks else STAGE_AWAITING_FIRST_AUDIO
        )
        return self

    def record_chunk(self, size_bytes: int = 0) -> None:
        self.audio_chunks += 1
        try:
            self.audio_bytes += int(size_bytes or 0)
        except (TypeError, ValueError):
            pass
        if self.stage == STAGE_AWAITING_FIRST_AUDIO:
            self.stage = STAGE_RECEIVING_AUDIO

    def record_retry(self) -> None:
        """A new attempt of this operation: earlier attempt details reset."""
        self.retry_count += 1
        self.server_error = None
        self.ws_close_code = None

    def mark_server_error(self, error: BaseException) -> None:
        """The server answered this operation with an error frame.

        Contract: a server error frame that fails the operation is
        ``request_failed`` (or ``retry_exhausted`` after a retry).
        """
        self.server_error = error

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._started) * 1000)

    def step(self, stage: Optional[str] = None) -> "_Step":
        """Guard a phase: sets the stage, reports a failure on the way out."""
        return _Step(self, stage)

    # -- settlement -------------------------------------------------------

    def succeed(self) -> None:
        if self.settled:
            return
        self.settled = True
        self._diagnostics.note_success()

    def cancel(self) -> None:
        """Caller-initiated cancellation: counted in ``sdk_stats``, never an event."""
        if self.settled:
            return
        self.settled = True
        try:
            self._diagnostics.note_cancelled()
        except Exception:  # pragma: no cover - defensive
            # KEEP-JUSTIFIED: runs on the caller's barge-in path; a telemetry
            # problem must never surface there.
            logger.debug("diagnostics: could not count cancellation", exc_info=True)

    def fail(self, exc: BaseException) -> None:
        """Report one event for *exc*. The caller re-raises; we never do."""
        # Closing an async generator early is normal control flow, not a
        # failure and not a cancellation: say nothing at all.
        if isinstance(exc, GeneratorExit):
            return
        if _is_cancellation(exc):
            self.cancel()
            return
        if self.settled:
            return
        self.settled = True
        reported = self.server_error if self.server_error is not None else exc
        try:
            self._diagnostics.note_failure()
            self._diagnostics.report(self._event_name(), self._attributes(reported))
        except Exception:  # pragma: no cover - defensive
            # KEEP-JUSTIFIED: this runs on the caller's exception path. The
            # original error is about to be re-raised and must reach them
            # unchanged, whatever happens here.
            logger.debug("diagnostics: could not report operation", exc_info=True)

    def _attributes(self, exc: BaseException) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {
            "kugel.operation_id": self.operation_id,
            "kugel.operation": self.operation,
            "kugel.transport": self.transport,
            "kugel.failure_stage": self.stage,
            "kugel.elapsed_ms": self.elapsed_ms(),
            "kugel.audio_chunks": self.audio_chunks,
            "kugel.audio_bytes": self.audio_bytes,
            "kugel.retry_count": self.retry_count,
            "kugel.outcome": OUTCOME_FAILED,
            "kugel.error_type": type(exc).__name__,
        }
        error_code = getattr(exc, "error_code", None)
        if isinstance(error_code, str):
            attributes["kugel.error_code"] = error_code
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            attributes["kugel.http_status"] = status
        request_id = getattr(exc, "request_id", None)
        if isinstance(request_id, str) and request_id:
            attributes["kugel.server_request_id"] = request_id
        close_code = self.ws_close_code
        if close_code is None:
            close_code = _close_code_from(exc)
        if close_code is not None:
            attributes["kugel.ws_close_code"] = close_code
        return attributes

    def _event_name(self) -> str:
        if self.retry_count > 0:
            return EVENT_RETRY_EXHAUSTED
        if self.server_error is not None:
            return EVENT_REQUEST_FAILED
        if self.stage in (STAGE_CONNECTING, STAGE_HANDSHAKE):
            return EVENT_CONNECTION_FAILED
        if self.stage in (
            STAGE_AWAITING_FIRST_AUDIO,
            STAGE_RECEIVING_AUDIO,
            STAGE_FINALIZING,
        ):
            return EVENT_STREAM_INTERRUPTED
        return EVENT_REQUEST_FAILED


def _is_cancellation(exc: BaseException) -> bool:
    import asyncio

    if isinstance(exc, asyncio.CancelledError):
        return True
    # A session may surface its own cancellation marker.
    return type(exc).__name__ == "CancelledError"


class _Step:
    """Sets a failure stage for the duration of a block, reports on exit."""

    __slots__ = ("_op", "_stage")

    def __init__(self, op: Operation, stage: Optional[str]) -> None:
        self._op = op
        self._stage = stage

    def __enter__(self) -> Operation:
        if self._stage is not None:
            self._op.stage = self._stage
        return self._op

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> bool:
        if exc is not None:
            self._op.fail(exc)
        return False

    async def __aenter__(self) -> Operation:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> bool:
        return self.__exit__(exc_type, exc, tb)


#: Shared no-op reporter for sessions constructed without a client.
DISABLED = Diagnostics(DiagnosticsConfig.disabled())


__all__ = [
    "ATTRIBUTE_TYPES",
    "DISABLED",
    "INTEGRATIONS",
    "MAX_CONSECUTIVE_NOT_FOUND",
    "Diagnostics",
    "Operation",
    "Sender",
    "build_payload",
    "encode_attributes",
]
