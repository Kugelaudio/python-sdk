"""Golden diagnostics batch: the SDK's own encoder must reproduce it exactly.

``services/ingress/tests/unit/test_sdk_diagnostics_contract.py`` loads the
same file and asserts the server keeps every attribute of every record, so
any encoder drift fails CI on one side or the other.

Regenerate after an intentional contract change::

    UPDATE_GOLDEN=1 uv run pytest tests/test_diagnostics_golden.py

then review the diff: every key in the contract allowlist must still appear.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from kugelaudio._diagnostics import (
    ATTRIBUTE_TYPES,
    EVENT_CONNECTION_FAILED,
    EVENT_REQUEST_FAILED,
    EVENT_RETRY_EXHAUSTED,
    EVENT_SDK_STATS,
    EVENT_STREAM_INTERRUPTED,
    Diagnostics,
    build_payload,
)

from .diagnostics_support import FakeSender, attrs_of, live_config

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "diagnostics_golden_batch.json"
GOLDEN_VERSION = "2.0.0"
GOLDEN_RUNTIME = "python/3.12.4"
#: 2026-01-01T00:00:00Z, one second apart per record.
GOLDEN_TIME_NS = 1_767_225_600_000_000_000

_FAILURE_COMMON = {"kugel.outcome": "failed", "kugel.runtime": GOLDEN_RUNTIME}

#: One record per event type; together they cover every allowlisted key.
GOLDEN_EVENTS: List[Tuple[str, Dict[str, Any]]] = [
    (
        EVENT_CONNECTION_FAILED,
        {
            **_FAILURE_COMMON,
            "kugel.event_id": "0a1b2c3d4e5f40718293a4b5c6d7e8f9",
            "kugel.operation_id": "1b2c3d4e5f60418293a4b5c6d7e8f90a",
            "kugel.operation": "stream_session",
            "kugel.transport": "websocket",
            "kugel.failure_stage": "handshake",
            "kugel.error_type": "AuthenticationError",
            "kugel.error_code": "UNAUTHORIZED",
            "kugel.http_status": 401,
            "kugel.server_request_id": "5e0c7d2a9b3f4e61a8c2d4f6b8e0a1c3",
            "kugel.elapsed_ms": 212,
            "kugel.audio_chunks": 0,
            "kugel.audio_bytes": 0,
            "kugel.retry_count": 0,
        },
    ),
    (
        EVENT_REQUEST_FAILED,
        {
            **_FAILURE_COMMON,
            "kugel.event_id": "2c3d4e5f6071429384a5b6c7d8e9f0a1",
            "kugel.operation_id": "3d4e5f607182439485b6c7d8e9f0a1b2",
            "kugel.operation": "models",
            "kugel.transport": "http",
            "kugel.failure_stage": "sending_request",
            "kugel.error_type": "RateLimitError",
            "kugel.error_code": "RATE_LIMITED",
            "kugel.http_status": 429,
            "kugel.server_request_id": "req_7f3a9c1e2b4d",
            "kugel.elapsed_ms": 87,
            "kugel.audio_chunks": 0,
            "kugel.audio_bytes": 0,
            "kugel.retry_count": 0,
        },
    ),
    (
        EVENT_STREAM_INTERRUPTED,
        {
            **_FAILURE_COMMON,
            "kugel.event_id": "4e5f60718293440596c7d8e9f0a1b2c3",
            "kugel.operation_id": "5f6071829304451607d8e9f0a1b2c3d4",
            "kugel.operation": "multi_context",
            "kugel.integration": "livekit",
            "kugel.transport": "websocket",
            "kugel.failure_stage": "receiving_audio",
            "kugel.error_type": "ConnectionError",
            "kugel.ws_close_code": 1006,
            "kugel.elapsed_ms": 1840,
            "kugel.audio_chunks": 12,
            "kugel.audio_bytes": 46080,
            "kugel.retry_count": 0,
        },
    ),
    (
        EVENT_RETRY_EXHAUSTED,
        {
            **_FAILURE_COMMON,
            "kugel.event_id": "607182930415462718e9f0a1b2c3d4e5",
            "kugel.operation_id": "7182930415264728a9f0a1b2c3d4e5f6",
            "kugel.operation": "multi_context",
            "kugel.integration": "pipecat",
            "kugel.transport": "websocket",
            "kugel.failure_stage": "awaiting_first_audio",
            "kugel.error_type": "ServerRestartingError",
            "kugel.ws_close_code": 1012,
            "kugel.elapsed_ms": 2310,
            "kugel.audio_chunks": 0,
            "kugel.audio_bytes": 0,
            "kugel.retry_count": 1,
        },
    ),
    (
        EVENT_SDK_STATS,
        {
            "kugel.runtime": GOLDEN_RUNTIME,
            "kugel.event_id": "82930415263748b9a0f1b2c3d4e5f607",
            "kugel.operation_id": "930415263748590a1b2c3d4e5f607182",
            "kugel.success_count": 41,
            "kugel.failure_count": 4,
            "kugel.cancelled_count": 3,
        },
    ),
]


def _encode_golden() -> str:
    reporter = Diagnostics(
        live_config(), FakeSender(), autostart=False, version=GOLDEN_VERSION
    )
    records = [
        reporter.record(event, attributes, GOLDEN_TIME_NS + index * 1_000_000_000)
        for index, (event, attributes) in enumerate(GOLDEN_EVENTS)
    ]
    payload = build_payload(records, GOLDEN_VERSION)
    return json.dumps(json.loads(payload), indent=2) + "\n"


def test_the_encoder_reproduces_the_golden_batch_exactly() -> None:
    encoded = _encode_golden()
    if os.environ.get("UPDATE_GOLDEN") == "1":
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(encoded, encoding="utf-8")
    assert GOLDEN_PATH.read_text(encoding="utf-8") == encoded


def test_the_golden_batch_covers_the_contract() -> None:
    body = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    records = body["resourceLogs"][0]["scopeLogs"][0]["logRecords"]
    flat = [attrs_of(record) for record in records]

    assert [f["kugel.event"] for f in flat] == [event for event, _ in GOLDEN_EVENTS]
    covered = {key for attributes in flat for key in attributes}
    assert covered == set(ATTRIBUTE_TYPES)
    assert {f["kugel.integration"] for f in flat} == {"none", "livekit", "pipecat"}
    # One batch is one POST: it must respect the per-POST cap.
    assert len(records) <= 8
