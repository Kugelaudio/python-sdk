"""Tests for the diagnostics reporter: enablement, target, encoding, delivery.

Operation-level behaviour (real SDK error paths, operation scope,
cancellation, request-id correlation, integration) lives in
``test_diagnostics_operations.py``; the golden batch in
``test_diagnostics_golden.py``. Contract:
``services/ingress/docs/sdk-diagnostics-contract.md``.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import httpx
import pytest

from kugelaudio import KugelAudio
from kugelaudio._diagnostics_config import DIAGNOSTICS_PATH, ENV_TELEMETRY, DiagnosticsConfig
from kugelaudio._diagnostics import (
    BATCH_SIZE,
    EVENT_REQUEST_FAILED,
    EVENT_SDK_STATS,
    MAX_CONSECUTIVE_NOT_FOUND,
    OP_MODELS,
    OP_STREAM,
    QUEUE_CAPACITY,
    TRANSPORT_WEBSOCKET,
    Diagnostics,
    encode_attributes,
)

from .diagnostics_support import (
    TEST_API_URL,
    TEST_AUTH_HEADERS,
    TEST_DIAGNOSTICS_URL,
    FakeSender,
    ScriptedSender,
    attrs_of,
    live_reporter,
)


def _reporter_source() -> str:
    """Source of both reporter modules, for "this is gone" assertions."""
    import kugelaudio._diagnostics as module
    import kugelaudio._diagnostics_config as config_module

    texts = []
    for mod in (module, config_module):
        with open(mod.__file__, "r", encoding="utf-8") as handle:
            texts.append(handle.read())
    return "\n".join(texts)


# ---------------------------------------------------------------------------
# Enablement matrix
# ---------------------------------------------------------------------------


class TestEnablement:
    def test_hosted_api_url_defaults_on(self):
        config = DiagnosticsConfig.resolve("https://api.kugelaudio.com", None, {})
        assert config.enabled is True
        assert config.endpoint_kind == "hosted"

    def test_custom_api_url_defaults_off(self):
        config = DiagnosticsConfig.resolve("https://tts.on-prem.internal", None, {})
        assert config.enabled is False
        assert config.endpoint_kind == "custom"

    def test_explicit_option_beats_the_custom_default(self):
        config = DiagnosticsConfig.resolve("https://tts.on-prem.internal", True, {})
        assert config.enabled is True

    def test_explicit_option_beats_the_hosted_default(self):
        config = DiagnosticsConfig.resolve("https://api.kugelaudio.com", False, {})
        assert config.enabled is False

    def test_env_off_beats_an_explicit_true(self):
        config = DiagnosticsConfig.resolve(
            "https://api.kugelaudio.com", True, {ENV_TELEMETRY: "off"}
        )
        assert config.enabled is False

    def test_env_on_beats_an_explicit_false(self):
        config = DiagnosticsConfig.resolve(
            "https://tts.on-prem.internal", False, {ENV_TELEMETRY: "yes"}
        )
        assert config.enabled is True

    @pytest.mark.parametrize("value", ["0", "false", "off", "no", "FALSE", " Off "])
    def test_falsey_env_values(self, value):
        config = DiagnosticsConfig.resolve(
            "https://api.kugelaudio.com", None, {ENV_TELEMETRY: value}
        )
        assert config.enabled is False

    @pytest.mark.parametrize("value", ["1", "true", "on", "yes", "TRUE"])
    def test_truthy_env_values(self, value):
        config = DiagnosticsConfig.resolve(
            "https://tts.on-prem.internal", None, {ENV_TELEMETRY: value}
        )
        assert config.enabled is True

    def test_client_constructor_option_is_honoured(self):
        assert KugelAudio(api_key="k")._diagnostics.enabled is True
        assert KugelAudio(api_key="k", telemetry=False)._diagnostics.enabled is False
        assert (
            KugelAudio(
                api_key="k", api_url="https://on-prem.internal"
            )._diagnostics.enabled
            is False
        )
        assert (
            KugelAudio(
                api_key="k", api_url="https://on-prem.internal", telemetry=True
            )._diagnostics.enabled
            is True
        )


# ---------------------------------------------------------------------------
# The target is the client's own authenticated API
# ---------------------------------------------------------------------------


class TestTargetDerivation:
    def test_url_is_derived_from_the_effective_api_url(self):
        config = DiagnosticsConfig.resolve(TEST_API_URL, None, {})
        assert config.url == "https://api.kugelaudio.com/v1/sdk-diagnostics"

    def test_a_trailing_slash_on_the_api_url_does_not_double_up(self):
        config = DiagnosticsConfig.resolve(TEST_API_URL + "/", None, {})
        assert config.url == "https://api.kugelaudio.com/v1/sdk-diagnostics"

    def test_a_custom_api_url_targets_that_same_host(self):
        config = DiagnosticsConfig.resolve(
            "https://tts.on-prem.internal", True, {}
        )
        assert config.url == "https://tts.on-prem.internal/v1/sdk-diagnostics"

    def test_there_is_no_endpoint_override(self):
        """Reports carry the API key, so no env var may redirect them.

        The removed override's name is assembled here so a repository grep
        for it stays at zero hits.
        """
        removed = ENV_TELEMETRY + "_" + "ENDPOINT"
        config = DiagnosticsConfig.resolve(
            TEST_API_URL, None, {removed: "https://collector.example.com/ingest"}
        )
        assert config.url == TEST_DIAGNOSTICS_URL

        assert removed not in _reporter_source()

    def test_without_an_api_url_there_is_nowhere_to_post(self):
        config = DiagnosticsConfig.resolve(None, True, {})
        assert config.url == ""
        assert config.inert is True

    def test_auth_headers_are_passed_through_verbatim(self):
        config = DiagnosticsConfig.resolve(
            TEST_API_URL, None, {}, auth_headers=TEST_AUTH_HEADERS
        )
        assert config.auth_headers == TEST_AUTH_HEADERS

    def test_the_client_targets_its_own_api_host_with_its_own_auth(self):
        sender = FakeSender()
        client = KugelAudio(api_key="secret-key")
        assert client._diagnostics.enabled is True
        assert client._diagnostics.inert is False
        assert client._diagnostics.url.endswith(DIAGNOSTICS_PATH)
        assert client._diagnostics.url.startswith(client._api_url)

        client._diagnostics = Diagnostics(
            client._diagnostics.config, sender, autostart=False, version="9.9.9"
        )
        client._diagnostics.report(EVENT_REQUEST_FAILED)
        client._diagnostics.flush(timeout=5.0)

        url, headers, _payload = sender.calls[0]
        assert url == client._api_url + DIAGNOSTICS_PATH
        assert headers["Authorization"] == "Bearer secret-key"
        assert headers["X-API-Key"] == "secret-key"
        assert headers["Content-Type"] == "application/json"

    def test_the_public_ingestion_credential_is_gone(self):
        """The public token, its env var and the OneUptime header are gone."""
        import kugelaudio._diagnostics as module
        import kugelaudio._diagnostics_config as config_module

        for gone in (
            "DEFAULT_TELEMETRY_TOKEN",
            "DEFAULT_TELEMETRY_ENDPOINT",
            "ENV_TELEMETRY_TOKEN",
            "TOKEN_HEADER",
        ):
            assert not hasattr(module, gone)
            assert not hasattr(config_module, gone)

        text = _reporter_source()
        assert "x-oneuptime-token" not in text
        assert "KUGELAUDIO_TELEMETRY_TOKEN" not in text
        assert "otel.kugelaudio.com" not in text


# ---------------------------------------------------------------------------
# Three consecutive 404s disable the reporter
# ---------------------------------------------------------------------------


class TestRouteMissing:
    def test_three_consecutive_404s_disable_the_reporter(self):
        sender = FakeSender(status=404)
        reporter = live_reporter(sender)

        for _ in range(MAX_CONSECUTIVE_NOT_FOUND):
            reporter.report(EVENT_REQUEST_FAILED, {"kugel.operation": OP_MODELS})
            reporter.flush(timeout=5.0)

        assert len(sender.calls) == MAX_CONSECUTIVE_NOT_FOUND == 3
        assert reporter.route_missing is True
        assert reporter.inert is True

        # Nothing more is queued and nothing more is sent, for the rest of
        # the process: an older ingress must not be pestered forever.
        for _ in range(50):
            reporter.report(EVENT_REQUEST_FAILED)
        reporter.flush(timeout=5.0)
        reporter.close(timeout=5.0)

        assert len(sender.calls) == MAX_CONSECUTIVE_NOT_FOUND
        assert reporter.pending() == 0

    def test_two_404s_alone_do_not_disable_the_reporter(self):
        sender = FakeSender(statuses=[404, 404, 202])
        reporter = live_reporter(sender)

        for _ in range(3):
            reporter.report(EVENT_REQUEST_FAILED)
            reporter.flush(timeout=5.0)

        assert len(sender.calls) == 3
        assert reporter.route_missing is False
        assert reporter.inert is False

    def test_any_other_status_resets_the_404_streak(self):
        sender = FakeSender(statuses=[404, 404, 202, 404, 404])
        reporter = live_reporter(sender)

        for _ in range(5):
            reporter.report(EVENT_REQUEST_FAILED)
            reporter.flush(timeout=5.0)

        assert len(sender.calls) == 5
        assert reporter.route_missing is False

    def test_a_404_batch_is_dropped_and_never_retried(self):
        sender = FakeSender(status=404)
        reporter = live_reporter(sender)
        reporter.report(EVENT_REQUEST_FAILED)
        reporter.flush(timeout=5.0)

        # One call, not two: a 404 is definitive. Retrying it would also make
        # "three consecutive 404s" mean one and a half batches.
        assert len(sender.calls) == 1
        assert reporter.pending() == 0

    def test_a_500_between_404s_resets_the_streak(self):
        # 404, 404, then a 500 (retried once, streak cleared), then two more
        # 404s: the streak restarts, so the reporter survives.
        sender = FakeSender(statuses=[404, 404, 500, 500, 404, 404])
        reporter = live_reporter(sender)

        for _ in range(5):
            reporter.report(EVENT_REQUEST_FAILED)
            reporter.flush(timeout=5.0)

        assert len(sender.calls) == 6
        assert reporter.route_missing is False

    def test_a_transport_failure_neither_bumps_nor_clears_the_streak(self):
        """A throw is not a response, so it must not touch the counter."""
        sender = ScriptedSender([404, 404, RuntimeError("down"), RuntimeError("down"), 404])
        reporter = live_reporter(sender)

        for _ in range(4):
            reporter.report(EVENT_REQUEST_FAILED)
            reporter.flush(timeout=5.0)

        # 1 + 1 + 2 (the throw is retried) + 1
        assert len(sender.calls) == 5
        assert reporter.route_missing is True

    def test_a_disabled_reporter_emits_no_closing_sdk_stats(self):
        sender = FakeSender(status=404)
        reporter = live_reporter(sender)
        for _ in range(MAX_CONSECUTIVE_NOT_FOUND):
            reporter.operation(OP_STREAM, TRANSPORT_WEBSOCKET).fail(
                RuntimeError("x")
            )
            reporter.flush(timeout=5.0)

        before = len(sender.calls)
        reporter.close(timeout=5.0)
        assert len(sender.calls) == before

    def test_the_kill_switch_is_per_reporter_not_global(self):
        """One process may hold clients pointed at different hosts."""
        stale = live_reporter(FakeSender(status=404))
        current = live_reporter(FakeSender(status=202))

        for _ in range(MAX_CONSECUTIVE_NOT_FOUND):
            stale.report(EVENT_REQUEST_FAILED)
            stale.flush(timeout=5.0)

        assert stale.route_missing is True
        assert current.route_missing is False
        assert current.inert is False


# ---------------------------------------------------------------------------
# OTLP envelope
# ---------------------------------------------------------------------------


class TestOtlpEncoding:
    def test_payload_has_the_contract_resource_and_scope_shape(self):
        sender = FakeSender()
        reporter = live_reporter(sender)
        reporter.report(EVENT_REQUEST_FAILED, {"kugel.operation": OP_MODELS})
        reporter.flush()

        url, headers, payload = sender.calls[0]
        assert url == TEST_DIAGNOSTICS_URL
        assert headers["Content-Type"] == "application/json"
        # The SDK's existing auth headers, verbatim — no second scheme.
        assert headers["Authorization"] == TEST_AUTH_HEADERS["Authorization"]
        assert headers["X-API-Key"] == TEST_AUTH_HEADERS["X-API-Key"]

        body = json.loads(payload.decode("utf-8"))
        resource = body["resourceLogs"][0]
        flat = {
            item["key"]: item["value"]["stringValue"]
            for item in resource["resource"]["attributes"]
        }
        assert flat == {
            "service.name": "kugelaudio-sdk",
            "service.version": "9.9.9",
            "telemetry.sdk.language": "python",
        }

        scope_logs = resource["scopeLogs"][0]
        assert scope_logs["scope"] == {
            "name": "kugelaudio.diagnostics",
            "version": "1",
        }

        record = scope_logs["logRecords"][0]
        assert record["severityNumber"] == 17
        assert record["severityText"] == "ERROR"
        assert record["body"] == {"stringValue": EVENT_REQUEST_FAILED}
        assert record["timeUnixNano"].isdigit()
        assert len(record["timeUnixNano"]) >= 19

    def test_required_attributes_are_always_present(self):
        sender = FakeSender()
        reporter = live_reporter(sender)
        reporter.report(EVENT_REQUEST_FAILED)
        reporter.flush()

        flat = attrs_of(sender.records[0])
        assert flat["kugel.event"] == EVENT_REQUEST_FAILED
        assert len(flat["kugel.event_id"]) == 32
        assert flat["kugel.event_id"] == flat["kugel.event_id"].lower()
        assert len(flat["kugel.operation_id"]) == 32
        assert flat["kugel.sdk.name"] == "python"
        assert flat["kugel.sdk.version"] == "9.9.9"
        major, minor, micro = __import__("sys").version_info[:3]
        assert flat["kugel.runtime"] == f"python/{major}.{minor}.{micro}"
        assert flat["kugel.integration"] == "none"
        assert flat["kugel.endpoint_kind"] == "hosted"

    def test_sdk_stats_is_info_and_failures_are_error(self):
        sender = FakeSender()
        reporter = live_reporter(sender)
        reporter.operation(OP_STREAM, TRANSPORT_WEBSOCKET).fail(RuntimeError("x"))
        reporter.close(timeout=5.0)

        by_event = {r["body"]["stringValue"]: r for r in sender.records}
        assert by_event[EVENT_SDK_STATS]["severityNumber"] == 9
        assert by_event[EVENT_SDK_STATS]["severityText"] == "INFO"
        failure = by_event["connection_failed"]
        assert (failure["severityNumber"], failure["severityText"]) == (17, "ERROR")

    def test_ints_are_encoded_as_otlp_string_int_values(self):
        encoded = encode_attributes(
            {"kugel.http_status": 401, "kugel.elapsed_ms": 12.9}
        )
        assert {"key": "kugel.http_status", "value": {"intValue": "401"}} in encoded
        assert {"key": "kugel.elapsed_ms", "value": {"intValue": "12"}} in encoded


# ---------------------------------------------------------------------------
# Attribute allowlist
# ---------------------------------------------------------------------------


class TestAttributeAllowlist:
    def test_unknown_keys_are_dropped_by_the_encoder(self):
        encoded = encode_attributes(
            {
                "kugel.operation": OP_STREAM,
                "kugel.text": "the secret invoice number is DE89 3704",
                "http.url": "https://api.kugelaudio.com/v1/tts",
                "api_key": "sk-live-deadbeef",
            }
        )
        assert [item["key"] for item in encoded] == ["kugel.operation"]

    def test_none_values_are_dropped(self):
        assert encode_attributes({"kugel.error_code": None}) == []

    def test_disallowed_values_never_reach_the_payload(self):
        sender = FakeSender()
        reporter = live_reporter(sender)
        reporter.report(
            EVENT_REQUEST_FAILED,
            {
                "kugel.operation": OP_STREAM,
                "text": "Guten Tag, hier ist die IBAN DE89370400440532013000",
                "api_key": "sk-live-deadbeef",
                "hostname": "customer-box.internal",
            },
        )
        reporter.flush()

        raw = sender.raw
        assert "IBAN" not in raw
        assert "sk-live-deadbeef" not in raw
        assert "customer-box.internal" not in raw
        assert OP_STREAM in raw

    def test_exception_message_is_never_sent_only_the_class_name(self):
        sender = FakeSender()
        reporter = live_reporter(sender)
        op = reporter.operation(OP_STREAM, TRANSPORT_WEBSOCKET)
        op.fail(ValueError("customer said: my card is 4111 1111 1111 1111"))
        reporter.flush()

        raw = sender.raw
        assert "4111" not in raw
        assert "customer said" not in raw
        assert attrs_of(sender.records[0])["kugel.error_type"] == "ValueError"


# ---------------------------------------------------------------------------
# Bounded queue and the per-POST cap
# ---------------------------------------------------------------------------


class TestQueueBound:
    def test_hundred_events_keep_the_newest_sixty_four(self):
        sender = FakeSender()
        reporter = live_reporter(sender)  # autostart=False: nothing drains it
        for index in range(100):
            reporter.report(EVENT_REQUEST_FAILED, {"kugel.error_code": str(index)})

        assert reporter.pending() == QUEUE_CAPACITY == 64

        reporter.flush(timeout=5.0)
        codes = [attrs_of(r)["kugel.error_code"] for r in sender.records]
        # The OLDEST were dropped; 36..99 survived, in order.
        assert codes == [str(i) for i in range(36, 100)]

    def test_flush_sends_in_batches_of_eight(self):
        sender = FakeSender()
        reporter = live_reporter(sender)
        for _ in range(BATCH_SIZE * 2):
            reporter.report(EVENT_REQUEST_FAILED)
        reporter.flush(timeout=5.0)

        assert len(sender.calls) == 2
        assert len(sender.records) == BATCH_SIZE * 2

    def test_every_post_carries_at_most_eight_records_on_every_path(self):
        """Report path, flush path and close path alike (contract: <= 8)."""
        sender = FakeSender()
        reporter = Diagnostics(
            DiagnosticsConfig(
                enabled=True, url=TEST_DIAGNOSTICS_URL, endpoint_kind="hosted"
            ),
            sender,
            version="9.9.9",
        )
        for _ in range(BATCH_SIZE * 2 + 3):
            reporter.operation(OP_STREAM, TRANSPORT_WEBSOCKET).fail(RuntimeError("x"))
        reporter.flush(timeout=5.0)
        for _ in range(BATCH_SIZE + 5):
            reporter.operation(OP_STREAM, TRANSPORT_WEBSOCKET).fail(RuntimeError("x"))
        reporter.close(timeout=5.0)

        assert sender.batches, "nothing was delivered"
        assert max(len(batch) for batch in sender.batches) <= BATCH_SIZE == 8
        # 19 + 13 failures + one sdk_stats, nothing lost.
        assert len(sender.records) == 33


# ---------------------------------------------------------------------------
# Delivery: all network I/O on the worker thread, close() bounded
# ---------------------------------------------------------------------------


class TestCloseIsBounded:
    def test_close_with_a_hanging_sender_returns_within_the_deadline(self):
        sender = FakeSender(delay_s=5.0)
        reporter = live_reporter(sender)
        reporter.report(EVENT_REQUEST_FAILED)

        started = time.monotonic()
        reporter.close()
        elapsed = time.monotonic() - started

        # 1 s budget plus runner slack; a missing cap would take the 5 s sleep.
        assert elapsed <= 1.5
        assert len(sender.calls) == 1  # the worker did pick the batch up

    def test_close_never_sends_on_the_caller_thread(self):
        sender = FakeSender()
        reporter = live_reporter(sender)
        reporter.operation(OP_MODELS, "http").succeed()
        reporter.report(EVENT_REQUEST_FAILED)
        reporter.close(timeout=5.0)

        assert len(sender.calls) == 1
        assert threading.current_thread() not in sender.threads

    def test_close_queues_sdk_stats_only_when_a_count_is_non_zero(self):
        idle = FakeSender()
        live_reporter(idle).close(timeout=5.0)
        assert idle.calls == []

        busy = FakeSender()
        reporter = live_reporter(busy)
        reporter.operation(OP_MODELS, "http").succeed()
        reporter.close(timeout=5.0)
        assert [attrs_of(r)["kugel.event"] for r in busy.records] == [EVENT_SDK_STATS]

    async def test_aclose_does_not_block_the_event_loop(self):
        client = KugelAudio(api_key="k")
        client._diagnostics = live_reporter(FakeSender(delay_s=5.0))
        client._diagnostics.report(EVENT_REQUEST_FAILED)

        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        ticking = asyncio.create_task(ticker())
        started = time.monotonic()
        await client.aclose()
        elapsed = time.monotonic() - started
        ticking.cancel()

        assert elapsed <= 1.6
        # ~1 s of bounded wait: a blocked loop would have ticked ~0 times.
        assert ticks >= 10


# ---------------------------------------------------------------------------
# A failing sender never reaches the caller
# ---------------------------------------------------------------------------


class TestFailuresAreSwallowed:
    def test_failing_sender_does_not_raise_and_retries_once(self):
        sender = FakeSender(fail=True)
        reporter = live_reporter(sender)
        reporter.report(EVENT_REQUEST_FAILED)
        reporter.flush(timeout=5.0)  # must not raise

        # Initial attempt + at most one retry, then the batch is dropped.
        assert len(sender.calls) == 2
        assert reporter.pending() == 0

    def test_close_with_a_failing_sender_is_silent(self):
        reporter = live_reporter(FakeSender(fail=True))
        reporter.report(EVENT_REQUEST_FAILED)
        reporter.close()  # must not raise

    def test_a_broken_operation_never_breaks_the_caller(self):
        reporter = live_reporter(FakeSender(fail=True))
        op = reporter.operation(OP_STREAM, TRANSPORT_WEBSOCKET)
        op.fail(RuntimeError("boom"))  # must not raise

    @pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
    def test_every_non_2xx_is_a_silent_drop(self, status):
        sender = FakeSender(status=status)
        reporter = live_reporter(sender)
        reporter.report(EVENT_REQUEST_FAILED)
        reporter.flush(timeout=5.0)  # must not raise

        # Initial attempt + the one allowed retry, then dropped.
        assert len(sender.calls) == 2
        assert reporter.pending() == 0
        assert reporter.route_missing is False  # only 404 counts
        assert reporter.inert is False

    def test_a_413_is_dropped_and_never_retried(self):
        sender = FakeSender(status=413)
        reporter = live_reporter(sender)
        reporter.report(EVENT_REQUEST_FAILED)
        reporter.flush(timeout=5.0)

        assert len(sender.calls) == 1
        assert reporter.pending() == 0
        assert reporter.route_missing is False

    def test_a_non_2xx_never_surfaces_on_the_caller_error_path(self):
        reporter = live_reporter(FakeSender(status=500))
        op = reporter.operation(OP_STREAM, TRANSPORT_WEBSOCKET)
        op.fail(RuntimeError("boom"))  # must not raise
        reporter.close(timeout=5.0)  # must not raise


# ---------------------------------------------------------------------------
# No network, no new dependency
# ---------------------------------------------------------------------------


class TestNoNetworkByDefault:
    def test_module_imports_only_stdlib_and_existing_dependencies(self):
        import kugelaudio._diagnostics as module

        source = module.__file__
        with open(source, "r", encoding="utf-8") as handle:
            text = handle.read()
        for forbidden in ("import requests", "import opentelemetry", "import aiohttp"):
            assert forbidden not in text

    def test_client_close_does_not_touch_the_network(self):
        sender = FakeSender()
        client = KugelAudio(api_key="k")
        client._diagnostics = live_reporter(sender)
        client.close()
        # No operations ran, so sdk_stats is not emitted either.
        assert sender.calls == []

    @pytest.mark.real_diagnostics_sender
    def test_default_sender_returns_the_status_code(self, monkeypatch):
        """The sender seam: the transport hands a status back, it never guesses."""
        import urllib.request

        from kugelaudio._diagnostics import urllib_sender

        class _Response:
            status = 202

            def read(self):
                return b""

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(
            urllib.request, "urlopen", lambda *a, **kw: _Response()
        )
        assert (
            urllib_sender(TEST_DIAGNOSTICS_URL, {}, b"{}") == 202
        )

    @pytest.mark.real_diagnostics_sender
    def test_default_sender_turns_a_404_into_a_status_not_an_exception(
        self, monkeypatch
    ):
        import io
        import urllib.error
        import urllib.request

        from kugelaudio._diagnostics import urllib_sender

        def _raise(*a, **kw):
            raise urllib.error.HTTPError(
                TEST_DIAGNOSTICS_URL, 404, "Not Found", {}, io.BytesIO(b"")
            )

        monkeypatch.setattr(urllib.request, "urlopen", _raise)
        assert urllib_sender(TEST_DIAGNOSTICS_URL, {}, b"{}") == 404

    def test_httpx_is_not_used_by_the_reporter(self):
        # The reporter must not borrow the client's httpx pool: telemetry has
        # to survive a closed client.
        import kugelaudio._diagnostics as module

        assert not hasattr(module, "httpx")
        assert httpx is not None  # keep the import meaningful
