"""Diagnostics over a real socket, with the real stdlib sender.

A ``ThreadingHTTPServer`` stands in for ingress (``diagnostics_wire_support``).
No sender is injected: these tests prove what actually leaves the process.
Failure modes run in subprocesses, so interpreter shutdown, stray threads and
anything printed to stdout/stderr are observed exactly as a user would.
Contract: ``services/ingress/docs/sdk-diagnostics-contract.md``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from kugelaudio import KugelAudio

from .diagnostics_wire_support import (  # noqa: F401  (fixtures)
    OPT_OUT_MATRIX,
    DiagnosticsStub,
    diagnostics_stub,
    hosted_probe,
)

pytestmark = pytest.mark.real_diagnostics_sender

SECRET_TEXT = "GEHEIM-EINGABETEXT-4711"


def _fail_once(client: KugelAudio) -> None:
    with pytest.raises(Exception):
        client.models.list()


# ---------------------------------------------------------------------------
# a/b: opt-out matrix and the hosted default, on the wire
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("env", "option", "hosted", "expect_posts"), OPT_OUT_MATRIX)
def test_core_client_opt_out_matrix_on_the_wire(
    diagnostics_stub, hosted_probe, monkeypatch, env, option, hosted, expect_posts
):
    if env is not None:
        monkeypatch.setenv("KUGELAUDIO_TELEMETRY", env)
    url = hosted_probe(diagnostics_stub) if hosted else diagnostics_stub.url()
    client = KugelAudio(api_key="wire-key", api_url=url, telemetry=option)

    _fail_once(client)
    client.close()

    assert bool(diagnostics_stub.posts) is expect_posts


# ---------------------------------------------------------------------------
# c: what a POST carries
# ---------------------------------------------------------------------------


def test_posts_are_json_authenticated_capped_and_free_of_input_text(
    diagnostics_stub, hosted_probe
):
    client = KugelAudio(api_key="wire-key", api_url=hosted_probe(diagnostics_stub))
    for _ in range(10):
        with pytest.raises(Exception):
            client.dictionaries.create(name=SECRET_TEXT)
    client.close()

    # The text really went to the API, so its absence below means something.
    assert any(SECRET_TEXT.encode() in body for body in diagnostics_stub.api_bodies)
    posts = diagnostics_stub.posts
    assert posts
    for post in posts:
        headers = {name.lower(): value for name, value in post.headers.items()}
        assert headers["content-type"] == "application/json"
        assert headers["authorization"] == "Bearer wire-key"
        assert headers["x-api-key"] == "wire-key"
        assert len(post.records) <= 8
        assert SECRET_TEXT.encode() not in post.body
    # 10 failures + sdk_stats, none lost to the cap.
    assert sum(len(post.records) for post in posts) == 11


# ---------------------------------------------------------------------------
# d/f: endpoint failure modes, in subprocesses
# ---------------------------------------------------------------------------

#: Runs one scenario and writes its measurements to argv[5] as JSON. Prints
#: nothing itself, so any output on stdout/stderr came from the SDK.
CHILD = r"""
import asyncio, json, logging, socket, sys, time

mode, level, base_url, dead_port, out_path = sys.argv[1:6]
logging.basicConfig(level=getattr(logging, level))

_real_getaddrinfo = socket.getaddrinfo
def _getaddrinfo(host, *args, **kwargs):
    if isinstance(host, str) and host.endswith(".invalid"):
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
    return _real_getaddrinfo(host, *args, **kwargs)
socket.getaddrinfo = _getaddrinfo

import kugelaudio._diagnostics as diagnostics
from kugelaudio import KugelAudio

def _boom(*args, **kwargs):
    raise RuntimeError("telemetry exploded")

if mode == "sender_raises":
    diagnostics.urllib_sender = _boom
if mode == "encoder_raises":
    diagnostics.build_payload = _boom

def url_for(flavor):
    if mode == "refused":
        return f"http://127.0.0.1:{dead_port}"
    if mode == "unresolvable":
        return f"http://ingress.invalid:{dead_port}"
    return f"{base_url}/{level}-{flavor}"

calls = 4 if mode == "404" else 1
result = {}

def sync_errors(telemetry):
    client = KugelAudio(api_key="wire-key", api_url=url_for("sync"), telemetry=telemetry)
    errors = []
    for _ in range(calls):
        try:
            client.models.list()
        except Exception as exc:
            errors.append(type(exc).__name__)
        if telemetry and mode == "404":
            client._diagnostics.flush(2.0)
    started = time.monotonic()
    client.close()
    return errors, time.monotonic() - started

async def async_errors(telemetry):
    client = KugelAudio(api_key="wire-key", api_url=url_for("async"), telemetry=telemetry)
    errors = []
    for _ in range(calls):
        session = client.tts.streaming_session(voice_id=1, language="de")
        try:
            await session.connect()
        except Exception as exc:
            errors.append(type(exc).__name__)
        if telemetry and mode == "404":
            await asyncio.to_thread(client._diagnostics.flush, 2.0)
    started = time.monotonic()
    await client.aclose()
    return errors, time.monotonic() - started

result["sync_off"], _ = sync_errors(False)
result["sync_on"], result["sync_close_s"] = sync_errors(True)
result["async_off"], _ = asyncio.run(async_errors(False))
result["async_on"], result["async_close_s"] = asyncio.run(async_errors(True))
result["done_at"] = time.time()
with open(out_path, "w") as handle:
    json.dump(result, handle)
"""

#: The contract's 1 s budget plus slack for loaded CI runners; still clearly
#: under the 3 s request timeout (and under 2 s, what two sequential 1 s
#: waits would take), so a missing cap fails.
CLOSE_BOUND_S = 1.5
#: "No delay at all" plus scheduling slack.
NO_DELAY_S = 0.5

MODES = ["404", "500", "413", "hang", "refused", "unresolvable", "sender_raises", "encoder_raises"]
LEVELS = ["WARNING", "INFO"]
#: POSTs expected per (level, flavor) run on the wire.
EXPECTED_POSTS = {"404": 3, "500": 2, "413": 1, "hang": 1}


def _dead_port() -> int:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _run_child(
    mode: str, level: str, base_url: str, dead_port: int, out_dir: str
) -> Tuple[subprocess.CompletedProcess, float, Path]:
    out_path = Path(out_dir) / f"{mode}-{level}.json"
    env = {k: v for k, v in os.environ.items() if k != "KUGELAUDIO_TELEMETRY"}
    completed = subprocess.run(
        [
            sys.executable,
            "-W",
            "always::ResourceWarning",
            "-c",
            CHILD,
            mode,
            level,
            base_url,
            str(dead_port),
            str(out_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
    )
    return completed, time.time(), out_path


def _unexpected_output(text: str, allowed: List[str]) -> List[str]:
    """Every non-empty line that is not an exact, pre-existing SDK hint."""
    return [line for line in text.splitlines() if line.strip() and line not in allowed]


def test_endpoint_failure_modes_are_silent_bounded_and_invisible(diagnostics_stub):
    dead_port = _dead_port()
    runs: Dict[Tuple[str, str], Any] = {}
    with tempfile.TemporaryDirectory() as out_dir, ThreadPoolExecutor(
        # Bounded, so a loaded CI runner is not asked for 16 interpreters at once.
        max_workers=8
    ) as pool:
        futures = {
            (mode, level): pool.submit(
                _run_child,
                mode,
                level,
                diagnostics_stub.url(mode if mode in EXPECTED_POSTS else "accept"),
                dead_port,
                out_dir,
            )
            for mode in MODES
            for level in LEVELS
        }
        for key, future in futures.items():
            completed, ended_at, out_path = future.result()
            result = json.loads(out_path.read_text()) if out_path.exists() else None
            runs[key] = (completed, ended_at, result)

    failures: List[str] = []
    for (mode, level), (completed, ended_at, result) in runs.items():
        tag = f"{mode}/{level}"
        if completed.returncode != 0 or result is None:
            failures.append(f"{tag}: exit {completed.returncode}: {completed.stderr[-2000:]}")
            continue
        base = diagnostics_stub.url(mode if mode in EXPECTED_POSTS else "accept")
        # httpx's own INFO request line is pre-existing and not telemetry.
        allowed = [
            f'INFO:httpx:HTTP Request: GET {base}/{level}-sync/v1/models '
            '"HTTP/1.1 401 Unauthorized"'
        ]
        noise = _unexpected_output(completed.stdout, []) + _unexpected_output(
            completed.stderr, allowed
        )
        if noise:
            failures.append(f"{tag}: printed {noise}")
        for flavor in ("sync", "async"):
            if result[f"{flavor}_on"] != result[f"{flavor}_off"]:
                failures.append(
                    f"{tag}: {flavor} error changed {result[f'{flavor}_off']} -> "
                    f"{result[f'{flavor}_on']}"
                )
            # Budget 1 s; the slack absorbs a loaded runner and still sits well
            # below the 3 s request timeout a missing cap would show.
            if result[f"{flavor}_close_s"] > CLOSE_BOUND_S:
                failures.append(f"{tag}: {flavor} close took {result[f'{flavor}_close_s']:.2f}s")
            if mode in EXPECTED_POSTS:
                seen = len(diagnostics_stub.posts_under(f"{mode}/{level}-{flavor}"))
                if seen != EXPECTED_POSTS[mode]:
                    failures.append(f"{tag}: {flavor} sent {seen} POSTs")
        exit_after_close = ended_at - result["done_at"]
        if exit_after_close > 2.0:
            failures.append(f"{tag}: exited {exit_after_close:.2f}s after close")

    assert failures == []
    # The failing modes all hit the API for real: the SDK error is the 401.
    _completed, _ended, result = runs[("500", "WARNING")]
    assert result["sync_on"] == ["AuthenticationError"]
    assert result["async_on"] == ["AuthenticationError"]


# ---------------------------------------------------------------------------
# e: the reporter never logs above DEBUG
# ---------------------------------------------------------------------------


def test_every_reporter_log_call_is_debug():
    import ast

    import kugelaudio._diagnostics as module
    import kugelaudio._diagnostics_config as config_module

    trees = [
        ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        for mod in (module, config_module)
    ]
    levels = {
        node.func.attr
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"logger", "logging"}
    }
    assert levels == {"debug", "getLogger"}


# ---------------------------------------------------------------------------
# Exit flush: uncaught errors on clients never closed
# ---------------------------------------------------------------------------

#: argv: scenario, api url, out path. Records when the uncaught exception hit
#: and when the exit hooks finished (its own hook is registered first, so it
#: runs last), so the delay added by the exit flush is measured in-process.
EXIT_CHILD = r"""
import atexit, json, sys, time

scenario, url, out_path = sys.argv[1:4]
marks = {}

def _record_exit():
    marks["hooks_done_at"] = time.time()
    with open(out_path, "w") as handle:
        json.dump(marks, handle)

atexit.register(_record_exit)
_excepthook = sys.excepthook
def _timed_excepthook(*args):
    marks["uncaught_at"] = time.time()
    _excepthook(*args)
sys.excepthook = _timed_excepthook

from kugelaudio import KugelAudio

client = KugelAudio(api_key="wire-key", api_url=url, telemetry=scenario != "off")
if scenario == "many":
    clients = [client] + [
        KugelAudio(api_key="wire-key", api_url=url, telemetry=True) for _ in range(2)
    ]
    for each in clients:
        try:
            each.models.list()
        except Exception:
            pass
    raise RuntimeError("the user's own bug")
if scenario in ("drained", "closed"):
    try:
        client.models.list()
    except Exception:
        pass
    if scenario == "drained":
        client._diagnostics.flush(2.0)
    else:
        client.close()
    raise RuntimeError("the user's own bug")
client.tts.generate("Hallo Welt", voice_id=1, language="de")
"""


def _run_exit_child(scenario: str, url: str, out_dir: str) -> Tuple[Any, Dict[str, float]]:
    out_path = Path(out_dir) / f"exit-{scenario}.json"
    env = {k: v for k, v in os.environ.items() if k != "KUGELAUDIO_TELEMETRY"}
    completed = subprocess.run(
        [sys.executable, "-c", EXIT_CHILD, scenario, url, str(out_path)],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
    )
    marks = json.loads(out_path.read_text()) if out_path.exists() else {}
    return completed, marks


def test_exit_flush_sends_uncaught_failures_and_bounds_the_delay(diagnostics_stub):
    scenarios = {
        "on": diagnostics_stub.url("accept/exit-on"),
        "off": diagnostics_stub.url("accept/exit-on"),
        "hang": diagnostics_stub.url("hang/exit"),
        "many": diagnostics_stub.url("hang/exit-many"),
        "drained": diagnostics_stub.url("accept/exit-drained"),
        "closed": diagnostics_stub.url("accept/exit-closed"),
    }
    with tempfile.TemporaryDirectory() as out_dir, ThreadPoolExecutor(
        max_workers=len(scenarios)
    ) as pool:
        futures = {
            name: pool.submit(_run_exit_child, name, url, out_dir)
            for name, url in scenarios.items()
        }
        runs = {name: future.result() for name, future in futures.items()}

    def delay(name: str) -> float:
        marks = runs[name][1]
        return marks["hooks_done_at"] - marks["uncaught_at"]

    for name, (completed, marks) in runs.items():
        assert completed.returncode == 1, (name, completed.stderr[-2000:])
        assert completed.stdout == "", name
        assert set(marks) == {"uncaught_at", "hooks_done_at"}, name

    # (1) The uncaught SDK failure reached the stub, within the 1 s budget,
    # and stderr is exactly the traceback the user gets without telemetry.
    on, off = runs["on"][0], runs["off"][0]
    events = [
        record["body"]["stringValue"]
        for post in diagnostics_stub.posts_under("accept/exit-on")
        for record in post.records
    ]
    assert set(events) & {"connection_failed", "request_failed"}
    assert delay("on") <= CLOSE_BOUND_S
    assert on.stderr == off.stderr
    assert on.stderr.startswith("Traceback (most recent call last):")
    assert on.stderr.rstrip().splitlines()[-1].startswith(
        "kugelaudio.exceptions.AuthenticationError"
    )
    # (2) Nothing queued: the hook adds no delay.
    assert delay("drained") <= NO_DELAY_S
    assert len(diagnostics_stub.posts_under("accept/exit-drained")) == 1
    # (3) A hung endpoint delays exit by about the 1 s budget, then it exits:
    # clearly bounded, and clearly below the 3 s request timeout.
    assert 0.7 <= delay("hang") <= 1.8
    # (1b) Three unclosed clients against the hung endpoint share ONE 1 s
    # deadline; per-client waits would add about 3 s.
    assert 0.7 <= delay("many") <= 1.8
    # (4) An explicit close() already flushed: the hook sends nothing more.
    assert delay("closed") <= NO_DELAY_S
    assert len(diagnostics_stub.posts_under("accept/exit-closed")) == 1


# ---------------------------------------------------------------------------
# Fork safety
# ---------------------------------------------------------------------------

#: argv: out path. Queues a record, forks while HOLDING the reporter's lock
#: (as if another thread held it at fork time), and records which process
#: ever called the sender. The child must neither deadlock nor re-send.
FORK_CHILD = r"""
import json, os, sys, time, warnings
warnings.simplefilter("ignore", DeprecationWarning)  # fork with threads alive

from kugelaudio._diagnostics import Diagnostics
from kugelaudio._diagnostics_config import DiagnosticsConfig

out_path = sys.argv[1]
senders = []

def sender(url, headers, payload):
    with open(out_path + ".sends", "a") as handle:
        handle.write(f"{os.getpid()}\n")
    return 202

config = DiagnosticsConfig(enabled=True, url="http://127.0.0.1:9/v1/sdk-diagnostics")
reporter = Diagnostics(config, sender, autostart=False)
reporter.report("request_failed")

reporter._changed.acquire()
pid = os.fork()
if pid == 0:
    reporter.report("request_failed")      # must not queue in the child
    reporter.flush(1.0)                    # must not block on the copied lock
    state = {"pending": reporter.pending(), "inert": reporter.inert}
    reporter.close()
    with open(out_path + ".child", "w") as handle:
        json.dump(state, handle)
    sys.exit(0)                            # runs the child's atexit hook too
reporter._changed.release()
deadline = time.monotonic() + 10
while time.monotonic() < deadline:
    done, status = os.waitpid(pid, os.WNOHANG)
    if done:
        break
    time.sleep(0.02)
else:
    os.kill(pid, 9)
    status = -1
reporter.close(2.0)
with open(out_path, "w") as handle:
    json.dump({"parent": os.getpid(), "child_status": status}, handle)
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="os.fork is POSIX-only")
def test_a_forked_child_never_resends_or_blocks_on_inherited_reporters():
    with tempfile.TemporaryDirectory() as out_dir:
        out = Path(out_dir) / "fork"
        completed = subprocess.run(
            [sys.executable, "-c", FORK_CHILD, str(out)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=Path(__file__).resolve().parents[1],
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        parent = json.loads(out.read_text())
        child = json.loads(Path(str(out) + ".child").read_text())
        senders = Path(str(out) + ".sends").read_text().split()

    assert parent["child_status"] == 0  # exited, did not deadlock
    assert child == {"pending": 0, "inert": True}
    # Only the parent ever sent, exactly its one batch.
    assert senders == [str(parent["parent"])]
