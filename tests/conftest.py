"""Pytest config for the Python SDK tests.

``test_local_kugel2.py`` and ``test_staging_kugel2.py`` are standalone CLI
scripts (``uv run python tests/test_foo.py``) that happen to define
``test_*`` coroutines as internal helpers. They are not pytest fixtures,
so pytest fails to resolve their ``client`` parameter at collection time.
Skip them from pytest discovery.
"""

from __future__ import annotations

from typing import Dict

import pytest

import kugelaudio._diagnostics as _diagnostics

collect_ignore = [
    "test_local_kugel2.py",
    "test_staging_kugel2.py",
]


REAL_SENDER_MARKER = "real_diagnostics_sender"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{REAL_SENDER_MARKER}: exercise the real stdlib sender "
        "(the test must stub urlopen itself)",
    )


def _accept_without_sending(url: str, headers: Dict[str, str], payload: bytes) -> int:
    """Stand-in for the stdlib sender: answers 202, touches no socket."""
    return 202


@pytest.fixture(autouse=True)
def _diagnostics_never_hits_the_network(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the diagnostics reporter off the wire in unit tests.

    The reporter targets the client's own API host and is enabled by default
    against ``*.kugelaudio.com``, so a client built in a unit test would
    really POST from its delivery thread. Swapping the default sender leaves
    enablement resolution untouched (that is what the tests assert on) while
    guaranteeing zero network I/O. Tests that assert on delivery inject their
    own sender; tests of the stdlib sender itself carry ``REAL_SENDER_MARKER``
    and stub ``urlopen`` instead, so they too touch no socket.
    """
    if REAL_SENDER_MARKER in request.keywords:
        return
    monkeypatch.setattr(_diagnostics, "urllib_sender", _accept_without_sending)
