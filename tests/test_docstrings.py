"""Public docstrings must name the current model and the server's real limits."""

from __future__ import annotations

import inspect

import pytest

import kugelaudio
from kugelaudio import client, models, streaming


@pytest.mark.parametrize("module", [kugelaudio, client, streaming])
def test_no_retired_model_ids_in_public_docs(module):
    """``kugel-1``/``kugel-1-turbo`` are retired; examples should show ``kugel-3``."""
    assert "kugel-1" not in inspect.getsource(module)


@pytest.mark.parametrize("module", [client, streaming])
def test_multi_context_limit_matches_server(module):
    """Ingress allows 20 contexts per multi-context socket, not 5."""
    source = inspect.getsource(module)
    assert "up to 5 " not in source
    assert "up to 20 independent" in source


@pytest.mark.parametrize("module", [client, models])
def test_speed_docs_do_not_name_wsola(module):
    """Ingress stretches with TDHS (services/ingress/native/src/time_stretch.rs), not WSOLA."""
    assert "WSOLA" not in inspect.getsource(module)
