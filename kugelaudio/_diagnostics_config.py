"""Diagnostics enablement and target resolution.

Contract ``services/ingress/docs/sdk-diagnostics-contract.md``, sections
"Enablement" and "Wire format". Kept apart from the reporter
(:mod:`kugelaudio._diagnostics`) so each module stays small; nothing here
touches the network.
"""

from __future__ import annotations

import os
from typing import Dict, Mapping, Optional
from urllib.parse import urlparse

#: Appended to the effective API base URL ("Wire format"). Diagnostics go to
#: the SAME authenticated host the SDK already calls, so the server can
#: attribute every event to the org whose key signed the request. There is
#: no override: the auth headers must never be sent anywhere else.
DIAGNOSTICS_PATH = "/v1/sdk-diagnostics"

ENV_TELEMETRY = "KUGELAUDIO_TELEMETRY"

HOSTED_HOST_SUFFIX = ".kugelaudio.com"
ENDPOINT_KIND_HOSTED = "hosted"
ENDPOINT_KIND_CUSTOM = "custom"

_TRUE_VALUES = frozenset({"1", "true", "on", "yes"})
_FALSE_VALUES = frozenset({"0", "false", "off", "no"})




def endpoint_kind_for(api_url: Optional[str]) -> str:
    """``hosted`` when the effective API host is ours, else ``custom``."""
    if not api_url:
        return ENDPOINT_KIND_CUSTOM
    try:
        host = (urlparse(api_url).hostname or "").lower()
    except ValueError:
        # An unparseable URL fails closed: ``custom`` means telemetry is off
        # by default, which is the safe direction.
        return ENDPOINT_KIND_CUSTOM
    return (
        ENDPOINT_KIND_HOSTED
        if host.endswith(HOSTED_HOST_SUFFIX)
        else ENDPOINT_KIND_CUSTOM
    )


def diagnostics_url_for(api_url: Optional[str]) -> str:
    """``<effective API base URL>/v1/sdk-diagnostics``.

    Returns ``""`` when there is no API URL to derive from: the reporter
    treats that as "nowhere to POST" and stays inert.
    """
    if not api_url:
        return ""
    return api_url.rstrip("/") + DIAGNOSTICS_PATH


class DiagnosticsConfig:
    """Resolved telemetry configuration.

    Resolution order (first match wins), identical in all three SDKs:

    1. ``KUGELAUDIO_TELEMETRY`` env var,
    2. the explicit ``telemetry=`` constructor option,
    3. enabled only when the effective API host ends in ``.kugelaudio.com``.

    ``auth_headers`` are the client's existing auth headers passed through
    verbatim: this class never builds credentials of its own.
    """

    __slots__ = ("enabled", "url", "auth_headers", "endpoint_kind")

    def __init__(
        self,
        *,
        enabled: bool,
        url: str = "",
        auth_headers: Optional[Mapping[str, str]] = None,
        endpoint_kind: str = ENDPOINT_KIND_CUSTOM,
    ) -> None:
        self.enabled = enabled
        self.url = url
        self.auth_headers: Dict[str, str] = dict(auth_headers or {})
        self.endpoint_kind = endpoint_kind

    @property
    def inert(self) -> bool:
        """True when the reporter must make no network call at all."""
        return not self.enabled or not self.url

    @classmethod
    def disabled(cls) -> DiagnosticsConfig:
        return cls(enabled=False)

    @classmethod
    def resolve(
        cls,
        api_url: Optional[str],
        telemetry: Optional[bool] = None,
        env: Optional[Mapping[str, str]] = None,
        auth_headers: Optional[Mapping[str, str]] = None,
    ) -> DiagnosticsConfig:
        environ: Mapping[str, str] = os.environ if env is None else env
        endpoint_kind = endpoint_kind_for(api_url)

        raw = (environ.get(ENV_TELEMETRY) or "").strip().lower()
        if raw in _FALSE_VALUES:
            enabled = False
        elif raw in _TRUE_VALUES:
            enabled = True
        elif telemetry is not None:
            enabled = bool(telemetry)
        else:
            enabled = endpoint_kind == ENDPOINT_KIND_HOSTED

        return cls(
            enabled=enabled,
            url=diagnostics_url_for(api_url),
            auth_headers=auth_headers,
            endpoint_kind=endpoint_kind,
        )
