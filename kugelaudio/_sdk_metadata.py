"""SDK metadata sent to ingress for observability."""

from __future__ import annotations

from importlib import metadata
from urllib.parse import urlencode

SDK_NAME = "python"


def sdk_version() -> str:
    try:
        return metadata.version("kugelaudio")
    except metadata.PackageNotFoundError:
        return "unknown"


def sdk_headers() -> dict[str, str]:
    version = sdk_version()
    return {
        "X-KugelAudio-SDK": SDK_NAME,
        "X-KugelAudio-SDK-Version": version,
        "User-Agent": f"kugelaudio-python/{version}",
    }


def sdk_query_string() -> str:
    return urlencode({"sdk": SDK_NAME, "sdk_version": sdk_version()})


__all__ = ["SDK_NAME", "sdk_headers", "sdk_query_string", "sdk_version"]
