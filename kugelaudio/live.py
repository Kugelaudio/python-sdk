"""OpenAI Live client configuration for KugelAudio speech-to-speech."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

from kugelaudio.client import _parse_api_key
from kugelaudio.exceptions import ValidationError

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.live import SessionConfigParam


KUGEL_LIVE_MODEL: str = "fluid-1"


class KugelLiveExtensions(TypedDict, total=False):
    """Additive Kugel options nested under ``session.kugel``."""

    overlap_control: Literal["cancel_on_speech", "suspend_and_classify"]


if TYPE_CHECKING:

    class KugelLiveSessionConfig(SessionConfigParam, total=False):
        """OpenAI Live session configuration with additive Kugel options."""

        kugel: KugelLiveExtensions

else:

    class KugelLiveSessionConfig(TypedDict, total=False):
        """Runtime representation of the typed Live session configuration."""

        kugel: KugelLiveExtensions


def create_live_client(
    api_key: str,
    *,
    base_url: str,
) -> AsyncOpenAI:
    """Return the official OpenAI async client configured for Kugel Live.

    ``base_url`` follows the OpenAI SDK convention and must include the API version,
    for example ``http://localhost:8080/v1``. It is required until the independently
    deployed Live service has a stable public hostname.

    Use ``delegation={"type": "client"}``: your application owns delegated
    reasoning, tools and business logic. Return context using the SDK's native
    ``connection.session.thinking.append`` and ``commentary.append`` methods.
    Provider-managed Responses delegation and stored sessions are not supported.
    """
    if not api_key:
        raise ValidationError(
            "KugelAudio API key is missing. Set the KUGELAUDIO_API_KEY "
            "environment variable or pass api_key=... to create_live_client()."
        )

    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError(
            "OpenAI Live support requires the optional dependency. "
            "Install with: pip install 'kugelaudio[live]'"
        ) from exc

    if not hasattr(AsyncOpenAI, "live"):
        raise ImportError(
            "The installed OpenAI SDK does not provide the Live API. "
            "Install or upgrade with: pip install 'kugelaudio[live]'"
        )

    clean_key, _detected_region = _parse_api_key(api_key)
    if not base_url.strip():
        raise ValidationError("Live base_url must not be empty")
    return AsyncOpenAI(api_key=clean_key, base_url=base_url.rstrip("/"))


__all__: list[str] = [
    "KUGEL_LIVE_MODEL",
    "KugelLiveExtensions",
    "KugelLiveSessionConfig",
    "create_live_client",
]
