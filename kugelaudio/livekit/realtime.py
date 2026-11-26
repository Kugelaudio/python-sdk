# Copyright 2026 KugelAudio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fluid 1 speech to speech as a LiveKit Agents realtime model.

KugelAudio's speech-to-speech API speaks the OpenAI Live protocol, so this model is
LiveKit's own ``GPTLiveModel`` with KugelAudio's endpoint, key and defaults::

    from livekit.agents import Agent, AgentSession, function_tool
    from kugelaudio.livekit import RealtimeModel

    session = AgentSession(llm=RealtimeModel(voice=973))
    await session.start(room=ctx.room, agent=Agent(instructions="..."))
"""

from __future__ import annotations

import os
from typing import Literal

import aiohttp
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.plugins.openai.realtime import GPTLiveModel, ResponsesDelegationOptions

from kugelaudio.client import _parse_api_key, _resolve_region_url
from kugelaudio.exceptions import ValidationError
from kugelaudio.live import KUGEL_LIVE_MODEL

DEFAULT_BACK_OFFICE_MODEL: str = "mistral-medium-3.5-128b"
"""The back-office model of ``delegation="responses"`` when ``responses_options`` names
none. LiveKit's plugin always sends a model, and its own default is not one KugelAudio
runs, so the server's documented default is named here."""

Delegation = Literal["responses", "client"]

_RESPONSES_KEYS: frozenset[str] = frozenset(
    {"model", "instructions", "tool_choice", "parallel_tool_calls", "reasoning"}
)
_TOOL_CHOICES: frozenset[str] = frozenset({"auto", "none", "required"})
_EFFORTS: frozenset[str] = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh"}
)


def _check_responses_options(options: ResponsesDelegationOptions) -> None:
    """Refuse what KugelAudio's back office does not take; the server would reject it.

    LiveKit's type also carries OpenAI-only settings (``text``, ``service_tier``,
    ``max_output_tokens``), and a ``tool_choice`` naming one function.
    """
    unsupported = sorted(set(options) - _RESPONSES_KEYS)
    if unsupported:
        raise ValueError(
            f"responses_options does not support {', '.join(unsupported)}; "
            f"supported: {', '.join(sorted(_RESPONSES_KEYS))}"
        )
    tool_choice = options.get("tool_choice")
    if tool_choice is not None and (
        not isinstance(tool_choice, str) or tool_choice not in _TOOL_CHOICES
    ):
        raise ValueError(
            'responses_options["tool_choice"] must be "auto", "none" or "required"'
        )
    reasoning = options.get("reasoning")
    if reasoning is not None and (
        set(reasoning) - {"effort"} or reasoning.get("effort", "none") not in _EFFORTS
    ):
        raise ValueError(
            'responses_options["reasoning"] takes only "effort": '
            + ", ".join(sorted(_EFFORTS))
        )


class RealtimeModel(GPTLiveModel):
    """Fluid 1, ready to pass to ``AgentSession(llm=...)``."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        region: str | None = None,
        base_url: str | None = None,
        model: str = KUGEL_LIVE_MODEL,
        voice: int | str | None = None,
        delegation: Delegation = "responses",
        responses_options: ResponsesDelegationOptions | None = None,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """Create a Fluid 1 realtime model.

        Args:
            api_key: KugelAudio API key. If not provided, reads the
                ``KUGELAUDIO_API_KEY`` environment variable. Prefix with ``"eu-"``
                to select the direct EU endpoint; the prefix is stripped before auth.
            region: API endpoint region. Use ``"eu"`` for the direct EU endpoint.
                Overrides any prefix detected from the API key. Ignored when
                *base_url* is set.
            base_url: API base URL without the version, such as
                ``https://api.kugelaudio.com``. Overrides region selection.
            model: Speech-to-speech model ID.
            voice: Numeric KugelAudio voice ID. ``None`` uses the default voice.
            delegation: ``"responses"`` (default): KugelAudio runs the model that
                works on a delegation, and the agent's ``@function_tool`` functions
                run in your worker. ``"client"``: your code answers each
                ``delegation_created`` event of the session itself.
            responses_options: Back-office settings under ``"responses"``, such as
                ``{"model": "gpt-6.1-terra"}``. Supported keys: ``model``,
                ``instructions``, ``tool_choice`` (``"auto"``, ``"none"`` or
                ``"required"``), ``parallel_tool_calls`` and ``reasoning``
                (``{"effort": ...}``). Without a model,
                :data:`DEFAULT_BACK_OFFICE_MODEL` runs.
            http_session: aiohttp session to reuse for the WebSocket.
            max_session_duration: Seconds before the connection is recycled.
            conn_options: Retry and timeout settings of the connection.

        Raises:
            ValueError: No API key, an unknown *region*, a voice that is not a
                numeric ID, or *responses_options* given with ``"client"`` or
                carrying a setting KugelAudio does not support.
        """
        kugelaudio_api_key = api_key or os.environ.get("KUGELAUDIO_API_KEY")
        if not kugelaudio_api_key:
            raise ValueError(
                "KUGELAUDIO_API_KEY must be set or api_key must be provided"
            )
        clean_key, detected_region = _parse_api_key(kugelaudio_api_key)
        if not clean_key:
            # An empty key makes LiveKit's plugin send OPENAI_API_KEY instead.
            raise ValueError(
                f"the API key is only the region prefix {detected_region}-"
            )
        if base_url:
            resolved_url = base_url.rstrip("/")
        else:
            try:
                resolved_url = _resolve_region_url(region or detected_region)
            except ValidationError as exc:
                raise ValueError(str(exc)) from exc

        if delegation == "client" and responses_options is not None:
            raise ValueError('responses_options applies only to delegation="responses"')
        if responses_options is not None:
            _check_responses_options(responses_options)
        options = (
            ResponsesDelegationOptions()
            if responses_options is None
            else responses_options.copy()
        )
        if delegation == "responses" and "model" not in options:
            options["model"] = DEFAULT_BACK_OFFICE_MODEL

        super().__init__(
            api_key=clean_key,
            base_url=f"{resolved_url}/v1",
            model=model,
            # RISK: leaving voice to the parent sends its OpenAI default name, which the
            # server answers with the account's default voice; a non-numeric name passed
            # here would do the same silently, so it is refused below instead.
            **({} if voice is None else {"voice": _voice_id(voice)}),
            delegation=delegation,
            responses_options=options if delegation == "responses" else NOT_GIVEN,
            http_session=http_session,
            max_session_duration=max_session_duration,
            conn_options=conn_options,
        )


def _voice_id(voice: int | str) -> str:
    """The wire form of a numeric KugelAudio voice ID."""
    text = str(voice).strip()
    if isinstance(voice, bool) or not text.isdecimal():
        raise ValueError(f"voice must be a numeric KugelAudio voice ID, got {voice!r}")
    return text


__all__ = ["DEFAULT_BACK_OFFICE_MODEL", "Delegation", "RealtimeModel"]
