# Copyright 2024 KugelAudio
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

"""
KugelAudio LiveKit Agents Plugin.

This module provides integration between KugelAudio TTS and LiveKit Agents.

Example usage:
    from kugelaudio.livekit import TTS
    from livekit.agents.pipeline import VoicePipelineAgent

    tts = TTS(api_key="your-api-key", model="kugel-3")

    agent = VoicePipelineAgent(
        tts=tts,
        ...
    )

Or register as a LiveKit plugin:
    from kugelaudio.livekit import register_plugin
    register_plugin()

    # Now available as livekit.plugins.kugelaudio
    from livekit.plugins import kugelaudio
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kugelaudio.livekit.tts import (
        TTS as TTS,
        ChunkedStream as ChunkedStream,
        SynthesizeStream as SynthesizeStream,
    )
    from kugelaudio.livekit.models import TTSModels as TTSModels
    from kugelaudio.livekit.turn import KugelTurnBridge as KugelTurnBridge

_LIVEKIT_AVAILABLE = (
    find_spec("livekit") is not None and find_spec("livekit.agents") is not None
)


def _check_livekit_installed() -> None:
    """Check if livekit-agents is installed."""
    if not _LIVEKIT_AVAILABLE:
        raise ImportError(
            "livekit-agents is required for LiveKit integration. "
            "Install with: pip install kugelaudio[livekit]"
        )


# Lazy imports to avoid requiring livekit-agents when just importing the SDK
def __getattr__(name: str):
    """Lazy import TTS and related classes."""
    if name in ("TTS", "ChunkedStream", "SynthesizeStream"):
        _check_livekit_installed()
        from kugelaudio.livekit.tts import TTS, ChunkedStream, SynthesizeStream

        return {
            "TTS": TTS,
            "ChunkedStream": ChunkedStream,
            "SynthesizeStream": SynthesizeStream,
        }[name]

    if name == "TTSModels":
        from kugelaudio.livekit.models import TTSModels

        return TTSModels

    if name == "KugelTurnBridge":
        _check_livekit_installed()
        try:
            from kugelaudio.livekit.turn import KugelTurnBridge
        except ImportError as exc:
            raise ImportError(
                "Kugel turn detection requires both kugelaudio[livekit] and "
                "kugelaudio[turn-detection]"
            ) from exc

        return KugelTurnBridge

    if name == "SUPPORTED_LANGUAGES":
        from kugelaudio.livekit.tts import SUPPORTED_LANGUAGES

        return SUPPORTED_LANGUAGES

    if name in (
        "DEFAULT_MODEL",
        "DEFAULT_SAMPLE_RATE",
        "DEFAULT_VOICE_ID",
        "DEFAULT_CFG_SCALE",
        "DEFAULT_MAX_NEW_TOKENS",
        "SUPPORTED_SAMPLE_RATES",
    ):
        from kugelaudio.livekit import models

        return getattr(models, name)

    if name == "register_plugin":
        _check_livekit_installed()
        from kugelaudio.livekit._plugin import register_plugin

        return register_plugin

    if name == "__version__":
        from kugelaudio import __version__

        return __version__

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TTS",
    "ChunkedStream",
    "SynthesizeStream",
    "TTSModels",
    "KugelTurnBridge",
    "DEFAULT_MODEL",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_VOICE_ID",
    "DEFAULT_CFG_SCALE",
    "DEFAULT_MAX_NEW_TOKENS",
    "SUPPORTED_SAMPLE_RATES",
    "SUPPORTED_LANGUAGES",
    "register_plugin",
    "__version__",
]
