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

"""LiveKit plugin registration for KugelAudio."""

import logging

from livekit.agents import Plugin

from kugelaudio import __version__

logger = logging.getLogger("kugelaudio.livekit")

_plugin_registered = False


class KugelAudioPlugin(Plugin):
    """KugelAudio plugin for LiveKit Agents."""

    def __init__(self) -> None:
        # Use a name that matches what livekit.plugins.kugelaudio would use
        super().__init__(
            "livekit.plugins.kugelaudio",
            __version__,
            "livekit.plugins.kugelaudio",
            logger,
        )


def register_plugin() -> None:
    """Register KugelAudio as a LiveKit plugin.

    After calling this, you can import the plugin via:
        from livekit.plugins import kugelaudio

    Example:
        from kugelaudio.livekit import register_plugin
        register_plugin()

        from livekit.plugins import kugelaudio
        tts = kugelaudio.TTS()
    """
    global _plugin_registered
    if not _plugin_registered:
        Plugin.register_plugin(KugelAudioPlugin())
        _plugin_registered = True
