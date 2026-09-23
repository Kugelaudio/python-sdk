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

"""Pronunciation dictionaries through the LiveKit plugin (ENG-572).

``/ws/tts/multi`` applies dictionaries only when the context's config frame
carries a top-level ``project_id``, and rejects a non-empty
``dictionary_ids`` without one. These tests assert the frames the real
``_send_loop`` writes, not the options object.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("livekit.agents")

from kugelaudio.livekit.tts import TTS, _Connection, _SynthesizeContent

from .test_livekit_speed_prewarm import _opts, _run_send_loop


class TestDictionaryWireFormat:
    async def test_config_frame_carries_project_and_dictionaries(self) -> None:
        conn = _Connection(_opts(project_id=42, dictionary_ids=[7, 9]), object())  # type: ignore[arg-type]
        frames = await _run_send_loop(
            conn,
            [
                _SynthesizeContent("ctx1", "Hallo"),
                _SynthesizeContent("ctx1", " Welt", flush=True),
            ],
        )

        assert frames[0]["project_id"] == 42
        assert frames[0]["dictionary_ids"] == [7, 9]
        assert "project_id" not in frames[0].get("voice_settings", {})
        # Session config rides only on a context's first frame.
        assert "project_id" not in frames[1]
        assert "dictionary_ids" not in frames[1]

    async def test_absent_when_not_set(self) -> None:
        conn = _Connection(_opts(), object())  # type: ignore[arg-type]
        frames = await _run_send_loop(
            conn, [_SynthesizeContent("ctx1", "Hallo", flush=True)]
        )

        assert "project_id" not in frames[0]
        assert "dictionary_ids" not in frames[0]

    async def test_empty_selection_is_sent_as_opt_out(self) -> None:
        # [] disables the project's defaults server-side; it is not "unset".
        conn = _Connection(_opts(dictionary_ids=[]), object())  # type: ignore[arg-type]
        frames = await _run_send_loop(
            conn, [_SynthesizeContent("ctx1", "Hallo", flush=True)]
        )

        assert frames[0]["dictionary_ids"] == []
        assert "project_id" not in frames[0]


class TestDictionaryOptions:
    def test_init_plumbs_through(self) -> None:
        tts = TTS(api_key="test-key", project_id=42, dictionary_ids=[7])
        assert tts._opts.project_id == 42
        assert tts._opts.dictionary_ids == [7]

    def test_init_rejects_selection_without_project(self) -> None:
        with pytest.raises(ValueError, match="project_id"):
            TTS(api_key="test-key", dictionary_ids=[7])

    def test_update_options_rejects_selection_without_project(self) -> None:
        tts = TTS(api_key="test-key")
        with pytest.raises(ValueError, match="project_id"):
            tts.update_options(dictionary_ids=[7])
        assert tts._opts.dictionary_ids is None, "rejected value must not be applied"

    def test_update_options_applies_and_invalidates_connection(self) -> None:
        tts = TTS(api_key="test-key")
        conn = MagicMock(spec=_Connection)
        conn.is_current = True
        conn._closed = False
        tts._current_connection = conn

        tts.update_options(project_id=42, dictionary_ids=[7])

        assert tts._opts.project_id == 42
        assert tts._opts.dictionary_ids == [7]
        conn.mark_non_current.assert_called_once()
        assert tts._current_connection is None
