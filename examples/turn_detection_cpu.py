"""Minimal event-driven CPU turn-detection integration.

Install with ``pip install 'kugelaudio[turn-detection]'`` and authenticate to
the private Hugging Face repository before the first download.
"""

from __future__ import annotations

from kugelaudio.turn import TurnDecision, TurnDetector, TurnLanguage, TurnSession


class VoiceTurnController:
    """Connect audio, interim ASR, and VAD callbacks to one turn session."""

    def __init__(self, detector: TurnDetector, *, language: TurnLanguage) -> None:
        self.turn: TurnSession = detector.create_session(language)

    def on_audio(self, pcm16: bytes) -> None:
        """Call for every mono 16 kHz user-audio chunk, including real silence."""
        self.turn.push_pcm16(pcm16, sample_rate=16_000)

    def on_interim_transcript(self, text: str) -> None:
        """Replace the current streaming-ASR hypothesis."""
        self.turn.update_transcript(text)

    def on_vad_silence(self, duration_ms: int) -> TurnDecision:
        """Call with increasing duration during one VAD silence span."""
        return self.turn.observe_silence(duration_ms)

    def on_vad_speech_resumed(self) -> None:
        """Cancel any pending endpoint when the user continues speaking."""
        self.turn.speech_resumed()

    def on_turn_consumed(self) -> None:
        """Clear state after the assistant accepts an end-turn decision."""
        self.turn.reset_turn()


if __name__ == "__main__":
    detector = TurnDetector.from_pretrained(cpu_threads=4)
    controller = VoiceTurnController(detector, language="en")
