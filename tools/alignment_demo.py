"""
Interactive alignment demo — uses the KugelAudio SDK to generate speech
with server-side word-level timestamps and serves a web UI with
waveform + word-boundary overlays.

Usage:
    cd packages/public/python-sdk
    uv run --extra tools python tools/alignment_demo.py [--port 8765]

Requires TTS_MASTER_API_KEY in the environment or a .env file in the repo root.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from kugelaudio import KugelAudio, WordTimestamp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────

SAMPLES = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming technology.",
    "In a hole in the ground there lived a hobbit.",
    "Good morning, how are you doing today?",
    "Guten Morgen, wie geht es Ihnen heute?",
    "Bonjour, comment allez-vous aujourd'hui?",
]

NATIVE_SR = 24_000  # TTS native sample rate


def _load_master_key() -> str:
    """Load master API key from env or .env file (searches upward from script)."""
    key = os.environ.get("TTS_MASTER_API_KEY")
    if key:
        return key
    # Walk up from the script location to find a .env with the key
    cur = Path(__file__).resolve().parent
    for _ in range(6):
        env_path = cur / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("TTS_MASTER_API_KEY="):
                    return line.split("=", 1)[1].strip()
        cur = cur.parent
    raise RuntimeError("TTS_MASTER_API_KEY not found in env or .env")


def _stamps_to_dicts(stamps: list[WordTimestamp]) -> list[dict[str, Any]]:
    """Convert WordTimestamp objects to plain dicts for JSON serialization."""
    return [
        {
            "word": w.word,
            "start_ms": w.start_ms,
            "end_ms": w.end_ms,
            "char_start": w.char_start,
            "char_end": w.char_end,
            "score": w.score,
        }
        for w in stamps
    ]


def pcm_to_wav_bytes(pcm_bytes: bytes, sr: int) -> bytes:
    """Wrap raw PCM16-LE bytes in a WAV container."""
    buf = io.BytesIO()
    audio_np = np.frombuffer(pcm_bytes, dtype="<i2")
    sf.write(buf, audio_np, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ── HTML template ─────────────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>KugelAudio Alignment Demo</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #0f0f0f; color: #e0e0e0; padding: 24px;
}
h1 { text-align: center; margin-bottom: 8px; color: #fff; font-size: 1.6rem; }
.subtitle { text-align: center; color: #888; margin-bottom: 32px; font-size: 0.9rem; }
.sample {
  background: #1a1a1a; border-radius: 12px; padding: 20px;
  margin-bottom: 24px; border: 1px solid #2a2a2a;
}
.sample-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 12px;
}
.sample-text { font-size: 1.1rem; color: #fff; font-weight: 500; }
.sample-meta { font-size: 0.8rem; color: #888; }
.waveform { width: 100%; margin-bottom: 12px; border-radius: 8px; overflow: hidden; }
.words-bar {
  display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px;
}
.word-chip {
  padding: 4px 10px; border-radius: 6px; font-size: 0.8rem;
  cursor: pointer; transition: transform 0.1s, box-shadow 0.1s;
  color: #fff; font-weight: 500;
}
.word-chip:hover { transform: scale(1.08); box-shadow: 0 2px 8px rgba(0,0,0,0.4); }
.word-chip .timing { font-size: 0.65rem; opacity: 0.7; display: block; }
.legend { text-align: center; margin-top: 32px; color: #666; font-size: 0.8rem; }
.playing { outline: 2px solid #fff; outline-offset: 1px; }
</style>
</head>
<body>
<h1>KugelAudio Word Alignment Inspector</h1>
<p class="subtitle">Click a word chip to seek to that position. Colored regions on the waveform show word boundaries.</p>

<div id="samples"></div>

<p class="legend">
  Word timestamps generated server-side via KugelAudio SDK
</p>

<script src="https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.min.js"></script>
<script src="https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.min.js"></script>
<script>
const SAMPLES = __SAMPLES_JSON__;

const COLORS = [
  'rgba(255, 107, 107, 0.35)', 'rgba(78, 205, 196, 0.35)',
  'rgba(255, 217, 61, 0.35)',  'rgba(162, 155, 254, 0.35)',
  'rgba(0, 206, 209, 0.35)',   'rgba(255, 154, 162, 0.35)',
  'rgba(144, 238, 144, 0.35)', 'rgba(255, 182, 108, 0.35)',
  'rgba(173, 216, 230, 0.35)', 'rgba(221, 160, 221, 0.35)',
];
const SOLID_COLORS = [
  '#ff6b6b', '#4ecdc4', '#ffd93d', '#a29bfe', '#00ced1',
  '#ff9aa2', '#90ee90', '#ffb66c', '#add8e6', '#dda0dd',
];

const container = document.getElementById('samples');

SAMPLES.forEach((sample, si) => {
  const div = document.createElement('div');
  div.className = 'sample';
  div.innerHTML = `
    <div class="sample-header">
      <span class="sample-text">"${sample.text}"</span>
      <span class="sample-meta">${sample.words.length} words &middot; ${sample.duration_ms.toFixed(0)}ms audio</span>
    </div>
    <div class="waveform" id="wave-${si}"></div>
    <div class="words-bar" id="words-${si}"></div>
  `;
  container.appendChild(div);

  const ws = WaveSurfer.create({
    container: `#wave-${si}`,
    waveColor: '#4a4a4a',
    progressColor: '#7c7cff',
    cursorColor: '#fff',
    height: 80,
    barWidth: 2,
    barRadius: 2,
    barGap: 1,
    url: `/audio/${si}`,
  });

  const regions = ws.registerPlugin(WaveSurfer.Regions.create());

  ws.on('ready', () => {
    const dur = ws.getDuration();
    const wordsBar = document.getElementById(`words-${si}`);

    sample.words.forEach((w, wi) => {
      const color = COLORS[wi % COLORS.length];
      const solidColor = SOLID_COLORS[wi % SOLID_COLORS.length];
      const startSec = w.start_ms / 1000;
      const endSec = w.end_ms / 1000;

      regions.addRegion({
        start: startSec,
        end: endSec,
        color: color,
        drag: false,
        resize: false,
      });

      const chip = document.createElement('span');
      chip.className = 'word-chip';
      chip.style.background = solidColor;
      chip.innerHTML = `${w.word}<span class="timing">${w.start_ms}-${w.end_ms}ms</span>`;
      chip.onclick = () => {
        ws.setTime(startSec);
        ws.play();
      };
      wordsBar.appendChild(chip);
    });
  });
});
</script>
</body>
</html>
"""  # noqa: E501


# ── Flask app ─────────────────────────────────────────────────────────


def build_app(samples_data: list[dict], wav_bytes_list: list[bytes]):
    """Build and return the Flask app."""
    from flask import Flask, Response

    app = Flask(__name__)

    @app.route("/")
    def index():
        html = HTML_TEMPLATE.replace(
            "__SAMPLES_JSON__", json.dumps(samples_data)
        )
        return Response(html, content_type="text/html")

    @app.route("/audio/<int:idx>")
    def audio(idx: int):
        if 0 <= idx < len(wav_bytes_list):
            return Response(wav_bytes_list[idx], content_type="audio/wav")
        return Response("Not found", status=404)

    return app


# ── Main ──────────────────────────────────────────────────────────────


def generate_all_samples(client: KugelAudio):
    """Generate TTS audio with server-side word timestamps for each sample."""
    samples_data: list[dict] = []
    wav_bytes_list: list[bytes] = []

    for text in SAMPLES:
        logger.info("Generating: %r", text[:50])

        response = client.tts.generate(
            text=text,
            model_id="kugel-1-turbo",
            sample_rate=NATIVE_SR,
            word_timestamps=True,
        )

        wav_bytes = pcm_to_wav_bytes(response.audio, NATIVE_SR)
        words = _stamps_to_dicts(response.word_timestamps)

        logger.info(
            "  -> %d words, %.0fms audio, RTF=%.2f",
            len(words), response.duration_ms, response.rtf,
        )

        samples_data.append({
            "text": text,
            "words": words,
            "duration_ms": response.duration_ms,
        })
        wav_bytes_list.append(wav_bytes)

    return samples_data, wav_bytes_list


def main():
    parser = argparse.ArgumentParser(description="Alignment demo server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    # 1. Load master key and create SDK client
    master_key = _load_master_key()
    logger.info("Master key loaded (length=%d)", len(master_key))
    client = KugelAudio(api_key=master_key)

    # 2. Generate audio with server-side word timestamps
    logger.info("Generating TTS audio with word timestamps via SDK...")
    samples_data, wav_bytes_list = generate_all_samples(client)

    # 3. Serve web UI
    app = build_app(samples_data, wav_bytes_list)
    logger.info("Starting server on http://%s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
