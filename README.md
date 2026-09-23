# KugelAudio Python SDK

Official Python SDK for the KugelAudio Text-to-Speech API.

## Installation

```bash
pip install kugelaudio
```

Or with `uv`:

```bash
uv add kugelaudio
```

## Quick Start

```python
from kugelaudio import KugelAudio

# Initialize the client - just needs an API key!
client = KugelAudio(api_key="your_api_key")

# Generate speech
audio = client.tts.generate(
    text="Hello, world!",
    model_id="kugel-3",
    voice_id=1071,     # browse voices with client.voices.list()
    language="en",     # skip auto-detection (~150ms) when you know the language
)

# Save to file
audio.save("output.wav")
```

`voice_id` is required — `1071` ("Samantha Ferris", en-US) is a public voice you
can call straight away. List every voice available to your key with
`client.voices.list()` — see [Voices](#voices).

## Local CPU Turn Detection

The optional turn-detection runtime downloads the private, version-pinned ONNX
bundle from Hugging Face, verifies every declared SHA-256 checksum, and then
runs fully locally on CPU. ONNX Runtime performs model inference; the extra also
loads the CPU PyTorch runtime to reproduce the native numerical environment used
for the published quality gates. It requires Python 3.11 or newer.

```bash
pip install "kugelaudio[turn-detection]"
hf auth login  # required while the model repository is private
```

Load one model per process and create cheap state for each conversation:

```python
from kugelaudio.turn import TurnDecisionReason, TurnDetector, TurnOutcomeKind

detector = TurnDetector.from_pretrained(cpu_threads=12)
turn = detector.create_session("en")

# Feed the current user's audio and interim ASR transcript continuously.
turn.push_pcm16(pcm_chunk, sample_rate=16_000)
turn.update_transcript("I think we should probably")

# Call with the current duration whenever VAD reports real silence.
decision = turn.observe_silence(duration_ms=200)

if not decision.end_turn:
    # Temporal bundles rescore ambiguous evidence at their later policy points.
    decision = turn.observe_silence(duration_ms=300)

if decision.end_turn:
    start_assistant_response()
    turn.reset_turn()
```

Stable `v2.1.1` contains both qualified quantization variants on one immutable
policy-only patch over the byte-identical v2.1.0 weights. W8A32 is recommended
and selected by default:

```python
detector = TurnDetector.from_pretrained(
    revision="v2.1.1",
    preset="responsive-300ms",  # or conservative-600ms
    cpu_threads=12,
)
```

The supported W8A8 QAT variant must be selected explicitly because W8A32 has
better endpoint quality:

```python
quantized = TurnDetector.from_pretrained(
    revision="v2.1.1",
    variant="models/w8a8-qat",
    preset="responsive-300ms",
    cpu_threads=12,
)
```

The SDK default pins the exact commit behind stable tag `v2.1.1`. Human-facing
release tags are immutable; `channels/stable` and `channels/preview` are mutable
operational pointers and must be selected explicitly. The W8A8 model is stable
supported but not `recommended`: its 11.21% responsive false-cutoff rate exceeds
the 9% gate, while W8A32 reaches 8.65%. Unqualified models live under
`experiments/*` and are never selected by the default runtime.

With correctly confirmed false-cutoff feedback, the W8A32 policy measures
7.94% at 296.5 ms and 3.55% at 594.5 ms on the same public validation grid.
These are conditional results selected on public validation, not an automatic
feedback classifier or an independent production-generalization claim.

On Linux x86_64 the turn-detection extra installs the calibrated OpenVINO
runtime; other platforms retain the CPU reference runtime. A policy requiring
OpenVINO fails explicitly when that provider is unavailable. For native macOS
development, opt into the verified but quality-unqualified CPU path explicitly:

```python
detector = TurnDetector.from_pretrained(execution_provider="cpu")
```

Omitting the override always retains the bundle-calibrated provider. The
qualified runtime uses BF16 activations only for the fixed-shape Whisper stage;
the dynamic-text fusion stage stays FP32 to avoid non-finite OpenVINO outputs.
Servers with a compatible `onnxruntime-gpu` installation may explicitly select
`execution_provider="cuda"`. CUDA selection fails during model loading if the
provider is unavailable. The qualified graphs keep their expensive encoder and
matrix kernels on CUDA while ONNX Runtime handles small dynamic-shape/control
nodes on an explicit CPU fallback; the standard `turn-detection` extra remains
the qualified local CPU/OpenVINO distribution.

`cpu_streams=1` gives the lowest single-conversation latency. Shared workers
can set `cpu_streams=2` for two concurrent turns or `cpu_streams=4` for four or
more; streams improve parallel request throughput but increase isolated-call
latency because the fixed CPU thread budget is divided between them.
On the qualified 26-core Xeon target, `cpu_threads=20` is the balanced optimum:
use it only when the worker owns at least 20 physical cores, and keep the
explicit lower setting for smaller allocations.

Temporal sessions reuse exact intermediate work without weakening rescoring.
If neither audio, transcript, nor supplied history changed, the next duration-specific threshold
reuses the identical probability; a transcript-only revision reruns Qwen while
reusing Whisper embeddings; appended audio always reruns both graphs. On the
qualified 20-core target these paths measured 0.0 ms, 20.4 ms, and 39.9 ms p50,
respectively, after a 45.3 ms first score.

For a checkpoint trained and evaluated with dialogue history, supply prior user
and assistant messages separately from the current interim transcript:

```python
from kugelaudio.turn import TurnMessage

# `turn` must use a bundle configured for more than one context message.
turn.update_history([
    TurnMessage("user", "I'd like to book a trip."),
    TurnMessage("assistant", "Where would you like to go?"),
])
turn.update_transcript("To Berlin, and then")
```

`update_history()` replaces the complete supplied history and copies it; the
session retains it across `reset_turn()` and clears it on `reset_conversation()`.
Only include messages already available to the application at the scoring time,
including the assistant text actually spoken before an interruption. The SDK
never adds a predicted endpoint or a partial transcript to history automatically.
An updated history invalidates pending scores and reruns text fusion while reusing
unchanged audio embeddings. False-cutoff feedback remains a separate policy input
through `record_outcome()`.

The bundle's `context_messages` limit includes the current user message; the
most recent prior messages and tokens are retained. Empty prior messages are
omitted; an empty current transcript retains its user marker when history exists.
Existing bundles default to one context message and reject nonempty history;
enabling this API does not retrain them or establish a quality improvement.
Direct predictor calls accept the same `history_messages` list in
`predict_proba()` and `predict_proba_from_audio()`.

For a latency-first application, commit a confident completion immediately
when the first 200 ms score returns while retaining the calibrated threshold
and incomplete-turn timeout:

```python
turn = detector.create_session("en", action_delay_ms=200)
```

The incomplete-turn fallback defaults to the bundle's calibrated timeout. Pass
`timeout_ms` only when the surrounding product has a measured override. Read
the resolved value from `effective_timeout_ms` on `TurnSession`,
`KugelTurnBridge`, or `KugelTurnStopStrategy`.

When VAD detects speech before the endpoint action, cancel the pending decision:

```python
turn.speech_resumed()
```

Temporal policies automatically learn a speaker's ordinary within-turn pause
lengths when speech resumes. If product-level evidence distinguishes a likely
false cutoff from an intentional barge-in, record that outcome explicitly:

```python
turn.record_outcome(
    TurnOutcomeKind.LIKELY_FALSE_CUTOFF,
    speaker_pause_ms=420,
)
```

Do not infer false-cutoff feedback from speech resumption alone; intentional
barge-ins must be recorded as `CONFIRMED_BARGE_IN` or left neutral.

Important input and lifecycle rules:

- Audio must be mono 16 kHz. `push_audio` accepts normalized `float32`; use
  `push_pcm16` for explicit little-endian signed PCM16 conversion.
- The transcript must be the current interim user transcript. Do not wait for a
  post-endpoint final transcript.
- Legacy policies score each silence episode once. Temporal policies rescore at
  each declared silence duration (for example 200/300/400/600 ms), using the
  latest audio/transcript snapshot and stopping after the first confident score.
- Call `observe_silence` with monotonically increasing durations for one silence
  span. Call `speech_resumed` before starting a new span.
- Only `P(complete)` can trigger a model endpoint. `incomplete`, `backchannel`,
  and `wait` keep listening until speech resumes or the measured timeout fires.
- One `TurnDetector` owns the roughly 1.55 GiB model runtime. Share it across
  sessions instead of loading one copy per conversation.
- No network is used after the immutable model revision is cached. Pass
  `local_files_only=True` to enforce offline startup.

The stable v2.1.1 policies cover English. Applications serving additional
languages must explicitly route those sessions to a calibrated multilingual
revision; the SDK never silently changes model identity. Unsupported languages,
missing private-repository access, corrupt bundles, wrong sample rates, and
invalid session ordering raise typed `TurnDetectionError` subclasses.

### LiveKit Agents

Install both optional integrations and LiveKit's VAD plugin:

```bash
pip install "kugelaudio[livekit,turn-detection]" "livekit-agents[silero]"
```

`KugelTurnBridge` keeps LiveKit's normal STT pipeline intact while teeing its
audio into KugelTurn. Configure LiveKit for manual endpointing so two detectors
cannot commit the same user turn:

```python
import logging
from collections.abc import AsyncIterable

from livekit import rtc
from livekit.agents import Agent, AgentSession, ModelSettings
from livekit.plugins import silero

from kugelaudio.livekit import KugelTurnBridge
from kugelaudio.turn import TurnDetector

logger = logging.getLogger(__name__)
detector = TurnDetector.from_pretrained(cpu_threads=12)  # once per worker


def observe_decision(decision):
    logger.info(
        "turn decision",
        extra={
            "reason": decision.reason.value,
            "end_turn": decision.end_turn,
            "silence_ms": decision.silence_ms,
            "inference_ms": decision.inference_ms,
            "probabilities": decision.probabilities,
        },
    )


bridge = KugelTurnBridge(
    detector,
    language="en",
    vad_silence_ms=200,
    action_delay_ms=200,  # omit to use the calibrated language delay
    on_decision=observe_decision,
)


class VoiceAgent(Agent):
    def stt_node(
        self,
        audio: AsyncIterable[rtc.AudioFrame],
        model_settings: ModelSettings,
    ):
        return bridge.stt_node(self, audio, model_settings)


session = AgentSession(
    turn_detection="manual",
    vad=silero.VAD.load(min_silence_duration=0.2),
    stt=stt,
    llm=llm,
    tts=tts,
)

try:
    await session.start(agent=VoiceAgent(instructions="..."), room=ctx.room)
finally:
    await bridge.aclose()
```

On LiveKit Agents 1.5+, the equivalent non-deprecated configuration is
`turn_handling=TurnHandlingOptions(turn_detection="manual")`. The bridge
resamples mono LiveKit input to 16 kHz, accumulates interim/final transcripts,
runs ONNX inference outside the event loop, cancels pending decisions when
speech resumes, and calls `commit_user_turn()` when the measured policy ends.

Keep `vad_silence_ms` equal to LiveKit's `min_silence_duration` in milliseconds.
The validated configuration is 200 ms.

`on_decision` runs for the first model score and each meaningful policy-state
change, including the final `model_complete` or `timeout`. Its `TurnDecision`
contains the transcript, threshold, four class probabilities, silence duration,
and `inference_ms` when a new model score was computed. Keep the callback
non-blocking; enqueue network or storage work instead of performing it inline.

### Pipecat

Install the Pipecat and turn-detection extras:

```bash
pip install "kugelaudio[pipecat,turn-detection]"
```

Use `KugelTurnStopStrategy` as the sole Pipecat user-turn stop strategy:

```python
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from kugelaudio.pipecat import KugelTurnStopStrategy
from kugelaudio.turn import TurnDetector

detector = TurnDetector.from_pretrained(cpu_threads=12)  # once per process
turn_strategy = KugelTurnStopStrategy(
    detector,
    language="en",
    vad_silence_ms=200,
    action_delay_ms=200,  # omit to use the calibrated language delay
)

user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(stop_secs=0.2),
        ),
        user_turn_strategies=UserTurnStrategies(
            stop=[turn_strategy],
        ),
    ),
)
```

The strategy supports Pipecat 0.0.101+ and 1.x turn-management APIs. It consumes
native `InputAudioRawFrame`, transcription, and VAD frames, explicitly rejects
multichannel input, resamples mono PCM to 16 kHz, and emits Pipecat's standard
`on_user_turn_stopped` event. For Pipecat 0.x releases whose VAD stop frame does
not carry its own duration, keep `vad_silence_ms` equal to `VADParams.stop_secs`.

## Client Configuration

```python
from kugelaudio import KugelAudio

# Simple setup - single URL handles everything
client = KugelAudio(api_key="your_api_key")

# Or with custom options
client = KugelAudio(
    api_key="your_api_key",           # Required: Your API key
    api_url="https://api.kugelaudio.com",  # Optional: API base URL (default)
    timeout=60.0,                      # Optional: Request timeout in seconds
    telemetry=None,                    # Optional: anonymous error diagnostics
)
```

`telemetry` is `None` (hosted API only), `True` or `False` — see
[Error diagnostics](#error-diagnostics).

### Region Selection

By default, KugelAudio uses the canonical geo-routed API endpoint. You can
select the direct EU endpoint when you need to pin traffic to Europe.

| Region hint | Endpoint |
|-------------|----------|
| default | `api.kugelaudio.com` (geo-routed) |
| `eu` | `api.eu.kugelaudio.com` |

**Option 1 — API key prefix** (simplest, works with env vars):

```python
client = KugelAudio(api_key="eu-ka_your_api_key")       # → EU
client = KugelAudio(api_key="ka_your_api_key")          # → canonical geo-routed API
```

**Option 2 — `region` parameter**:

```python
client = KugelAudio(api_key="ka_your_api_key", region="eu")
```

The prefix is always stripped before authentication. Priority: `api_url` > `region` > key prefix > default.

### Single URL Architecture

The SDK uses a **single URL** for both REST API and WebSocket streaming. The TTS server provides both REST endpoints (`/v1/models`, `/v1/voices`) and WebSocket (`/ws/tts`) - no proxy needed, minimal latency.

### Local Development

For local development, point directly to your TTS server:

```python
client = KugelAudio(
    api_key="your_api_key",
    api_url="http://localhost:8000",   # TTS server handles everything
)
```

Or if you have separate backend and TTS servers:

```python
client = KugelAudio(
    api_key="your_api_key",
    api_url="http://localhost:8001",   # Backend for REST API
    tts_url="http://localhost:8000",   # TTS server for WebSocket streaming
)
```

## Available Models

| Model ID | Name | Best for |
|----------|------|----------|
| `kugel-3` | Kugel 3 | Voice agents, narration, brand voices, streaming, multilingual TTS |

`kugel-3` is the current production model — use it for new integrations. Legacy
IDs (`kugel-2.5`, `kugel-2-turbo`, `kugel-2`, `kugel-1`, `kugel-1-turbo`) are
still accepted for backwards compatibility; see
[Models](https://docs.kugelaudio.com/models).

### List Available Models

```python
models = client.models.list()

for model in models:
    print(f"{model.id}: {model.name}")
    print(f"  Description: {model.description}")
    print(f"  Max Input: {model.max_input_length} characters")
    print(f"  Sample Rate: {model.sample_rate} Hz")
```

## Voices

### List Available Voices

```python
# List all available voices (paginated)
result = client.voices.list()

for voice in result.voices:
    print(f"{voice.id}: {voice.name}")
    print(f"  Category: {voice.category}")
    print(f"  Languages: {', '.join(voice.supported_languages)}")
print(f"Showing {len(result.voices)} of {result.total} voices")

# Filter by language
result = client.voices.list(language="de")

# Get only public voices
result = client.voices.list(include_public=True)

# Paginate through results
page1 = client.voices.list(limit=10, offset=0)
page2 = client.voices.list(limit=10, offset=10)
```

### Get a Specific Voice

```python
voice = client.voices.get(voice_id=1071)
print(f"Voice: {voice.name}")
print(f"Sample text: {voice.sample_text}")
```

## Text-to-Speech Generation

### Basic Generation (Non-Streaming)

Generate complete audio and receive it all at once:

```python
audio = client.tts.generate(
    text="Hello, this is a test of the KugelAudio text-to-speech system.",
    model_id="kugel-3",        # Current model (see "Available Models")
    voice_id=1071,             # Required: voice to speak with (client.voices.list())
    cfg_scale=2.0,             # Guidance scale (1.2-2.5)
    max_new_tokens=2048,       # Maximum tokens to generate
    sample_rate=24000,         # Output sample rate
    output_format=None,        # Optional: 'pcm_24000', 'ulaw_8000', 'alaw_8000', ...
    normalize=True,            # Enable text normalization (see below)
    language="en",             # Language for normalization
)

# Audio properties
print(f"Duration: {audio.duration_seconds:.2f}s")
print(f"Samples: {audio.samples}")
print(f"Sample rate: {audio.sample_rate} Hz")
print(f"Generation time: {audio.generation_ms:.0f}ms")
print(f"RTF: {audio.rtf:.2f}")  # Real-time factor

# Save to WAV file
audio.save("output.wav")

# Get raw PCM bytes
pcm_data = audio.audio

# Get WAV bytes (with header)
wav_bytes = audio.to_wav_bytes()
```

### Streaming Audio Output

Receive audio chunks as they are generated for lower latency:

```python
# Synchronous streaming
for item in client.tts.stream(
    text="Hello, this is streaming audio.",
    model_id="kugel-3",
    voice_id=1071,
    language="en",
):
    if hasattr(item, 'audio'):  # AudioChunk
        # Process audio chunk immediately
        print(f"Chunk {item.index}: {len(item.audio)} bytes, {item.samples} samples")
        # play_audio(item.audio)
    elif isinstance(item, dict) and item.get('final'):
        # Final stats
        print(f"Total duration: {item.get('dur_ms', 0):.0f}ms")
        print(f"Generation time: {item.get('gen_ms', 0):.0f}ms")
```

### Async Streaming

For async applications:

```python
import asyncio

async def generate_speech():
    async for item in client.tts.stream_async(
        text="Async streaming example.",
        model_id="kugel-3",
        voice_id=1071,
        language="en",
    ):
        if hasattr(item, 'audio'):
            # Process chunk
            pass

asyncio.run(generate_speech())
```

### Async Generation

```python
import asyncio

async def main():
    audio = await client.tts.generate_async(
        text="Async generation example.",
        model_id="kugel-3",
        voice_id=1071,
        language="en",
    )
    audio.save("async_output.wav")

asyncio.run(main())
```

## Text Normalization

Text normalization converts numbers, dates, times, and other non-verbal text into spoken words. For example:
- "I have 3 apples" → "I have three apples"
- "The meeting is at 2:30 PM" → "The meeting is at two thirty PM"
- "€50.99" → "fifty euros and ninety-nine cents"

### Usage

```python
# With explicit language (recommended - fastest)
audio = client.tts.generate(
    text="I bought 3 items for €50.99 on 01/15/2024.",
    voice_id=1071,
    normalize=True,
    language="en",  # Specify language for best performance
)

# With auto-detection (adds ~150ms latency)
audio = client.tts.generate(
    text="Ich habe 3 Artikel für 50,99€ gekauft.",
    voice_id=1705,  # a German voice — client.voices.list(language="de")
    normalize=True,
    # language not specified - will auto-detect
)
```

### Supported Languages

| Code | Language | Code | Language |
|------|----------|------|----------|
| `de` | German | `nl` | Dutch |
| `en` | English | `pl` | Polish |
| `fr` | French | `sv` | Swedish |
| `es` | Spanish | `da` | Danish |
| `it` | Italian | `no` | Norwegian |
| `pt` | Portuguese | `fi` | Finnish |
| `cs` | Czech | `hu` | Hungarian |
| `ro` | Romanian | `el` | Greek |
| `uk` | Ukrainian | `bg` | Bulgarian |
| `tr` | Turkish | `vi` | Vietnamese |
| `ar` | Arabic | `hi` | Hindi |
| `zh` | Chinese | `ja` | Japanese |
| `ko` | Korean | | |

### Performance Warning

> ⚠️ **Latency Warning**: Using `normalize=True` without specifying `language` adds approximately **150ms latency** for language auto-detection. For best performance in latency-sensitive applications, always specify the `language` parameter.

## LLM Integration: Streaming Text Input

For real-time TTS when streaming text from an LLM (GPT-4, Claude, etc.),
use a `StreamingSession`. Forward LLM tokens directly to `session.send()`
**without** `flush=True` — the server accumulates them and starts
generation at natural sentence boundaries. Flush exactly once at the end
of the assistant turn.

> ⚠️ **Do not call `session.send(text, flush=True)` between sentences or
> words.** Each explicit flush is a separate TTS request that pays the
> full model time-to-first-audio (TTFA) again and produces an audible
> gap. See [Chunking & per-segment latency](https://docs.kugelaudio.com/streaming/chunking-and-latency)
> for the full rationale and ElevenLabs migration notes.

### Async Streaming Session

```python
import asyncio

async def speak_turn(llm_token_stream):
    async with client.tts.streaming_session(
        voice_id=1071,
        model_id="kugel-3",
        language="en",
    ) as session:
        # Forward every LLM token directly. No flush=True per token —
        # the server's text buffer chunks at sentence boundaries.
        async for token in llm_token_stream:
            async for chunk in session.send(token):
                play_audio(chunk.audio)

        # Single flush at turn end emits any trailing text.
        async for chunk in session.flush():
            play_audio(chunk.audio)

        # Per-session usage — bill your own customers per conversation.
        # cost_cents is the actual charge in EUR cents (None if undetermined).
        usage = session.last_usage
        if usage:
            print(f"audio: {usage.audio_seconds}s, cost: {usage.cost_cents} ct")

asyncio.run(speak_turn(my_llm_stream()))
```

### Synchronous Streaming Session

```python
with client.tts.streaming_session_sync(
    voice_id=1071,
    model_id="kugel-3",
    language="en",
) as session:
    for token in llm_token_stream:
        for chunk in session.send(token):  # no flush per token
            play_audio(chunk.audio)

    for chunk in session.flush():  # single flush at turn end
        play_audio(chunk.audio)
```

### Pipecat Word Timestamps

The Pipecat service can forward server word timings with their KugelAudio
context ID. Supplying the callback enables timestamp generation; omitting it
keeps the existing audio-only behavior.

```python
from kugelaudio.models import WordTimestamp
from kugelaudio.pipecat import KugelAudioTTSService

def on_word_timestamps(
    context_id: str, timestamps: list[WordTimestamp]
) -> None:
    for timestamp in timestamps:
        record_timing(context_id, timestamp)

tts = KugelAudioTTSService(
    api_key="your_api_key",
    voice_id=123,
    on_word_timestamps=on_word_timestamps,
)
```

## Error Handling

```python
from kugelaudio import KugelAudio
from kugelaudio.exceptions import (
    KugelAudioError,
    AuthenticationError,
    RateLimitError,
    InsufficientCreditsError,
    ValidationError,
    NotFoundError,
)

try:
    audio = client.tts.generate(text="Hello!", voice_id=1071)
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded, please wait")
except InsufficientCreditsError:
    print("Not enough credits, please top up")
except ValidationError as e:
    print(f"Invalid request: {e}")
except NotFoundError as e:
    print(f"Resource not found (e.g. unknown voice_id): {e}")
except KugelAudioError as e:
    print(f"API error: {e}")
```

### Rolling deploys

When a replica restarts, the server closes the WebSocket with code `1012` (or
refuses the upgrade with `1013`). `StreamingSession` and `MultiContextSession`
handle that for you: if no audio for the current turn has been delivered yet,
they wait one second, reconnect to another replica and resend the turn, session
config and open contexts included, then keep streaming. The replay is logged at
`INFO` on the `kugelaudio.streaming` logger and your code sees nothing else.

A turn whose audio already started is not replayed, because that would repeat
what the listener just heard, and no turn is replayed twice. Those two cases
raise `ServerRestartingError`, a `ConnectionError` with `retry_after = 1`:

```python
from kugelaudio.exceptions import ServerRestartingError

try:
    async for chunk in session.send("Hello!", flush=True):
        play(chunk)
except ServerRestartingError as e:
    await asyncio.sleep(e.retry_after or 1)
    await session.connect()  # a fresh replica; resend what was not spoken
```

### Request IDs

Every error the server produces carries a `request_id` you can quote to
support. It is read from the `x-request-id` response header on HTTP calls and
from the `request_id` field of a WebSocket error frame, and it is appended to
the exception message automatically:

```python
try:
    audio = client.tts.generate(text="Hello!", voice_id=1071)
except KugelAudioError as e:
    print(e.request_id)  # -> "9f2c1ab4..." (None if the server sent none)
```

### Error diagnostics

The SDK can report anonymous diagnostics when a call fails, so recurring client
errors show up before anyone opens a ticket. One report per failed operation
carries the error class name, the stage it failed in (`connecting`,
`handshake`, `sending_request`, `awaiting_first_audio`, `receiving_audio`,
`finalizing`), elapsed time, audio chunk/byte counts, HTTP status or WebSocket
close code, and the server request id.

It never sends your text, audio, API key, URLs, host names or exception
messages. Delivery is off the calling thread, bounded, and failures are
swallowed: a telemetry problem can never break or slow a synthesis call.

Reports go to `POST <your api_url>/v1/sdk-diagnostics` — the same host and the
same API key as every other call, so nothing leaves for a third-party endpoint
and no extra credential is involved. An API that does not serve the route
answers `404`, and after three of them the SDK stops reporting for the rest of
the process.

Reporting is on by default only against the hosted `*.kugelaudio.com` API. A
custom or on-premise `api_url` is off by default.

```python
client = KugelAudio(api_key="your_api_key", telemetry=False)  # opt out
client = KugelAudio(api_key="...", api_url="https://tts.internal", telemetry=True)
```

| Variable | Effect |
|---|---|
| `KUGELAUDIO_TELEMETRY` | `0/false/off/no` or `1/true/on/yes`. Overrides the `telemetry=` argument. |

There is no endpoint override: reports carry your API key, so they only ever
go to the API host the client already talks to.

The LiveKit and Pipecat plugins report the same way, tagged with the
integration they came through, and take the same `telemetry=` argument with
the same precedence (`KUGELAUDIO_TELEMETRY` still wins):

```python
from kugelaudio.livekit import TTS
from kugelaudio.pipecat import KugelAudioTTSService

tts = TTS(api_key="...", telemetry=False)
service = KugelAudioTTSService(api_key="...", voice_id=280, telemetry=False)
```

## Data Models

### AudioChunk

Represents a single audio chunk from streaming:

```python
class AudioChunk:
    audio: bytes          # Raw PCM16 audio data
    encoding: str         # 'pcm_s16le' | 'mulaw' | 'alaw' (G.711 when output_format set)
    index: int           # Chunk index (0-based)
    sample_rate: int     # Sample rate (24000)
    samples: int         # Number of samples in chunk

    @property
    def duration_seconds(self) -> float:
        """Duration of this chunk in seconds."""
```

### AudioResponse

Complete audio response from generation:

```python
class AudioResponse:
    audio: bytes              # Complete PCM16 audio
    sample_rate: int          # Sample rate (24000)
    samples: int              # Total samples
    duration_ms: float        # Duration in milliseconds
    generation_ms: float      # Generation time in milliseconds
    rtf: float               # Real-time factor

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""

    def save(self, path: str) -> None:
        """Save as WAV file."""

    def to_wav_bytes(self) -> bytes:
        """Get WAV file as bytes."""
```

### Model

TTS model information:

```python
class Model:
    id: str                   # 'kugel-3' (or a legacy ID such as 'kugel-2.5')
    name: str                 # Human-readable name
    description: str          # Model description
    max_input_length: int     # Maximum input characters
    sample_rate: int          # Output sample rate
```

Supported native output format tokens are `pcm_8000`, `pcm_16000`, `pcm_22050`,
`pcm_24000`, `ulaw_8000`, and `alaw_8000`.

### Voice

Voice information:

```python
class Voice:
    id: int                          # Voice ID
    name: str                        # Voice name
    description: Optional[str]       # Description
    category: Optional[VoiceCategory]  # 'premade', 'cloned', 'generated'
    sex: Optional[VoiceSex]          # 'male', 'female', 'neutral'
    age: Optional[VoiceAge]          # 'young', 'middle_aged', 'old'
    supported_languages: List[str]   # ['en', 'de', ...]
    sample_text: Optional[str]       # Sample text for preview
    avatar_url: Optional[str]        # Avatar image URL
    sample_url: Optional[str]        # Sample audio URL
    is_public: bool                  # Whether voice is public
    verified: bool                   # Whether voice is verified
```

## Complete Example

```python
from kugelaudio import KugelAudio

# Initialize client
client = KugelAudio(api_key="your_api_key")

# List available models
print("Available Models:")
for model in client.models.list():
    print(f"  - {model.id}: {model.name}")

# List available voices
print("\nAvailable Voices:")
for voice in client.voices.list(limit=5).voices:
    print(f"  - {voice.id}: {voice.name}")

# Generate audio
print("\nGenerating audio...")
audio = client.tts.generate(
    text="Welcome to KugelAudio. This is an example of high-quality text-to-speech synthesis.",
    model_id="kugel-3",
    voice_id=1071,
    language="en",
)

print(f"Generated {audio.duration_seconds:.2f}s of audio in {audio.generation_ms:.0f}ms")
print(f"Real-time factor: {audio.rtf:.2f}x")

# Save to file
audio.save("example.wav")
print("Saved to example.wav")

# Close client
client.close()
```

## Verifying the AI-generated watermark

Audio produced by the API carries an in-band watermark (EU AI Act Art. 50)
alongside the `X-KugelAudio-AI-Generated: true` response header. The optional
`watermark` extra verifies a clip locally.

```bash
pip install "kugelaudio[watermark]"
```

```python
from kugelaudio.watermark import WatermarkDetector

detector = WatermarkDetector()          # load once, reuse
result = detector.detect_file("speech.wav")

if result.ai_generated:
    print(f"AI-generated (confidence {result.confidence:.3f})")
    print(f"source id {result.customer_id}")
```

`detect(samples, sample_rate)` scores an in-memory mono float array at any
rate. It resamples to the watermark's native 24 kHz before scoring. There is
one bundled KugelAudio detector; it is numpy-only, works offline, and does not
take a backend argument.

The payload is a 12-bit source id (`0`–`4095`) encoded as a codeword across 30
detector windows (roughly four seconds), rather than as two raw bits. Presence
can be reported from a shorter clip, but `customer_id` remains unset until
there is enough aligned payload evidence to attribute the source.

Scoring is duration-dependent — more audio means more windows to average — so
clips shorter than one second are rejected rather than answered unreliably;
pass `min_seconds=` if you accept a weaker verdict. Multi-channel, integer,
empty, non-finite, and out-of-range audio raises `WatermarkAudioError` instead
of being silently coerced, because each of those would change the verdict you
are about to act on.

## Agent skill

If you build with a coding agent, install the bundled skill so it gets the TTFA
rules, streaming semantics, and text-formatting constraints without you
re-explaining them:

```bash
kugelaudio-skills install     # → ./.claude/skills/kugelaudio-tts/
```

`--global` installs to `~/.claude/skills/`, and `--dest <dir>` selects another
target.

## License

MIT
