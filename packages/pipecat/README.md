# deepslate-pipecat

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-deepslate.eu-green)](https://docs.deepslate.eu/)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Pipecat plugin for [Deepslate's](https://deepslate.eu/) realtime speech-to-speech AI API.

`deepslate-pipecat` provides a `DeepslateRealtimeLLMService` implementation for the [Pipecat](https://github.com/pipecat-ai/pipecat) framework, connecting your Pipecat pipelines to Deepslate's unified voice AI infrastructure. The plugin handles bidirectional audio streaming, frame translation, WebSocket connection management, server-side VAD, and optional ElevenLabs TTS — all transparently, through a Pipecat-native interface.

---

## Features

- **Realtime Audio Streaming** — Low-latency bidirectional PCM audio over WebSockets
- **Server-side VAD** — Voice Activity Detection handled by Deepslate with configurable sensitivity
- **Function Calling** — Full tool/function calling support via Pipecat's `register_function` API
- **Flexible TTS** — Choose server-side Deepslate-hosted (cloned) voice TTS, ElevenLabs TTS, or any downstream Pipecat TTS service
- **Automatic Interruption Handling** — Native support for interruptions with buffer clearing
- **Dynamic Context Injection** — Append user or system messages to an active session mid-conversation via `LLMMessagesAppendFrame`
- **Frame-based Architecture** — Seamless integration with Pipecat's pipeline model
- **Dynamic Audio Configuration** — Automatically adapts to audio format changes at runtime

---

## Installation

```bash
pip install deepslate-pipecat
```

### Requirements

- Python 3.11 or higher

### Dependencies (installed automatically)

- `deepslate-core` — Shared Deepslate models and base client
- `pipecat-ai>=0.0.40` — Core Pipecat framework
- `loguru>=0.7.2` — Structured logging
- `websockets>=16.0` — WebSocket client

---

## Prerequisites

### Deepslate Account

Sign up at [deepslate.eu](https://deepslate.eu) and set the following environment variables:

```bash
DEEPSLATE_VENDOR_ID=your_vendor_id
DEEPSLATE_ORGANIZATION_ID=your_organization_id
DEEPSLATE_API_KEY=your_api_key
```

### ElevenLabs TTS (Optional)

For server-side TTS with automatic interruption handling:

```bash
ELEVENLABS_API_KEY=your_elevenlabs_api_key
ELEVENLABS_VOICE_ID=your_voice_id        # e.g., '21m00Tcm4TlvDq8ikWAM' for Rachel
ELEVENLABS_MODEL_ID=eleven_turbo_v2      # optional
```

> **Note:** Without `ElevenLabsTtsConfig`, the service emits `TTSTextFrame` objects for downstream Pipecat TTS services (Cartesia, Azure TTS, etc.). Context truncation on interruption requires server-side TTS.

---

## Quick Start

A complete voice bot using Daily.co WebRTC transport, ElevenLabs TTS, and function calling:

```python
import asyncio
import os
import random
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import LLMSetToolsFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.services.daily import DailyParams, DailyTransport

from deepslate.pipecat import DeepslateOptions, DeepslateRealtimeLLMService, ElevenLabsLocation, ElevenLabsTtsConfig

load_dotenv(override=True)

logger.remove()
logger.add(sys.stderr, level="DEBUG")

# Tool definitions (OpenAI function-calling JSON schema format)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Get the current weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city to look up."}
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_location",
            "description": "Get the user's current location.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


async def lookup_weather(params: FunctionCallParams):
    result = {
        "location": params.arguments.get("location", "unknown"),
        "temperature_celsius": random.randint(10, 35),
        "precipitation": random.choice(["none", "light", "moderate", "heavy"]),
        "air_pressure_hpa": random.randint(900, 1100),
    }
    await params.result_callback(result)


async def get_current_location(params: FunctionCallParams):
    await params.result_callback({"location": "Berlin"})


async def main():
    daily_api_key = os.getenv("DAILY_API_KEY")
    daily_room_url = os.getenv("DAILY_ROOM_URL")

    async with aiohttp.ClientSession() as session:
        headers = {"Authorization": f"Bearer {daily_api_key}"}
        room_name = daily_room_url.split("/")[-1]
        async with session.post(
            "https://api.daily.co/v1/meeting-tokens",
            headers=headers,
            json={"properties": {"room_name": room_name}},
        ) as r:
            token = (await r.json())["token"]

    transport = DailyTransport(
        room_url=daily_room_url,
        token=token,
        bot_name="Deepslate Bot",
        params=DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            camera_out_enabled=False,
            vad_enabled=False,  # Deepslate handles VAD server-side
        ),
    )

    opts = DeepslateOptions.from_env(
        system_prompt="You are a friendly and helpful AI assistant. Keep your answers concise."
    )
    tts = ElevenLabsTtsConfig.from_env()
    llm = DeepslateRealtimeLLMService(options=opts, tts_config=tts)

    llm.register_function("lookup_weather", lookup_weather)
    llm.register_function("get_current_location", get_current_location)

    pipeline = Pipeline([transport.input(), llm, transport.output()])
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    await task.queue_frame(LLMSetToolsFrame(tools=TOOLS))

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info(f"Participant {participant['id']} joined.")

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await task.cancel()

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Configuration

### `DeepslateOptions`

| Parameter         | Type            | Default                          | Description                                                    |
|-------------------|-----------------|----------------------------------|----------------------------------------------------------------|
| `vendor_id`       | `str`           | env: `DEEPSLATE_VENDOR_ID`       | Deepslate vendor ID                                            |
| `organization_id` | `str`           | env: `DEEPSLATE_ORGANIZATION_ID` | Deepslate organization ID                                      |
| `api_key`         | `str`           | env: `DEEPSLATE_API_KEY`         | Deepslate API key                                              |
| `base_url`        | `str`           | `"https://app.deepslate.eu"`     | Base URL for Deepslate API                                     |
| `system_prompt`   | `str`           | `"You are a helpful assistant."` | System prompt for the AI assistant                             |
| `ws_url`          | `Optional[str]` | `None`                           | Direct WebSocket URL (overrides `base_url`; for local dev/testing) |
| `max_retries`     | `int`           | `3`                              | Maximum reconnection attempts before giving up                 |

Use `DeepslateOptions.from_env()` to load credentials from environment variables:

```python
from deepslate.pipecat import DeepslateOptions

opts = DeepslateOptions.from_env(
    system_prompt="You are a customer service agent. Be professional and helpful.",
    max_retries=5,
)
```

### VAD Configuration

Pass a `VadConfig` (also aliased as `DeepslateVadConfig` for backwards compatibility) to tune server-side voice activity detection:

```python
from deepslate.pipecat import DeepslateRealtimeLLMService, VadConfig

llm = DeepslateRealtimeLLMService(
    options=opts,
    vad_config=VadConfig(
        confidence_threshold=0.3,  # Lower = more sensitive
        min_volume=0.005,
        start_duration_ms=100,
        stop_duration_ms=300,
        backbuffer_duration_ms=500,
    ),
)
```

| Parameter                | Type    | Default | Description                                                       |
|--------------------------|---------|---------|-------------------------------------------------------------------|
| `confidence_threshold`   | `float` | `0.5`   | Minimum confidence required to classify audio as speech (0.0–1.0) |
| `min_volume`             | `float` | `0.01`  | Minimum volume level to classify audio as speech (0.0–1.0)        |
| `start_duration_ms`      | `int`   | `200`   | Duration of speech (ms) required to trigger speech start          |
| `stop_duration_ms`       | `int`   | `500`   | Duration of silence (ms) required to trigger speech end           |
| `backbuffer_duration_ms` | `int`   | `1000`  | Audio (ms) buffered before speech detection triggers              |

**Tuning tips:**
- **Noisy environments:** Increase `confidence_threshold` (0.6–0.8) and `min_volume` (0.02–0.05)
- **Lower latency:** Decrease `start_duration_ms` (100–150) and `stop_duration_ms` (200–300)
- **Natural conversations:** Slightly increase `stop_duration_ms` (600–800)
- **Capture sentence starts:** Increase `backbuffer_duration_ms` (1500–2000)

### `HostedTtsConfig`

Use a voice cloned and hosted within Deepslate. No external TTS credentials required.

```python
from deepslate.pipecat import DeepslateRealtimeLLMService, HostedTtsConfig, HostedTtsMode

llm = DeepslateRealtimeLLMService(
    options=opts,
    tts_config=HostedTtsConfig(
        voice_id="c3dfa73f-a1ab-4aad-b48a-0e9b9fe4a69f",
        mode=HostedTtsMode.HIGH_QUALITY,  # or LOW_LATENCY
    ),
)
```

| Parameter  | Type            | Default                      | Description |
|------------|-----------------|------------------------------|-------------|
| `voice_id` | `str`           | required                     | ID of the hosted (cloned) voice |
| `mode`     | `HostedTtsMode` | `HostedTtsMode.HIGH_QUALITY` | Quality/latency tradeoff for highest response speed |

**`HostedTtsMode` values:**

| Value | Description |
|---|---|
| `HIGH_QUALITY` | Best output quality with still relatively low latency. Recommended for most use cases (default).                                                                  |
| `LOW_LATENCY` | Low latency generation mode that takes next to no time to complete. Output quality may be significantly reduced. |

### `ElevenLabsTtsConfig`

| Parameter  | Type                 | Default                    | Description                                                           |
|------------|----------------------|----------------------------|-----------------------------------------------------------------------|
| `api_key`  | `str`                | env: `ELEVENLABS_API_KEY`  | ElevenLabs API key                                                    |
| `voice_id` | `str`                | env: `ELEVENLABS_VOICE_ID` | Voice ID (e.g., `'21m00Tcm4TlvDq8ikWAM'` for Rachel)                |
| `model_id` | `Optional[str]`      | env: `ELEVENLABS_MODEL_ID` | Model ID (e.g., `'eleven_turbo_v2'`); uses ElevenLabs default if unset |
| `location` | `ElevenLabsLocation` | `ElevenLabsLocation.US`    | Regional endpoint: US (all accounts), EU or INDIA (enterprise only)  |

#### Server-side vs Client-side TTS

**Server-side TTS (recommended — best interruption handling):**

```python
from deepslate.pipecat import DeepslateRealtimeLLMService, ElevenLabsTtsConfig, HostedTtsConfig, HostedTtsMode

# Option A: Deepslate-hosted (cloned) voice — no external credentials needed
tts_config = HostedTtsConfig(voice_id="your-voice-id", mode=HostedTtsMode.HIGH_QUALITY)

# Option B: ElevenLabs
tts_config = ElevenLabsTtsConfig.from_env()

llm = DeepslateRealtimeLLMService(options=opts, tts_config=tts_config)

pipeline = Pipeline([transport.input(), llm, transport.output()])
```

**Client-side TTS (e.g., Cartesia):**

```python
from pipecat.services.cartesia import CartesiaTTSService

llm = DeepslateRealtimeLLMService(options=opts)  # No tts_config — emits TTSTextFrame
tts = CartesiaTTSService(...)

pipeline = Pipeline([transport.input(), llm, tts, transport.output()])
```

> **Important:** Server-side TTS enables Deepslate to truncate the response context when a user interrupts, ensuring the model stays in sync with what was actually spoken. Client-side TTS does not support this.

---

## Function Calling

Define tools as OpenAI-style JSON schemas, register async handlers, and sync the definitions to Deepslate via `LLMSetToolsFrame`:

```python
from pipecat.frames.frames import LLMSetToolsFrame
from pipecat.services.llm_service import FunctionCallParams

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Get the current weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city to look up."}
                },
                "required": ["location"],
            },
        },
    },
]

async def lookup_weather(params: FunctionCallParams):
    await params.result_callback({"temperature_celsius": 22, "condition": "sunny"})

llm.register_function("lookup_weather", lookup_weather)

# Queue tool definitions — synced to Deepslate after the pipeline starts
await task.queue_frame(LLMSetToolsFrame(tools=TOOLS))
```

---

## Dynamic Context Injection

Inject messages into an active session mid-conversation using `LLMMessagesAppendFrame`.

| Role        | Behaviour                                                       | Triggers inference? |
|-------------|-----------------------------------------------------------------|---------------------|
| `user`      | Appended to conversation history as a silent user input         | Only if `run_llm=True` |
| `system`    | Forwarded as `extra_instructions` on the next inference turn    | Only if `run_llm=True` |
| `assistant` | Not supported — logged as a warning                             | —                   |

> **Note:** `system` instructions via `LLMMessagesAppendFrame` are ephemeral — they affect only the triggered inference turn. To set a persistent system prompt, use `DeepslateOptions.system_prompt`.

**Silent context injection:**

```python
from pipecat.frames.frames import LLMMessagesAppendFrame

await task.queue_frame(
    LLMMessagesAppendFrame(
        messages=[{"role": "user", "content": "My name is Alice and I'm from Paris."}],
        run_llm=False,
    )
)
```

**Immediate inference with a system instruction:**

```python
await task.queue_frame(
    LLMMessagesAppendFrame(
        messages=[{
            "role": "system",
            "content": "You are now a professional chef assistant. Greet the user and ask how you can help with their cooking.",
        }],
        run_llm=True,
    )
)
```

---

## Sending a Welcome Message

`DeepslateSessionInitializedFrame` is emitted once the WebSocket session is fully initialized and ready to accept messages. Use it to send a welcome message instead of relying on a fixed delay:

```python
from deepslate.pipecat import DeepslateRealtimeLLMService, DeepslateSessionInitializedFrame, DeepslateDirectSpeechFrame

class MyPipeline(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, DeepslateSessionInitializedFrame):
            # Session is ready — send a welcome message
            await self.push_frame(DeepslateDirectSpeechFrame(text="Hello! How can I help you today?"))
```

---

## Transport Integration

### Daily.co (WebRTC)

```python
from pipecat.transports.services.daily import DailyTransport, DailyParams

transport = DailyTransport(
    room_url=daily_room_url,
    token=token,
    bot_name="My Voice Bot",
    params=DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_enabled=False,  # Deepslate handles VAD
    ),
)
```

### Twilio

```python
from pipecat.transports.services.twilio import TwilioTransport

transport = TwilioTransport(
    account_sid=twilio_account_sid,
    auth_token=twilio_auth_token,
    from_number=twilio_from_number,
)
```

### Generic WebSocket

```python
from pipecat.transports.network.websocket import WebsocketTransport, WebsocketParams

transport = WebsocketTransport(
    host="0.0.0.0",
    port=8765,
    params=WebsocketParams(audio_in_enabled=True, audio_out_enabled=True),
)
```

---

## Frame Reference

**Input frames consumed by `DeepslateRealtimeLLMService`:**

| Frame | Description |
|---|---|
| `AudioRawFrame` | PCM audio from user (forwarded to Deepslate for STT + inference) |
| `TextFrame` | Text input from user |
| `FunctionCallResultFrame` | Result of an executed function tool |
| `LLMMessagesAppendFrame` | Injects user/system messages mid-conversation |
| `LLMSetToolsFrame` | Updates active tool/function definitions |
| `StartFrame`, `EndFrame`, `CancelFrame` | Pipeline lifecycle management |

**Output frames emitted:**

| Frame | Description |
|---|---|
| `DeepslateSessionInitializedFrame` | Session is fully initialized and ready to accept messages |
| `LLMFullResponseStartFrame` / `LLMFullResponseEndFrame` | Marks the start/end of an AI response |
| `LLMTextFrame` | Streaming text transcript of the AI response |
| `OutputAudioRawFrame` | PCM audio output (only with server-side TTS configured) |
| `InterruptionFrame` | User interrupted — signals buffer clearing |
| `FunctionCallRequestFrame` | Request to execute a function tool |
| `ErrorFrame` | An error occurred during processing |

---

## Troubleshooting

### Connection Failures

Verify `DEEPSLATE_VENDOR_ID`, `DEEPSLATE_ORGANIZATION_ID`, and `DEEPSLATE_API_KEY` are set. The plugin retries with exponential backoff (2 s → 4 s → 8 s, capped at 30 s). Increase the retry limit if needed:

```python
opts = DeepslateOptions.from_env(max_retries=5)
```

### Audio Issues

Deepslate expects signed 16-bit PCM audio. Verify sample rate (common: 16000, 24000, 48000 Hz) and channel count (mono = 1) match between your transport and Deepslate. Enable `DEBUG` logging to inspect detected audio configuration:

```python
from loguru import logger
import sys
logger.remove()
logger.add(sys.stderr, level="DEBUG")
```

### No LLM Response

- Check VAD settings — they may be too strict (lower `confidence_threshold` or `min_volume`)
- Ensure sufficient audio duration is being sent
- Check for `ErrorFrame` output in the pipeline

### Protobuf Version Conflicts

```bash
pip install --upgrade "protobuf>=5.26.0"
```

---

## Examples

The [`examples/`](examples/) directory contains a ready-to-run bot you can use as a starting point.

### `simple_bot.py` — Daily.co voice bot with function calling

A fully working Pipecat pipeline that demonstrates:
- Daily.co WebRTC transport (swap for Twilio, WebSocket, etc.)
- Server-side ElevenLabs TTS with interruption handling
- Two example function tools: `lookup_weather` and `get_current_location`

```
packages/pipecat/examples/
├── simple_bot.py      # The bot
└── .env.example       # Required environment variables
```

**Setup:**

```bash
# 1. Install dependencies
pip install deepslate-pipecat "pipecat-ai[daily]" aiohttp python-dotenv loguru

# 2. Configure credentials
cd packages/pipecat/examples
cp .env.example .env
# Edit .env and fill in your credentials

# 3. Run
python simple_bot.py
```

---

## Documentation

- [Deepslate Documentation](https://docs.deepslate.eu/)
- [Pipecat Documentation](https://docs.pipecat.ai/)
- [API Reference](https://docs.deepslate.eu/api-reference/)

---

## Support

- **Issues:** [GitHub Issues](https://github.com/deepslate-labs/deepslate-sdks/issues)
- **Documentation:** [docs.deepslate.eu](https://docs.deepslate.eu/)
- **Email:** info@deepslate.eu

---

## License

Apache License 2.0 — see [LICENSE](../../LICENSE) for details.