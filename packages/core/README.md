# deepslate-core

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-deepslate.eu-green)](https://docs.deepslate.eu/)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Shared core library for [Deepslate's](https://deepslate.eu/) realtime voice AI SDKs.

> **You probably don't need to install this directly.**
>
> Install [`deepslate-livekit`](../livekit/README.md) or [`deepslate-pipecat`](../pipecat/README.md) instead — they pull in `deepslate-core` automatically. This package is for developers building **custom Deepslate integrations** outside of LiveKit or Pipecat.

---

## What's Inside

`deepslate-core` provides everything needed to connect to the Deepslate Realtime API from any async Python application:

- **`DeepslateSession`** — High-level session that manages the full WebSocket lifecycle, protobuf framing, session initialization, reconnection, and callback dispatch. The primary building block for custom integrations.
- **`DeepslateOptions`** — API credentials and connection configuration
- **`VadConfig`** — Server-side Voice Activity Detection parameters
- **`ElevenLabsTtsConfig` / `ElevenLabsLocation`** — ElevenLabs TTS configuration
- **`BaseDeepslateClient`** — Low-level async WebSocket client with exponential-backoff reconnection (used internally by `DeepslateSession`)
- **Protobuf definitions** — Compiled `.proto` bindings for the Deepslate realtime wire protocol

---

## Installation

```bash
pip install deepslate-core
```

### Requirements

- Python 3.11 or higher

### Dependencies

- `aiohttp>=3.10.0` — Async HTTP and WebSocket client
- `protobuf>=5.26.0` — Protocol buffer serialization

---

## Usage

> This section is for custom integration authors. For standard usage see the [LiveKit](../livekit/README.md) or [Pipecat](../pipecat/README.md) packages.

### Credentials

```python
from deepslate.core import DeepslateOptions

# Load from environment variables (recommended)
opts = DeepslateOptions.from_env()

# Or configure manually
opts = DeepslateOptions(
    vendor_id="your_vendor_id",
    organization_id="your_org_id",
    api_key="your_api_key",
    system_prompt="You are a helpful assistant.",
)
```

Required environment variables:

```bash
DEEPSLATE_VENDOR_ID=your_vendor_id
DEEPSLATE_ORGANIZATION_ID=your_organization_id
DEEPSLATE_API_KEY=your_api_key
```

### VAD Configuration

```python
from deepslate.core import VadConfig

vad = VadConfig(
    confidence_threshold=0.5,   # 0.0–1.0
    min_volume=0.01,            # 0.0–1.0
    start_duration_ms=200,
    stop_duration_ms=500,
    backbuffer_duration_ms=1000,
)
```

### ElevenLabs TTS Configuration

```python
from deepslate.core import ElevenLabsTtsConfig, ElevenLabsLocation

# Load from environment variables
tts = ElevenLabsTtsConfig.from_env()

# Or configure manually
tts = ElevenLabsTtsConfig(
    api_key="your_elevenlabs_key",
    voice_id="21m00Tcm4TlvDq8ikWAM",
    model_id="eleven_turbo_v2",
    location=ElevenLabsLocation.US,  # US (default), EU, or INDIA
)
```

### `DeepslateSession`

`DeepslateSession` is the recommended entry point for custom integrations. It handles the full protocol lifecycle — session initialization, protobuf serialization, reconnection, and server-event routing — exposing a clean callback interface so your code only deals with application logic.

Use `DeepslateSession.create()` when your code owns the connection. It creates its own `BaseDeepslateClient` and closes it automatically when `close()` is called:

```python
import asyncio
from deepslate.core import DeepslateOptions, DeepslateSession, ElevenLabsTtsConfig

async def main():
    opts = DeepslateOptions.from_env(
        system_prompt="You are a helpful assistant."
    )
    tts = ElevenLabsTtsConfig.from_env()

    async def on_text_fragment(text: str) -> None:
        print(text, end="", flush=True)

    async def on_audio_chunk(
        pcm_bytes: bytes, sample_rate: int, channels: int, transcript: str | None
    ) -> None:
        # Forward PCM audio to your output device / transport
        ...

    async def on_tool_call(call_id: str, name: str, params: dict) -> None:
        result = await dispatch_tool(name, params)
        await session.send_tool_response(call_id, result)

    async def on_error(category: str, message: str, trace_id: str | None) -> None:
        print(f"[{category}] {message}")

    session = DeepslateSession.create(
        opts,
        tts_config=tts,
        user_agent="MyApp/1.0",
        on_text_fragment=on_text_fragment,
        on_audio_chunk=on_audio_chunk,
        on_tool_call=on_tool_call,
        on_error=on_error,
        on_response_begin=lambda: print("\n--- response start ---"),
        on_response_end=lambda: print("--- response end ---\n"),
    )

    session.start()

    # Initialize for text-only interaction (audio sessions initialize automatically)
    await session.initialize(sample_rate=24000, channels=1)

    # Send a text prompt and trigger a reply
    await session.send_text("What is the capital of France?")

    await asyncio.sleep(5)  # Wait for response
    await session.close()


asyncio.run(main())
```

#### Sending audio

Audio sessions initialize automatically on the first `send_audio()` call. The session also sends a `ReconfigureSessionRequest` transparently if the audio format changes mid-session:

```python
# pcm_bytes: raw signed 16-bit PCM
await session.send_audio(pcm_bytes, sample_rate=16000, channels=1)
```

#### Updating tools at runtime

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"],
            },
        },
    }
]

await session.update_tools(tools)
```

Tool definitions are re-synced automatically after every reconnect.

#### Reconnection

`start()` drives `BaseDeepslateClient.run_with_retry()` internally. On a dropped connection the session resets its state, re-sends `InitializeSessionRequest`, re-syncs tool definitions, and resumes — all without any action required from your code. Call `close()` to stop the retry loop permanently.

---

## API Reference

### `DeepslateSession.create()`

Factory that creates a session together with its own `BaseDeepslateClient`. The session owns the client and closes it when `close()` is called.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `options` | `DeepslateOptions` | required | API credentials and settings |
| `vad_config` | `VadConfig \| None` | `None` | VAD settings (uses defaults if omitted) |
| `tts_config` | `ElevenLabsTtsConfig \| None` | `None` | Enables server-side TTS audio output |
| `user_agent` | `str` | `"DeepslateCore"` | HTTP `User-Agent` header sent on connect |
| `http_session` | `aiohttp.ClientSession \| None` | `None` | Shared aiohttp session (not closed by the session) |
| `on_*` | callbacks | `None` | See callback table below |

### `DeepslateSession` — send methods

| Method | Description |
|---|---|
| `await session.send_audio(pcm, sr, ch)` | Send PCM audio; auto-initializes and auto-reconfigures on format change |
| `await session.send_text(text, trigger=IMMEDIATE)` | Send a text `UserInput` |
| `await session.initialize(sr=24000, ch=1)` | Explicitly initialize (for text-only sessions) |
| `await session.trigger_inference(instructions=None)` | Manually trigger a model reply |
| `await session.send_tool_response(call_id, result)` | Return a tool result to the server |
| `await session.update_tools(tools)` | Sync tool definitions (persisted across reconnects) |
| `await session.reconfigure(system_prompt=None, temperature=None)` | Live-update inference settings |
| `await session.send_direct_speech(text, include_in_history=True)` | Speak text directly via TTS, bypassing the LLM |
| `await session.export_chat_history(await_pending=False)` | Request a history export; result delivered via `on_chat_history` |
| `await session.send_conversation_query(query_id, prompt, instructions)` | Side-channel inference; at least one of `prompt`/`instructions` required; result via `on_conversation_query_result` |
| `await session.report_playback_position(bytes_played)` | Report audio playback position for server-side truncation |

### `DeepslateSession` — callbacks

All callbacks are optional `async` callables, set as constructor parameters or reassigned as attributes after construction.

| Callback | Signature | Fired when |
|---|---|---|
| `on_text_fragment` | `(text: str)` | Model streams a text token |
| `on_audio_chunk` | `(pcm: bytes, sr: int, ch: int, transcript: str \| None)` | Model streams a TTS audio chunk |
| `on_tool_call` | `(call_id: str, name: str, params: dict)` | Model requests a tool invocation |
| `on_response_begin` | `()` | Model response starts |
| `on_response_end` | `()` | Model response ends |
| `on_user_transcription` | `(text: str, language: str \| None)` | User speech transcription result arrives |
| `on_playback_buffer_clear` | `()` | Server cleared its audio playback buffer |
| `on_chat_history` | `(messages: list[ChatMessageDict])` | Chat history export received |
| `on_conversation_query_result` | `(query_id: str, text: str)` | Side-channel query result received |
| `on_error` | `(category: str, message: str, trace_id: str \| None)` | Server sent an error notification |
| `on_fatal_error` | `(exc: Exception)` | All reconnect retries exhausted |

### `DeepslateOptions`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vendor_id` | `str` | env: `DEEPSLATE_VENDOR_ID` | Deepslate vendor ID |
| `organization_id` | `str` | env: `DEEPSLATE_ORGANIZATION_ID` | Deepslate organization ID |
| `api_key` | `str` | env: `DEEPSLATE_API_KEY` | Deepslate API key |
| `base_url` | `str` | `"https://app.deepslate.eu"` | Base URL for Deepslate API |
| `system_prompt` | `str` | `"You are a helpful assistant."` | Default system prompt |
| `ws_url` | `str \| None` | `None` | Direct WebSocket URL (overrides `base_url`; for local dev) |
| `max_retries` | `int` | `3` | Maximum reconnection attempts before giving up |

### `VadConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `confidence_threshold` | `float` | `0.5` | Minimum confidence to classify audio as speech (0–1) |
| `min_volume` | `float` | `0.01` | Minimum volume to classify audio as speech (0–1) |
| `start_duration_ms` | `int` | `200` | Duration of speech required to trigger start event |
| `stop_duration_ms` | `int` | `500` | Duration of silence required to trigger stop event |
| `backbuffer_duration_ms` | `int` | `1000` | Audio buffer captured before speech detection triggers |

---

## Documentation

- [Deepslate Documentation](https://docs.deepslate.eu/)
- [API Reference](https://docs.deepslate.eu/api-reference/)

---

## License

Apache License 2.0 — see [LICENSE](../../LICENSE) for details.