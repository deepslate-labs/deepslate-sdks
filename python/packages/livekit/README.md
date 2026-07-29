# deepslate-livekit

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-deepslate.eu-green)](https://docs.deepslate.eu/)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

LiveKit Agents plugin for [Deepslate's](https://deepslate.eu/) realtime voice AI API.

`deepslate-livekit` provides a `RealtimeModel` implementation for the [LiveKit Agents](https://github.com/livekit/agents) framework, enabling seamless integration with Deepslate's unified voice AI infrastructure — speech-to-speech streaming, server-side VAD, LLM inference, and optional ElevenLabs TTS, all in a single WebSocket connection.

---

## Features

- **Realtime Voice AI Streaming** — Low-latency bidirectional audio streaming over WebSockets
- **Server-side VAD** — Voice Activity Detection handled by Deepslate with configurable sensitivity
- **Function Tools** — Define and invoke tools using LiveKit's `@function_tool()` decorator
- **Flexible TTS** — Server-side TTS via Deepslate-hosted (cloned) voices or ElevenLabs, with automatic context truncation on interruption
- **Automatic Interruption Handling** — Truncates the in-flight response when users interrupt

---

## Installation

```bash
pip install deepslate-livekit
```

### Requirements

- Python 3.11 or higher

### Dependencies (installed automatically)

- `deepslate-core` — Shared Deepslate models and base client
- `livekit-agents>=1.3.8` — LiveKit Agents framework

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

For server-side text-to-speech with automatic interruption handling:

```bash
ELEVENLABS_API_KEY=your_elevenlabs_api_key
ELEVENLABS_VOICE_ID=your_voice_id
ELEVENLABS_MODEL_ID=eleven_turbo_v2  # optional
```

> **Note:** You can alternatively use LiveKit's built-in client-side TTS. However, context truncation on interruption only works with server-side TTS configured via `ElevenLabsTtsConfig`.

---

## Quick Start

```python
from livekit import agents
from livekit.agents import AgentServer, AgentSession, Agent, room_io

from deepslate.livekit import RealtimeModel, ElevenLabsTtsConfig


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")


server = AgentServer()


@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    session = AgentSession(
        llm=RealtimeModel(
            tts_config=ElevenLabsTtsConfig.from_env()
        ),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
```

---

## Configuration

### `RealtimeModel`

| Parameter                | Type                  | Default                          | Description                                             |
|--------------------------|-----------------------|----------------------------------|---------------------------------------------------------|
| `vendor_id`              | `str`                 | env: `DEEPSLATE_VENDOR_ID`       | Deepslate vendor ID                                     |
| `organization_id`        | `str`                 | env: `DEEPSLATE_ORGANIZATION_ID` | Deepslate organization ID                               |
| `api_key`                | `str`                 | env: `DEEPSLATE_API_KEY`         | Deepslate API key                                       |
| `base_url`               | `str`                 | `"https://app.deepslate.eu"`     | Base URL for Deepslate API                              |
| `system_prompt`          | `str`                 | `"You are a helpful assistant."` | System prompt for the model                             |
| `generate_reply_timeout` | `float`               | `30.0`                           | Timeout in seconds for `generate_reply` (0 = no limit) |
| `tts_config`             | `ElevenLabsTtsConfig \| HostedTtsConfig` | `None`          | TTS configuration (enables server-side audio output)    |
| `vad_config`             | `VadConfig`           | `None`                            | Voice activity detection tuning                         |

Pass a `VadConfig` instance to tune voice activity detection — see [VAD Configuration](#vad-configuration) below.

### VAD Configuration

```python
from deepslate.livekit import RealtimeModel, VadConfig

llm = RealtimeModel(
    vad_config=VadConfig(
        confidence_threshold=0.4,   # 0.0–1.0: minimum confidence to classify as speech
        min_volume=0.0,             # 0.0–1.0: minimum volume to classify as speech
        start_duration_ms=150,      # ms of speech required to trigger start
        stop_duration_ms=390,       # ms of silence required to trigger stop
        backbuffer_duration_ms=1000 # ms of audio buffered before detection triggers
    )
)
```

| Parameter                    | Type    | Default | Description                                               |
|-------------------------------|---------|---------|-----------------------------------------------------------|
| `confidence_threshold`        | `float` | `0.4`   | Minimum confidence to consider audio as speech (0.0–1.0)  |
| `min_volume`                  | `float` | `0.0`   | Minimum volume threshold (0.0–1.0)                        |
| `start_duration_ms`           | `int`   | `150`   | Duration of speech required to detect start (ms)          |
| `stop_duration_ms`            | `int`   | `390`   | Duration of silence required to detect end (ms)           |
| `backbuffer_duration_ms`      | `int`   | `1000`  | Audio buffer captured before speech detection triggers    |

**Tuning tips:**
- **Noisy environments:** Increase `confidence_threshold` (0.6–0.8) and `min_volume` (0.02–0.05)
- **Lower latency:** Decrease `start_duration_ms` (100–150) and `stop_duration_ms` (200–300)
- **Natural pacing:** Slightly increase `stop_duration_ms` (600–800)

### `HostedTtsConfig`

Use a voice cloned and hosted within Deepslate. No external TTS credentials required.

```python
from deepslate.livekit import RealtimeModel, HostedTtsConfig, HostedTtsMode

llm = RealtimeModel(
    tts_config=HostedTtsConfig(
        voice_id="c3dfa73f-a1ab-4aad-b48a-0e9b9fe4a69f",
        mode=HostedTtsMode.HIGH_QUALITY,  # or LOW_LATENCY
    )
)
```

| Parameter  | Type            | Default                      | Description |
|------------|-----------------|------------------------------|-------------|
| `voice_id` | `str`           | required                     | ID of the hosted (cloned) voice |
| `mode`     | `HostedTtsMode` | `HostedTtsMode.HIGH_QUALITY` | Quality/latency tradeoff for highest response speed |

**`HostedTtsMode` values:**

| Value | Description                                                                                                                                                      |
|---|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `HIGH_QUALITY` | Best output quality with still relatively low latency. Recommended for most use cases (default).                                                                 |
| `LOW_LATENCY` | Low latency generation mode that takes next to no time to complete. Output quality may be significantly reduced. |

### `ElevenLabsTtsConfig`

| Parameter  | Type                 | Default                   | Description                                                          |
|------------|----------------------|---------------------------|----------------------------------------------------------------------|
| `api_key`  | `str`                | env: `ELEVENLABS_API_KEY` | ElevenLabs API key                                                   |
| `voice_id` | `str`                | env: `ELEVENLABS_VOICE_ID` | Voice ID (e.g., `'21m00Tcm4TlvDq8ikWAM'` for Rachel)               |
| `model_id` | `str \| None`        | env: `ELEVENLABS_MODEL_ID` | Model ID, e.g., `'eleven_turbo_v2'`; uses ElevenLabs default if unset |
| `location` | `ElevenLabsLocation` | `ElevenLabsLocation.US`   | Regional API endpoint (US works with all accounts; EU/INDIA require enterprise) |

Use `ElevenLabsTtsConfig.from_env()` to load from environment variables.

---

## Function Tools

Use LiveKit's `@function_tool()` decorator to expose tools to the model:

```python
from livekit.agents import Agent, function_tool, RunContext
from deepslate.livekit import RealtimeModel


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful assistant.")

    @function_tool()
    async def get_weather(self, context: RunContext, location: str) -> str:
        """Get the current weather for a given city."""
        # Your implementation here
        return f"It's sunny and 22°C in {location}."
```

---

## Sending a Welcome Message

To greet the user, speak directly the moment the agent becomes active. Override
`Agent.on_enter()` and call `speak_direct()` on the realtime session that the
`AgentSession` created for you — reachable via `self.realtime_llm_session`.
`speak_direct()` buffers the utterance until the session is ready, so no fixed
delay or event handling is needed:

```python
from typing import cast

from livekit.agents import Agent
from deepslate.livekit import DeepslateRealtimeSession


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")

    async def on_enter(self) -> None:
        session = cast(DeepslateRealtimeSession, self.realtime_llm_session)
        await session.speak_direct(
            "Please note that this call is handled by an AI and may be recorded.",
            uninterruptable=True,
        )


@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    model = RealtimeModel(tts_config=ElevenLabsTtsConfig.from_env())
    session = AgentSession(llm=model)
    await session.start(room=ctx.room, agent=Assistant())
```

---

## Exporting Chat History

Call `export_chat_history()` on the realtime session to request the current
conversation from the server. It returns the exported messages directly, and
also emits a `chat_history_exported` event for listeners that prefer the
event-based style:

```python
from typing import cast

from deepslate.livekit import DeepslateRealtimeSession


rt = cast(DeepslateRealtimeSession, session.current_agent.realtime_llm_session)

history = await rt.export_chat_history(
    await_pending=True,   # wait for any in-flight turn to settle first
    exclude_audio=True,   # omit tts_audio/input_audio bytes, transcripts only
)

# Option 1: inspect the raw content blocks (text, tool_call, tool_result, ...)
for msg in history:
    print(msg["role"], msg["content"])

# Option 2: print just the text portions of each message
for msg in history:
    text = " ".join(c["text"] for c in msg["content"] if c["type"] == "text")
    print(f"[{msg['role']}] {text}")
```

Each item is a `ChatMessageDict` (importable from `deepslate.core`) with:

| Field | Description                                                                                                                              |
|---|------------------------------------------------------------------------------------------------------------------------------------------|
| `role` | `"system"` \| `"user"` \| `"assistant"`                                                                                                  |
| `delivery_status` | `DELIVERY_COMPLETE` \| `DELIVERY_IN_PROGRESS` \| `DELIVERY_INTERRUPTED`                                                                  |
| `ephemeral` | `true` when the message was spoken via `DirectSpeech` with `include_in_history: false`. Audible to the user but not in the LLM’s context |
| `content` | Ordered content blocks: `text` (with optional `tts_audio`), `input_audio`, `tool_call`, `tool_result`, `thoughts`, `instructions`        |
| `turn_id` | The model turn this message belongs to, or `None`                                                                                        |
| `truncated_at_response_turn_id` | Set if this message was cut off by a later interruption                                                                                  |

---

## Examples

The [`examples/`](examples/) directory contains a ready-to-run agent you can use as a starting point.

### `chat_agent.py` — Voice assistant with function tools

A fully working LiveKit agent that demonstrates:
- Connecting to a LiveKit room
- Server-side ElevenLabs TTS with interruption handling
- Two example function tools: `lookup_weather` and `get_current_location`

```
packages/livekit/examples/
├── chat_agent.py      # The agent
└── .env.example       # Required environment variables
```

**Setup:**

```bash
# 1. Install dependencies
pip install deepslate-livekit python-dotenv

# 2. Configure credentials
cd packages/livekit/examples
cp .env.example .env
# Edit .env and fill in your credentials

# 3. Run
python chat_agent.py dev
```

---

## Documentation

- [Deepslate Documentation](https://docs.deepslate.eu/)
- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [API Reference](https://docs.deepslate.eu/api-reference/)

---

## License

Apache License 2.0 — see [LICENSE](../../LICENSE) for details.