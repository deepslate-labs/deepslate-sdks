# deepslate-core

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-deepslate.eu-green)](https://docs.deepslate.eu/)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Shared core library for [Deepslate's](https://deepslate.eu/) realtime voice AI SDKs.

> **You probably don't need to install this directly.**
>
> Install [`deepslate-livekit`](../livekit/README.md) or [`deepslate-pipecat`](../pipecat/README.md) instead — they pull in `deepslate-core` automatically. This package is intended for developers building **custom Deepslate integrations** outside of LiveKit or Pipecat.

---

## What's Inside

`deepslate-core` contains the shared building blocks used by all Deepslate SDK integrations:

- **`DeepslateOptions`** — API credentials and connection configuration
- **`VadConfig`** — Server-side Voice Activity Detection parameters
- **`ElevenLabsTtsConfig` / `ElevenLabsLocation`** — ElevenLabs TTS configuration
- **`BaseDeepslateClient`** — Base async WebSocket client with connection management and retry logic
- **Protobuf definitions** — Compiled `.proto` bindings for the Deepslate realtime wire protocol (`deepslate_core.proto`)

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

> This section is for custom integration authors. For standard usage, see the [LiveKit](../livekit/README.md) or [Pipecat](../pipecat/README.md) packages.

### Credentials

```python
from deepslate_core import DeepslateOptions

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
from deepslate_core import VadConfig

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
from deepslate_core import ElevenLabsTtsConfig, ElevenLabsLocation

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

### Base Client

`BaseDeepslateClient` provides the async WebSocket connection lifecycle, exponential-backoff reconnection, and protobuf message framing. Subclass it to build a custom integration:

```python
from deepslate_core import BaseDeepslateClient, DeepslateOptions

class MyCustomClient(BaseDeepslateClient):
    async def _on_message(self, message):
        # Handle incoming protobuf messages
        ...

opts = DeepslateOptions.from_env()
client = MyCustomClient(options=opts)

async with client:
    # Connection is established; send messages via client._send(proto_message)
    ...
```

---

## API Reference

### `DeepslateOptions`

| Parameter         | Type            | Default                          | Description                                           |
|-------------------|-----------------|----------------------------------|-------------------------------------------------------|
| `vendor_id`       | `str`           | env: `DEEPSLATE_VENDOR_ID`       | Deepslate vendor ID                                   |
| `organization_id` | `str`           | env: `DEEPSLATE_ORGANIZATION_ID` | Deepslate organization ID                             |
| `api_key`         | `str`           | env: `DEEPSLATE_API_KEY`         | Deepslate API key                                     |
| `base_url`        | `str`           | `"https://app.deepslate.eu"`     | Base URL for Deepslate API                            |
| `system_prompt`   | `str`           | `"You are a helpful assistant."` | Default system prompt                                 |
| `ws_url`          | `Optional[str]` | `None`                           | Direct WebSocket URL (overrides `base_url`; for local dev) |
| `max_retries`     | `int`           | `3`                              | Maximum reconnection attempts before giving up        |

### `VadConfig`

| Parameter                | Type    | Default | Description                                            |
|--------------------------|---------|---------|--------------------------------------------------------|
| `confidence_threshold`   | `float` | `0.5`   | Minimum confidence to classify audio as speech (0–1)   |
| `min_volume`             | `float` | `0.01`  | Minimum volume to classify audio as speech (0–1)       |
| `start_duration_ms`      | `int`   | `200`   | Duration of speech required to trigger start event     |
| `stop_duration_ms`       | `int`   | `500`   | Duration of silence required to trigger stop event     |
| `backbuffer_duration_ms` | `int`   | `1000`  | Audio buffer captured before speech detection triggers |

---

## Documentation

- [Deepslate Documentation](https://docs.deepslate.eu/)
- [API Reference](https://docs.deepslate.eu/api-reference/)

---

## License

Apache License 2.0 — see [LICENSE](../../LICENSE) for details.