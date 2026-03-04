# Deepslate Realtime SDKs

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-deepslate.eu-green)](https://docs.deepslate.eu/)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Official Python SDKs for [Deepslate's](https://deepslate.eu/) realtime voice AI API.

Deepslate provides unified voice AI infrastructure — combining speech-to-speech streaming, server-side VAD, LLM inference, and ElevenLabs TTS — accessible through your preferred agent framework.

---

## Packages

This monorepo contains three published packages:

| Package | PyPI | Description |
|---|---|---|
| [`deepslate-livekit`](packages/livekit/README.md) | `pip install deepslate-livekit` | `RealtimeModel` plugin for [LiveKit Agents](https://github.com/livekit/agents) |
| [`deepslate-pipecat`](packages/pipecat/README.md) | `pip install deepslate-pipecat` | `LLMService` plugin for [Pipecat](https://github.com/pipecat-ai/pipecat) |
| [`deepslate-core`](packages/core/README.md) | `pip install deepslate-core` | Shared models, base client, and protobuf definitions (internal) |

### Repository Structure

```
deepslate-sdks/
├── packages/
│   ├── core/          # deepslate-core — shared internals
│   ├── livekit/       # deepslate-livekit — LiveKit Agents plugin
│   └── pipecat/       # deepslate-pipecat — Pipecat plugin
├── pyproject.toml     # uv workspace root
└── uv.lock
```

---

## Installation

Install only the package you need. Each plugin brings in `deepslate-core` automatically as a dependency.

**For LiveKit Agents projects:**

```bash
pip install deepslate-livekit
```

**For Pipecat projects:**

```bash
pip install deepslate-pipecat
```

---

## Quick Start

Set your credentials as environment variables:

```bash
DEEPSLATE_VENDOR_ID=your_vendor_id
DEEPSLATE_ORGANIZATION_ID=your_organization_id
DEEPSLATE_API_KEY=your_api_key
```

Then see the package-specific README for a complete usage example:

- **LiveKit** → [`packages/livekit/README.md`](packages/livekit/README.md)
- **Pipecat** → [`packages/pipecat/README.md`](packages/pipecat/README.md)

---

## Documentation

- [Deepslate Documentation](https://docs.deepslate.eu/)
- [API Reference](https://docs.deepslate.eu/api-reference/)

---

## Contributing

Contributions are welcome. Please open an issue or pull request on GitHub.

This project uses [`uv`](https://docs.astral.sh/uv/) for workspace management. To set up a local development environment:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and sync all workspace packages
git clone https://github.com/deepslate-labs/deepslate-sdks.git
cd deepslate-sdks
uv sync --all-packages
```

Individual packages can be worked on in isolation:

```bash
uv sync --package deepslate-livekit
uv sync --package deepslate-pipecat
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.