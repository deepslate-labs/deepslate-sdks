# Deepslate Realtime SDKs

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-deepslate.eu-green)](https://docs.deepslate.eu/)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Node](https://img.shields.io/badge/node-18+-blue.svg)](https://nodejs.org/)

Official SDKs for [Deepslate's](https://deepslate.eu/) realtime voice AI API, for **Python** and **Node/TypeScript**.

Deepslate provides unified voice AI infrastructure — combining speech-to-speech streaming, server-side VAD, LLM inference, and ElevenLabs TTS — accessible through your preferred agent framework.

---

## Languages

This is a polyglot monorepo. Both languages share a single wire contract — [`proto/realtime.proto`](proto/realtime.proto) — and mirror the same architecture (a shared `core` package plus framework-specific plugins).

| Language | Location | Tooling | Docs |
|---|---|---|---|
| **Python** | [`python/`](python/) | [`uv`](https://docs.astral.sh/uv/) workspace | see tables below |
| **Node / TypeScript** | [`node/`](node/) | [`pnpm`](https://pnpm.io/) workspace | [`node/README.md`](node/README.md) |

### Python packages

| Package | PyPI | Description |
|---|---|--|
| [`deepslate-livekit`](python/packages/livekit/README.md) | `pip install deepslate-livekit` | `RealtimeModel` plugin for [LiveKit Agents](https://github.com/livekit/agents) |
| [`deepslate-pipecat`](python/packages/pipecat/README.md) | `pip install deepslate-pipecat` | `LLMService` plugin for [Pipecat](https://github.com/pipecat-ai/pipecat) |
| [`deepslate-core`](python/packages/core/README.md) | `pip install deepslate-core` | Shared implementation for integrations; prefer a higher-level package |

### Node / TypeScript packages

| Package | npm | Description |
|---|---|---|
| [`@deepslate/livekit`](node/packages/livekit/README.md) | `npm install @deepslate/livekit` | `RealtimeModel` plugin for [LiveKit Agents](https://github.com/livekit/agents) (Node) |
| [`@deepslate/core`](node/packages/core/README.md) | `npm install @deepslate/core` | Shared implementation for integrations; prefer a higher-level package |
| [`@deepslate/proto`](node/packages/proto/README.md) | (internal) | Generated protobuf bindings; consumed by `@deepslate/core` only |

---

## Repository structure

```text
deepslate-sdks/
├── proto/
│   └── realtime.proto       # single wire-contract source of truth (both languages)
├── python/                  # uv workspace
│   ├── pyproject.toml
│   ├── uv.lock
│   └── packages/
│       ├── core/            # deepslate-core
│       ├── livekit/         # deepslate-livekit
│       └── pipecat/         # deepslate-pipecat
└── node/                    # pnpm workspace
    ├── package.json
    ├── pnpm-workspace.yaml
    └── packages/
        ├── proto/           # @deepslate/proto (generated)
        ├── core/            # @deepslate/core
        └── livekit/         # @deepslate/livekit
```

The proto is compiled by each language with its own toolchain (Python via `grpc_tools`, Node via
[`buf`](https://buf.build/)) but the `.proto` file is never duplicated. See
[`proto/README.md`](proto/README.md) for both regeneration commands.

---

## Quick Start

Set your credentials as environment variables (same for both languages):

```bash
DEEPSLATE_VENDOR_ID=your_vendor_id
DEEPSLATE_ORGANIZATION_ID=your_organization_id
DEEPSLATE_API_KEY=your_api_key
```

**Python**

```bash
pip install deepslate-livekit   # or deepslate-pipecat
```

**Node / TypeScript**

```bash
npm install @deepslate/livekit @livekit/agents @livekit/rtc-node
```

Then see the package-specific README for a complete usage example:

- **Python LiveKit** → [`python/packages/livekit/README.md`](python/packages/livekit/README.md)
- **Python Pipecat** → [`python/packages/pipecat/README.md`](python/packages/pipecat/README.md)
- **Node LiveKit** → [`node/packages/livekit/README.md`](node/packages/livekit/README.md)

---

## Documentation

- [Deepslate Documentation](https://docs.deepslate.eu/)
- [API Reference](https://docs.deepslate.eu/api-reference/)

---

## Contributing

Contributions are welcome. Please open an issue or pull request on GitHub.

**Python** — uses [`uv`](https://docs.astral.sh/uv/) for workspace management:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and sync all workspace packages
git clone https://github.com/deepslate-labs/deepslate-sdks.git
cd deepslate-sdks/python
uv sync --all-packages
```

**Node / TypeScript** — uses [`pnpm`](https://pnpm.io/) (provided by corepack):

```bash
corepack enable
cd deepslate-sdks/node
pnpm install
pnpm -r run build
```

See [`node/README.md`](node/README.md) for the full Node development workflow.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.