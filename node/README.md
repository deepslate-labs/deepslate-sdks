# Deepslate Realtime SDKs — Node / TypeScript

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-deepslate.eu-green)](https://docs.deepslate.eu/)
[![Node](https://img.shields.io/badge/node-18+-blue.svg)](https://nodejs.org/)

Official Node/TypeScript SDKs for [Deepslate's](https://deepslate.eu/) realtime voice AI API.

Deepslate provides unified voice AI infrastructure — combining speech-to-speech streaming, server-side VAD, LLM inference, and ElevenLabs TTS — accessible through your preferred agent framework.

> These are the TypeScript counterparts to the [Python SDKs](../README.md). Both languages share a single wire contract: [`proto/realtime.proto`](../proto/realtime.proto).

---

## Packages

This is a [pnpm](https://pnpm.io/) workspace containing three packages:

| Package | npm | Description |
|---|---|---|
| [`@deepslate/livekit`](packages/livekit/README.md) | `npm install @deepslate/livekit` | `RealtimeModel` plugin for [LiveKit Agents](https://github.com/livekit/agents) (Node) |
| [`@deepslate/core`](packages/core/README.md) | `npm install @deepslate/core` | Shared implementation for integrations — prefer a higher-level package |
| [`@deepslate/proto`](packages/proto/README.md) | (internal) | Generated protobuf bindings; consumed by `@deepslate/core` only |

### Repository structure

```text
node/
├── package.json            # private workspace root (scripts + dev tooling)
├── pnpm-workspace.yaml
├── tsconfig.base.json       # shared compiler options, extended by each package
├── buf.gen.yaml             # protobuf-es codegen config
└── packages/
    ├── proto/               # @deepslate/proto   — generated TS from ../../proto
    ├── core/                # @deepslate/core    — transport + session
    └── livekit/             # @deepslate/livekit — LiveKit Agents plugin
```

---

## Installation

Install only the package you need. Each plugin brings in `@deepslate/core` automatically.

**For LiveKit Agents projects:**

```bash
npm install @deepslate/livekit
# plus the framework peers:
npm install @livekit/agents @livekit/rtc-node
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
- **Custom integrations** → [`packages/core/README.md`](packages/core/README.md)

---

## Documentation

- [Deepslate Documentation](https://docs.deepslate.eu/)
- [API Reference](https://docs.deepslate.eu/api-reference/)

---

## Contributing

This project uses [pnpm](https://pnpm.io/) for workspace management and [`buf`](https://buf.build/) +
[`@bufbuild/protobuf`](https://github.com/bufbuild/protobuf-es) (protobuf-es v2) for TypeScript
protobuf codegen. To set up a local development environment:

```bash
# pnpm is provided by corepack (bundled with Node 18+)
corepack enable

# Clone and install all workspace packages
git clone https://github.com/deepslate-labs/deepslate-sdks.git
cd deepslate-sdks/node
pnpm install
```

Common workspace scripts:

```bash
pnpm run generate      # regenerate @deepslate/proto from ../proto/realtime.proto
pnpm -r run build      # build every package (proto → core → livekit)
pnpm --filter @deepslate/core run test   # run the core unit tests (vitest)
```

> **Regenerating protobuf bindings:** the proto is shared with Python. See
> [`proto/README.md`](../proto/README.md) for both the Node (`buf`) and Python (`grpc_tools`)
> regeneration commands.

---

## License

Apache License 2.0 — see [LICENSE](../LICENSE) for details.