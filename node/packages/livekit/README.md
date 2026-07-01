# @deepslate-labs/livekit

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-deepslate.eu-green)](https://docs.deepslate.eu/)
[![Node](https://img.shields.io/badge/node-18+-blue.svg)](https://nodejs.org/)

LiveKit Agents plugin for [Deepslate's](https://deepslate.eu/) realtime voice AI API (TypeScript).

`@deepslate-labs/livekit` provides a `RealtimeModel` implementation for the [LiveKit Agents](https://github.com/livekit/agents)
Node framework, enabling seamless integration with Deepslate's unified voice AI infrastructure —
speech-to-speech streaming, server-side VAD, LLM inference, and optional ElevenLabs TTS, all in a single
WebSocket connection.

---

## Features

- **Realtime Voice AI Streaming** — Low-latency bidirectional audio streaming over WebSockets
- **Server-side VAD** — Voice Activity Detection handled by Deepslate with configurable sensitivity
- **Function Tools** — Define and invoke tools using LiveKit's `llm.tool()` helper
- **Flexible TTS** — Server-side TTS via Deepslate-hosted (cloned) voices or ElevenLabs, with automatic context truncation on interruption
- **Automatic Interruption Handling** — Truncates the in-flight response when users interrupt

---

## Installation

```bash
npm install @deepslate-labs/livekit
```

### Requirements

- Node.js 18 or higher

### Peer dependencies

- `@livekit/agents` `^1.0.7` — LiveKit Agents framework (Node)
- `@livekit/rtc-node` `^0.13.27` — LiveKit realtime audio frames

```bash
npm install @livekit/agents @livekit/rtc-node
```

`@deepslate-labs/core` is pulled in automatically.

---

## Prerequisites

### Deepslate Account

Sign up at [deepslate.eu](https://deepslate.eu) and set the following environment variables:

```bash
DEEPSLATE_VENDOR_ID=your_vendor_id
DEEPSLATE_ORGANIZATION_ID=your_organization_id
DEEPSLATE_API_KEY=your_api_key
```

### ElevenLabs TTS (optional)

For server-side text-to-speech with automatic interruption handling:

```bash
ELEVENLABS_API_KEY=your_elevenlabs_api_key
ELEVENLABS_VOICE_ID=your_voice_id
ELEVENLABS_MODEL_ID=eleven_turbo_v2  # optional
```

> **Note:** You can alternatively use LiveKit's built-in client-side TTS. However, context truncation on
> interruption only works with server-side TTS configured via `ElevenLabsTtsConfig` / `HostedTtsConfig`.

---

## Quick Start

```ts
import { fileURLToPath } from "node:url";
import { type JobContext, ServerOptions, cli, defineAgent, voice } from "@livekit/agents";
import { RealtimeModel, elevenLabsConfigFromEnv } from "@deepslate-labs/livekit";

export default defineAgent({
  entry: async (ctx: JobContext) => {
    await ctx.connect();

    const session = new voice.AgentSession({
      llm: new RealtimeModel({
        ttsConfig: elevenLabsConfigFromEnv(),
      }),
    });

    await session.start({
      agent: new voice.Agent({ instructions: "You are a helpful voice AI assistant." }),
      room: ctx.room,
    });

    session.generateReply({ instructions: "Greet the user and offer your assistance." });
  },
});

cli.runApp(new ServerOptions({ agent: fileURLToPath(import.meta.url) }));
```

---

## Configuration

### `RealtimeModel`

The constructor takes a single options object (`RealtimeModelOptions`):

| Field | Type | Default | Description |
|---|---|---|---|
| `vendorId` | `string` | env: `DEEPSLATE_VENDOR_ID` | Deepslate vendor ID |
| `organizationId` | `string` | env: `DEEPSLATE_ORGANIZATION_ID` | Deepslate organization ID |
| `apiKey` | `string` | env: `DEEPSLATE_API_KEY` | Deepslate API key |
| `baseUrl` | `string` | `"https://app.deepslate.eu"` | Base URL for Deepslate API |
| `systemPrompt` | `string` | `"You are a helpful assistant."` | System prompt for the model |
| `temperature` | `number` | `1.0` | Sampling temperature (0.0–2.0) |
| `generateReplyTimeout` | `number` | `30.0` | Timeout in seconds for `generateReply` (0 = no limit) |
| `vad` | `VadConfig` | defaults | Voice activity detection tuning |
| `ttsConfig` | `TtsConfig` | `undefined` | TTS configuration (enables server-side audio output) |
| `wsUrl` | `string` | `undefined` | Direct WebSocket URL (for local dev/testing) |

### VAD Configuration

```ts
import { RealtimeModel } from "@deepslate-labs/livekit";

const model = new RealtimeModel({
  vad: {
    confidenceThreshold: 0.5,   // 0.0–1.0: minimum confidence to classify as speech
    minVolume: 0.01,            // 0.0–1.0: minimum volume to classify as speech
    startDurationMs: 200,       // ms of speech required to trigger start
    stopDurationMs: 500,        // ms of silence required to trigger stop
    backbufferDurationMs: 1000, // ms of audio buffered before detection triggers
  },
});
```

**Tuning tips:**
- **Noisy environments:** increase `confidenceThreshold` (0.6–0.8) and `minVolume` (0.02–0.05)
- **Lower latency:** decrease `startDurationMs` (100–150) and `stopDurationMs` (200–300)
- **Natural pacing:** slightly increase `stopDurationMs` (600–800)

### `HostedTtsConfig`

Use a voice cloned and hosted within Deepslate. No external TTS credentials required.

```ts
import { RealtimeModel, HostedTtsMode } from "@deepslate-labs/livekit";

const model = new RealtimeModel({
  ttsConfig: {
    provider: "hosted",
    voiceId: "c3dfa73f-a1ab-4aad-b48a-0e9b9fe4a69f",
    mode: HostedTtsMode.HIGH_QUALITY, // or LOW_LATENCY
  },
});
```

### `ElevenLabsTtsConfig`

```ts
import { RealtimeModel, ElevenLabsLocation, elevenLabsConfigFromEnv } from "@deepslate-labs/livekit";

// Load from environment variables
const model = new RealtimeModel({ ttsConfig: elevenLabsConfigFromEnv() });

// Or configure manually
const model = new RealtimeModel({
  ttsConfig: {
    provider: "eleven_labs",
    apiKey: "your_elevenlabs_key",
    voiceId: "21m00Tcm4TlvDq8ikWAM",
    modelId: "eleven_turbo_v2",
    location: ElevenLabsLocation.US, // US (default), EU, or INDIA
  },
});
```

---

## Function Tools

Use LiveKit's `llm.tool()` helper to expose tools to the model. Tool parameters are described with a
[zod](https://zod.dev/) schema:

```ts
import { llm, voice } from "@livekit/agents";
import { z } from "zod";

const lookupWeather = llm.tool({
  description: "Get the current weather for a given location.",
  parameters: z.object({
    location: z.string().describe("The city or location to look up weather for."),
  }),
  execute: async ({ location }) => `It's sunny and 22°C in ${location}.`,
});

const agent = new voice.Agent({
  instructions: "You are a helpful assistant.",
  tools: { lookupWeather },
});
```

---

## Sending a Welcome Message

To greet the user, speak directly the moment the agent becomes active. Subclass `voice.Agent`, override
`onEnter()`, and call `speakDirect()` on the realtime session that the `AgentSession` created for you —
reachable via `getActivityOrThrow().realtimeLLMSession`:

```ts
import { voice } from "@livekit/agents";
import { RealtimeModel, DeepslateRealtimeSession, elevenLabsConfigFromEnv } from "@deepslate-labs/livekit";

class Assistant extends voice.Agent {
  constructor() {
    super({ instructions: "You are a helpful voice AI assistant." });
  }

  async onEnter(): Promise<void> {
    // This is the SAME session AgentSession is driving, so its audio is wired
    // to the room. speakDirect() initializes the session if needed and buffers
    // the utterance until it is ready - no fixed delay and no
    // "session_initialized" event handling required.
    const session = this.getActivityOrThrow().realtimeLLMSession as DeepslateRealtimeSession;
    // The third argument (uninterruptable) is set to true so the greeting is
    // spoken in full even if the user starts talking over it. The second
    // argument (includeInHistory) keeps its default of true.
    await session.speakDirect("Hello! How can I help you today?", true, true);
  }
}

const session = new voice.AgentSession({
  llm: new RealtimeModel({ ttsConfig: elevenLabsConfigFromEnv() }),
});
await session.start({ agent: new Assistant(), room: ctx.room });
```

> **Do not call `model.session()` yourself here.** `AgentSession.start()` internally calls
> `model.session()` to create the realtime session it connects to the room. Calling `model.session()`
> again opens a *second, independent* WebSocket session whose audio is never routed to the room - your
> welcome message would be spoken into the void while a duplicate session runs in parallel. Always reach
> the active session through `getActivityOrThrow().realtimeLLMSession`.

---

## Examples

The [`examples/`](examples/) directory contains a ready-to-run agent you can use as a starting point.

### `chat-agent.ts` — Voice assistant with function tools

A fully working LiveKit agent that demonstrates:
- Connecting to a LiveKit room
- Server-side ElevenLabs TTS with interruption handling
- Two example function tools: `lookupWeather` and `getCurrentLocation`

```
packages/livekit/examples/
├── chat-agent.ts      # The agent
└── .env.example       # Required environment variables
```

**Setup:**

```bash
# 1. Configure credentials
cd packages/livekit/examples
cp .env.example .env
# Edit .env and fill in your credentials

# 2. Run (tsx is included as a workspace dev dependency)
pnpm tsx examples/chat-agent.ts dev
```

---

## Documentation

- [Deepslate Documentation](https://docs.deepslate.eu/)
- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [API Reference](https://docs.deepslate.eu/api-reference/)

---

## License

Apache License 2.0 — see [LICENSE](../../../LICENSE) for details.