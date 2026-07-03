# @deepslate-labs/core

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-deepslate.eu-green)](https://docs.deepslate.eu/)
[![Node](https://img.shields.io/badge/node-18+-blue.svg)](https://nodejs.org/)

Shared core library for [Deepslate's](https://deepslate.eu/) realtime voice AI SDKs (TypeScript).

> **You probably don't need to install this directly.**
>
> Install [`@deepslate-labs/livekit`](../livekit/README.md) instead — it pulls in `@deepslate-labs/core`
> automatically. This package is for developers building **custom Deepslate integrations** outside of
> LiveKit.

`DeepslateSession` is a typed `EventEmitter` — you subscribe to server events with
`session.on(event, …)`.

---

## What's Inside

`@deepslate-labs/core` provides everything needed to connect to the Deepslate Realtime API from any Node application:

- **`DeepslateSession`** — High-level session that manages the full WebSocket lifecycle, protobuf framing, session initialization, reconnection, and event dispatch. A typed `EventEmitter`. The primary building block for custom integrations.
- **`DeepslateOptions`** — API credentials and connection configuration, with `resolveOptions()` / `optionsFromEnv()` normalizers.
- **`VadConfig`** — Server-side Voice Activity Detection parameters.
- **`HostedTtsConfig` / `HostedTtsMode`** — Deepslate-hosted (cloned) voice TTS configuration.
- **`ElevenLabsTtsConfig` / `ElevenLabsLocation`** — ElevenLabs TTS configuration (`elevenLabsConfigFromEnv()` loads from env).
- **`BaseDeepslateClient`** — Low-level WebSocket client with exponential-backoff reconnection (used internally by `DeepslateSession`).
- **Protobuf bindings** — re-exported via [`@deepslate-labs/proto`](../proto/README.md) for the Deepslate realtime wire protocol.

---

## Installation

```bash
npm install @deepslate-labs/core
```

### Requirements

- Node.js 18 or higher

### Dependencies

- `ws` — WebSocket client
- `@deepslate-labs/proto` — generated protobuf bindings (protobuf-es v2)

---

## Usage

> This section is for custom integration authors. For standard usage see the [LiveKit](../livekit/README.md) package.

### Credentials

```ts
import { optionsFromEnv, resolveOptions } from "@deepslate-labs/core";

// Load from environment variables (recommended)
const opts = optionsFromEnv();

// Or configure manually
const opts = resolveOptions({
  vendorId: "your_vendor_id",
  organizationId: "your_org_id",
  apiKey: "your_api_key",
  systemPrompt: "You are a helpful assistant.",
});
```

Required environment variables:

```bash
DEEPSLATE_VENDOR_ID=your_vendor_id
DEEPSLATE_ORGANIZATION_ID=your_organization_id
DEEPSLATE_API_KEY=your_api_key
```

### TTS Configuration

Two TTS providers are supported. Pass either config as `ttsConfig` to `DeepslateSession.create()`.

**Deepslate-hosted (cloned) voice** — no external provider credentials required:

```ts
import { HostedTtsMode, type HostedTtsConfig } from "@deepslate-labs/core";

const tts: HostedTtsConfig = {
  provider: "hosted",
  voiceId: "c3dfa73f-a1ab-4aad-b48a-0e9b9fe4a69f",
  mode: HostedTtsMode.HIGH_QUALITY, // or HostedTtsMode.LOW_LATENCY
};
```

**ElevenLabs:**

```ts
import { elevenLabsConfigFromEnv, ElevenLabsLocation } from "@deepslate-labs/core";

// Load from environment variables
const tts = elevenLabsConfigFromEnv();

// Or configure manually
const tts = {
  provider: "eleven_labs" as const,
  apiKey: "your_elevenlabs_key",
  voiceId: "21m00Tcm4TlvDq8ikWAM",
  modelId: "eleven_turbo_v2",
  location: ElevenLabsLocation.US, // US (default), EU, or INDIA
};
```

### `DeepslateSession`

`DeepslateSession` is the recommended entry point for custom integrations. It handles the full protocol
lifecycle — session initialization, protobuf serialization, reconnection, and server-event routing —
emitting strongly-typed events so your code only deals with application logic.

Use `DeepslateSession.create()` when your code owns the connection: it creates its own
`BaseDeepslateClient` and closes it automatically when `close()` is called.

```ts
import { DeepslateSession, optionsFromEnv, elevenLabsConfigFromEnv } from "@deepslate-labs/core";

const session = DeepslateSession.create(
  optionsFromEnv({ systemPrompt: "You are a helpful assistant." }),
  { ttsConfig: elevenLabsConfigFromEnv(), userAgent: "MyApp/1.0" },
);

session.on("sessionInitialized", () => {
  // Session is ready — safe to send the first message
  void session.sendText("What is the capital of France?");
});

session.on("textFragment", (text) => process.stdout.write(text));

session.on("audioChunk", (pcm, sampleRate, channels, transcript) => {
  // Forward PCM audio to your output device / transport
});

session.on("toolCall", async (callId, name, params) => {
  const result = await dispatchTool(name, params);
  await session.sendToolResponse(callId, result);
});

session.on("error", (category, message, traceId) => {
  console.error(`[${category}] ${message}`);
});

session.start();

// Initialize for text-only interaction (audio sessions initialize automatically).
// "sessionInitialized" fires once the session is ready.
await session.initialize(24000, 1);

// ... later
await session.close();
```

#### Sending audio

Audio sessions initialize automatically on the first `sendAudio()` call. The session also reconfigures
transparently if the audio format changes mid-session:

```ts
// pcm: raw signed 16-bit PCM (Uint8Array)
await session.sendAudio(pcm, 16000, 1);
```

#### Updating tools at runtime

```ts
await session.updateTools([
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "Get the weather for a city.",
      parameters: {
        type: "object",
        properties: { city: { type: "string" } },
        required: ["city"],
      },
    },
  },
]);
```

Tool definitions are re-synced automatically after every reconnect.

#### Reconnection

`start()` drives `BaseDeepslateClient.runWithRetry()` internally. On a dropped connection the session
resets its state, re-sends the initialize request, re-syncs tool definitions, and resumes — all without
any action required from your code. Call `close()` to stop the retry loop permanently.

---

## API Reference

### `DeepslateSession.create(options, opts?)`

Factory that creates a session together with its own `BaseDeepslateClient`. The session owns the client
and closes it when `close()` is called.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `options` | `DeepslateOptions` | required | API credentials and settings |
| `opts.vadConfig` | `VadConfig` | defaults | VAD settings |
| `opts.ttsConfig` | `TtsConfig` | `undefined` | Enables server-side TTS audio output |
| `opts.userAgent` | `string` | `"DeepslateCore"` | `User-Agent` header sent on connect |

### `DeepslateSession` — send methods

| Method | Description |
|---|---|
| `session.sendAudio(pcm, sr, ch)` | Send PCM audio; auto-initializes and auto-reconfigures on format change |
| `session.sendText(text, trigger?)` | Send a text `UserInput` (default trigger: `IMMEDIATE`) |
| `session.initialize(sr?, ch?)` | Explicitly initialize (for text-only sessions) |
| `session.triggerInference(instructions?)` | Manually trigger a model reply |
| `session.sendToolResponse(callId, result)` | Return a tool result to the server |
| `session.updateTools(tools)` | Sync tool definitions (persisted across reconnects) |
| `session.reconfigure(systemPrompt?, temperature?)` | Live-update inference settings |
| `session.sendDirectSpeech(text, includeInHistory?, uninterruptable?)` | Speak text directly via TTS, bypassing the LLM |
| `session.exportChatHistory(awaitPending?, excludeAudio?)` | Request a history export; result via the `chatHistory` event |
| `session.sendConversationQuery(queryId, prompt?, instructions?)` | Side-channel inference; result via the `conversationQueryResult` event |

### `DeepslateSession` — events

Subscribe with `session.on(event, listener)`. Event payloads are strongly typed via `DeepslateSessionEvents`.

| Event | Listener signature | Emitted when |
|---|---|---|
| `sessionInitialized` | `()` | Session is fully initialized and ready to accept messages |
| `textFragment` | `(text: string)` | Model streams a text token |
| `audioChunk` | `(pcm: Uint8Array, sampleRate: number, channels: number, transcript: string \| null)` | Model streams a TTS audio chunk |
| `toolCall` | `(callId: string, name: string, params: Record<string, unknown>)` | Model requests a tool invocation |
| `responseBegin` | `()` | Model response starts |
| `responseEnd` | `()` | Model response ends |
| `userTranscription` | `(text: string, language: string \| null, turnId: number)` | User speech transcription arrives |
| `playbackBufferClear` | `()` | Server cleared its audio playback buffer |
| `chatHistory` | `(messages: ChatMessage[])` | Chat history export received |
| `conversationQueryResult` | `(queryId: string, text: string)` | Side-channel query result received |
| `error` | `(category: string, message: string, traceId: string \| null)` | Server sent an error notification |
| `fatalError` | `(err: Error)` | All reconnect retries exhausted |

### `DeepslateOptions`

| Field | Type | Default | Description |
|---|---|---|---|
| `vendorId` | `string` | env: `DEEPSLATE_VENDOR_ID` | Deepslate vendor ID |
| `organizationId` | `string` | env: `DEEPSLATE_ORGANIZATION_ID` | Deepslate organization ID |
| `apiKey` | `string` | env: `DEEPSLATE_API_KEY` | Deepslate API key |
| `baseUrl` | `string` | `"https://app.deepslate.eu"` | Base URL for Deepslate API |
| `systemPrompt` | `string` | `"You are a helpful assistant."` | Default system prompt |
| `temperature` | `number` | `1.0` | Sampling temperature (0.0–2.0) |
| `wsUrl` | `string` | `undefined` | Direct WebSocket URL (overrides `baseUrl`; for local dev) |
| `maxRetries` | `number` | `3` | Maximum reconnection attempts before giving up |
| `generateReplyTimeout` | `number` | `30.0` | Timeout in seconds for reply generation (0 = no timeout) |

### `VadConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `confidenceThreshold` | `number` | `0.5` | Minimum confidence to classify audio as speech (0–1) |
| `minVolume` | `number` | `0.01` | Minimum volume to classify audio as speech (0–1) |
| `startDurationMs` | `number` | `200` | Duration of speech required to trigger start event |
| `stopDurationMs` | `number` | `500` | Duration of silence required to trigger stop event |
| `backbufferDurationMs` | `number` | `1000` | Audio buffer captured before speech detection triggers |

---

## Documentation

- [Deepslate Documentation](https://docs.deepslate.eu/)
- [API Reference](https://docs.deepslate.eu/api-reference/)

---

## License

Apache License 2.0 — see [LICENSE](../../../LICENSE) for details.