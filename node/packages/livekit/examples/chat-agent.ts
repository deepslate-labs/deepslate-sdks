// Copyright 2026 Deepslate
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * Deepslate + LiveKit Agents (Node) — Chat Agent Example
 * ======================================================
 * A voice AI assistant that joins a LiveKit room and responds to speech.
 * Includes two example function tools (weather lookup and location detection)
 * to demonstrate Deepslate's function-calling support.
 *
 * The TypeScript port of python/packages/livekit/examples/chat_agent.py.
 *
 * Setup
 * -----
 *   1. Copy .env.example to .env and fill in your credentials.
 *   2. Start a local LiveKit server (or point LIVEKIT_URL at a hosted one).
 *   3. Run:  pnpm tsx examples/chat-agent.ts dev
 */
import { fileURLToPath } from "node:url";

import {
  type JobContext,
  ServerOptions,
  cli,
  defineAgent,
  llm,
  voice,
} from "@livekit/agents";
import { config as loadEnv } from "dotenv";
import { z } from "zod";

import { RealtimeModel, elevenLabsConfigFromEnv } from "@deepslate-labs/livekit";

loadEnv();

// ---------------------------------------------------------------------------
// Function tools
// ---------------------------------------------------------------------------

const lookupWeather = llm.tool({
  description: "Get the current weather for a given location.",
  parameters: z.object({
    location: z.string().describe("The location to look up the weather for."),
  }),
  execute: async ({ location }) => ({
    location,
    temperatureCelsius: Math.floor(Math.random() * 26) + 10,
    precipitation: ["none", "light", "moderate", "heavy"][
      Math.floor(Math.random() * 4)
    ],
    airPressureHpa: Math.floor(Math.random() * 201) + 900,
  }),
});

const getCurrentLocation = llm.tool({
  description: "Get the user's current location.",
  execute: async () => "Berlin",
});

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

export default defineAgent({
  entry: async (ctx: JobContext) => {
    await ctx.connect();

    const session = new voice.AgentSession({
      llm: new RealtimeModel({
        // DEEPSLATE_WS_URL can override the default endpoint (local dev/testing).
        wsUrl: process.env.DEEPSLATE_WS_URL,
        // Enable server-side TTS only when ElevenLabs credentials are present.
        ttsConfig: process.env.ELEVENLABS_API_KEY
          ? elevenLabsConfigFromEnv()
          : undefined,
      }),
    });

    await session.start({
      agent: new voice.Agent({
        instructions: "You are a helpful voice AI assistant.",
        tools: { lookupWeather, getCurrentLocation },
      }),
      room: ctx.room,
    });

    session.generateReply({
      instructions: "Greet the user and offer your assistance.",
    });
  },
});

cli.runApp(new ServerOptions({ agent: fileURLToPath(import.meta.url) }));
