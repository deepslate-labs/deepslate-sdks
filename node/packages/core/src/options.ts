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

// Core Deepslate connection and model options.

/** Inference trigger mode for user input. */
export enum TriggerMode {
  NO_TRIGGER = "no_trigger",
  QUEUE = "queue",
  IMMEDIATE = "immediate",
}

/** ElevenLabs API endpoint region.
 *  See https://elevenlabs.io/docs/overview/administration/data-residency */
export enum ElevenLabsLocation {
  US = "US",
  EU = "EU",
  INDIA = "INDIA",
}

/** Quality/latency mode for Deepslate-hosted TTS synthesis. */
export enum HostedTtsMode {
  HIGH_QUALITY = "HIGH_QUALITY",
  LOW_LATENCY = "LOW_LATENCY",
}

/** Core Deepslate connection and model options (as provided by the caller). */
export interface DeepslateOptions {
  vendorId: string;
  organizationId: string;
  apiKey: string;
  /** Base URL for the Deepslate API. Default: https://app.deepslate.eu */
  baseUrl?: string;
  /** System prompt dictating model behavior. */
  systemPrompt?: string;
  /** Sampling temperature (0.0 to 2.0). */
  temperature?: number;
  /** Optional direct WebSocket URL (bypasses standard auth URL construction). */
  wsUrl?: string;
  /** Maximum number of reconnection attempts before giving up. */
  maxRetries?: number;
  /** Timeout in seconds for generate_reply (0 = no timeout). */
  generateReplyTimeout?: number;
}

/** Fully-populated options with all defaults applied. */
export interface ResolvedDeepslateOptions {
  vendorId: string;
  organizationId: string;
  apiKey: string;
  baseUrl: string;
  systemPrompt: string;
  temperature: number;
  wsUrl?: string;
  maxRetries: number;
  generateReplyTimeout: number;
}

export const DEEPSLATE_DEFAULTS = {
  baseUrl: "https://app.deepslate.eu",
  systemPrompt: "You are a helpful assistant.",
  temperature: 1.0,
  maxRetries: 3,
  generateReplyTimeout: 30.0,
} as const;

/** Apply defaults to a partially-specified options object. */
export function resolveOptions(opts: DeepslateOptions): ResolvedDeepslateOptions {
  return {
    vendorId: opts.vendorId,
    organizationId: opts.organizationId,
    apiKey: opts.apiKey,
    baseUrl: opts.baseUrl ?? DEEPSLATE_DEFAULTS.baseUrl,
    systemPrompt: opts.systemPrompt ?? DEEPSLATE_DEFAULTS.systemPrompt,
    temperature: opts.temperature ?? DEEPSLATE_DEFAULTS.temperature,
    wsUrl: opts.wsUrl,
    maxRetries: opts.maxRetries ?? DEEPSLATE_DEFAULTS.maxRetries,
    generateReplyTimeout:
      opts.generateReplyTimeout ?? DEEPSLATE_DEFAULTS.generateReplyTimeout,
  };
}

/** Build options from explicit values, falling back to DEEPSLATE_* env vars. */
export function optionsFromEnv(
  overrides: Partial<DeepslateOptions> = {},
): ResolvedDeepslateOptions {
  const vendorId = overrides.vendorId ?? process.env.DEEPSLATE_VENDOR_ID;
  if (!vendorId) {
    throw new Error(
      "Deepslate vendor ID required. Provide vendorId or set DEEPSLATE_VENDOR_ID.",
    );
  }
  const organizationId =
    overrides.organizationId ?? process.env.DEEPSLATE_ORGANIZATION_ID;
  if (!organizationId) {
    throw new Error(
      "Deepslate organization ID required. Provide organizationId or set DEEPSLATE_ORGANIZATION_ID.",
    );
  }
  const apiKey = overrides.apiKey ?? process.env.DEEPSLATE_API_KEY;
  if (!apiKey) {
    throw new Error(
      "Deepslate API key required. Provide apiKey or set DEEPSLATE_API_KEY.",
    );
  }
  return resolveOptions({ ...overrides, vendorId, organizationId, apiKey });
}

/** Voice Activity Detection configuration handled server-side by Deepslate. */
export interface VadConfig {
  confidenceThreshold?: number;
  minVolume?: number;
  startDurationMs?: number;
  stopDurationMs?: number;
  backbufferDurationMs?: number;
}

export const VAD_DEFAULTS = {
  confidenceThreshold: 0.5,
  minVolume: 0.01,
  startDurationMs: 200,
  stopDurationMs: 500,
  backbufferDurationMs: 1000,
} as const;

/** ElevenLabs voice settings for fine-grained TTS control. */
export interface ElevenLabsVoiceSettings {
  stability?: number;
  similarityBoost?: number;
  style?: number;
  useSpeakerBoost?: boolean;
  speed?: number;
}

/** ElevenLabs TTS configuration for Deepslate-hosted TTS. */
export interface ElevenLabsTtsConfig {
  provider: "eleven_labs";
  apiKey: string;
  voiceId: string;
  modelId?: string;
  location?: ElevenLabsLocation;
  voiceSettings?: ElevenLabsVoiceSettings;
}

/** Deepslate-hosted TTS configuration using a cloned voice. */
export interface HostedTtsConfig {
  provider: "hosted";
  voiceId: string;
  mode?: HostedTtsMode;
}

export type TtsConfig = ElevenLabsTtsConfig | HostedTtsConfig;

/** Build an ElevenLabs TTS config from explicit values or ELEVENLABS_* env vars. */
export function elevenLabsConfigFromEnv(
  overrides: Partial<Omit<ElevenLabsTtsConfig, "provider">> = {},
): ElevenLabsTtsConfig {
  const apiKey = overrides.apiKey ?? process.env.ELEVENLABS_API_KEY;
  if (!apiKey) {
    throw new Error(
      "ElevenLabs API key required. Provide apiKey or set ELEVENLABS_API_KEY.",
    );
  }
  const voiceId = overrides.voiceId ?? process.env.ELEVENLABS_VOICE_ID;
  if (!voiceId) {
    throw new Error(
      "ElevenLabs voice ID required. Provide voiceId or set ELEVENLABS_VOICE_ID.",
    );
  }
  return {
    provider: "eleven_labs",
    apiKey,
    voiceId,
    modelId: overrides.modelId ?? process.env.ELEVENLABS_MODEL_ID,
    location: overrides.location ?? ElevenLabsLocation.US,
    voiceSettings: overrides.voiceSettings,
  };
}