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

// Protobuf construction/parsing helpers.
//
// Note: protobuf-es v2 represents google.protobuf.Struct fields directly as
// plain JsonObject values, so Struct conversion helpers are unnecessary —
// objects pass through unchanged.
import { create } from "@bufbuild/protobuf";
import {
  type ChatHistory,
  type InitializeSessionRequest,
  InitializeSessionRequestSchema,
  ChatMessageRole,
  ChatDeliveryStatus,
  ElevenLabsLocation as ProtoElevenLabsLocation,
  HostedTtsMode as ProtoHostedTtsMode,
  InferenceTriggerMode,
  SampleFormat,
} from "@deepslate/proto";

import {
  type ResolvedDeepslateOptions,
  type TtsConfig,
  type VadConfig,
  ElevenLabsLocation,
  HostedTtsMode,
  TriggerMode,
  VAD_DEFAULTS,
} from "./options.js";
import type { ChatMessage, ContentBlock } from "./types.js";

/** Map the public string TriggerMode to the numeric proto enum. */
export const TRIGGER_MODE_MAP: Record<TriggerMode, InferenceTriggerMode> = {
  [TriggerMode.NO_TRIGGER]: InferenceTriggerMode.NO_TRIGGER,
  [TriggerMode.QUEUE]: InferenceTriggerMode.QUEUE,
  [TriggerMode.IMMEDIATE]: InferenceTriggerMode.IMMEDIATE,
};

const ELEVENLABS_LOCATION_MAP: Record<ElevenLabsLocation, ProtoElevenLabsLocation> =
  {
    [ElevenLabsLocation.US]: ProtoElevenLabsLocation.US,
    [ElevenLabsLocation.EU]: ProtoElevenLabsLocation.EU,
    [ElevenLabsLocation.INDIA]: ProtoElevenLabsLocation.INDIA,
  };

const HOSTED_TTS_MODE_MAP: Record<HostedTtsMode, ProtoHostedTtsMode> = {
  [HostedTtsMode.HIGH_QUALITY]: ProtoHostedTtsMode.HIGH_QUALITY,
  [HostedTtsMode.LOW_LATENCY]: ProtoHostedTtsMode.LOW_LATENCY,
};

/** Convert milliseconds to a proto Duration init ({ seconds: bigint, nanos }). */
export function durationFromMs(ms: number): { seconds: bigint; nanos: number } {
  return {
    seconds: BigInt(Math.floor(ms / 1000)),
    nanos: (ms % 1000) * 1_000_000,
  };
}

/**
 * Build the WebSocket URL for the Deepslate realtime endpoint.
 * https → wss, http → ws; preserves host and port.
 */
export function buildWsUrl(
  baseUrl: string,
  vendorId: string,
  organizationId: string,
): string {
  const parsed = new URL(baseUrl);
  let scheme: string;
  if (parsed.protocol === "https:") scheme = "wss:";
  else if (parsed.protocol === "http:") scheme = "ws:";
  else scheme = parsed.protocol;
  return `${scheme}//${parsed.host}/api/v1/vendors/${vendorId}/organizations/${organizationId}/realtime`;
}

/** Build an InitializeSessionRequest from core configuration objects. */
export function buildInitializeRequest(params: {
  sampleRate: number;
  numChannels: number;
  vadConfig: VadConfig;
  systemPrompt: string;
  ttsConfig?: TtsConfig;
  temperature?: number;
}): InitializeSessionRequest {
  const audioLine = {
    sampleRate: params.sampleRate,
    channelCount: params.numChannels,
    sampleFormat: SampleFormat.SIGNED_16_BIT,
  };

  const vad = params.vadConfig;
  const ttsConfiguration = buildTtsConfiguration(params.ttsConfig);

  return create(InitializeSessionRequestSchema, {
    inputAudioLine: audioLine,
    outputAudioLine: audioLine,
    vadConfiguration: {
      confidenceThreshold:
        vad.confidenceThreshold ?? VAD_DEFAULTS.confidenceThreshold,
      minVolume: vad.minVolume ?? VAD_DEFAULTS.minVolume,
      startDuration: durationFromMs(
        vad.startDurationMs ?? VAD_DEFAULTS.startDurationMs,
      ),
      stopDuration: durationFromMs(
        vad.stopDurationMs ?? VAD_DEFAULTS.stopDurationMs,
      ),
      backbufferDuration: durationFromMs(
        vad.backbufferDurationMs ?? VAD_DEFAULTS.backbufferDurationMs,
      ),
    },
    inferenceConfiguration: {
      systemPrompt: params.systemPrompt,
      temperature: params.temperature ?? 1.0,
    },
    ttsConfiguration,
  });
}

/** Build the InitializeSessionRequest from resolved options + configs. */
export function buildInitializeRequestFromOptions(
  options: ResolvedDeepslateOptions,
  vadConfig: VadConfig,
  sampleRate: number,
  numChannels: number,
  ttsConfig?: TtsConfig,
): InitializeSessionRequest {
  return buildInitializeRequest({
    sampleRate,
    numChannels,
    vadConfig,
    systemPrompt: options.systemPrompt,
    ttsConfig,
    temperature: options.temperature,
  });
}

type TtsConfigurationInit = NonNullable<
  Parameters<typeof create<typeof InitializeSessionRequestSchema>>[1]
>["ttsConfiguration"];

function buildTtsConfiguration(ttsConfig?: TtsConfig): TtsConfigurationInit {
  if (!ttsConfig) return undefined;

  if (ttsConfig.provider === "eleven_labs") {
    const voiceSettings = ttsConfig.voiceSettings
      ? {
          stability: ttsConfig.voiceSettings.stability ?? 0,
          similarityBoost: ttsConfig.voiceSettings.similarityBoost ?? 0,
          style: ttsConfig.voiceSettings.style ?? 0,
          useSpeakerBoost: ttsConfig.voiceSettings.useSpeakerBoost ?? false,
          speed: ttsConfig.voiceSettings.speed ?? 0,
        }
      : undefined;
    return {
      provider: {
        case: "elevenLabs",
        value: {
          apiKey: ttsConfig.apiKey,
          voiceId: ttsConfig.voiceId,
          modelId: ttsConfig.modelId,
          location:
            ELEVENLABS_LOCATION_MAP[ttsConfig.location ?? ElevenLabsLocation.US],
          voiceSettings,
        },
      },
    };
  }

  // hosted
  return {
    provider: {
      case: "hosted",
      value: {
        voice: {
          case: "voiceRef",
          value: { voiceId: ttsConfig.voiceId },
        },
        mode: HOSTED_TTS_MODE_MAP[ttsConfig.mode ?? HostedTtsMode.HIGH_QUALITY],
      },
    },
  };
}

/** Convert a proto ChatHistory into plain ChatMessage objects. */
export function parseChatHistory(chatHistory: ChatHistory): ChatMessage[] {
  const messages: ChatMessage[] = [];

  for (const msg of chatHistory.messages) {
    const content: ContentBlock[] = [];

    for (const block of msg.content) {
      switch (block.content.case) {
        case "textContent": {
          const tc = block.content.value;
          content.push({
            type: "text",
            text: tc.text,
            ttsAudio: tc.ttsAudio
              ? {
                  audio: tc.ttsAudio.audio?.data ?? new Uint8Array(),
                  transcription: tc.ttsAudio.transcription,
                }
              : null,
          });
          break;
        }
        case "inputAudio": {
          const ia = block.content.value;
          content.push({
            type: "input_audio",
            audio: ia.audio?.data ?? new Uint8Array(),
            transcription: ia.transcription,
          });
          break;
        }
        case "toolCall": {
          const tc = block.content.value;
          content.push({
            type: "tool_call",
            id: tc.id,
            name: tc.name,
            parameters: (tc.parameters as Record<string, unknown>) ?? {},
          });
          break;
        }
        case "toolResult": {
          const tr = block.content.value;
          content.push({ type: "tool_result", id: tr.id, result: tr.result });
          break;
        }
        case "thoughts":
          content.push({ type: "thoughts", text: block.content.value });
          break;
        case "instructions":
          content.push({ type: "instructions", text: block.content.value });
          break;
        default:
          break;
      }
    }

    const roleName = ChatMessageRole[msg.role] ?? "UNKNOWN";
    const deliveryName = ChatDeliveryStatus[msg.deliveryStatus] ?? "UNKNOWN";
    messages.push({
      role: roleName.toLowerCase(),
      deliveryStatus: deliveryName,
      ephemeral: msg.ephemeral,
      content,
    });
  }

  return messages;
}