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

// Public API surface for @deepslate-labs/core.

export { BaseDeepslateClient, RetriableError } from "./client.js";
export type { RunWithRetryHandlers } from "./client.js";

export {
  DeepslateSession,
  type DeepslateSessionCreateOptions,
} from "./session.js";

export {
  TriggerMode,
  ElevenLabsLocation,
  HostedTtsMode,
  resolveOptions,
  optionsFromEnv,
  elevenLabsConfigFromEnv,
  DEEPSLATE_DEFAULTS,
  VAD_DEFAULTS,
} from "./options.js";
export type {
  DeepslateOptions,
  ResolvedDeepslateOptions,
  VadConfig,
  ElevenLabsVoiceSettings,
  ElevenLabsTtsConfig,
  HostedTtsConfig,
  TtsConfig,
} from "./options.js";

export type {
  FunctionTool,
  FunctionDefinition,
  TtsAudio,
  TextContent,
  InputAudioContent,
  ToolCallContent,
  ToolResultContent,
  ThoughtsContent,
  InstructionsContent,
  ContentBlock,
  ChatMessage,
} from "./types.js";

export { TypedEventEmitter } from "./events.js";
export type { DeepslateSessionEvents } from "./events.js";

export { setLogger, consoleLogger } from "./log.js";
export type { Logger } from "./log.js";

export {
  buildWsUrl,
  durationFromMs,
  parseChatHistory,
  buildInitializeRequest,
  TRIGGER_MODE_MAP,
} from "./utils.js";