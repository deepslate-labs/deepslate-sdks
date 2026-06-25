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

// Public API for @deepslate/livekit.
export {
  RealtimeModel,
  DeepslateRealtimeSession,
  type RealtimeModelOptions,
} from "./realtime/model.js";

// Re-export the core config types callers need to construct a model.
export {
  TriggerMode,
  ElevenLabsLocation,
  HostedTtsMode,
  elevenLabsConfigFromEnv,
} from "@deepslate/core";
export type {
  DeepslateOptions,
  VadConfig,
  TtsConfig,
  ElevenLabsTtsConfig,
  HostedTtsConfig,
} from "@deepslate/core";