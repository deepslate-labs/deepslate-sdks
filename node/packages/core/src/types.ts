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

// Plain serializable shapes surfaced to callers. These intentionally avoid any
// generated protobuf types so consumers never touch @deepslate-labs/proto.

/** A tool the model may call. */
export interface FunctionDefinition {
  name: string;
  description?: string;
  parameters?: Record<string, unknown>;
}

export interface FunctionTool {
  type: "function";
  function: FunctionDefinition;
}

/** TTS-synthesized audio attached to a text content block. */
export interface TtsAudio {
  audio: Uint8Array;
  transcription: string;
}

export interface TextContent {
  type: "text";
  text: string;
  ttsAudio: TtsAudio | null;
}

export interface InputAudioContent {
  type: "input_audio";
  audio: Uint8Array;
  transcription: string;
}

export interface ToolCallContent {
  type: "tool_call";
  id: string;
  name: string;
  parameters: Record<string, unknown>;
}

export interface ToolResultContent {
  type: "tool_result";
  id: string;
  result: string;
}

export interface ThoughtsContent {
  type: "thoughts";
  text: string;
}

export interface InstructionsContent {
  type: "instructions";
  text: string;
}

export type ContentBlock =
  | TextContent
  | InputAudioContent
  | ToolCallContent
  | ToolResultContent
  | ThoughtsContent
  | InstructionsContent;

export interface ChatMessage {
  role: string;
  deliveryStatus: string;
  ephemeral: boolean;
  content: ContentBlock[];
}