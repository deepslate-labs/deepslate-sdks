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

// Typed EventEmitter for Deepslate session events.
import { EventEmitter } from "node:events";

import type { ChatMessage } from "./types.js";

// Declared as a `type` (not an interface) so it satisfies the
// `Record<string, (...args) => void>` constraint of TypedEventEmitter:
// type literals get an implicit index signature, interfaces do not.
export type DeepslateSessionEvents = {
  /** A fragment of model text streamed as tokens arrive. */
  textFragment: (text: string) => void;
  /** A chunk of model TTS audio (PCM), with optional alignment transcript. */
  audioChunk: (
    pcm: Uint8Array,
    sampleRate: number,
    channels: number,
    transcript: string | null,
  ) => void;
  /** The model requested a tool call. */
  toolCall: (callId: string, name: string, params: Record<string, unknown>) => void;
  /** A server-side error notification (connection stays as-is). */
  error: (category: string, message: string, traceId: string | null) => void;
  /** The model began its response. */
  responseBegin: () => void;
  /** The model finished its response. */
  responseEnd: () => void;
  /** A user audio turn was transcribed. */
  userTranscription: (
    text: string,
    language: string | null,
    turnId: number,
  ) => void;
  /** The playback buffer should be cleared (user started speaking). */
  playbackBufferClear: () => void;
  /** An exported chat history arrived. */
  chatHistory: (messages: ChatMessage[]) => void;
  /** A side-channel conversation query completed. */
  conversationQueryResult: (queryId: string, text: string) => void;
  /** The session is initialized and ready (server sent SessionReady). */
  sessionInitialized: () => void;
  /** Reconnection exhausted or an unexpected error ended the session. */
  fatalError: (err: Error) => void;
};

/* eslint-disable @typescript-eslint/no-explicit-any */
type EventMap = Record<string, (...args: any[]) => void>;

/** A minimal strongly-typed wrapper over Node's EventEmitter. */
export class TypedEventEmitter<TEvents extends EventMap> extends EventEmitter {
  override on<K extends keyof TEvents & string>(
    event: K,
    listener: TEvents[K],
  ): this {
    return super.on(event, listener as (...args: any[]) => void);
  }

  override once<K extends keyof TEvents & string>(
    event: K,
    listener: TEvents[K],
  ): this {
    return super.once(event, listener as (...args: any[]) => void);
  }

  override off<K extends keyof TEvents & string>(
    event: K,
    listener: TEvents[K],
  ): this {
    return super.off(event, listener as (...args: any[]) => void);
  }

  override emit<K extends keyof TEvents & string>(
    event: K,
    ...args: Parameters<TEvents[K]>
  ): boolean {
    return super.emit(event, ...args);
  }
}