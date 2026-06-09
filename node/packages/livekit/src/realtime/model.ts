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

// Deepslate realtime model for @livekit/agents (Node).
//
// RealtimeModel resolves core options; DeepslateRealtimeSession subscribes to the
// core DeepslateSession events and translates each into the framework's
// generation/stream primitives. All protobuf details stay in @deepslate/core.
import { AudioFrame } from "@livekit/rtc-node";
// The realtime API is exposed under the `llm` namespace of @livekit/agents, so
// we re-bind the names this module uses. (Values are destructured; the
// type-only members are aliased.)
import { llm } from "@livekit/agents";

const { ChatContext, FunctionCall, isFunctionTool, toJsonSchema } = llm;
type ChatContext = llm.ChatContext;
type FunctionCall = llm.FunctionCall;
type GenerationCreatedEvent = llm.GenerationCreatedEvent;
type MessageGeneration = llm.MessageGeneration;
type RealtimeCapabilities = llm.RealtimeCapabilities;
type RealtimeModelError = llm.RealtimeModelError;
type ToolChoice = llm.ToolChoice;
type ToolContext = llm.ToolContext;
import {
  DeepslateSession,
  TriggerMode,
  optionsFromEnv,
  type FunctionTool as DeepslateFunctionTool,
  type ResolvedDeepslateOptions,
  type TtsConfig,
  type VadConfig,
} from "@deepslate/core";

import { logger } from "../log.js";
import { createPushable, type Pushable } from "../stream.js";

const DEEPSLATE_BASE_URL = "https://app.deepslate.eu";

export interface RealtimeModelOptions {
  vendorId?: string;
  organizationId?: string;
  apiKey?: string;
  baseUrl?: string;
  systemPrompt?: string;
  temperature?: number;
  generateReplyTimeout?: number;
  vad?: VadConfig;
  ttsConfig?: TtsConfig;
  wsUrl?: string;
}

/** Internal state for an in-flight response generation. */
interface ResponseGeneration {
  messageStream: Pushable<MessageGeneration>;
  functionStream: Pushable<FunctionCall>;
  // We only ever emit plain strings; ReadableStream<string> is assignable to
  // the framework's ReadableStream<string | TimedString> sink.
  textStream: Pushable<string>;
  audioStream: Pushable<AudioFrame>;
  responseId: string;
  firstTokenAt?: number;
}

let responseCounter = 0;
const shortId = (prefix: string): string => `${prefix}${(responseCounter++).toString(36)}_${Date.now().toString(36)}`;

export class RealtimeModel extends llm.RealtimeModel {
  /** Resolved core options; shared with each session. */
  readonly opts: ResolvedDeepslateOptions;
  readonly vad?: VadConfig;
  readonly ttsConfig?: TtsConfig;

  constructor(options: RealtimeModelOptions = {}) {
    const audioOutput = options.ttsConfig != null;
    const capabilities: RealtimeCapabilities = {
      messageTruncation: true,
      turnDetection: true,
      userTranscription: true,
      // Deepslate auto-generates a reply after a ToolCallResponse, so the
      // framework must NOT also call generateReply() (that would send a
      // duplicate TriggerInference).
      autoToolReplyGeneration: false,
      audioOutput,
      manualFunctionCalls: false,
      perResponseToolChoice: false,
    };
    super(capabilities);

    this.opts = optionsFromEnv({
      vendorId: options.vendorId,
      organizationId: options.organizationId,
      apiKey: options.apiKey,
      baseUrl: options.baseUrl ?? DEEPSLATE_BASE_URL,
      systemPrompt: options.systemPrompt,
      temperature: options.temperature,
      generateReplyTimeout: options.generateReplyTimeout,
      wsUrl: options.wsUrl,
    });
    this.vad = options.vad;
    this.ttsConfig = options.ttsConfig;
  }

  get model(): string {
    return "deepslate-realtime";
  }

  override get provider(): string {
    return "deepslate";
  }

  session(): DeepslateRealtimeSession {
    return new DeepslateRealtimeSession(this);
  }

  async close(): Promise<void> {
    // No shared resources to release; sessions own their core session.
  }
}

export class DeepslateRealtimeSession extends llm.RealtimeSession {
  private readonly model: RealtimeModel;
  private readonly session: DeepslateSession;

  private _chatCtx: ChatContext = ChatContext.empty();
  private _tools: ToolContext = {};
  private toolsDicts: DeepslateFunctionTool[] = [];
  private toolChoice: ToolChoice | null = null;

  private currentGeneration: ResponseGeneration | null = null;
  private pendingUserText: string | null = null;
  private pendingUserGeneration = false;
  private replyWaiters: Array<(ev: GenerationCreatedEvent) => void> = [];

  private audioChain: Promise<unknown> = Promise.resolve();

  // queryId → settlers for in-flight side-channel conversation queries.
  private readonly pendingQueries = new Map<
    string,
    { resolve: (text: string) => void; reject: (err: Error) => void }
  >();

  constructor(model: RealtimeModel) {
    super(model);
    this.model = model;

    this.session = DeepslateSession.create(
      {
        vendorId: model.opts.vendorId,
        organizationId: model.opts.organizationId,
        apiKey: model.opts.apiKey,
        baseUrl: model.opts.baseUrl,
        systemPrompt: model.opts.systemPrompt,
        temperature: model.opts.temperature,
        wsUrl: model.opts.wsUrl,
        maxRetries: model.opts.maxRetries,
        generateReplyTimeout: model.opts.generateReplyTimeout,
      },
      {
        vadConfig: model.vad,
        ttsConfig: model.ttsConfig,
        userAgent: "DeepslateLiveKit",
      },
    );

    this.wireCoreEvents();
    this.session.start();
  }

  get chatCtx(): ChatContext {
    return this._chatCtx.copy();
  }

  get tools(): ToolContext {
    return { ...this._tools };
  }

  async updateInstructions(instructions: string): Promise<void> {
    this.model.opts.systemPrompt = instructions;
    await this.session.reconfigure(instructions);
  }

  async updateChatCtx(chatCtx: ChatContext): Promise<void> {
    const existing = new Set(this._chatCtx.items.map((i) => i.id));
    for (const item of chatCtx.items) {
      if (existing.has(item.id)) continue;
      if (item.type === "message" && item.role === "user") {
        const text = item.textContent;
        if (text) this.pendingUserText = text;
      } else if (item.type === "function_call_output") {
        await this.session.sendToolResponse(item.callId, item.output);
      }
    }
    this._chatCtx = chatCtx.copy();
  }

  async updateTools(tools: ToolContext): Promise<void> {
    const dicts: DeepslateFunctionTool[] = [];
    for (const [name, tool] of Object.entries(tools)) {
      if (!isFunctionTool(tool)) continue;
      const schema = toJsonSchema(tool);
      dicts.push({
        type: "function",
        function: {
          name: schema.name ?? name,
          description: schema.description ?? "",
          parameters: schema.parameters ?? {},
        },
      });
    }
    this.toolsDicts = dicts;
    this._tools = { ...tools };
    await this.syncToolChoice();
    logger.debug("updated tools:", dicts.map((d) => d.function.name));
  }

  updateOptions(options: { toolChoice?: ToolChoice | null }): void {
    if (options.toolChoice !== undefined) {
      this.toolChoice = options.toolChoice;
      void this.syncToolChoice();
    }
  }

  pushAudio(frame: AudioFrame): void {
    // Copy out of the frame's backing buffer — it may be reused after return.
    const pcm = new Uint8Array(
      frame.data.buffer.slice(
        frame.data.byteOffset,
        frame.data.byteOffset + frame.data.byteLength,
      ),
    );
    this.audioChain = this.audioChain
      .then(() => this.session.sendAudio(pcm, frame.sampleRate, frame.channels))
      .catch((err) => logger.error("audio dispatch failed:", err));
  }

  async generateReply(
    instructions?: string,
    options?: { signal?: AbortSignal },
  ): Promise<GenerationCreatedEvent> {
    this.pendingUserGeneration = true;
    const promise = new Promise<GenerationCreatedEvent>((resolve, reject) => {
      this.replyWaiters.push(resolve);
      const timeout = this.model.opts.generateReplyTimeout;
      if (timeout > 0) {
        const t = setTimeout(() => {
          this.removeWaiter(resolve);
          reject(new Error(`generateReply timed out after ${timeout}s`));
        }, timeout * 1000);
        if (typeof t === "object" && "unref" in t) t.unref();
      }
      options?.signal?.addEventListener("abort", () => {
        this.removeWaiter(resolve);
        reject(new Error("generateReply aborted"));
      });
    });

    if (this.pendingUserText) {
      const text = this.pendingUserText;
      this.pendingUserText = null;
      if (instructions) {
        await this.session.sendText(text, TriggerMode.NO_TRIGGER);
        await this.session.triggerInference(instructions);
      } else {
        await this.session.initialize();
        await this.session.sendText(text, TriggerMode.IMMEDIATE);
      }
    } else {
      await this.session.initialize();
      await this.session.triggerInference(instructions);
    }

    return promise;
  }

  async commitAudio(): Promise<void> {
    // Deepslate uses server-side VAD for auto-commit.
  }

  async clearAudio(): Promise<void> {
    logger.warn("clearAudio not yet supported by the Deepslate backend");
  }

  async interrupt(): Promise<void> {
    this.closeCurrentGeneration();
  }

  async truncate(_options: {
    messageId: string;
    audioEndMs: number;
    modalities?: ("text" | "audio")[];
    audioTranscript?: string;
  }): Promise<void> {
    // Deepslate handles truncation server-side automatically.
  }

  override async close(): Promise<void> {
    this.closeCurrentGeneration();
    for (const waiter of this.pendingQueries.values()) {
      waiter.reject(new Error("session closed"));
    }
    this.pendingQueries.clear();
    await this.session.close();
    await super.close();
  }

  // ---- extra Deepslate-specific helpers ----

  async sendText(
    text: string,
    mode: TriggerMode = TriggerMode.NO_TRIGGER,
  ): Promise<void> {
    await this.session.initialize();
    await this.session.sendText(text, mode);
  }

  async speakDirect(text: string, includeInHistory = true): Promise<void> {
    await this.session.initialize();
    await this.session.sendDirectSpeech(text, includeInHistory);
  }

  /**
   * Run a one-shot side-channel inference against the current conversation.
   * Resolves with the model's complete text reply (via the
   * `conversationQueryResult` core event).
   */
  async queryConversation(prompt?: string, instructions?: string): Promise<string> {
    const queryId = shortId("query_");
    const result = new Promise<string>((resolve, reject) => {
      this.pendingQueries.set(queryId, { resolve, reject });
    });
    await this.session.initialize();
    await this.session.sendConversationQuery(queryId, prompt, instructions);
    return result;
  }

  async exportChatHistory(awaitPending = false, excludeAudio = false): Promise<void> {
    await this.session.exportChatHistory(awaitPending, excludeAudio);
  }

  // ---- core event wiring ----

  private wireCoreEvents(): void {
    this.session.on("textFragment", (text) => {
      const gen = this.ensureGeneration();
      gen.textStream.push(text);
      gen.firstTokenAt ??= Date.now();
    });

    this.session.on("audioChunk", (pcm, sampleRate, channels, transcript) => {
      const gen = this.ensureGeneration();
      const int16 = new Int16Array(
        pcm.buffer.slice(pcm.byteOffset, pcm.byteOffset + pcm.byteLength),
      );
      const samplesPerChannel = int16.length / Math.max(channels, 1);
      gen.audioStream.push(
        new AudioFrame(int16, sampleRate, channels, samplesPerChannel),
      );
      gen.firstTokenAt ??= Date.now();
      if (transcript) this.emit("audio_transcript", transcript);
    });

    this.session.on("toolCall", (callId, name, params) => {
      const gen = this.ensureGeneration();
      gen.functionStream.push(
        FunctionCall.create({ callId, name, args: JSON.stringify(params) }),
      );
      logger.debug(`tool call request: ${name}(${callId})`);
      this.closeCurrentGeneration();
    });

    this.session.on("responseBegin", () => {
      this.ensureGeneration();
    });

    this.session.on("responseEnd", () => {
      this.closeCurrentGeneration();
    });

    this.session.on("playbackBufferClear", () => {
      if (this.currentGeneration) {
        this.emit("input_speech_started", {});
        this.closeCurrentGeneration();
      }
    });

    this.session.on("userTranscription", (text) => {
      this.emit("input_audio_transcription_completed", {
        itemId: shortId("item_"),
        transcript: text,
        isFinal: true,
      });
    });

    this.session.on("error", (category, message, traceId) => {
      const suffix = traceId ? ` (trace_id=${traceId})` : "";
      this.emitError(
        new Error(`[Deepslate] ${category}: ${message}${suffix}`),
        false,
      );
    });

    this.session.on("fatalError", (err) => {
      this.emitError(err, false);
    });

    // ---- Deepslate-specific extras (not consumed by the agent loop) ----

    this.session.on("sessionInitialized", () => {
      this.emit("session_initialized", undefined);
    });

    this.session.on("chatHistory", (messages) => {
      this.emit("chat_history_exported", messages);
    });

    this.session.on("conversationQueryResult", (queryId, text) => {
      const waiter = this.pendingQueries.get(queryId);
      if (waiter) {
        this.pendingQueries.delete(queryId);
        waiter.resolve(text);
      } else {
        logger.warn(
          `received conversationQueryResult for unknown queryId: '${queryId}'`,
        );
      }
    });
  }

  private emitError(error: Error, recoverable: boolean): void {
    logger.error(error.message);
    const payload: RealtimeModelError = {
      type: "realtime_model_error",
      timestamp: Date.now(),
      label: this.model.label(),
      error,
      recoverable,
    };
    this.emit("error", payload);
  }

  private effectiveToolsDicts(): DeepslateFunctionTool[] {
    const tc = this.toolChoice;
    if (tc === "none") return [];
    if (tc && typeof tc === "object" && tc.type === "function") {
      const name = tc.function.name;
      return this.toolsDicts.filter((t) => t.function.name === name);
    }
    return this.toolsDicts;
  }

  private async syncToolChoice(): Promise<void> {
    await this.session.updateTools(this.effectiveToolsDicts());
  }

  private ensureGeneration(): ResponseGeneration {
    if (this.currentGeneration) return this.currentGeneration;
    return this.createGeneration();
  }

  private createGeneration(): ResponseGeneration {
    const userInitiated = this.pendingUserGeneration;
    this.pendingUserGeneration = false;

    const responseId = shortId("resp_");
    const hasAudio = this.model.ttsConfig != null;

    const gen: ResponseGeneration = {
      messageStream: createPushable<MessageGeneration>(),
      functionStream: createPushable<FunctionCall>(),
      textStream: createPushable<string>(),
      audioStream: createPushable<AudioFrame>(),
      responseId,
    };
    this.currentGeneration = gen;

    const modalities: ("text" | "audio")[] = hasAudio
      ? ["audio", "text"]
      : ["text"];
    if (!hasAudio) gen.audioStream.close();

    gen.messageStream.push({
      messageId: responseId,
      textStream: gen.textStream.stream,
      audioStream: gen.audioStream.stream,
      modalities: Promise.resolve(modalities),
    });

    const ev: GenerationCreatedEvent = {
      messageStream: gen.messageStream.stream,
      functionStream: gen.functionStream.stream,
      userInitiated,
      responseId,
    };

    this.emit("generation_created", ev);

    const waiters = this.replyWaiters;
    this.replyWaiters = [];
    for (const resolve of waiters) resolve(ev);

    return gen;
  }

  private closeCurrentGeneration(): void {
    const gen = this.currentGeneration;
    if (!gen) return;
    this.currentGeneration = null;
    gen.textStream.close();
    gen.audioStream.close();
    gen.functionStream.close();
    gen.messageStream.close();
  }

  private removeWaiter(resolve: (ev: GenerationCreatedEvent) => void): void {
    this.replyWaiters = this.replyWaiters.filter((w) => w !== resolve);
  }
}