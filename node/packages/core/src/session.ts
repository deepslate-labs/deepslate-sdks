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

// Manages a single logical Deepslate realtime connection. Server events are
// delivered through a typed EventEmitter (see events.ts). Messages sent before
// the session is ready are buffered and flushed on SessionReady.
import {
  create,
  fromBinary,
  toBinary,
  type JsonObject,
  type MessageInitShape,
} from "@bufbuild/protobuf";
import {
  type ClientBoundMessage,
  ClientBoundMessageSchema,
  type ServiceBoundMessage,
  ServiceBoundMessageSchema,
  SampleFormat,
  SessionErrorCategory,
} from "@deepslate/proto";
import WebSocket from "ws";

import { BaseDeepslateClient, RetriableError } from "./client.js";
import { DeepslateSessionEvents, TypedEventEmitter } from "./events.js";
import { logger } from "./log.js";
import {
  type DeepslateOptions,
  type ResolvedDeepslateOptions,
  type TtsConfig,
  type VadConfig,
  TriggerMode,
  resolveOptions,
} from "./options.js";
import type { FunctionTool } from "./types.js";
import {
  TRIGGER_MODE_MAP,
  buildInitializeRequestFromOptions,
  parseChatHistory,
} from "./utils.js";

type ServiceBoundInit = MessageInitShape<typeof ServiceBoundMessageSchema>;

export interface DeepslateSessionCreateOptions {
  vadConfig?: VadConfig;
  ttsConfig?: TtsConfig;
  userAgent?: string;
}

function rawDataToUint8(data: WebSocket.RawData): Uint8Array {
  if (Buffer.isBuffer(data)) {
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  }
  if (Array.isArray(data)) return new Uint8Array(Buffer.concat(data));
  return new Uint8Array(data as ArrayBuffer);
}

export class DeepslateSession extends TypedEventEmitter<DeepslateSessionEvents> {
  private ownsClient = false;
  private currentTools: FunctionTool[] = [];
  private shouldStop = false;
  private closing = false;

  private ws: WebSocket | null = null;
  private sessionInitializedFlag = false;
  private initRequestSent = false;
  private sampleRateValue: number | null = null;
  private channelsValue: number | null = null;
  private packetIdCounter = 0;
  private pendingBeforeInit: ServiceBoundMessage[] = [];
  private pendingQueryIds: string[] = [];

  private mainPromise: Promise<void> | undefined;

  constructor(
    private readonly client: BaseDeepslateClient,
    private readonly options: ResolvedDeepslateOptions,
    private readonly vadConfig: VadConfig = {},
    private readonly ttsConfig?: TtsConfig,
  ) {
    super();
  }

  /** Create a session together with its own BaseDeepslateClient. */
  static create(
    options: DeepslateOptions,
    opts: DeepslateSessionCreateOptions = {},
  ): DeepslateSession {
    const resolved = resolveOptions(options);
    const client = new BaseDeepslateClient(
      resolved,
      opts.userAgent ?? "DeepslateCore",
    );
    const session = new DeepslateSession(
      client,
      resolved,
      opts.vadConfig,
      opts.ttsConfig,
    );
    session.ownsClient = true;
    return session;
  }

  get sessionInitialized(): boolean {
    return this.sessionInitializedFlag;
  }

  get sampleRate(): number | null {
    return this.sampleRateValue;
  }

  get channels(): number | null {
    return this.channelsValue;
  }

  /** Spawn the background loop that drives client.runWithRetry(). Idempotent. */
  start(): void {
    if (this.mainPromise) return;
    this.shouldStop = false;
    this.closing = false;
    this.mainPromise = this.client.runWithRetry((ws) => this.runWs(ws), {
      shouldContinue: () => !this.shouldStop,
      onFatalError: (err) => this.fire("fatalError", err),
    });
  }

  /** Stop the session and (if owned) close the client. */
  async close(): Promise<void> {
    this.shouldStop = true;
    this.closing = true;
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.close();
    }
    if (this.mainPromise) {
      try {
        await this.mainPromise;
      } catch {
        // run loop errors are surfaced via the fatalError event
      }
    }
    this.mainPromise = undefined;
    if (this.ownsClient) await this.client.aclose();
  }

  // ---- public send API ----

  async sendAudio(
    pcm: Uint8Array,
    sampleRate: number,
    channels: number,
    trigger: TriggerMode = TriggerMode.IMMEDIATE,
  ): Promise<number> {
    this.ensureInitialized(sampleRate, channels);

    if (sampleRate !== this.sampleRateValue || channels !== this.channelsValue) {
      this.sampleRateValue = sampleRate;
      this.channelsValue = channels;
      this.enqueueOrBuffer({
        payload: {
          case: "reconfigureSessionRequest",
          value: {
            inputAudioLine: {
              sampleRate,
              channelCount: channels,
              sampleFormat: SampleFormat.SIGNED_16_BIT,
            },
          },
        },
      });
    }

    const packetId = this.nextPacketId();
    this.enqueueOrBuffer({
      payload: {
        case: "userInput",
        value: {
          packetId: BigInt(packetId),
          mode: TRIGGER_MODE_MAP[trigger],
          input: { case: "audioData", value: { data: pcm } },
        },
      },
    });
    return packetId;
  }

  async sendText(
    text: string,
    trigger: TriggerMode = TriggerMode.IMMEDIATE,
  ): Promise<number> {
    const packetId = this.nextPacketId();
    this.enqueueOrBuffer({
      payload: {
        case: "userInput",
        value: {
          packetId: BigInt(packetId),
          mode: TRIGGER_MODE_MAP[trigger],
          input: { case: "textData", value: { data: text } },
        },
      },
    });
    return packetId;
  }

  /** Explicitly initialize the session without sending audio (text-only flows). */
  async initialize(sampleRate = 24000, channels = 1): Promise<void> {
    this.ensureInitialized(sampleRate, channels);
  }

  async triggerInference(instructions?: string): Promise<void> {
    this.enqueueOrBuffer({
      payload: {
        case: "triggerInference",
        value: instructions !== undefined ? { extraInstructions: instructions } : {},
      },
    });
  }

  async sendToolResponse(callId: string, result: unknown): Promise<void> {
    const resultStr =
      typeof result === "string" ? result : JSON.stringify(result);
    this.enqueueOrBuffer({
      payload: { case: "toolCallResponse", value: { id: callId, result: resultStr } },
    });
  }

  /** Persist tool definitions and sync them to the server. */
  async updateTools(tools: FunctionTool[]): Promise<void> {
    this.currentTools = tools;
    if (!this.sessionInitializedFlag) return;
    const msg = this.buildUpdateToolsMessage(tools);
    if (msg) this.send(msg);
  }

  async reconfigure(
    systemPrompt?: string,
    temperature?: number,
  ): Promise<void> {
    if (systemPrompt === undefined && temperature === undefined) return;
    const inferenceConfiguration: { systemPrompt?: string; temperature?: number } =
      {};
    if (systemPrompt !== undefined) inferenceConfiguration.systemPrompt = systemPrompt;
    if (temperature !== undefined) inferenceConfiguration.temperature = temperature;
    this.enqueueOrBuffer({
      payload: { case: "reconfigureSessionRequest", value: { inferenceConfiguration } },
    });
  }

  async sendDirectSpeech(text: string, includeInHistory = true): Promise<void> {
    this.enqueueOrBuffer({
      payload: { case: "directSpeech", value: { text, includeInHistory } },
    });
  }

  async exportChatHistory(
    awaitPending = false,
    excludeAudio = false,
  ): Promise<void> {
    this.enqueueOrBuffer({
      payload: {
        case: "exportChatHistoryRequest",
        value: { awaitPending, excludeAudio },
      },
    });
  }

  async sendConversationQuery(
    queryId: string,
    prompt?: string,
    instructions?: string,
  ): Promise<void> {
    if (prompt === undefined && instructions === undefined) {
      throw new Error("At least one of 'prompt' or 'instructions' must be provided.");
    }
    this.pendingQueryIds.push(queryId);
    const value: { prompt?: string; instructions?: string } = {};
    if (prompt !== undefined) value.prompt = prompt;
    if (instructions !== undefined) value.instructions = instructions;
    this.enqueueOrBuffer({ payload: { case: "conversationQuery", value } });
  }

  async reportPlaybackPosition(bytesPlayed: number): Promise<void> {
    this.enqueueOrBuffer({
      payload: {
        case: "playbackPositionReport",
        value: { bytesPlayed: BigInt(bytesPlayed) },
      },
    });
  }

  // ---- internals ----

  private nextPacketId(): number {
    this.packetIdCounter += 1;
    return this.packetIdCounter;
  }

  private resetState(): void {
    this.ws = null;
    this.sessionInitializedFlag = false;
    this.initRequestSent = false;
    this.sampleRateValue = null;
    this.channelsValue = null;
    this.packetIdCounter = 0;
    this.pendingBeforeInit = [];
    this.pendingQueryIds = [];
    this.closing = false;
  }

  private ensureInitialized(sampleRate: number, channels: number): void {
    if (this.sessionInitializedFlag || this.initRequestSent || this.ws === null) {
      return;
    }
    this.sampleRateValue = sampleRate;
    this.channelsValue = channels;
    this.initRequestSent = true;

    const initRequest = buildInitializeRequestFromOptions(
      this.options,
      this.vadConfig,
      sampleRate,
      channels,
      this.ttsConfig,
    );
    this.send(
      create(ServiceBoundMessageSchema, {
        payload: { case: "initializeSessionRequest", value: initRequest },
      }),
    );
    logger.debug(
      `DeepslateSession: initializing session (${sampleRate}Hz, ${channels}ch)`,
    );

    if (this.currentTools.length > 0) {
      const toolsMsg = this.buildUpdateToolsMessage(this.currentTools);
      if (toolsMsg) this.send(toolsMsg);
    }
  }

  private enqueueOrBuffer(init: ServiceBoundInit): void {
    const msg = create(ServiceBoundMessageSchema, init);
    if (this.sessionInitializedFlag) this.send(msg);
    else this.pendingBeforeInit.push(msg);
  }

  private send(msg: ServiceBoundMessage): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      logger.error("DeepslateSession: send called with no open socket");
      return;
    }
    try {
      this.ws.send(toBinary(ServiceBoundMessageSchema, msg));
    } catch (err) {
      logger.error(`DeepslateSession: send error: ${String(err)}`);
    }
  }

  private buildUpdateToolsMessage(
    tools: FunctionTool[],
  ): ServiceBoundMessage | null {
    if (tools.length === 0) return null;
    const toolDefinitions = tools.map((tool) => ({
      name: tool.function.name ?? "",
      description: tool.function.description ?? "",
      // proto Struct field is typed as JsonObject in protobuf-es.
      parameters: (tool.function.parameters ?? {}) as JsonObject,
    }));
    return create(ServiceBoundMessageSchema, {
      payload: {
        case: "updateToolDefinitionsRequest",
        value: { toolDefinitions },
      },
    });
  }

  private fire<K extends keyof DeepslateSessionEvents & string>(
    event: K,
    ...args: Parameters<DeepslateSessionEvents[K]>
  ): void {
    if (this.listenerCount(event) === 0) {
      if (event === "error" || event === "fatalError") {
        logger.error(`DeepslateSession: unhandled ${event}`, ...args);
      }
      return;
    }
    try {
      this.emit(event, ...args);
    } catch (err) {
      logger.error("DeepslateSession: unhandled exception in listener", err);
    }
  }

  private runWs(ws: WebSocket): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      this.resetState();
      this.ws = ws;
      let settled = false;
      logger.info("DeepslateSession: connected to Deepslate Realtime API");

      const cleanup = () => {
        ws.off("message", onMessage);
        ws.off("close", onClose);
        ws.off("error", onError);
      };
      const settle = (fn: () => void) => {
        if (settled) return;
        settled = true;
        cleanup();
        fn();
      };

      const onMessage = (data: WebSocket.RawData) => {
        try {
          const msg = fromBinary(ClientBoundMessageSchema, rawDataToUint8(data));
          this.handleServerMessage(msg);
        } catch (err) {
          logger.error(`DeepslateSession: failed to handle message: ${String(err)}`);
        }
      };
      const onClose = (code: number, reason: Buffer) => {
        if (this.shouldStop || this.closing) {
          settle(resolve);
          return;
        }
        const reasonStr = reason.length ? reason.toString() : "no reason provided";
        logger.error(
          `DeepslateSession: WebSocket closed unexpectedly: code=${code}, reason=${reasonStr}`,
        );
        settle(() =>
          reject(
            new RetriableError(
              `WebSocket closed unexpectedly: code=${code}, reason=${reasonStr}`,
            ),
          ),
        );
      };
      const onError = (err: Error) => {
        settle(() => reject(new RetriableError(err.message)));
      };

      ws.on("message", onMessage);
      ws.on("close", onClose);
      ws.on("error", onError);
    });
  }

  private handleServerMessage(msg: ClientBoundMessage): void {
    switch (msg.payload.case) {
      case "sessionReady": {
        logger.info("DeepslateSession: session ready");
        for (const pending of this.pendingBeforeInit) this.send(pending);
        this.pendingBeforeInit = [];
        this.sessionInitializedFlag = true;
        this.fire("sessionInitialized");
        break;
      }
      case "responseBegin":
        this.fire("responseBegin");
        break;
      case "responseEnd":
        this.fire("responseEnd");
        break;
      case "modelTextFragment":
        this.fire("textFragment", msg.payload.value.text);
        break;
      case "modelAudioChunk": {
        const chunk = msg.payload.value;
        if (chunk.audio && chunk.audio.data.length > 0) {
          this.fire(
            "audioChunk",
            chunk.audio.data,
            this.sampleRateValue ?? 24000,
            this.channelsValue ?? 1,
            chunk.transcript ? chunk.transcript : null,
          );
        }
        break;
      }
      case "userTranscriptionResult": {
        const result = msg.payload.value;
        this.fire(
          "userTranscription",
          result.text,
          result.language ? result.language : null,
          result.turnId,
        );
        break;
      }
      case "playbackClearBuffer":
        this.fire("playbackBufferClear");
        break;
      case "toolCallRequest": {
        const req = msg.payload.value;
        const params = (req.parameters as Record<string, unknown>) ?? {};
        this.fire("toolCall", req.id, req.name, params);
        break;
      }
      case "conversationQueryResult": {
        const text = msg.payload.value.text;
        const queryId = this.pendingQueryIds.shift() ?? "";
        if (!queryId) {
          logger.warn(
            "DeepslateSession: received conversation_query_result with no pending query_id",
          );
        }
        this.fire("conversationQueryResult", queryId, text);
        break;
      }
      case "chatHistory":
        this.fire("chatHistory", parseChatHistory(msg.payload.value));
        break;
      case "error": {
        const notification = msg.payload.value;
        const categoryName = SessionErrorCategory[notification.category];
        const traceId = notification.traceId ? notification.traceId : null;
        logger.error(
          `DeepslateSession: server error [${categoryName}]: ${notification.message}` +
            (traceId ? ` (trace_id=${traceId})` : ""),
        );
        this.fire("error", categoryName, notification.message, traceId);
        break;
      }
      default:
        logger.debug(`DeepslateSession: unhandled payload type: ${msg.payload.case}`);
        break;
    }
  }
}