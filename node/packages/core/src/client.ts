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

// WebSocket connectivity with exponential-backoff reconnection.
import type { ClientRequest, IncomingMessage } from "node:http";

import WebSocket from "ws";

import { TypedEventEmitter } from "./events.js";
import { logger } from "./log.js";
import type { ResolvedDeepslateOptions } from "./options.js";
import { buildWsUrl } from "./utils.js";

/**
 * Marks an error as retriable. Connection failures and unexpected socket
 * closes are retriable; anything else is treated as fatal and ends the session.
 */
export class RetriableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RetriableError";
  }
}

/**
 * The Deepslate server permanently rejected the WebSocket handshake.
 *
 * Raised for 4xx responses other than 408/429 (e.g. bad credentials, an
 * unknown vendor/organization, or a model the organization cannot use).
 */
export class HandshakeRejectedError extends Error {
  constructor(
    readonly status: number,
    readonly model?: string,
  ) {
    super(handshakeRejectedMessage(status, model));
    this.name = "HandshakeRejectedError";
  }
}

function handshakeRejectedMessage(status: number, model?: string): string {
  const prefix = `Deepslate rejected the WebSocket handshake (HTTP ${status})`;
  if (status === 401) return `${prefix}: check DEEPSLATE_API_KEY / apiKey.`;
  if (status === 403 || status === 404) {
    if (model) {
      return (
        `${prefix} for model '${model}': check vendorId / organizationId, ` +
        "or the organization may not have access to this model."
      );
    }
    return `${prefix}: check vendorId / organizationId.`;
  }
  return `${prefix}.`;
}

/** 4xx statuses that are transient and still worth retrying. */
const RETRIABLE_4XX = new Set([408, 429]);

export interface RunWithRetryHandlers {
  /** Checked before every attempt; return false to stop the loop. */
  shouldContinue: () => boolean;
  /**
   * Called once when retries are exhausted, the handshake is rejected
   * (HandshakeRejectedError), or an unexpected error occurs.
   */
  onFatalError: (err: Error) => void | Promise<void>;
}

export type BaseDeepslateClientEvents = {
  /**
   * Emitted when a raw WebSocket error occurs outside the standard request lifecycle.
   *
   * Because connection and session errors are handled via the 'runWithRetry' promise loop,
   * this event safely catches orphaned errors (e.g., a socket emitting an error after
   * a rejected handshake) to prevent unhandled process exceptions.
   */
  socketError: (err: Error) => void;
};

/** Max time to wait for the WebSocket handshake before aborting the attempt. */
const HANDSHAKE_TIMEOUT_MS = 10_000;

/**
 * Manages WebSocket connectivity to the Deepslate Realtime API: URL
 * construction, auth headers, and exponential-backoff reconnection. Used by
 * DeepslateSession via composition so all transport logic lives in one place.
 *
 * A client backs a single session run loop. Call `requestShutdown()` to make an
 * in-flight `connect()` / backoff `sleep()` return promptly during shutdown.
 */
export class BaseDeepslateClient extends TypedEventEmitter<BaseDeepslateClientEvents> {
  /** Socket created by connect() that hasn't opened yet (still CONNECTING). */
  private pendingWs: WebSocket | null = null;
  /** Set once shutdown has been requested; unblocks connect()/sleep(). */
  private aborted = false;
  /** Callbacks that wake any in-flight backoff sleep on shutdown. */
  private readonly abortListeners = new Set<() => void>();

  constructor(
    private readonly opts: ResolvedDeepslateOptions,
    readonly userAgent: string,
  ) {
    super();
  }

  /**
   * Request shutdown: abort an in-flight CONNECTING handshake and wake any
   * pending backoff sleep so `runWithRetry()` can exit promptly.
   */
  requestShutdown(): void {
    this.aborted = true;
    if (this.pendingWs) {
      try {
        this.pendingWs.terminate();
      } catch {
        // best effort
      }
      this.pendingWs = null;
    }
    for (const wake of this.abortListeners) wake();
    this.abortListeners.clear();
  }

  /** Sleep that resolves early when shutdown is requested. */
  private abortableSleep(ms: number): Promise<void> {
    return new Promise((resolve) => {
      if (this.aborted) {
        resolve();
        return;
      }
      const timer = setTimeout(() => {
        this.abortListeners.delete(wake);
        resolve();
      }, ms);
      const wake = () => {
        clearTimeout(timer);
        resolve();
      };
      this.abortListeners.add(wake);
    });
  }

  private buildWsUrl(): string {
    if (this.opts.wsUrl) return this.opts.wsUrl;
    return buildWsUrl(
      this.opts.baseUrl,
      this.opts.vendorId,
      this.opts.organizationId,
      this.opts.model,
    );
  }

  private buildHeaders(): Record<string, string> {
    const headers: Record<string, string> = { "User-Agent": this.userAgent };
    if (this.opts.apiKey) headers["Authorization"] = `Bearer ${this.opts.apiKey}`;
    return headers;
  }

  /**
   * Open a WebSocket connection and resolve once it is OPEN.
   *
   * Rejects with HandshakeRejectedError when the server answers the handshake
   * with a non-retriable 4xx status, and with RetriableError otherwise.
   */
  connect(): Promise<WebSocket> {
    const url = this.buildWsUrl();
    const headers = this.buildHeaders();
    logger.debug(`connecting to Deepslate: ${url}`);

    return new Promise<WebSocket>((resolve, reject) => {
      if (this.aborted) {
        reject(new RetriableError("connection aborted: client shutting down"));
        return;
      }
      const ws = new WebSocket(url, { headers, handshakeTimeout: HANDSHAKE_TIMEOUT_MS });
      ws.on("error", (err) => {
        logger.debug(`WebSocket error: ${err.message}`);
        if (ws.listenerCount("error") === 1 && this.listenerCount("socketError") > 0) {
          this.emit("socketError", err);
        }
      });
      this.pendingWs = ws;
      const onOpen = () => {
        this.pendingWs = null;
        ws.off("error", onError);
        ws.off("unexpected-response", onUnexpectedResponse);
        resolve(ws);
      };
      const onError = (err: Error) => {
        this.pendingWs = null;
        ws.off("open", onOpen);
        ws.off("unexpected-response", onUnexpectedResponse);
        // Connection-time failures (incl. handshake timeout and shutdown abort)
        // are retriable; runWithRetry() decides whether to stop.
        reject(new RetriableError(err.message));
      };
      const onUnexpectedResponse = (req: ClientRequest, res: IncomingMessage) => {
        this.pendingWs = null;
        ws.off("open", onOpen);
        ws.off("error", onError);
        res.on("error", () => {});
        res.resume();
        req.destroy();
        const status = res.statusCode ?? 0;
        if (status >= 400 && status < 500 && !RETRIABLE_4XX.has(status)) {
          reject(new HandshakeRejectedError(status, this.opts.model));
        } else {
          reject(new RetriableError(`Unexpected server response: ${status}`));
        }
      };
      ws.once("open", onOpen);
      ws.once("error", onError);
      ws.once("unexpected-response", onUnexpectedResponse);
    });
  }

  /**
   * Connect and run `runSession` with exponential-backoff retries.
   *
   * `runSession` receives the open socket and should return a promise that
   * settles when the connection ends — resolve on a clean/intentional close,
   * reject with a RetriableError on an unexpected close so the loop reconnects.
   * Any non-retriable rejection (or exhausting maxRetries) calls onFatalError
   * and exits.
   */
  async runWithRetry(
    runSession: (ws: WebSocket) => Promise<void>,
    handlers: RunWithRetryHandlers,
  ): Promise<void> {
    const { shouldContinue, onFatalError } = handlers;
    let numRetries = 0;
    const maxRetries = this.opts.maxRetries;

    while (shouldContinue() && !this.aborted) {
      try {
        const ws = await this.connect();
        await runSession(ws);
        numRetries = 0; // reset on clean exit
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));

        // Shutdown requested while connecting/running — exit without retrying.
        if (this.aborted || !shouldContinue()) return;

        if (error instanceof HandshakeRejectedError) {
          logger.error(`connection rejected: ${error.message}`);
          await onFatalError(error);
          return;
        }

        if (!(error instanceof RetriableError)) {
          logger.error(`unexpected error in Deepslate session: ${error.message}`);
          await onFatalError(error);
          return;
        }

        if (numRetries >= maxRetries) {
          logger.error(`connection failed after ${numRetries} retries: ${error.message}`);
          await onFatalError(error);
          return;
        }

        numRetries += 1;
        const retryInterval = Math.min(2 ** numRetries, 30);
        logger.warn(
          `connection failed (attempt ${numRetries}/${maxRetries}), ` +
            `retrying in ${retryInterval}s: ${error.message}`,
        );
        await this.abortableSleep(retryInterval * 1000);
      }
    }
  }

  /** No shared resources to release; ensure shutdown is signalled. */
  async aclose(): Promise<void> {
    this.requestShutdown();
  }
}