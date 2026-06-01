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
import WebSocket from "ws";

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

export interface RunWithRetryHandlers {
  /** Checked before every attempt; return false to stop the loop. */
  shouldContinue: () => boolean;
  /** Called once when retries are exhausted or an unexpected error occurs. */
  onFatalError: (err: Error) => void | Promise<void>;
}

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Manages WebSocket connectivity to the Deepslate Realtime API: URL
 * construction, auth headers, and exponential-backoff reconnection. Used by
 * DeepslateSession via composition so all transport logic lives in one place.
 */
export class BaseDeepslateClient {
  constructor(
    private readonly opts: ResolvedDeepslateOptions,
    private readonly userAgent: string,
  ) {}

  private buildWsUrl(): string {
    if (this.opts.wsUrl) return this.opts.wsUrl;
    return buildWsUrl(
      this.opts.baseUrl,
      this.opts.vendorId,
      this.opts.organizationId,
    );
  }

  private buildHeaders(): Record<string, string> {
    const headers: Record<string, string> = { "User-Agent": this.userAgent };
    if (this.opts.apiKey) headers["Authorization"] = `Bearer ${this.opts.apiKey}`;
    return headers;
  }

  /** Open a WebSocket connection and resolve once it is OPEN. */
  connect(): Promise<WebSocket> {
    const url = this.buildWsUrl();
    const headers = this.buildHeaders();
    logger.debug(`connecting to Deepslate: ${url}`);

    return new Promise<WebSocket>((resolve, reject) => {
      const ws = new WebSocket(url, { headers });
      const onOpen = () => {
        ws.off("error", onError);
        resolve(ws);
      };
      const onError = (err: Error) => {
        ws.off("open", onOpen);
        // Connection-time failures are retriable.
        reject(new RetriableError(err.message));
      };
      ws.once("open", onOpen);
      ws.once("error", onError);
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

    while (shouldContinue()) {
      try {
        const ws = await this.connect();
        await runSession(ws);
        numRetries = 0; // reset on clean exit
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));

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
        await sleep(retryInterval * 1000);
      }
    }
  }

  /** No shared resources to release in the ws-based client. */
  async aclose(): Promise<void> {
    // intentionally empty
  }
}