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

/** Logging sink for the SDK. Methods mirror `console`. */
export interface Logger {
  debug(...args: unknown[]): void;
  info(...args: unknown[]): void;
  warn(...args: unknown[]): void;
  error(...args: unknown[]): void;
}

const noop = (): void => {};

const SILENT_LOGGER: Logger = { debug: noop, info: noop, warn: noop, error: noop };

/** Console logger. Opt in with `setLogger(consoleLogger)`. */
export const consoleLogger: Logger = {
  debug: (...args) => console.debug("[deepslate.core]", ...args),
  info: (...args) => console.info("[deepslate.core]", ...args),
  warn: (...args) => console.warn("[deepslate.core]", ...args),
  error: (...args) => console.error("[deepslate.core]", ...args),
};

let active: Logger = SILENT_LOGGER;

/**
 * Install a logger for the SDK. The SDK is silent by default; pass
 * `undefined`/`null` to reset back to silent.
 */
export function setLogger(custom?: Logger | null): void {
  active = custom ?? SILENT_LOGGER;
}

export const logger: Logger = {
  debug: (...args) => active.debug(...args),
  info: (...args) => active.info(...args),
  warn: (...args) => active.warn(...args),
  error: (...args) => active.error(...args),
};