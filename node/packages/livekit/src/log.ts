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

// Logs through LiveKit's own logger (from @livekit/agents) and routes
// @deepslate/core's logs through it too, rather than writing to stdout directly.
import { format } from "node:util";

import { log } from "@livekit/agents";
import { setLogger, type Logger } from "@deepslate/core";

type Level = "debug" | "info" | "warn" | "error";

function emit(level: Level, args: unknown[]): void {
  try {
    log()[level](format(...args));
  } catch {
    // log() throws until the framework runs initializeLogger(); drop until then.
  }
}

export const logger: Logger = {
  debug: (...args) => emit("debug", args),
  info: (...args) => emit("info", args),
  warn: (...args) => emit("warn", args),
  error: (...args) => emit("error", args),
};

setLogger(logger);