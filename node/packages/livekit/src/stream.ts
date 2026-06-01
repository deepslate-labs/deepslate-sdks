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

// A push-driven ReadableStream for enqueuing items imperatively. The
// @livekit/agents realtime API hands generations to the framework as web
// ReadableStreams; this lets us feed them from the core session's event
// callbacks.
import { ReadableStream, type ReadableStreamDefaultController } from "node:stream/web";

export interface Pushable<T> {
  readonly stream: ReadableStream<T>;
  push(item: T): void;
  close(): void;
  readonly closed: boolean;
}

export function createPushable<T>(): Pushable<T> {
  let controller: ReadableStreamDefaultController<T> | undefined;
  let closed = false;
  const pending: T[] = [];

  const stream = new ReadableStream<T>({
    start(c) {
      controller = c;
      for (const item of pending) c.enqueue(item);
      pending.length = 0;
      if (closed) c.close();
    },
  });

  return {
    stream,
    push(item: T): void {
      if (closed) return;
      if (controller) controller.enqueue(item);
      else pending.push(item);
    },
    close(): void {
      if (closed) return;
      closed = true;
      if (controller) controller.close();
    },
    get closed(): boolean {
      return closed;
    },
  };
}