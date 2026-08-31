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

// Encoding of the session `experiments` map into protobuf Values.
import { fromJson, type JsonValue } from "@bufbuild/protobuf";
import { type Value, ValueSchema } from "@bufbuild/protobuf/wkt";

import type { Experiments } from "./options.js";

/**
 * Encode a caller's experiments map into protobuf `Value` entries.
 */
export function encodeExperiments(
  experiments: Experiments | undefined,
): Record<string, Value> {
  if (experiments === undefined) return {};

  const encoded: Record<string, Value> = {};
  for (const [name, value] of Object.entries(experiments)) {
    encoded[name] = fromJson(
      ValueSchema,
      normalize(value, `experiments[${JSON.stringify(name)}]`),
    );
  }
  return encoded;
}

/**
 * Reduce a caller value to plain JSON data, rejecting anything else.
 */
function normalize(value: unknown, path: string): JsonValue {
  if (value === undefined || value === null) return null;

  if (typeof value === "boolean" || typeof value === "string") return value;

  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new Error(
        `${path}: non-finite numbers (NaN, Infinity) have no JSON representation`,
      );
    }
    return value;
  }

  if (Array.isArray(value)) {
    return value.map((item, index) => normalize(item, `${path}[${index}]`));
  }

  if (isPlainObject(value)) {
    const normalized: Record<string, JsonValue> = {};
    for (const [key, item] of Object.entries(value)) {
      normalized[key] = normalize(item, `${path}.${key}`);
    }
    return normalized;
  }

  throw new Error(
    `${path}: ${describeType(value)} is not JSON-encodable. Experiment values ` +
      `accept null, booleans, numbers, strings, arrays and plain objects.`,
  );
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function describeType(value: unknown): string {
  const type = typeof value;
  if (type !== "object") return type;
  return (value as object).constructor?.name ?? "object";
}
