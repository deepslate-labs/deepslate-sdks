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

import { VERSION } from "./version.js";

const CORE_PRODUCT = "@deepslate-labs/core";

export interface UserAgentProduct {
  name: string;
  version: string;
}

/**
 * Build an RFC 7231-style User-Agent for Deepslate realtime connections.
 *
 * Versions are compiled-in constants (see `version.ts`). Callers pass the
 * value their own package exports rather than having this module discover it
 * at runtime.
 */
export function buildUserAgent(opts?: {
  product: UserAgentProduct;
  framework?: UserAgentProduct;
}): string {
  const core = `${CORE_PRODUCT}/${VERSION}`;
  const runtime = `node/${process.versions.node} ${process.platform}/${process.arch}`;
  if (!opts) return `${core} ${runtime}`;
  const comment = [core];
  if (opts.framework) {
    comment.push(`${opts.framework.name}/${opts.framework.version}`);
  }
  return `${opts.product.name}/${opts.product.version} (${comment.join("; ")}) ${runtime}`;
}
