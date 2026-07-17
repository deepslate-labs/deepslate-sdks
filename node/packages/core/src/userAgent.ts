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

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const CORE_PRODUCT = "@deepslate-labs/core";

function readPackageJson(dir: string): { version?: string } | null {
  try {
    return JSON.parse(readFileSync(path.join(dir, "package.json"), "utf8"));
  } catch {
    return null;
  }
}

function nearestPackageVersion(fromFile: string): string {
  let dir = path.dirname(fromFile);
  for (let i = 0; i < 10; i++) {
    const version = readPackageJson(dir)?.version;
    if (version) return version;
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return "unknown";
}

export function ownPackageVersion(importMetaUrl: string): string {
  try {
    return nearestPackageVersion(fileURLToPath(importMetaUrl));
  } catch {
    return "unknown";
  }
}

export function dependencyVersion(
  packageName: string,
  importMetaUrl: string,
): string {
  try {
    const require_ = createRequire(importMetaUrl);
    try {
      const version = (
        require_(`${packageName}/package.json`) as { version?: string }
      ).version;
      if (version) return version;
    } catch {
      // package.json not in the package's exports map; fall back to
      // resolving the entry point and walking up to its package.json.
    }
    return nearestPackageVersion(require_.resolve(packageName));
  } catch {
    return "unknown";
  }
}

export interface UserAgentProduct {
  name: string;
  version: string;
}

/**
 * Build an RFC 7231-style User-Agent for Deepslate realtime connections.
 */
export function buildUserAgent(opts?: {
  product: UserAgentProduct;
  framework?: UserAgentProduct;
}): string {
  const core = `${CORE_PRODUCT}/${ownPackageVersion(import.meta.url)}`;
  const runtime = `node/${process.versions.node} ${process.platform}/${process.arch}`;
  if (!opts) return `${core} ${runtime}`;
  const comment = [core];
  if (opts.framework) {
    comment.push(`${opts.framework.name}/${opts.framework.version}`);
  }
  return `${opts.product.name}/${opts.product.version} (${comment.join("; ")}) ${runtime}`;
}
