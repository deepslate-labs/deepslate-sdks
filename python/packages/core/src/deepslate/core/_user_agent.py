# Copyright 2026 Deepslate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib.metadata
import platform
import sys
from typing import Optional

_CORE_DIST = "deepslate-core"


def _dist_version(dist_name: str) -> str:
    """Return the installed version of a distribution, or ``"unknown"``."""
    try:
        return importlib.metadata.version(dist_name)
    except Exception:
        return "unknown"


def build_user_agent(
    product_dist: Optional[str] = None,
    framework_dist: Optional[str] = None,
) -> str:
    """Build an RFC 7231-style User-Agent for Deepslate realtime connections."""
    core = f"{_CORE_DIST}/{_dist_version(_CORE_DIST)}"
    runtime = f"python/{platform.python_version()} {sys.platform}/{platform.machine()}"
    if product_dist is None:
        return f"{core} {runtime}"
    comment = [core]
    if framework_dist is not None:
        comment.append(f"{framework_dist}/{_dist_version(framework_dist)}")
    return f"{product_dist}/{_dist_version(product_dist)} ({'; '.join(comment)}) {runtime}"
