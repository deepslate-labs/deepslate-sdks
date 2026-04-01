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

import importlib.metadata

try:
    __version__ = importlib.metadata.version("deepslate-pipecat")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from deepslate.core.options import (
    DeepslateOptions,
    ElevenLabsLocation,
    ElevenLabsTtsConfig,
    HostedTtsConfig,
    VadConfig,
)

from ._service import DeepslateRealtimeLLMService
from .frames import (
    DeepslateExportChatHistoryFrame,
    DeepslateChatHistoryFrame,
    DeepslateDirectSpeechFrame,
    DeepslateSessionInitializedFrame,
    DeepslateUserTranscriptionFrame,
    DeepslateModelTranscriptionFrame,
    DeepslateConversationQueryFrame,
    DeepslateConversationQueryResultFrame,
)

# Backward-compatible alias — legacy code that imported DeepslateVadConfig continues to work.
DeepslateVadConfig = VadConfig

__all__ = [
    "__version__",
    "DeepslateOptions",
    "DeepslateVadConfig",
    "DeepslateUserTranscriptionFrame",
    "DeepslateModelTranscriptionFrame",
    "ElevenLabsLocation",
    "ElevenLabsTtsConfig",
    "HostedTtsConfig",
    "DeepslateRealtimeLLMService",
    "DeepslateExportChatHistoryFrame",
    "DeepslateChatHistoryFrame",
    "DeepslateDirectSpeechFrame",
    "DeepslateConversationQueryFrame",
    "DeepslateConversationQueryResultFrame",
    "DeepslateSessionInitializedFrame",
]
