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
    __version__ = importlib.metadata.version("deepslate-core")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from .client import BaseDeepslateClient
from .session import DeepslateSession
from .options import (
    DeepslateOptions,
    ElevenLabsLocation,
    ElevenLabsTtsConfig,
    ElevenLabsVoiceSettingsConfig,
    VadConfig,
)
from ._types import (
    ChatMessageDict,
    ContentBlockDict,
    DeepslateSessionListener,
    FunctionDefinitionDict,
    FunctionToolDict,
    InputAudioContentDict,
    InstructionsContentDict,
    TextContentDict,
    ThoughtsContentDict,
    ToolCallContentDict,
    ToolResultContentDict,
    TriggerMode,
    TtsAudioDict,
)

__all__ = [
    "__version__",
    "BaseDeepslateClient",
    "DeepslateSession",
    "DeepslateSessionListener",
    "TriggerMode",
    "ChatMessageDict",
    "ContentBlockDict",
    "FunctionDefinitionDict",
    "FunctionToolDict",
    "DeepslateOptions",
    "ElevenLabsLocation",
    "ElevenLabsTtsConfig",
    "ElevenLabsVoiceSettingsConfig",
    "InputAudioContentDict",
    "InstructionsContentDict",
    "TextContentDict",
    "ThoughtsContentDict",
    "ToolCallContentDict",
    "ToolResultContentDict",
    "TtsAudioDict",
    "VadConfig",
]
