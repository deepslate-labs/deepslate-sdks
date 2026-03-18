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