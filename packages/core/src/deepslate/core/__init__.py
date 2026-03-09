import importlib.metadata

try:
    __version__ = importlib.metadata.version("deepslate-core")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from .client import BaseDeepslateClient
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
    InputAudioContentDict,
    InstructionsContentDict,
    TextContentDict,
    ThoughtsContentDict,
    ToolCallContentDict,
    ToolResultContentDict,
    TtsAudioDict,
)

__all__ = [
    "__version__",
    "BaseDeepslateClient",
    "ChatMessageDict",
    "ContentBlockDict",
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