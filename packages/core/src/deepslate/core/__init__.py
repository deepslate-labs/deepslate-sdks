import importlib.metadata

try:
    __version__ = importlib.metadata.version("deepslate-core")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from .client import BaseDeepslateClient
from .proto.realtime_pb2 import InferenceTriggerMode
from .session import (
    DeepslateSession,
    OnAudioChunk,
    OnChatHistory,
    OnConversationQueryResult,
    OnError,
    OnFatalError,
    OnPlaybackBufferClear,
    OnResponseBegin,
    OnResponseEnd,
    OnTextFragment,
    OnToolCall,
    OnUserTranscription,
)
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
    FunctionDefinitionDict,
    FunctionToolDict,
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
    "DeepslateSession",
    "InferenceTriggerMode",
    "OnAudioChunk",
    "OnChatHistory",
    "OnConversationQueryResult",
    "OnError",
    "OnFatalError",
    "OnPlaybackBufferClear",
    "OnResponseBegin",
    "OnResponseEnd",
    "OnTextFragment",
    "OnToolCall",
    "OnUserTranscription",
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