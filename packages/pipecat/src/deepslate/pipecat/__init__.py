import importlib.metadata

try:
    __version__ = importlib.metadata.version("deepslate-pipecat")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from deepslate.core.options import (
    DeepslateOptions,
    ElevenLabsLocation,
    ElevenLabsTtsConfig,
    VadConfig,
)

from ._service import DeepslateRealtimeLLMService
from .frames import (
    DeepslateExportChatHistoryFrame,
    DeepslateChatHistoryFrame,
    DeepslateDirectSpeechFrame,
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
    "DeepslateRealtimeLLMService",
    "DeepslateExportChatHistoryFrame",
    "DeepslateChatHistoryFrame",
    "DeepslateDirectSpeechFrame",
    "DeepslateConversationQueryFrame",
    "DeepslateConversationQueryResultFrame",
]