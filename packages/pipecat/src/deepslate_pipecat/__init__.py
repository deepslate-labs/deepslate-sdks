import importlib.metadata

try:
    __version__ = importlib.metadata.version("deepslate-pipecat")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from deepslate_core.options import (
    DeepslateOptions,
    ElevenLabsLocation,
    ElevenLabsTtsConfig,
    VadConfig,
)

from ._service import DeepslateRealtimeLLMService

# Backward-compatible alias — legacy code that imported DeepslateVadConfig continues to work.
DeepslateVadConfig = VadConfig

__all__ = [
    "__version__",
    "DeepslateOptions",
    "DeepslateVadConfig",
    "ElevenLabsLocation",
    "ElevenLabsTtsConfig",
    "DeepslateRealtimeLLMService",
]