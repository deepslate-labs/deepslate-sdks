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
    VadConfig,
)

__all__ = [
    "__version__",
    "BaseDeepslateClient",
    "DeepslateOptions",
    "ElevenLabsLocation",
    "ElevenLabsTtsConfig",
    "VadConfig",
]