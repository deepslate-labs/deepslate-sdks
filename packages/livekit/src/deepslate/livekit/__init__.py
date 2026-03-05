import importlib.metadata

try:
    __version__ = importlib.metadata.version("deepslate-livekit")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from deepslate.core.options import (
    DeepslateOptions,
    ElevenLabsLocation,
    ElevenLabsTtsConfig,
    VadConfig,
)

from .realtime import DeepslateRealtimeSession, RealtimeModel

from . import _plugin  # noqa: F401 – triggers Plugin.register_plugin on import

__all__ = [
    "__version__",
    "DeepslateOptions",
    "ElevenLabsLocation",
    "ElevenLabsTtsConfig",
    "VadConfig",
    "RealtimeModel",
    "DeepslateRealtimeSession",
]