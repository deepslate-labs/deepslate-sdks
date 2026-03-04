from livekit.agents import Plugin

import importlib.metadata

from ._log import logger

try:
    __version__ = importlib.metadata.version("deepslate-livekit")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"


class DeepslatePlugin(Plugin):
    def __init__(self) -> None:
        super().__init__("deepslate_livekit", __version__, "deepslate_livekit", logger)


Plugin.register_plugin(DeepslatePlugin())