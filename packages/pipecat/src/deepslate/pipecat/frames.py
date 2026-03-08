from dataclasses import dataclass
from typing import Optional

from pipecat.frames.frames import Frame


@dataclass
class DeepslateTranscriptionFrame(Frame):
    """Carries a speech transcription produced by Deepslate.

    Attributes:
        text:     The transcribed text.
        role:     ``"user"`` for STT results, ``"model"`` for TTS word-alignment.
        language: ISO 639-1 language code (e.g. ``"en"``), if provided by the server.
    """

    text: str
    role: str
    language: Optional[str] = None