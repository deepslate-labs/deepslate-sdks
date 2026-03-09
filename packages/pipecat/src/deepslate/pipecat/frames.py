from dataclasses import dataclass

from pipecat.frames.frames import Frame


@dataclass
class DeepslateDirectSpeechFrame(Frame):
    """Instructs Deepslate to speak text directly via TTS, bypassing the LLM.

    Any active inference is cancelled before synthesis begins.

    Attributes:
        text: The text to synthesize and speak.
        include_in_history: When True (default), the spoken text is recorded as
            an assistant message in the chat history so the LLM is aware of it.
            When False, the text is spoken ephemerally and the LLM has no
            knowledge that it was said.
    """

    text: str
    include_in_history: bool = True