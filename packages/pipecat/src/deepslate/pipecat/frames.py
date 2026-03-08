from dataclasses import dataclass, field

from pipecat.frames.frames import Frame


@dataclass
class DeepslateExportChatHistoryFrame(Frame):
    """Triggers a chat history export request to the Deepslate backend.

    Attributes:
        await_pending: When True, the server waits for all in-flight async
            operations (e.g. transcriptions) to complete before responding.
    """

    await_pending: bool = False


@dataclass
class DeepslateChatHistoryFrame(Frame):
    """Carries the exported chat history pushed down the pipeline.

    Each entry in ``messages`` is a plain dict with the following shape::

        {
            "role": "user" | "assistant" | "system",
            "delivery_status": "DELIVERY_COMPLETE" | ...,
            "ephemeral": bool,
            "content": [
                # text block (LLM output)
                {
                    "type": "text",
                    "text": str,
                    "tts_audio": {          # present only when TTS was active
                        "transcription": str,
                    } | None,
                },
                # raw audio block (user speech or non-TTS model audio)
                {
                    "type": "input_audio",
                    "transcription": str,
                },
                # model tool call
                {
                    "type": "tool_call",
                    "id": str,
                    "name": str,
                    "parameters": dict,
                },
                # tool execution result
                {
                    "type": "tool_result",
                    "id": str,
                    "result": str,
                },
                # chain-of-thought / reasoning
                {
                    "type": "thoughts",
                    "text": str,
                },
                # injected instructions (e.g. from trigger_inference)
                {
                    "type": "instructions",
                    "text": str,
                },
            ],
        }

    Attributes:
        messages: Ordered list of parsed chat message dicts.
    """

    messages: list = field(default_factory=list)