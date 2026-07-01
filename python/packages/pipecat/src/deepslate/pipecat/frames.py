# Copyright 2026 Deepslate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional

from pipecat.frames.frames import Frame, TranscriptionFrame

from deepslate.core import ChatMessageDict


@dataclass
class DeepslateExportChatHistoryFrame(Frame):
    """Triggers a chat history export request to the Deepslate backend."""

    await_pending: bool = False
    """When True, the server waits for all in-flight async operations (e.g. transcriptions) to complete before responding."""

    exclude_audio: bool = False
    """When True, audio data is omitted from the exported history (transcripts only)."""


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

    messages: list[ChatMessageDict] = field(default_factory=list)


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
        uninterruptable: When True, the utterance plays to completion and
            overlapping user speech is ignored until playback finishes (e.g. for
            compliance announcements). Defaults to False (interruptible).
    """

    text: str
    include_in_history: bool = True
    uninterruptable: bool = False


@dataclass
class DeepslateUserTranscriptionFrame(TranscriptionFrame):
    """User STT transcription produced by Deepslate."""


@dataclass
class DeepslateModelTranscriptionFrame(Frame):
    """TTS word-alignment transcription for model audio produced by Deepslate."""

    text: str
    """The transcribed text aligned to the model's TTS output."""


@dataclass
class DeepslateConversationQueryFrame(Frame):
    """Triggers a one-shot side-channel inference query to the Deepslate backend."""

    prompt: Optional[str] = None
    """Optional system prompt override for this query."""

    instructions: Optional[str] = None
    """Optional extra instructions appended to the conversation turns."""


@dataclass
class DeepslateSessionInitializedFrame(Frame):
    """Emitted once when the Deepslate session is fully initialized.

    Push a welcome message or any first-turn logic in response to this frame.
    """


@dataclass
class DeepslateConversationQueryResultFrame(Frame):
    """Carries the result of a side-channel conversation query pushed down the pipeline."""

    text: str = ""
    """The LLM's complete text reply to the conversation query."""


@dataclass
class DeepslateVadStateEventFrame(Frame):
    """Emitted whenever the server-side VAD state machine transitions between states.

    Always emitted (independent of ``enable_vad_frame_telemetry``), so this is
    the primary signal for speech-start and speech-end boundaries.

    States: ``"SILENCE"``, ``"SPEECH_STARTING"``, ``"SPEECH"``, ``"SPEECH_ENDING"``.
    """

    from_state: str = ""
    """The VAD state before the transition."""

    to_state: str = ""
    """The VAD state after the transition."""

    session_time_ms: int = 0
    """Wall-clock time in milliseconds since the input pipeline started."""

    packet_id: int = 0
    """Packet ID of the audio that triggered the transition."""


@dataclass
class DeepslateContextTruncatedFrame(Frame):
    """Emitted when the inference engine removes older turns from the LLM context window.

    Only includes turns **newly** truncated in this inference cycle.
    """

    truncated_turn_ids: list[int] = None  # type: ignore[assignment]
    """Turn IDs that were removed from the model's context window."""

    response_turn_id: int = 0
    """The assistant turn that was generated with the truncated context."""

    def __post_init__(self):
        if self.truncated_turn_ids is None:
            self.truncated_turn_ids = []
