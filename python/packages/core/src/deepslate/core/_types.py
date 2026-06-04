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

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional, TypedDict, Union


class TriggerMode(str, Enum):
    NO_TRIGGER = "no_trigger"
    QUEUE = "queue"
    IMMEDIATE = "immediate"


class TtsAudioDict(TypedDict):
    audio: bytes
    transcription: str


class TextContentDict(TypedDict):
    type: Literal["text"]
    text: str
    tts_audio: Optional[TtsAudioDict]


class InputAudioContentDict(TypedDict):
    type: Literal["input_audio"]
    audio: bytes
    transcription: str


class ToolCallContentDict(TypedDict):
    type: Literal["tool_call"]
    id: str
    name: str
    parameters: dict[str, object]


class ToolResultContentDict(TypedDict):
    type: Literal["tool_result"]
    id: str
    result: str


class ThoughtsContentDict(TypedDict):
    type: Literal["thoughts"]
    text: str


class InstructionsContentDict(TypedDict):
    type: Literal["instructions"]
    text: str


ContentBlockDict = Union[
    TextContentDict,
    InputAudioContentDict,
    ToolCallContentDict,
    ToolResultContentDict,
    ThoughtsContentDict,
    InstructionsContentDict,
]


class _FunctionDefinitionRequired(TypedDict):
    name: str


class FunctionDefinitionDict(_FunctionDefinitionRequired, total=False):
    description: str
    parameters: dict[str, object]


class FunctionToolDict(TypedDict):
    type: Literal["function"]
    function: FunctionDefinitionDict


class ChatMessageDict(TypedDict):
    role: str
    delivery_status: str
    ephemeral: bool
    content: list[ContentBlockDict]
    turn_id: Optional[int]
    truncated_at_response_turn_id: Optional[int]


class DeepslateSessionListener:
    """Base class for receiving events from a :class:`DeepslateSession`."""

    async def on_text_fragment(self, text: str) -> None:
        pass

    async def on_audio_chunk(
        self,
        pcm_bytes: bytes,
        sample_rate: int,
        channels: int,
        transcript: Optional[str],
    ) -> None:
        pass

    async def on_tool_call(
        self,
        call_id: str,
        name: str,
        params: dict,
        turn_id: Optional[int] = None,
    ) -> None:
        pass

    async def on_error(
        self, category: str, message: str, trace_id: Optional[str]
    ) -> None:
        pass

    async def on_response_begin(self, turn_id: int = 0) -> None:
        pass

    async def on_response_end(self, turn_id: int = 0) -> None:
        pass

    async def on_user_transcription(
        self, text: str, language: Optional[str], turn_id: int
    ) -> None:
        pass

    async def on_playback_buffer_clear(self) -> None:
        pass

    async def on_chat_history(self, messages: list[ChatMessageDict]) -> None:
        pass

    async def on_conversation_query_result(self, query_id: str, text: str) -> None:
        pass

    async def on_session_initialized(self) -> None:
        pass

    async def on_fatal_error(self, e: Exception) -> None:
        pass

    async def on_vad_state_event(
        self,
        from_state: str,
        to_state: str,
        session_time_ms: int,
        packet_id: int,
    ) -> None:
        """Called when the VAD state machine transitions between states.

        States: ``"SILENCE"``, ``"SPEECH_STARTING"``, ``"SPEECH"``, ``"SPEECH_ENDING"``.
        ``session_time_ms`` is wall-clock time since the input pipeline started.
        ``packet_id`` is the last packet that contributed to the transition.
        Always emitted regardless of ``enable_vad_frame_telemetry``.
        """
        pass

    async def on_context_truncated(
        self,
        truncated_turn_ids: list[int],
        response_turn_id: int,
    ) -> None:
        """Called when the inference engine removes older turns from the LLM context.

        ``truncated_turn_ids`` contains the turn IDs newly removed in this cycle.
        ``response_turn_id`` is the assistant turn that was generated without them.
        """
        pass
