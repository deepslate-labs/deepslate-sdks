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

    async def on_tool_call(self, call_id: str, name: str, params: dict) -> None:
        pass

    async def on_error(
        self, category: str, message: str, trace_id: Optional[str]
    ) -> None:
        pass

    async def on_response_begin(self) -> None:
        pass

    async def on_response_end(self) -> None:
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

    async def on_fatal_error(self, e: Exception) -> None:
        pass
