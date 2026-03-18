from __future__ import annotations

from typing import Literal, Optional, TypedDict, Union


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