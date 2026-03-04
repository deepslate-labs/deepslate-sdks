from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SampleFormat(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNSIGNED_8_BIT: _ClassVar[SampleFormat]
    SIGNED_16_BIT: _ClassVar[SampleFormat]
    SIGNED_32_BIT: _ClassVar[SampleFormat]
    FLOAT_32_BIT: _ClassVar[SampleFormat]
    FLOAT_64_BIT: _ClassVar[SampleFormat]

class ElevenLabsLocation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    US: _ClassVar[ElevenLabsLocation]
    EU: _ClassVar[ElevenLabsLocation]
    INDIA: _ClassVar[ElevenLabsLocation]

class InferenceTriggerMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NO_TRIGGER: _ClassVar[InferenceTriggerMode]
    QUEUE: _ClassVar[InferenceTriggerMode]
    IMMEDIATE: _ClassVar[InferenceTriggerMode]

class ChatMessageRole(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SYSTEM: _ClassVar[ChatMessageRole]
    USER: _ClassVar[ChatMessageRole]
    ASSISTANT: _ClassVar[ChatMessageRole]

class ChatDeliveryStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DELIVERY_IN_PROGRESS: _ClassVar[ChatDeliveryStatus]
    DELIVERY_COMPLETE: _ClassVar[ChatDeliveryStatus]
    DELIVERY_INTERRUPTED: _ClassVar[ChatDeliveryStatus]

class SessionErrorCategory(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ERROR_UNKNOWN: _ClassVar[SessionErrorCategory]
    ERROR_SESSION: _ClassVar[SessionErrorCategory]
    ERROR_CONFIGURATION: _ClassVar[SessionErrorCategory]
    ERROR_PROTOCOL: _ClassVar[SessionErrorCategory]
    ERROR_INFERENCE: _ClassVar[SessionErrorCategory]
    ERROR_AUDIO: _ClassVar[SessionErrorCategory]
    ERROR_TTS: _ClassVar[SessionErrorCategory]
    ERROR_INTERNAL: _ClassVar[SessionErrorCategory]
UNSIGNED_8_BIT: SampleFormat
SIGNED_16_BIT: SampleFormat
SIGNED_32_BIT: SampleFormat
FLOAT_32_BIT: SampleFormat
FLOAT_64_BIT: SampleFormat
US: ElevenLabsLocation
EU: ElevenLabsLocation
INDIA: ElevenLabsLocation
NO_TRIGGER: InferenceTriggerMode
QUEUE: InferenceTriggerMode
IMMEDIATE: InferenceTriggerMode
SYSTEM: ChatMessageRole
USER: ChatMessageRole
ASSISTANT: ChatMessageRole
DELIVERY_IN_PROGRESS: ChatDeliveryStatus
DELIVERY_COMPLETE: ChatDeliveryStatus
DELIVERY_INTERRUPTED: ChatDeliveryStatus
ERROR_UNKNOWN: SessionErrorCategory
ERROR_SESSION: SessionErrorCategory
ERROR_CONFIGURATION: SessionErrorCategory
ERROR_PROTOCOL: SessionErrorCategory
ERROR_INFERENCE: SessionErrorCategory
ERROR_AUDIO: SessionErrorCategory
ERROR_TTS: SessionErrorCategory
ERROR_INTERNAL: SessionErrorCategory

class ToolDefinition(_message.Message):
    __slots__ = ("name", "description", "parameters")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    name: str
    description: str
    parameters: _struct_pb2.Struct
    def __init__(self, name: _Optional[str] = ..., description: _Optional[str] = ..., parameters: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class UpdateToolDefinitionsRequest(_message.Message):
    __slots__ = ("tool_definitions",)
    TOOL_DEFINITIONS_FIELD_NUMBER: _ClassVar[int]
    tool_definitions: _containers.RepeatedCompositeFieldContainer[ToolDefinition]
    def __init__(self, tool_definitions: _Optional[_Iterable[_Union[ToolDefinition, _Mapping]]] = ...) -> None: ...

class ToolCallRequest(_message.Message):
    __slots__ = ("id", "name", "parameters")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    parameters: _struct_pb2.Struct
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., parameters: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class ToolCallResponse(_message.Message):
    __slots__ = ("id", "result")
    ID_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    id: str
    result: str
    def __init__(self, id: _Optional[str] = ..., result: _Optional[str] = ...) -> None: ...

class Duration(_message.Message):
    __slots__ = ("seconds", "nanos")
    SECONDS_FIELD_NUMBER: _ClassVar[int]
    NANOS_FIELD_NUMBER: _ClassVar[int]
    seconds: int
    nanos: int
    def __init__(self, seconds: _Optional[int] = ..., nanos: _Optional[int] = ...) -> None: ...

class AudioLineConfiguration(_message.Message):
    __slots__ = ("sample_rate", "channel_count", "sample_format")
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_COUNT_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    sample_rate: int
    channel_count: int
    sample_format: SampleFormat
    def __init__(self, sample_rate: _Optional[int] = ..., channel_count: _Optional[int] = ..., sample_format: _Optional[_Union[SampleFormat, str]] = ...) -> None: ...

class VadConfiguration(_message.Message):
    __slots__ = ("confidence_threshold", "min_volume", "start_duration", "stop_duration", "backbuffer_duration")
    CONFIDENCE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    MIN_VOLUME_FIELD_NUMBER: _ClassVar[int]
    START_DURATION_FIELD_NUMBER: _ClassVar[int]
    STOP_DURATION_FIELD_NUMBER: _ClassVar[int]
    BACKBUFFER_DURATION_FIELD_NUMBER: _ClassVar[int]
    confidence_threshold: float
    min_volume: float
    start_duration: Duration
    stop_duration: Duration
    backbuffer_duration: Duration
    def __init__(self, confidence_threshold: _Optional[float] = ..., min_volume: _Optional[float] = ..., start_duration: _Optional[_Union[Duration, _Mapping]] = ..., stop_duration: _Optional[_Union[Duration, _Mapping]] = ..., backbuffer_duration: _Optional[_Union[Duration, _Mapping]] = ...) -> None: ...

class InferenceConfiguration(_message.Message):
    __slots__ = ("system_prompt",)
    SYSTEM_PROMPT_FIELD_NUMBER: _ClassVar[int]
    system_prompt: str
    def __init__(self, system_prompt: _Optional[str] = ...) -> None: ...

class ElevenLabsTtsConfiguration(_message.Message):
    __slots__ = ("api_key", "voice_id", "model_id", "voice_settings", "location")
    API_KEY_FIELD_NUMBER: _ClassVar[int]
    VOICE_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    VOICE_SETTINGS_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    api_key: str
    voice_id: str
    model_id: str
    voice_settings: ElevenLabsVoiceSettings
    location: ElevenLabsLocation
    def __init__(self, api_key: _Optional[str] = ..., voice_id: _Optional[str] = ..., model_id: _Optional[str] = ..., voice_settings: _Optional[_Union[ElevenLabsVoiceSettings, _Mapping]] = ..., location: _Optional[_Union[ElevenLabsLocation, str]] = ...) -> None: ...

class ElevenLabsVoiceSettings(_message.Message):
    __slots__ = ("stability", "similarity_boost", "style", "use_speaker_boost", "speed")
    STABILITY_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_BOOST_FIELD_NUMBER: _ClassVar[int]
    STYLE_FIELD_NUMBER: _ClassVar[int]
    USE_SPEAKER_BOOST_FIELD_NUMBER: _ClassVar[int]
    SPEED_FIELD_NUMBER: _ClassVar[int]
    stability: float
    similarity_boost: float
    style: float
    use_speaker_boost: bool
    speed: float
    def __init__(self, stability: _Optional[float] = ..., similarity_boost: _Optional[float] = ..., style: _Optional[float] = ..., use_speaker_boost: bool = ..., speed: _Optional[float] = ...) -> None: ...

class HostedVoiceRef(_message.Message):
    __slots__ = ("voice_id",)
    VOICE_ID_FIELD_NUMBER: _ClassVar[int]
    voice_id: str
    def __init__(self, voice_id: _Optional[str] = ...) -> None: ...

class HostedTtsConfiguration(_message.Message):
    __slots__ = ("voice_ref",)
    VOICE_REF_FIELD_NUMBER: _ClassVar[int]
    voice_ref: HostedVoiceRef
    def __init__(self, voice_ref: _Optional[_Union[HostedVoiceRef, _Mapping]] = ...) -> None: ...

class TtsConfiguration(_message.Message):
    __slots__ = ("eleven_labs", "hosted")
    ELEVEN_LABS_FIELD_NUMBER: _ClassVar[int]
    HOSTED_FIELD_NUMBER: _ClassVar[int]
    eleven_labs: ElevenLabsTtsConfiguration
    hosted: HostedTtsConfiguration
    def __init__(self, eleven_labs: _Optional[_Union[ElevenLabsTtsConfiguration, _Mapping]] = ..., hosted: _Optional[_Union[HostedTtsConfiguration, _Mapping]] = ...) -> None: ...

class InitializeSessionRequest(_message.Message):
    __slots__ = ("input_audio_line", "output_audio_line", "vad_configuration", "inference_configuration", "tts_configuration")
    INPUT_AUDIO_LINE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_AUDIO_LINE_FIELD_NUMBER: _ClassVar[int]
    VAD_CONFIGURATION_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_CONFIGURATION_FIELD_NUMBER: _ClassVar[int]
    TTS_CONFIGURATION_FIELD_NUMBER: _ClassVar[int]
    input_audio_line: AudioLineConfiguration
    output_audio_line: AudioLineConfiguration
    vad_configuration: VadConfiguration
    inference_configuration: InferenceConfiguration
    tts_configuration: TtsConfiguration
    def __init__(self, input_audio_line: _Optional[_Union[AudioLineConfiguration, _Mapping]] = ..., output_audio_line: _Optional[_Union[AudioLineConfiguration, _Mapping]] = ..., vad_configuration: _Optional[_Union[VadConfiguration, _Mapping]] = ..., inference_configuration: _Optional[_Union[InferenceConfiguration, _Mapping]] = ..., tts_configuration: _Optional[_Union[TtsConfiguration, _Mapping]] = ...) -> None: ...

class ReconfigureSessionRequest(_message.Message):
    __slots__ = ("input_audio_line", "inference_configuration")
    INPUT_AUDIO_LINE_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_CONFIGURATION_FIELD_NUMBER: _ClassVar[int]
    input_audio_line: AudioLineConfiguration
    inference_configuration: InferenceConfiguration
    def __init__(self, input_audio_line: _Optional[_Union[AudioLineConfiguration, _Mapping]] = ..., inference_configuration: _Optional[_Union[InferenceConfiguration, _Mapping]] = ...) -> None: ...

class AudioData(_message.Message):
    __slots__ = ("data",)
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    def __init__(self, data: _Optional[bytes] = ...) -> None: ...

class TextData(_message.Message):
    __slots__ = ("data",)
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: str
    def __init__(self, data: _Optional[str] = ...) -> None: ...

class UserInput(_message.Message):
    __slots__ = ("packet_id", "mode", "audio_data", "text_data")
    PACKET_ID_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    TEXT_DATA_FIELD_NUMBER: _ClassVar[int]
    packet_id: int
    mode: InferenceTriggerMode
    audio_data: AudioData
    text_data: TextData
    def __init__(self, packet_id: _Optional[int] = ..., mode: _Optional[_Union[InferenceTriggerMode, str]] = ..., audio_data: _Optional[_Union[AudioData, _Mapping]] = ..., text_data: _Optional[_Union[TextData, _Mapping]] = ...) -> None: ...

class TriggerInference(_message.Message):
    __slots__ = ("extra_instructions",)
    EXTRA_INSTRUCTIONS_FIELD_NUMBER: _ClassVar[int]
    extra_instructions: str
    def __init__(self, extra_instructions: _Optional[str] = ...) -> None: ...

class ExportChatHistoryRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ServiceBoundMessage(_message.Message):
    __slots__ = ("initialize_session_request", "reconfigure_session_request", "user_input", "update_tool_definitions_request", "tool_call_response", "trigger_inference", "export_chat_history_request")
    INITIALIZE_SESSION_REQUEST_FIELD_NUMBER: _ClassVar[int]
    RECONFIGURE_SESSION_REQUEST_FIELD_NUMBER: _ClassVar[int]
    USER_INPUT_FIELD_NUMBER: _ClassVar[int]
    UPDATE_TOOL_DEFINITIONS_REQUEST_FIELD_NUMBER: _ClassVar[int]
    TOOL_CALL_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    TRIGGER_INFERENCE_FIELD_NUMBER: _ClassVar[int]
    EXPORT_CHAT_HISTORY_REQUEST_FIELD_NUMBER: _ClassVar[int]
    initialize_session_request: InitializeSessionRequest
    reconfigure_session_request: ReconfigureSessionRequest
    user_input: UserInput
    update_tool_definitions_request: UpdateToolDefinitionsRequest
    tool_call_response: ToolCallResponse
    trigger_inference: TriggerInference
    export_chat_history_request: ExportChatHistoryRequest
    def __init__(self, initialize_session_request: _Optional[_Union[InitializeSessionRequest, _Mapping]] = ..., reconfigure_session_request: _Optional[_Union[ReconfigureSessionRequest, _Mapping]] = ..., user_input: _Optional[_Union[UserInput, _Mapping]] = ..., update_tool_definitions_request: _Optional[_Union[UpdateToolDefinitionsRequest, _Mapping]] = ..., tool_call_response: _Optional[_Union[ToolCallResponse, _Mapping]] = ..., trigger_inference: _Optional[_Union[TriggerInference, _Mapping]] = ..., export_chat_history_request: _Optional[_Union[ExportChatHistoryRequest, _Mapping]] = ...) -> None: ...

class ModelTextFragment(_message.Message):
    __slots__ = ("text",)
    TEXT_FIELD_NUMBER: _ClassVar[int]
    text: str
    def __init__(self, text: _Optional[str] = ...) -> None: ...

class ModelAudioChunk(_message.Message):
    __slots__ = ("audio", "transcript")
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    TRANSCRIPT_FIELD_NUMBER: _ClassVar[int]
    audio: AudioData
    transcript: str
    def __init__(self, audio: _Optional[_Union[AudioData, _Mapping]] = ..., transcript: _Optional[str] = ...) -> None: ...

class ChatHistory(_message.Message):
    __slots__ = ("messages",)
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    messages: _containers.RepeatedCompositeFieldContainer[ChatMessage]
    def __init__(self, messages: _Optional[_Iterable[_Union[ChatMessage, _Mapping]]] = ...) -> None: ...

class ChatMessage(_message.Message):
    __slots__ = ("role", "content", "delivery_status")
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    DELIVERY_STATUS_FIELD_NUMBER: _ClassVar[int]
    role: ChatMessageRole
    content: _containers.RepeatedCompositeFieldContainer[ChatMessageContent]
    delivery_status: ChatDeliveryStatus
    def __init__(self, role: _Optional[_Union[ChatMessageRole, str]] = ..., content: _Optional[_Iterable[_Union[ChatMessageContent, _Mapping]]] = ..., delivery_status: _Optional[_Union[ChatDeliveryStatus, str]] = ...) -> None: ...

class ChatMessageContent(_message.Message):
    __slots__ = ("text_content", "input_audio", "thoughts", "tool_call", "tool_result", "instructions")
    TEXT_CONTENT_FIELD_NUMBER: _ClassVar[int]
    INPUT_AUDIO_FIELD_NUMBER: _ClassVar[int]
    THOUGHTS_FIELD_NUMBER: _ClassVar[int]
    TOOL_CALL_FIELD_NUMBER: _ClassVar[int]
    TOOL_RESULT_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTIONS_FIELD_NUMBER: _ClassVar[int]
    text_content: ChatTextContent
    input_audio: ChatAudioData
    thoughts: str
    tool_call: ToolCallRequest
    tool_result: ToolCallResponse
    instructions: str
    def __init__(self, text_content: _Optional[_Union[ChatTextContent, _Mapping]] = ..., input_audio: _Optional[_Union[ChatAudioData, _Mapping]] = ..., thoughts: _Optional[str] = ..., tool_call: _Optional[_Union[ToolCallRequest, _Mapping]] = ..., tool_result: _Optional[_Union[ToolCallResponse, _Mapping]] = ..., instructions: _Optional[str] = ...) -> None: ...

class ChatTextContent(_message.Message):
    __slots__ = ("text", "tts_audio")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TTS_AUDIO_FIELD_NUMBER: _ClassVar[int]
    text: str
    tts_audio: ChatAudioData
    def __init__(self, text: _Optional[str] = ..., tts_audio: _Optional[_Union[ChatAudioData, _Mapping]] = ...) -> None: ...

class ChatAudioData(_message.Message):
    __slots__ = ("audio", "format")
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    audio: AudioData
    format: AudioLineConfiguration
    def __init__(self, audio: _Optional[_Union[AudioData, _Mapping]] = ..., format: _Optional[_Union[AudioLineConfiguration, _Mapping]] = ...) -> None: ...

class PlaybackClearBuffer(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ResponseBegin(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ResponseEnd(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SessionErrorNotification(_message.Message):
    __slots__ = ("category", "message", "trace_id")
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    category: SessionErrorCategory
    message: str
    trace_id: str
    def __init__(self, category: _Optional[_Union[SessionErrorCategory, str]] = ..., message: _Optional[str] = ..., trace_id: _Optional[str] = ...) -> None: ...

class ClientBoundMessage(_message.Message):
    __slots__ = ("tool_call_request", "model_text_fragment", "model_audio_chunk", "playback_clear_buffer", "response_begin", "response_end", "chat_history", "error")
    TOOL_CALL_REQUEST_FIELD_NUMBER: _ClassVar[int]
    MODEL_TEXT_FRAGMENT_FIELD_NUMBER: _ClassVar[int]
    MODEL_AUDIO_CHUNK_FIELD_NUMBER: _ClassVar[int]
    PLAYBACK_CLEAR_BUFFER_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_BEGIN_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_END_FIELD_NUMBER: _ClassVar[int]
    CHAT_HISTORY_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    tool_call_request: ToolCallRequest
    model_text_fragment: ModelTextFragment
    model_audio_chunk: ModelAudioChunk
    playback_clear_buffer: PlaybackClearBuffer
    response_begin: ResponseBegin
    response_end: ResponseEnd
    chat_history: ChatHistory
    error: SessionErrorNotification
    def __init__(self, tool_call_request: _Optional[_Union[ToolCallRequest, _Mapping]] = ..., model_text_fragment: _Optional[_Union[ModelTextFragment, _Mapping]] = ..., model_audio_chunk: _Optional[_Union[ModelAudioChunk, _Mapping]] = ..., playback_clear_buffer: _Optional[_Union[PlaybackClearBuffer, _Mapping]] = ..., response_begin: _Optional[_Union[ResponseBegin, _Mapping]] = ..., response_end: _Optional[_Union[ResponseEnd, _Mapping]] = ..., chat_history: _Optional[_Union[ChatHistory, _Mapping]] = ..., error: _Optional[_Union[SessionErrorNotification, _Mapping]] = ...) -> None: ...
