from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
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

class VadState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SILENCE: _ClassVar[VadState]
    SPEECH_STARTING: _ClassVar[VadState]
    SPEECH: _ClassVar[VadState]
    SPEECH_ENDING: _ClassVar[VadState]

class ElevenLabsLocation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    US: _ClassVar[ElevenLabsLocation]
    EU: _ClassVar[ElevenLabsLocation]
    INDIA: _ClassVar[ElevenLabsLocation]

class HostedTtsMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    HIGH_QUALITY: _ClassVar[HostedTtsMode]
    LOW_LATENCY: _ClassVar[HostedTtsMode]

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
SILENCE: VadState
SPEECH_STARTING: VadState
SPEECH: VadState
SPEECH_ENDING: VadState
US: ElevenLabsLocation
EU: ElevenLabsLocation
INDIA: ElevenLabsLocation
HIGH_QUALITY: HostedTtsMode
LOW_LATENCY: HostedTtsMode
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
    __slots__ = ("id", "name", "parameters", "turn_id")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    TURN_ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    parameters: _struct_pb2.Struct
    turn_id: int
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., parameters: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., turn_id: _Optional[int] = ...) -> None: ...

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

class VadAnalysisFrame(_message.Message):
    __slots__ = ("frame_index", "session_time", "confidence", "volume", "state", "source_packet_ids")
    FRAME_INDEX_FIELD_NUMBER: _ClassVar[int]
    SESSION_TIME_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    VOLUME_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_PACKET_IDS_FIELD_NUMBER: _ClassVar[int]
    frame_index: int
    session_time: Duration
    confidence: float
    volume: float
    state: VadState
    source_packet_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, frame_index: _Optional[int] = ..., session_time: _Optional[_Union[Duration, _Mapping]] = ..., confidence: _Optional[float] = ..., volume: _Optional[float] = ..., state: _Optional[_Union[VadState, str]] = ..., source_packet_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class VadStateEvent(_message.Message):
    __slots__ = ("session_time", "from_state", "to_state", "packet_id")
    SESSION_TIME_FIELD_NUMBER: _ClassVar[int]
    FROM_STATE_FIELD_NUMBER: _ClassVar[int]
    TO_STATE_FIELD_NUMBER: _ClassVar[int]
    PACKET_ID_FIELD_NUMBER: _ClassVar[int]
    session_time: Duration
    from_state: VadState
    to_state: VadState
    packet_id: int
    def __init__(self, session_time: _Optional[_Union[Duration, _Mapping]] = ..., from_state: _Optional[_Union[VadState, str]] = ..., to_state: _Optional[_Union[VadState, str]] = ..., packet_id: _Optional[int] = ...) -> None: ...

class InferenceConfiguration(_message.Message):
    __slots__ = ("system_prompt", "temperature")
    SYSTEM_PROMPT_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    system_prompt: str
    temperature: float
    def __init__(self, system_prompt: _Optional[str] = ..., temperature: _Optional[float] = ...) -> None: ...

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

class HostedVoiceCloneV1(_message.Message):
    __slots__ = ("audio_data", "audio_format", "ref_text")
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FORMAT_FIELD_NUMBER: _ClassVar[int]
    REF_TEXT_FIELD_NUMBER: _ClassVar[int]
    audio_data: bytes
    audio_format: AudioLineConfiguration
    ref_text: str
    def __init__(self, audio_data: _Optional[bytes] = ..., audio_format: _Optional[_Union[AudioLineConfiguration, _Mapping]] = ..., ref_text: _Optional[str] = ...) -> None: ...

class HostedTtsConfiguration(_message.Message):
    __slots__ = ("voice_ref", "voice_clone_v1", "mode")
    VOICE_REF_FIELD_NUMBER: _ClassVar[int]
    VOICE_CLONE_V1_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    voice_ref: HostedVoiceRef
    voice_clone_v1: HostedVoiceCloneV1
    mode: HostedTtsMode
    def __init__(self, voice_ref: _Optional[_Union[HostedVoiceRef, _Mapping]] = ..., voice_clone_v1: _Optional[_Union[HostedVoiceCloneV1, _Mapping]] = ..., mode: _Optional[_Union[HostedTtsMode, str]] = ...) -> None: ...

class TtsConfiguration(_message.Message):
    __slots__ = ("eleven_labs", "hosted")
    ELEVEN_LABS_FIELD_NUMBER: _ClassVar[int]
    HOSTED_FIELD_NUMBER: _ClassVar[int]
    eleven_labs: ElevenLabsTtsConfiguration
    hosted: HostedTtsConfiguration
    def __init__(self, eleven_labs: _Optional[_Union[ElevenLabsTtsConfiguration, _Mapping]] = ..., hosted: _Optional[_Union[HostedTtsConfiguration, _Mapping]] = ...) -> None: ...

class InitializeSessionRequest(_message.Message):
    __slots__ = ("input_audio_line", "output_audio_line", "vad_configuration", "inference_configuration", "tts_configuration", "supports_playback_reporting", "enable_vad_frame_telemetry")
    INPUT_AUDIO_LINE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_AUDIO_LINE_FIELD_NUMBER: _ClassVar[int]
    VAD_CONFIGURATION_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_CONFIGURATION_FIELD_NUMBER: _ClassVar[int]
    TTS_CONFIGURATION_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_PLAYBACK_REPORTING_FIELD_NUMBER: _ClassVar[int]
    ENABLE_VAD_FRAME_TELEMETRY_FIELD_NUMBER: _ClassVar[int]
    input_audio_line: AudioLineConfiguration
    output_audio_line: AudioLineConfiguration
    vad_configuration: VadConfiguration
    inference_configuration: InferenceConfiguration
    tts_configuration: TtsConfiguration
    supports_playback_reporting: bool
    enable_vad_frame_telemetry: bool
    def __init__(self, input_audio_line: _Optional[_Union[AudioLineConfiguration, _Mapping]] = ..., output_audio_line: _Optional[_Union[AudioLineConfiguration, _Mapping]] = ..., vad_configuration: _Optional[_Union[VadConfiguration, _Mapping]] = ..., inference_configuration: _Optional[_Union[InferenceConfiguration, _Mapping]] = ..., tts_configuration: _Optional[_Union[TtsConfiguration, _Mapping]] = ..., supports_playback_reporting: bool = ..., enable_vad_frame_telemetry: bool = ...) -> None: ...

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
    __slots__ = ("extra_instructions", "flush_vad")
    EXTRA_INSTRUCTIONS_FIELD_NUMBER: _ClassVar[int]
    FLUSH_VAD_FIELD_NUMBER: _ClassVar[int]
    extra_instructions: str
    flush_vad: bool
    def __init__(self, extra_instructions: _Optional[str] = ..., flush_vad: bool = ...) -> None: ...

class ExportChatHistoryRequest(_message.Message):
    __slots__ = ("await_pending", "exclude_audio")
    AWAIT_PENDING_FIELD_NUMBER: _ClassVar[int]
    EXCLUDE_AUDIO_FIELD_NUMBER: _ClassVar[int]
    await_pending: bool
    exclude_audio: bool
    def __init__(self, await_pending: bool = ..., exclude_audio: bool = ...) -> None: ...

class PlaybackPositionReport(_message.Message):
    __slots__ = ("bytes_played",)
    BYTES_PLAYED_FIELD_NUMBER: _ClassVar[int]
    bytes_played: int
    def __init__(self, bytes_played: _Optional[int] = ...) -> None: ...

class DirectSpeech(_message.Message):
    __slots__ = ("text", "include_in_history")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_IN_HISTORY_FIELD_NUMBER: _ClassVar[int]
    text: str
    include_in_history: bool
    def __init__(self, text: _Optional[str] = ..., include_in_history: bool = ...) -> None: ...

class ConversationQuery(_message.Message):
    __slots__ = ("prompt", "instructions")
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTIONS_FIELD_NUMBER: _ClassVar[int]
    prompt: str
    instructions: str
    def __init__(self, prompt: _Optional[str] = ..., instructions: _Optional[str] = ...) -> None: ...

class ConversationQueryResult(_message.Message):
    __slots__ = ("text",)
    TEXT_FIELD_NUMBER: _ClassVar[int]
    text: str
    def __init__(self, text: _Optional[str] = ...) -> None: ...

class ServiceBoundMessage(_message.Message):
    __slots__ = ("initialize_session_request", "reconfigure_session_request", "user_input", "update_tool_definitions_request", "tool_call_response", "trigger_inference", "export_chat_history_request", "playback_position_report", "direct_speech", "conversation_query")
    INITIALIZE_SESSION_REQUEST_FIELD_NUMBER: _ClassVar[int]
    RECONFIGURE_SESSION_REQUEST_FIELD_NUMBER: _ClassVar[int]
    USER_INPUT_FIELD_NUMBER: _ClassVar[int]
    UPDATE_TOOL_DEFINITIONS_REQUEST_FIELD_NUMBER: _ClassVar[int]
    TOOL_CALL_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    TRIGGER_INFERENCE_FIELD_NUMBER: _ClassVar[int]
    EXPORT_CHAT_HISTORY_REQUEST_FIELD_NUMBER: _ClassVar[int]
    PLAYBACK_POSITION_REPORT_FIELD_NUMBER: _ClassVar[int]
    DIRECT_SPEECH_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_QUERY_FIELD_NUMBER: _ClassVar[int]
    initialize_session_request: InitializeSessionRequest
    reconfigure_session_request: ReconfigureSessionRequest
    user_input: UserInput
    update_tool_definitions_request: UpdateToolDefinitionsRequest
    tool_call_response: ToolCallResponse
    trigger_inference: TriggerInference
    export_chat_history_request: ExportChatHistoryRequest
    playback_position_report: PlaybackPositionReport
    direct_speech: DirectSpeech
    conversation_query: ConversationQuery
    def __init__(self, initialize_session_request: _Optional[_Union[InitializeSessionRequest, _Mapping]] = ..., reconfigure_session_request: _Optional[_Union[ReconfigureSessionRequest, _Mapping]] = ..., user_input: _Optional[_Union[UserInput, _Mapping]] = ..., update_tool_definitions_request: _Optional[_Union[UpdateToolDefinitionsRequest, _Mapping]] = ..., tool_call_response: _Optional[_Union[ToolCallResponse, _Mapping]] = ..., trigger_inference: _Optional[_Union[TriggerInference, _Mapping]] = ..., export_chat_history_request: _Optional[_Union[ExportChatHistoryRequest, _Mapping]] = ..., playback_position_report: _Optional[_Union[PlaybackPositionReport, _Mapping]] = ..., direct_speech: _Optional[_Union[DirectSpeech, _Mapping]] = ..., conversation_query: _Optional[_Union[ConversationQuery, _Mapping]] = ...) -> None: ...

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
    __slots__ = ("role", "content", "delivery_status", "ephemeral", "created_at", "turn_id", "truncated_at_response_turn_id")
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    DELIVERY_STATUS_FIELD_NUMBER: _ClassVar[int]
    EPHEMERAL_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    TURN_ID_FIELD_NUMBER: _ClassVar[int]
    TRUNCATED_AT_RESPONSE_TURN_ID_FIELD_NUMBER: _ClassVar[int]
    role: ChatMessageRole
    content: _containers.RepeatedCompositeFieldContainer[ChatMessageContent]
    delivery_status: ChatDeliveryStatus
    ephemeral: bool
    created_at: _timestamp_pb2.Timestamp
    turn_id: int
    truncated_at_response_turn_id: int
    def __init__(self, role: _Optional[_Union[ChatMessageRole, str]] = ..., content: _Optional[_Iterable[_Union[ChatMessageContent, _Mapping]]] = ..., delivery_status: _Optional[_Union[ChatDeliveryStatus, str]] = ..., ephemeral: bool = ..., created_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., turn_id: _Optional[int] = ..., truncated_at_response_turn_id: _Optional[int] = ...) -> None: ...

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
    __slots__ = ("audio", "format", "transcription")
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    TRANSCRIPTION_FIELD_NUMBER: _ClassVar[int]
    audio: AudioData
    format: AudioLineConfiguration
    transcription: str
    def __init__(self, audio: _Optional[_Union[AudioData, _Mapping]] = ..., format: _Optional[_Union[AudioLineConfiguration, _Mapping]] = ..., transcription: _Optional[str] = ...) -> None: ...

class PlaybackClearBuffer(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ResponseBegin(_message.Message):
    __slots__ = ("turn_id",)
    TURN_ID_FIELD_NUMBER: _ClassVar[int]
    turn_id: int
    def __init__(self, turn_id: _Optional[int] = ...) -> None: ...

class ResponseEnd(_message.Message):
    __slots__ = ("turn_id",)
    TURN_ID_FIELD_NUMBER: _ClassVar[int]
    turn_id: int
    def __init__(self, turn_id: _Optional[int] = ...) -> None: ...

class SessionErrorNotification(_message.Message):
    __slots__ = ("category", "message", "trace_id")
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    category: SessionErrorCategory
    message: str
    trace_id: str
    def __init__(self, category: _Optional[_Union[SessionErrorCategory, str]] = ..., message: _Optional[str] = ..., trace_id: _Optional[str] = ...) -> None: ...

class UserTranscriptionResult(_message.Message):
    __slots__ = ("turn_id", "text", "language")
    TURN_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    turn_id: int
    text: str
    language: str
    def __init__(self, turn_id: _Optional[int] = ..., text: _Optional[str] = ..., language: _Optional[str] = ...) -> None: ...

class SessionReady(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ContextTruncated(_message.Message):
    __slots__ = ("truncated_turn_ids", "response_turn_id")
    TRUNCATED_TURN_IDS_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_TURN_ID_FIELD_NUMBER: _ClassVar[int]
    truncated_turn_ids: _containers.RepeatedScalarFieldContainer[int]
    response_turn_id: int
    def __init__(self, truncated_turn_ids: _Optional[_Iterable[int]] = ..., response_turn_id: _Optional[int] = ...) -> None: ...

class ClientBoundMessage(_message.Message):
    __slots__ = ("tool_call_request", "model_text_fragment", "model_audio_chunk", "playback_clear_buffer", "response_begin", "response_end", "chat_history", "error", "user_transcription_result", "conversation_query_result", "session_ready", "vad_analysis_frame", "vad_state_event", "context_truncated")
    TOOL_CALL_REQUEST_FIELD_NUMBER: _ClassVar[int]
    MODEL_TEXT_FRAGMENT_FIELD_NUMBER: _ClassVar[int]
    MODEL_AUDIO_CHUNK_FIELD_NUMBER: _ClassVar[int]
    PLAYBACK_CLEAR_BUFFER_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_BEGIN_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_END_FIELD_NUMBER: _ClassVar[int]
    CHAT_HISTORY_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    USER_TRANSCRIPTION_RESULT_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_QUERY_RESULT_FIELD_NUMBER: _ClassVar[int]
    SESSION_READY_FIELD_NUMBER: _ClassVar[int]
    VAD_ANALYSIS_FRAME_FIELD_NUMBER: _ClassVar[int]
    VAD_STATE_EVENT_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_TRUNCATED_FIELD_NUMBER: _ClassVar[int]
    tool_call_request: ToolCallRequest
    model_text_fragment: ModelTextFragment
    model_audio_chunk: ModelAudioChunk
    playback_clear_buffer: PlaybackClearBuffer
    response_begin: ResponseBegin
    response_end: ResponseEnd
    chat_history: ChatHistory
    error: SessionErrorNotification
    user_transcription_result: UserTranscriptionResult
    conversation_query_result: ConversationQueryResult
    session_ready: SessionReady
    vad_analysis_frame: VadAnalysisFrame
    vad_state_event: VadStateEvent
    context_truncated: ContextTruncated
    def __init__(self, tool_call_request: _Optional[_Union[ToolCallRequest, _Mapping]] = ..., model_text_fragment: _Optional[_Union[ModelTextFragment, _Mapping]] = ..., model_audio_chunk: _Optional[_Union[ModelAudioChunk, _Mapping]] = ..., playback_clear_buffer: _Optional[_Union[PlaybackClearBuffer, _Mapping]] = ..., response_begin: _Optional[_Union[ResponseBegin, _Mapping]] = ..., response_end: _Optional[_Union[ResponseEnd, _Mapping]] = ..., chat_history: _Optional[_Union[ChatHistory, _Mapping]] = ..., error: _Optional[_Union[SessionErrorNotification, _Mapping]] = ..., user_transcription_result: _Optional[_Union[UserTranscriptionResult, _Mapping]] = ..., conversation_query_result: _Optional[_Union[ConversationQueryResult, _Mapping]] = ..., session_ready: _Optional[_Union[SessionReady, _Mapping]] = ..., vad_analysis_frame: _Optional[_Union[VadAnalysisFrame, _Mapping]] = ..., vad_state_event: _Optional[_Union[VadStateEvent, _Mapping]] = ..., context_truncated: _Optional[_Union[ContextTruncated, _Mapping]] = ...) -> None: ...
