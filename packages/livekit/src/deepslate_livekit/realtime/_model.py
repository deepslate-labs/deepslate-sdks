from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

import aiohttp
from livekit import rtc
from livekit.agents import llm, utils, FunctionTool, NOT_GIVEN, NotGivenOr
from livekit.agents.llm import (
    FunctionCall,
    GenerationCreatedEvent,
    InputSpeechStartedEvent,
    MessageGeneration,
    RawFunctionTool,
    ToolChoice,
    ToolContext,
)
from livekit.agents.llm.tool_context import (
    get_raw_function_info,
    is_function_tool,
    is_raw_function_tool,
)

import importlib.metadata

try:
    __version__ = importlib.metadata.version("deepslate-livekit")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from deepslate_core._utils import dict_to_struct, duration_from_ms, ELEVENLABS_LOCATION_MAP, struct_to_dict
from deepslate_core.client import BaseDeepslateClient
from deepslate_core.options import DeepslateOptions, ElevenLabsTtsConfig, VadConfig
from deepslate_core.proto import realtime_pb2 as proto

from .._log import logger

DEEPSLATE_BASE_URL = "https://app.deepslate.eu"


@dataclass
class _ResponseGeneration:
    """Internal state for a response being generated."""

    message_ch: utils.aio.Chan["MessageGeneration"]
    function_ch: utils.aio.Chan["FunctionCall"]
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    done_fut: asyncio.Future[None]
    response_id: str
    created_timestamp: float
    first_token_timestamp: float | None = None
    audio_transcript: str = ""


class RealtimeModel(llm.RealtimeModel):
    """Real-time language model using Deepslate.

    Connects to Deepslate's WebSocket API for streaming LLM responses.
    Audio format is auto-detected from the first audio frame.
    """

    def __init__(
        self,
        vendor_id: str | None = None,
        organization_id: str | None = None,
        api_key: str | None = None,
        base_url: str = DEEPSLATE_BASE_URL,
        system_prompt: str = "You are a helpful assistant.",
        generate_reply_timeout: float = 30.0,
        # VAD configuration
        vad_confidence_threshold: float = 0.5,
        vad_min_volume: float = 0.01,
        vad_start_duration_ms: int = 200,
        vad_stop_duration_ms: int = 500,
        vad_backbuffer_duration_ms: int = 1000,
        # TTS configuration
        tts_config: ElevenLabsTtsConfig | None = None,
        http_session: aiohttp.ClientSession | None = None,
        # Internal use only - direct WebSocket URL (bypass standard auth)
        ws_url: str | None = None,
    ):
        """Initialize a Deepslate RealtimeModel.

        Args:
            vendor_id: Deepslate vendor ID. Falls back to DEEPSLATE_VENDOR_ID env var.
            organization_id: Deepslate organization ID. Falls back to DEEPSLATE_ORGANIZATION_ID env var.
            api_key: Deepslate API key. Falls back to DEEPSLATE_API_KEY env var.
            base_url: Base URL for Deepslate API.
            system_prompt: System prompt for the model.
            generate_reply_timeout: Timeout in seconds for generate_reply (0 = no timeout).
            vad_confidence_threshold: VAD confidence threshold (0.0 to 1.0).
            vad_min_volume: VAD minimum volume threshold (0.0 to 1.0).
            vad_start_duration_ms: Duration of speech to detect start (milliseconds).
            vad_stop_duration_ms: Duration of silence to detect end (milliseconds).
            vad_backbuffer_duration_ms: Audio buffer duration before speech detection (milliseconds).
            tts_config: ElevenLabs TTS configuration. When provided, audio output is enabled
                        and Deepslate will use ElevenLabs for text-to-speech synthesis.
                        When None (default), only text output is provided.
            http_session: Optional shared aiohttp session.
        """
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=True,
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=True,
                audio_output=tts_config is not None,
                manual_function_calls=False,
            )
        )

        self._tts_config = tts_config

        if ws_url:
            # Used for local development or self-hosting Deepslate infrastructure
            deepslate_vendor_id = vendor_id or ""
            deepslate_organization_id = organization_id or ""
            deepslate_api_key = api_key or ""
        else:
            deepslate_vendor_id = vendor_id or os.environ.get("DEEPSLATE_VENDOR_ID")
            if not deepslate_vendor_id:
                raise ValueError(
                    "Deepslate vendor ID is required. "
                    "Provide it via the vendor_id parameter or set the DEEPSLATE_VENDOR_ID environment variable."
                )

            deepslate_organization_id = organization_id or os.environ.get("DEEPSLATE_ORGANIZATION_ID")
            if not deepslate_organization_id:
                raise ValueError(
                    "Deepslate organization ID is required. "
                    "Provide it via the organization_id parameter or set the DEEPSLATE_ORGANIZATION_ID environment variable."
                )

            deepslate_api_key = api_key or os.environ.get("DEEPSLATE_API_KEY")
            if not deepslate_api_key:
                raise ValueError(
                    "Deepslate API key is required. "
                    "Provide it via the api_key parameter or set the DEEPSLATE_API_KEY environment variable."
                )

        self._opts = DeepslateOptions(
            vendor_id=deepslate_vendor_id,
            organization_id=deepslate_organization_id,
            api_key=deepslate_api_key,
            base_url=base_url,
            system_prompt=system_prompt,
            ws_url=ws_url,
            generate_reply_timeout=generate_reply_timeout,
        )

        self._vad_config = VadConfig(
            confidence_threshold=vad_confidence_threshold,
            min_volume=vad_min_volume,
            start_duration_ms=vad_start_duration_ms,
            stop_duration_ms=vad_stop_duration_ms,
            backbuffer_duration_ms=vad_backbuffer_duration_ms,
        )

        self._client = BaseDeepslateClient(
            opts=self._opts,
            user_agent=f"DeepslateLiveKit/{__version__}",
            http_session=http_session,
        )

    @property
    def provider(self) -> str:
        return "deepslate"

    def session(self) -> DeepslateRealtimeSession:
        """Create a new Deepslate real-time session."""
        return DeepslateRealtimeSession(realtime_model=self)

    def update_options(
        self,
        *,
        system_prompt: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Update model options."""
        if utils.is_given(system_prompt):
            self._opts.system_prompt = system_prompt

    async def aclose(self) -> None:
        await self._client.aclose()


class DeepslateRealtimeSession(
    llm.RealtimeSession[Literal["deepslate_server_event_received", "deepslate_client_event_sent"]]
):
    """A session for the Deepslate Realtime API.

    This class manages WebSocket communication with Deepslate,
    handling audio streaming, tool calls, and response generation.
    """

    def __init__(self, realtime_model: RealtimeModel):
        super().__init__(realtime_model)
        self._realtime_model: RealtimeModel = realtime_model
        self._opts = realtime_model._opts
        self._vad_config = realtime_model._vad_config

        # Context
        self._tools = llm.ToolContext.empty()
        self._chat_ctx = llm.ChatContext.empty()
        self._instructions: str | None = None

        # Message channel for outgoing messages
        self._msg_ch: utils.aio.Chan[proto.ServiceBoundMessage] = utils.aio.Chan()

        # Audio state - will be initialized on first audio frame
        self._session_initialized = False
        self._detected_sample_rate: int | None = None
        self._detected_num_channels: int | None = None
        self._packet_id_counter: int = 0

        # Buffer for messages that need to be sent after initialization
        self._pending_messages: list[proto.ServiceBoundMessage] = []

        # Pending user text to send with next generate_reply
        self._pending_user_text: str | None = None

        # Generation tracking
        self._current_generation: _ResponseGeneration | None = None
        self._response_created_futures: dict[str, asyncio.Future[GenerationCreatedEvent]] = {}
        self._pending_user_generation: bool = False

        # Main task
        self._main_atask = asyncio.create_task(
            self._main_task(), name="DeepslateRealtimeSession._main_task"
        )

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    @property
    def tools(self) -> ToolContext:
        return self._tools.copy()

    async def update_instructions(self, instructions: str) -> None:
        """Update system prompt."""
        self._instructions = instructions
        self._opts.system_prompt = instructions
        logger.debug("instructions updated (will take effect on next session)")

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        """Update chat context.

        Deepslate manages its own context, but we capture new user messages
        to send them via TextData when generate_reply is called.
        We also handle function call outputs to send tool results back to the server.
        """
        existing_ids = {item.id for item in self._chat_ctx.items}

        for item in chat_ctx.items:
            if item.id not in existing_ids:
                if item.type == "message" and item.role == "user":
                    if text := item.text_content:
                        self._pending_user_text = text
                elif item.type == "function_call_output":
                    self._send_tool_response(item.call_id, item.output)

        self._chat_ctx = chat_ctx.copy()

    async def update_tools(self, tools: list[FunctionTool | RawFunctionTool | Any]) -> None:
        """Update tool definitions."""
        tool_definitions = []

        for tool in tools:
            if is_function_tool(tool):
                tool_schema = llm.utils.build_legacy_openai_schema(tool, internally_tagged=True)
                tool_def = proto.ToolDefinition(
                    name=tool_schema["name"],
                    description=tool_schema.get("description", ""),
                    parameters=dict_to_struct(tool_schema.get("parameters", {})),
                )
                tool_definitions.append(tool_def)
            elif is_raw_function_tool(tool):
                info = get_raw_function_info(tool)
                tool_def = proto.ToolDefinition(
                    name=info.name,
                    description=info.raw_schema.get("description", ""),
                    parameters=dict_to_struct(info.raw_schema.get("parameters", {})),
                )
                tool_definitions.append(tool_def)

        update_request = proto.UpdateToolDefinitionsRequest(tool_definitions=tool_definitions)
        msg = proto.ServiceBoundMessage(update_tool_definitions_request=update_request)

        if not self._session_initialized:
            self._pending_messages.append(msg)
        else:
            self._send_message(msg)

        self._tools = llm.ToolContext(tools)
        logger.debug(f"updated tools: {[t.name for t in tool_definitions]}")

    def update_options(self, *, tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN) -> None:
        """Update session options. Dynamic tool_choice updates not supported."""
        pass

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        """Push audio frame to Deepslate."""
        if not self._session_initialized:
            self._detected_sample_rate = frame.sample_rate
            self._detected_num_channels = frame.num_channels
            self._queue_initialize_session()
            self._session_initialized = True
            logger.debug(
                f"session initialized with audio format: {frame.sample_rate}Hz, {frame.num_channels}ch"
            )
        elif (
            frame.sample_rate != self._detected_sample_rate
            or frame.num_channels != self._detected_num_channels
        ):
            self._detected_sample_rate = frame.sample_rate
            self._detected_num_channels = frame.num_channels
            reconfig = proto.ReconfigureSessionRequest(
                input_audio_line=proto.AudioLineConfiguration(
                    sample_rate=frame.sample_rate,
                    channel_count=frame.num_channels,
                    sample_format=proto.SampleFormat.SIGNED_16_BIT,
                )
            )
            msg = proto.ServiceBoundMessage(reconfigure_session_request=reconfig)
            self._send_message(msg)
            logger.debug(f"reconfigured audio to {frame.sample_rate}Hz, {frame.num_channels}ch")

        self._packet_id_counter += 1
        user_input = proto.UserInput(
            packet_id=self._packet_id_counter,
            mode=proto.InferenceTriggerMode.IMMEDIATE,
            audio_data=proto.AudioData(data=frame.data.tobytes()),
        )
        msg = proto.ServiceBoundMessage(user_input=user_input)
        self._send_message(msg)

    def push_video(self, frame: rtc.VideoFrame) -> None:
        """Push video frame. Not supported by Deepslate."""
        logger.warning("Deepslate does not support video input")

    def send_text(
        self,
        text: str,
        mode: proto.InferenceTriggerMode = proto.InferenceTriggerMode.NO_TRIGGER,
    ) -> None:
        """Send text input to Deepslate."""
        self._ensure_session_initialized()

        self._packet_id_counter += 1
        user_input = proto.UserInput(
            packet_id=self._packet_id_counter,
            mode=mode,
            text_data=proto.TextData(data=text),
        )
        msg = proto.ServiceBoundMessage(user_input=user_input)
        self._send_message(msg)

    def generate_reply(
        self, *, instructions: NotGivenOr[str] = NOT_GIVEN
    ) -> asyncio.Future[GenerationCreatedEvent]:
        """Request model to generate a reply."""
        fut: asyncio.Future[GenerationCreatedEvent] = asyncio.Future()
        request_id = utils.shortuuid("gen_")
        self._response_created_futures[request_id] = fut
        self._pending_user_generation = True

        if utils.is_given(instructions):
            self._instructions = instructions

        if self._pending_user_text:
            if utils.is_given(instructions):
                self.send_text(self._pending_user_text, mode=proto.InferenceTriggerMode.NO_TRIGGER)
                trigger = proto.TriggerInference(extra_instructions=instructions)
                msg = proto.ServiceBoundMessage(trigger_inference=trigger)
                self._send_auto_initialize(msg)
            else:
                self.send_text(self._pending_user_text, mode=proto.InferenceTriggerMode.IMMEDIATE)
            self._pending_user_text = None
        else:
            trigger = proto.TriggerInference()
            if utils.is_given(instructions):
                trigger = proto.TriggerInference(extra_instructions=instructions)
            msg = proto.ServiceBoundMessage(trigger_inference=trigger)
            self._send_auto_initialize(msg)

        timeout = self._opts.generate_reply_timeout
        if timeout > 0:
            loop = asyncio.get_event_loop()

            def _on_timeout() -> None:
                if not fut.done():
                    fut.set_exception(
                        TimeoutError(f"generate_reply timed out after {timeout}s")
                    )

            handle = loop.call_later(timeout, _on_timeout)
            fut.add_done_callback(lambda _: handle.cancel())

        return fut

    def commit_audio(self) -> None:
        """Commit audio buffer. Deepslate uses VAD for auto-commit."""
        pass

    def clear_audio(self) -> None:
        """Clear audio buffer. TODO: Backend doesn't support yet."""
        logger.warning("clear_audio not yet supported by Deepslate backend")

    def interrupt(self) -> None:
        """Interrupt current generation."""
        if self._current_generation:
            self._close_current_generation()

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Truncate message. Deepslate does automatic truncation server side."""
        pass

    async def aclose(self) -> None:
        """Close the session."""
        self._msg_ch.close()
        if self._current_generation:
            with contextlib.suppress(asyncio.InvalidStateError):
                self._current_generation.done_fut.set_result(None)
        await utils.aio.cancel_and_wait(self._main_atask)

    # --- Internal methods ---

    def _send_message(self, msg: proto.ServiceBoundMessage) -> None:
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(msg)

    def _send_auto_initialize(self, msg: proto.ServiceBoundMessage) -> None:
        self._ensure_session_initialized()
        self._send_message(msg)

    def _ensure_session_initialized(self) -> None:
        """Ensure session is initialized, using defaults if no audio detected."""
        if self._session_initialized:
            return

        self._detected_sample_rate = 24000
        self._detected_num_channels = 1

        self._queue_initialize_session()
        self._session_initialized = True
        logger.debug("session initialized for text-only mode (24kHz mono)")

    def _queue_initialize_session(self) -> None:
        """Queue the InitializeSessionRequest message."""
        if self._detected_sample_rate is None or self._detected_num_channels is None:
            logger.error("cannot initialize session: audio format not detected")
            return

        tts_config = None
        if self._realtime_model._tts_config is not None:
            el_config = self._realtime_model._tts_config
            eleven_labs_config = proto.ElevenLabsTtsConfiguration(
                api_key=el_config.api_key,
                voice_id=el_config.voice_id,
                location=ELEVENLABS_LOCATION_MAP[el_config.location],
            )
            if el_config.model_id:
                eleven_labs_config.model_id = el_config.model_id
            tts_config = proto.TtsConfiguration(eleven_labs=eleven_labs_config)

        init_request = proto.InitializeSessionRequest(
            input_audio_line=proto.AudioLineConfiguration(
                sample_rate=self._detected_sample_rate,
                channel_count=self._detected_num_channels,
                sample_format=proto.SampleFormat.SIGNED_16_BIT,
            ),
            output_audio_line=proto.AudioLineConfiguration(
                sample_rate=self._detected_sample_rate,
                channel_count=self._detected_num_channels,
                sample_format=proto.SampleFormat.SIGNED_16_BIT,
            ),
            vad_configuration=proto.VadConfiguration(
                confidence_threshold=self._vad_config.confidence_threshold,
                min_volume=self._vad_config.min_volume,
                start_duration=duration_from_ms(self._vad_config.start_duration_ms),
                stop_duration=duration_from_ms(self._vad_config.stop_duration_ms),
                backbuffer_duration=duration_from_ms(self._vad_config.backbuffer_duration_ms),
            ),
            inference_configuration=proto.InferenceConfiguration(
                system_prompt=self._opts.system_prompt,
            ),
            tts_configuration=tts_config,
        )

        msg = proto.ServiceBoundMessage(initialize_session_request=init_request)
        logger.debug(
            f"initializing session: {self._detected_sample_rate}Hz, {self._detected_num_channels}ch"
        )
        self._send_message(msg)

        for pending_msg in self._pending_messages:
            self._send_message(pending_msg)
        self._pending_messages.clear()

    def _send_tool_response(self, call_id: str, result: str) -> None:
        """Send tool call response back to Deepslate."""
        response = proto.ToolCallResponse(id=call_id, result=result)
        msg = proto.ServiceBoundMessage(tool_call_response=response)
        self._send_message(msg)

    def _create_generation(self) -> None:
        """Create a new response generation."""
        is_user_initiated = self._pending_user_generation
        self._pending_user_generation = False

        response_id = utils.shortuuid("resp_")

        self._current_generation = _ResponseGeneration(
            message_ch=utils.aio.Chan(),
            function_ch=utils.aio.Chan(),
            text_ch=utils.aio.Chan(),
            audio_ch=utils.aio.Chan(),
            done_fut=asyncio.Future(),
            response_id=response_id,
            created_timestamp=time.time(),
        )

        has_audio = self._realtime_model._tts_config is not None
        msg_modalities: asyncio.Future[list[Literal["text", "audio"]]] = asyncio.Future()
        if has_audio:
            msg_modalities.set_result(["audio", "text"])
        else:
            msg_modalities.set_result(["text"])
            self._current_generation.audio_ch.close()

        self._current_generation.message_ch.send_nowait(
            MessageGeneration(
                message_id=response_id,
                text_stream=self._current_generation.text_ch,
                audio_stream=self._current_generation.audio_ch,
                modalities=msg_modalities,
            )
        )

        generation_ev = GenerationCreatedEvent(
            message_stream=self._current_generation.message_ch,
            function_stream=self._current_generation.function_ch,
            user_initiated=is_user_initiated,
            response_id=response_id,
        )

        self.emit("generation_created", generation_ev)

        for fut in list(self._response_created_futures.values()):
            if not fut.done():
                fut.set_result(generation_ev)
        self._response_created_futures.clear()

    def _close_current_generation(self) -> None:
        """Close the current generation and its channels."""
        if self._current_generation is None:
            return

        self._current_generation.text_ch.close()
        self._current_generation.audio_ch.close()
        self._current_generation.function_ch.close()
        self._current_generation.message_ch.close()

        with contextlib.suppress(asyncio.InvalidStateError):
            self._current_generation.done_fut.set_result(None)

        self._current_generation = None

    # --- WebSocket lifecycle ---

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        """Main event loop with reconnection support."""

        async def _on_fatal_error(e: Exception) -> None:
            self.emit(
                "error",
                llm.RealtimeModelError(
                    timestamp=time.time(),
                    label=self._realtime_model.label,
                    error=e,
                    recoverable=False,
                ),
            )

        await self._realtime_model._client.run_with_retry(
            self._run_ws,
            should_continue=lambda: not self._msg_ch.closed,
            on_fatal_error=_on_fatal_error,
        )

    async def _run_ws(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        """Run WebSocket communication tasks."""
        closing = False

        @utils.log_exceptions(logger=logger)
        async def _send_task() -> None:
            nonlocal closing
            async for msg in self._msg_ch:
                try:
                    data = msg.SerializeToString()
                    await ws_conn.send_bytes(data)
                    self.emit("deepslate_client_event_sent", msg)
                except Exception as e:
                    logger.error(f"failed to send message: {e}")
                    break

            closing = True
            await ws_conn.close()

        @utils.log_exceptions(logger=logger)
        async def _recv_task() -> None:
            while True:
                msg = await ws_conn.receive()

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing:
                        return
                    close_code = ws_conn.close_code
                    close_reason = msg.extra if msg.extra else "no reason provided"
                    logger.error(
                        f"WebSocket closed: code={close_code}, reason={close_reason}"
                    )
                    raise aiohttp.ClientError(
                        f"WebSocket connection closed unexpectedly: code={close_code}, reason={close_reason}"
                    )

                if msg.type == aiohttp.WSMsgType.ERROR:
                    raise aiohttp.ClientError(f"WebSocket error: {ws_conn.exception()}")

                if msg.type != aiohttp.WSMsgType.BINARY:
                    logger.warning(f"unexpected message type: {msg.type}")
                    continue

                client_msg = proto.ClientBoundMessage()
                client_msg.ParseFromString(msg.data)
                self.emit("deepslate_server_event_received", client_msg)

                self._handle_server_message(client_msg)

        tasks = [
            asyncio.create_task(_send_task(), name="_send_task"),
            asyncio.create_task(_recv_task(), name="_recv_task"),
        ]

        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await ws_conn.close()

    def _handle_server_message(self, msg: proto.ClientBoundMessage) -> None:
        """Handle messages from Deepslate server."""
        payload_type = msg.WhichOneof("payload")

        if payload_type == "model_text_fragment":
            self._handle_model_text_fragment(msg.model_text_fragment)
        elif payload_type == "model_audio_chunk":
            self._handle_model_audio_chunk(msg.model_audio_chunk)
        elif payload_type == "tool_call_request":
            self._handle_tool_call_request(msg.tool_call_request)
        elif payload_type == "playback_clear_buffer":
            self._handle_playback_clear_buffer(msg.playback_clear_buffer)
        elif payload_type == "response_begin":
            self._handle_response_begin()
        elif payload_type == "response_end":
            self._handle_response_end()
        else:
            logger.debug(f"unhandled message type: {payload_type}")

    def _handle_model_text_fragment(self, fragment: proto.ModelTextFragment) -> None:
        """Handle streaming text output from model."""
        if self._current_generation is None:
            self._create_generation()

        if self._current_generation is None:
            logger.warning("failed to create generation for text fragment")
            return

        self._current_generation.text_ch.send_nowait(fragment.text)
        self._current_generation.audio_transcript += fragment.text

        if self._current_generation.first_token_timestamp is None:
            self._current_generation.first_token_timestamp = time.time()

    def _handle_model_audio_chunk(self, chunk: proto.ModelAudioChunk) -> None:
        """Handle streaming TTS audio output from Deepslate."""
        if self._current_generation is None:
            self._create_generation()

        if self._current_generation is None:
            logger.warning("failed to create generation for audio chunk")
            return

        if not chunk.audio or not chunk.audio.data:
            return

        audio_bytes = chunk.audio.data

        frame = rtc.AudioFrame(
            data=audio_bytes,
            sample_rate=self._detected_sample_rate or 24000,
            num_channels=self._detected_num_channels or 1,
            samples_per_channel=len(audio_bytes) // 2,
        )

        self._current_generation.audio_ch.send_nowait(frame)

        if self._current_generation.first_token_timestamp is None:
            self._current_generation.first_token_timestamp = time.time()

        if chunk.transcript:
            self._current_generation.text_ch.send_nowait(chunk.transcript)
            self._current_generation.audio_transcript += chunk.transcript

    def _handle_tool_call_request(self, request: proto.ToolCallRequest) -> None:
        """Handle tool call requests from model."""
        if self._current_generation is None:
            self._create_generation()

        if self._current_generation is None:
            return

        params_dict = {}
        if request.HasField("parameters"):
            params_dict = struct_to_dict(request.parameters)

        self._current_generation.function_ch.send_nowait(
            FunctionCall(
                call_id=request.id,
                name=request.name,
                arguments=json.dumps(params_dict),
            )
        )

        logger.debug(f"tool call request: {request.name}({request.id})")
        self._close_current_generation()

    def _handle_playback_clear_buffer(self, _: proto.PlaybackClearBuffer) -> None:
        """Handle playback clear buffer signal."""
        if self._current_generation is not None:
            self.emit("input_speech_started", InputSpeechStartedEvent())
            self._close_current_generation()

    def _handle_response_begin(self) -> None:
        """Handle response begin signal."""
        if self._current_generation is None:
            self._create_generation()

    def _handle_response_end(self) -> None:
        """Handle response end - close the current generation."""
        self._close_current_generation()