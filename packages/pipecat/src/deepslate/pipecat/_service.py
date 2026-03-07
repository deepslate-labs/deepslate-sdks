import asyncio
import json
from typing import Any, List, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    OutputAudioRawFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    StartFrame,
    TextFrame,
)
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams, LLMService

from deepslate.core._utils import build_initialize_request, dict_to_struct, struct_to_dict
from deepslate.core.client import BaseDeepslateClient
from deepslate.core.options import DeepslateOptions, ElevenLabsTtsConfig, VadConfig
from deepslate.core.proto import realtime_pb2 as proto
from .frames import DeepslateDirectSpeechFrame


class DeepslateRealtimeLLMService(LLMService):
    """Pipecat service for Deepslate's end-to-end Speech-to-Speech Realtime API."""

    def __init__(
        self,
        options: DeepslateOptions,
        vad_config: Optional[VadConfig] = None,
        tts_config: Optional[ElevenLabsTtsConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._opts = options
        self._vad_config = vad_config or VadConfig()
        self._tts_config = tts_config

        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._client: Optional[BaseDeepslateClient] = None

        self._receive_task: Optional[asyncio.Task] = None
        self._main_task: Optional[asyncio.Task] = None
        self._session_initialized = False
        self._should_stop = False
        self._tools: List[dict] = []

        self._detected_sample_rate: Optional[int] = None
        self._detected_num_channels: Optional[int] = None
        self._packet_id_counter: int = 0

    async def start(self, frame: StartFrame):
        """Starts the Pipecat service and initializes the WebSocket connection."""
        await super().start(frame)
        self._client = BaseDeepslateClient(self._opts, user_agent="PipecatDeepslate/1.0")
        self._should_stop = False
        self._main_task = asyncio.create_task(self._main_event_loop())

    async def stop(self, frame: EndFrame):
        """Stops the Pipecat service."""
        self._should_stop = True
        await self._disconnect()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancels the Pipecat service immediately."""
        self._should_stop = True
        await self._disconnect()
        await super().cancel(frame)

    async def _main_event_loop(self):
        """Main event loop with reconnection support."""

        async def _on_fatal_error(e: Exception) -> None:
            error_msg = f"Connection failed: {e}"
            await self.push_frame(ErrorFrame(error_msg))

        await self._client.run_with_retry(
            self._run_session,
            should_continue=lambda: not self._should_stop,
            on_fatal_error=_on_fatal_error,
        )
        logger.info("Main event loop terminated")

    async def _run_session(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Run one WebSocket session until it closes."""
        self._ws = ws
        logger.info("Successfully connected to Deepslate Realtime API.")
        self._receive_task = asyncio.create_task(self._receive_loop())
        try:
            await self._receive_task
        finally:
            self._ws = None
            self._session_initialized = False
            if not self._should_stop:
                logger.info("WebSocket connection closed, attempting to reconnect...")

    async def _disconnect(self):
        """Close tasks and connections cleanly."""
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._main_task and not self._main_task.done():
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass

        if self._ws and not self._ws.closed:
            await self._ws.close()

        if self._client is not None:
            await self._client.aclose()
            self._client = None

        self._session_initialized = False

    async def process_frame(self, frame: Frame, direction: Any):
        """Handle incoming frames from the Pipecat pipeline."""
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            await self._handle_audio_input(frame)

        elif isinstance(frame, TextFrame):
            await self._handle_text_input(frame.text)

        elif isinstance(frame, FunctionCallResultFrame):
            await self._handle_function_result(frame)

        elif isinstance(frame, LLMSetToolsFrame):
            # Capture tool definitions and sync with server.
            # In pipecat >=0.0.104, frame.tools may be a ToolsSchema object
            # instead of a plain List[dict].  Normalise to List[dict] so that
            # _sync_tools can iterate and call .get() on each entry.
            if isinstance(frame.tools, ToolsSchema):
                self._tools = [
                    {"type": "function", "function": schema.to_default_dict()}
                    for schema in frame.tools.standard_tools
                ]
            else:
                self._tools = frame.tools
            if self._session_initialized:
                await self._sync_tools()

        elif isinstance(frame, LLMMessagesAppendFrame):
            await self._handle_messages_append(frame)

        elif isinstance(frame, LLMMessagesUpdateFrame):
            await self._handle_messages_update(frame)

        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._handle_update_settings(frame)

        elif isinstance(frame, DeepslateDirectSpeechFrame):
            await self._handle_direct_speech(frame)

        else:
            await self.push_frame(frame, direction)

    async def _sync_tools(self):
        """Syncs tool definitions with the Deepslate server."""
        if not self._ws or not self._tools:
            return

        tool_definitions = []
        for tool in self._tools:
            func = tool.get("function", {})
            tool_definitions.append(proto.ToolDefinition(
                name=func.get("name"),
                description=func.get("description", ""),
                parameters=dict_to_struct(func.get("parameters", {})),
            ))

        update_request = proto.UpdateToolDefinitionsRequest(tool_definitions=tool_definitions)
        await self._send_msg(proto.ServiceBoundMessage(update_tool_definitions_request=update_request))

    async def _handle_messages_append(self, frame: LLMMessagesAppendFrame):
        """Handle LLMMessagesAppendFrame by injecting messages into the active session."""
        if not self._ws:
            return

        if not self._session_initialized:
            self._detected_sample_rate = 16000
            self._detected_num_channels = 1
            await self._send_initialize_session()
            self._session_initialized = True

        system_parts: List[str] = []

        for message in frame.messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if isinstance(content, list):
                text_parts = [
                    block.get("text", "") for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                content = " ".join(text_parts)

            if not content:
                continue

            if role == "user":
                self._packet_id_counter += 1
                user_input = proto.UserInput(
                    packet_id=self._packet_id_counter,
                    mode=proto.InferenceTriggerMode.NO_TRIGGER,
                    text_data=proto.TextData(data=content),
                )
                await self._send_msg(proto.ServiceBoundMessage(user_input=user_input))

            elif role == "system":
                system_parts.append(content)

            else:
                logger.warning(
                    f"LLMMessagesAppendFrame: role '{role}' cannot be injected into "
                    "Deepslate session — only 'user' and 'system' roles are supported."
                )

        if frame.run_llm and self._session_initialized and self._ws:
            extra = " ".join(system_parts) if system_parts else None
            trigger = proto.TriggerInference()
            if extra:
                trigger.extra_instructions = extra
            await self._send_msg(proto.ServiceBoundMessage(trigger_inference=trigger))

    async def _handle_messages_update(self, frame: LLMMessagesUpdateFrame):
        """Handle LLMMessagesUpdateFrame by re-syncing tools and optionally triggering inference."""
        if self._session_initialized:
            await self._sync_tools()

        if frame.run_llm and self._session_initialized and self._ws:
            await self._send_msg(proto.ServiceBoundMessage(trigger_inference=proto.TriggerInference()))

    async def _handle_direct_speech(self, frame: DeepslateDirectSpeechFrame):
        """Send a DirectSpeech message to bypass the LLM and speak text via TTS."""
        if not self._ws:
            return

        if not self._session_initialized:
            self._detected_sample_rate = 16000
            self._detected_num_channels = 1
            await self._send_initialize_session()
            self._session_initialized = True

        direct_speech = proto.DirectSpeech(
            text=frame.text,
            include_in_history=frame.include_in_history,
        )
        await self._send_msg(proto.ServiceBoundMessage(direct_speech=direct_speech))

    async def _handle_update_settings(self, frame: LLMUpdateSettingsFrame):
        """Apply runtime setting changes. Currently handles system_prompt."""
        new_prompt = frame.settings.get("system_prompt")
        if new_prompt is None:
            return
        self._opts.system_prompt = new_prompt
        if self._session_initialized:
            await self._sync_system_prompt()

    async def _sync_system_prompt(self):
        """Send the current system prompt to the server via ReconfigureSessionRequest."""
        if not self._ws:
            return
        reconfig = proto.ReconfigureSessionRequest(
            inference_configuration=proto.InferenceConfiguration(
                system_prompt=self._opts.system_prompt,
            )
        )
        await self._send_msg(proto.ServiceBoundMessage(reconfigure_session_request=reconfig))

    async def _handle_audio_input(self, frame: AudioRawFrame):
        """Forward PCM audio from Pipecat to Deepslate."""
        if not self._ws:
            return

        if not self._session_initialized:
            self._detected_sample_rate = frame.sample_rate
            self._detected_num_channels = frame.num_channels
            await self._send_initialize_session()
            self._session_initialized = True

        elif (frame.sample_rate != self._detected_sample_rate or
              frame.num_channels != self._detected_num_channels):
            self._detected_sample_rate = frame.sample_rate
            self._detected_num_channels = frame.num_channels

            reconfig = proto.ReconfigureSessionRequest(
                input_audio_line=proto.AudioLineConfiguration(
                    sample_rate=frame.sample_rate,
                    channel_count=frame.num_channels,
                    sample_format=proto.SampleFormat.SIGNED_16_BIT,
                )
            )
            await self._send_msg(proto.ServiceBoundMessage(reconfigure_session_request=reconfig))

        self._packet_id_counter += 1
        user_input = proto.UserInput(
            packet_id=self._packet_id_counter,
            mode=proto.InferenceTriggerMode.IMMEDIATE,
            audio_data=proto.AudioData(data=frame.audio),
        )
        await self._send_msg(proto.ServiceBoundMessage(user_input=user_input))

    async def _handle_text_input(self, text: str):
        """Forward Text frames as trigger inputs."""
        if not self._ws:
            return

        if not self._session_initialized:
            self._detected_sample_rate = 16000
            self._detected_num_channels = 1
            await self._send_initialize_session()
            self._session_initialized = True

        self._packet_id_counter += 1
        user_input = proto.UserInput(
            packet_id=self._packet_id_counter,
            mode=proto.InferenceTriggerMode.IMMEDIATE,
            text_data=proto.TextData(data=text),
        )
        await self._send_msg(proto.ServiceBoundMessage(user_input=user_input))

    async def _dispatch_function_call(self, call_id: str, function_name: str, args: dict):
        """Look up a registered function handler and execute it."""
        item = self._functions.get(function_name) or self._functions.get(None)
        if not item:
            logger.warning(f"Received tool call for unregistered function: '{function_name}'")
            return

        async def result_callback(result, *, properties=None):
            result_str = result if isinstance(result, str) else json.dumps(result)
            response = proto.ToolCallResponse(id=call_id, result=result_str)
            await self._send_msg(proto.ServiceBoundMessage(tool_call_response=response))

        try:
            await item.handler(FunctionCallParams(
                function_name=function_name,
                tool_call_id=call_id,
                arguments=args,
                llm=self,
                context=None,
                result_callback=result_callback,
            ))
        except Exception as e:
            logger.error(f"Error executing function '{function_name}': {e}")

    async def _handle_function_result(self, frame: FunctionCallResultFrame):
        """Forward function return values to Deepslate."""
        if not self._ws:
            return

        result_str = frame.result if isinstance(frame.result, str) else json.dumps(frame.result)
        response = proto.ToolCallResponse(id=frame.tool_call_id, result=result_str)
        await self._send_msg(proto.ServiceBoundMessage(tool_call_response=response))

    async def _send_initialize_session(self):
        """Constructs and sends the initialize payload based on config."""
        init_request = build_initialize_request(
            sample_rate=self._detected_sample_rate,
            num_channels=self._detected_num_channels,
            vad_config=self._vad_config,
            system_prompt=self._opts.system_prompt,
            tts_config=self._tts_config,
        )

        msg = proto.ServiceBoundMessage(initialize_session_request=init_request)
        await self._send_msg(msg)

        if self._tools:
            await self._sync_tools()

    async def _send_msg(self, msg: proto.ServiceBoundMessage):
        """Serialize and send a proto message over the websocket."""
        if not self._ws or self._ws.closed:
            logger.debug("Dropping outbound message: WebSocket is not open.")
            return
        try:
            await self._ws.send_bytes(msg.SerializeToString())
        except Exception as e:
            logger.error(f"Error sending message to Deepslate: {e}")

    async def _receive_loop(self):
        """Long running task to receive and handle websocket messages from Deepslate."""
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    client_msg = proto.ClientBoundMessage()
                    client_msg.ParseFromString(msg.data)
                    await self._handle_server_message(client_msg)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.warning(f"WebSocket closed or errored: {msg.data}")
                    break
        except Exception as e:
            logger.error(f"WebSocket receive error: {e}")

    async def _handle_server_message(self, msg: proto.ClientBoundMessage):
        """Map server protobuf events to Pipecat Frames."""
        payload_type = msg.WhichOneof("payload")

        if payload_type == "response_begin":
            await self.push_frame(LLMFullResponseStartFrame())

        elif payload_type == "response_end":
            await self.push_frame(LLMFullResponseEndFrame())

        elif payload_type == "model_text_fragment":
            text = msg.model_text_fragment.text
            await self.push_frame(LLMTextFrame(text))

        elif payload_type == "model_audio_chunk":
            audio_bytes = msg.model_audio_chunk.audio.data
            transcript = msg.model_audio_chunk.transcript

            frame = OutputAudioRawFrame(
                audio=audio_bytes,
                sample_rate=self._detected_sample_rate or 16000,
                num_channels=self._detected_num_channels or 1,
            )
            await self.push_frame(frame)

            if transcript:
                await self.push_frame(LLMTextFrame(transcript))

        elif payload_type == "playback_clear_buffer":
            await self.push_frame(InterruptionFrame())

        elif payload_type == "tool_call_request":
            req = msg.tool_call_request
            args_dict = struct_to_dict(req.parameters) if req.HasField("parameters") else {}
            asyncio.create_task(self._dispatch_function_call(req.id, req.name, args_dict))

        elif payload_type == "error":
            notification = msg.error
            category_name = proto.SessionErrorCategory.Name(notification.category)
            trace_id = notification.trace_id if notification.HasField("trace_id") else None
            trace_suffix = f" (trace_id={trace_id})" if trace_id else ""
            error_msg = f"[Deepslate] {category_name}: {notification.message}{trace_suffix}"
            logger.error(error_msg)
            await self.push_frame(ErrorFrame(error_msg))