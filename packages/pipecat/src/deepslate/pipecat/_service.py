import asyncio
import time
from typing import Any, List, Optional

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

from deepslate.core import (
    DeepslateOptions,
    DeepslateSession,
    ElevenLabsTtsConfig,
    InferenceTriggerMode,
    VadConfig,
)
from .frames import (
    DeepslateExportChatHistoryFrame,
    DeepslateChatHistoryFrame,
    DeepslateDirectSpeechFrame,
    DeepslateUserTranscriptionFrame,
    DeepslateModelTranscriptionFrame,
    DeepslateConversationQueryFrame,
    DeepslateConversationQueryResultFrame,
)


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

        self._session: Optional[DeepslateSession] = None
        self._tools: List[dict] = []

    async def start(self, frame: StartFrame):
        """Starts the Pipecat service and opens the Deepslate connection."""
        await super().start(frame)
        self._session = DeepslateSession.create(
            self._opts,
            vad_config=self._vad_config,
            tts_config=self._tts_config,
            user_agent="PipecatDeepslate/1.0",
            on_text_fragment=self._on_text_fragment,
            on_audio_chunk=self._on_audio_chunk,
            on_tool_call=self._on_tool_call,
            on_error=self._on_error,
            on_response_begin=self._on_response_begin,
            on_response_end=self._on_response_end,
            on_user_transcription=self._on_user_transcription,
            on_playback_buffer_clear=self._on_playback_buffer_clear,
            on_chat_history=self._on_chat_history,
            on_conversation_query_result=self._on_conversation_query_result,
            on_fatal_error=self._on_fatal_error,
        )
        self._session.start()

    async def stop(self, frame: EndFrame):
        """Stops the Pipecat service."""
        await self._disconnect()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancels the Pipecat service immediately."""
        await self._disconnect()
        await super().cancel(frame)

    async def _disconnect(self):
        if self._session is not None:
            await self._session.close()  # also closes the owned client
            self._session = None

    async def process_frame(self, frame: Frame, direction: Any):
        """Handle incoming frames from the Pipecat pipeline."""
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            if self._session is not None:
                await self._session.send_audio(
                    frame.audio, frame.sample_rate, frame.num_channels
                )

        elif isinstance(frame, TextFrame):
            if self._session is not None:
                await self._session.send_text(frame.text)

        elif isinstance(frame, FunctionCallResultFrame):
            if self._session is not None:
                await self._session.send_tool_response(frame.tool_call_id, frame.result)

        elif isinstance(frame, LLMSetToolsFrame):
            # Normalise ToolsSchema (pipecat >=0.0.104) to list[dict].
            if isinstance(frame.tools, ToolsSchema):
                self._tools = [
                    {"type": "function", "function": schema.to_default_dict()}
                    for schema in frame.tools.standard_tools
                ]
            else:
                self._tools = frame.tools
            if self._session is not None:
                await self._session.update_tools(self._tools)

        elif isinstance(frame, LLMMessagesAppendFrame):
            await self._handle_messages_append(frame)

        elif isinstance(frame, LLMMessagesUpdateFrame):
            await self._handle_messages_update(frame)

        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._handle_update_settings(frame)

        elif isinstance(frame, DeepslateExportChatHistoryFrame):
            if self._session is not None:
                await self._session.export_chat_history(frame.await_pending)

        elif isinstance(frame, DeepslateDirectSpeechFrame):
            if self._session is not None:
                await self._session.send_direct_speech(frame.text, frame.include_in_history)

        elif isinstance(frame, DeepslateConversationQueryFrame):
            if self._session is not None:
                # Pipecat result frames don't require correlation by ID; pass
                # an empty string — on_conversation_query_result ignores it.
                await self._session.send_conversation_query(
                    "", frame.prompt, frame.instructions
                )

        else:
            await self.push_frame(frame, direction)

    async def _on_response_begin(self) -> None:
        await self.push_frame(LLMFullResponseStartFrame())

    async def _on_response_end(self) -> None:
        await self.push_frame(LLMFullResponseEndFrame())

    async def _on_text_fragment(self, text: str) -> None:
        await self.push_frame(LLMTextFrame(text))

    async def _on_audio_chunk(
        self,
        pcm_bytes: bytes,
        sample_rate: int,
        num_channels: int,
        transcript: Optional[str],
    ) -> None:
        await self.push_frame(
            OutputAudioRawFrame(
                audio=pcm_bytes,
                sample_rate=sample_rate,
                num_channels=num_channels,
            )
        )
        if transcript:
            await self.push_frame(DeepslateModelTranscriptionFrame(text=transcript))

    async def _on_tool_call(
        self, call_id: str, function_name: str, params: dict
    ) -> None:
        asyncio.create_task(
            self._dispatch_function_call(call_id, function_name, params)
        )

    async def _on_error(
        self, category: str, message: str, trace_id: Optional[str]
    ) -> None:
        trace_suffix = f" (trace_id={trace_id})" if trace_id else ""
        await self.push_frame(
            ErrorFrame(f"[Deepslate] {category}: {message}{trace_suffix}")
        )

    async def _on_user_transcription(self, text: str, language: Optional[str], turn_id: int = 0) -> None:
        await self.push_frame(
            DeepslateUserTranscriptionFrame(
                text=text,
                user_id="user",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                language=language,
            )
        )

    async def _on_playback_buffer_clear(self) -> None:
        await self.push_frame(InterruptionFrame())

    async def _on_chat_history(self, messages) -> None:
        await self.push_frame(DeepslateChatHistoryFrame(messages=messages))

    async def _on_conversation_query_result(self, query_id: str, text: str) -> None:
        await self.push_frame(DeepslateConversationQueryResultFrame(text=text))

    async def _on_fatal_error(self, e: Exception) -> None:
        await self.push_frame(ErrorFrame(f"Connection failed: {e}"))

    async def _handle_messages_append(self, frame: LLMMessagesAppendFrame):
        """Inject messages into the active session."""
        if self._session is None:
            return

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
                await self._session.send_text(
                    content, trigger=InferenceTriggerMode.NO_TRIGGER
                )
            elif role == "system":
                system_parts.append(content)
            else:
                logger.warning(
                    f"LLMMessagesAppendFrame: role '{role}' cannot be injected into "
                    "Deepslate session — only 'user' and 'system' roles are supported."
                )

        if frame.run_llm:
            extra = " ".join(system_parts) if system_parts else None
            await self._session.trigger_inference(instructions=extra)

    async def _handle_messages_update(self, frame: LLMMessagesUpdateFrame):
        """Re-sync tools and optionally trigger inference."""
        if self._session is None:
            return
        await self._session.update_tools(self._tools)
        if frame.run_llm:
            await self._session.trigger_inference()

    async def _handle_update_settings(self, frame: LLMUpdateSettingsFrame):
        """Apply runtime changes to system_prompt and/or temperature."""
        if self._session is None:
            return
        new_prompt = frame.settings.get("system_prompt")
        new_temp = frame.settings.get("temperature")
        if new_prompt is not None:
            self._opts.system_prompt = new_prompt
        if new_temp is not None:
            self._opts.temperature = new_temp
        if (
            (new_prompt is not None or new_temp is not None)
            and self._session.session_initialized
        ):
            await self._session.reconfigure(
                system_prompt=new_prompt,
                temperature=new_temp,
            )

    async def _dispatch_function_call(
        self, call_id: str, function_name: str, args: dict
    ):
        """Look up a registered function handler and execute it."""
        item = self._functions.get(function_name) or self._functions.get(None)
        if not item:
            logger.warning(
                f"Received tool call for unregistered function: '{function_name}'"
            )
            return

        async def result_callback(result, *, properties=None):
            if self._session is not None:
                await self._session.send_tool_response(call_id, result)

        try:
            await item.handler(
                FunctionCallParams(
                    function_name=function_name,
                    tool_call_id=call_id,
                    arguments=args,
                    llm=self,
                    context=None,
                    result_callback=result_callback,
                )
            )
        except Exception as e:
            logger.error(f"Error executing function '{function_name}': {e}")