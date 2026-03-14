from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

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

from deepslate.core import (
    BaseDeepslateClient,
    DeepslateOptions,
    DeepslateSession,
    ElevenLabsTtsConfig,
    InferenceTriggerMode,
    VadConfig,
)

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
        temperature: float = 1.0,
        generate_reply_timeout: float = 30.0,
        # VAD configuration
        vad_confidence_threshold: float = 0.5,
        vad_min_volume: float = 0.01,
        vad_start_duration_ms: int = 200,
        vad_stop_duration_ms: int = 500,
        vad_backbuffer_duration_ms: int = 1000,
        # TTS configuration
        tts_config: ElevenLabsTtsConfig | None = None,
        http_session: Any = None,
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
            temperature: Sampling temperature (0.0 to 2.0). Higher values produce more random output.
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
            temperature=temperature,
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

    def session(self) -> "DeepslateRealtimeSession":
        """Create a new Deepslate real-time session."""
        return DeepslateRealtimeSession(realtime_model=self)

    def update_options(
        self,
        *,
        system_prompt: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """Update model options.

        Changes take effect on the next session initialization (e.g., after reconnect).
        To apply immediately to an active session use
        ``DeepslateRealtimeSession.update_instructions()`` or send a
        ``ReconfigureSessionRequest`` via the session.
        """
        if utils.is_given(system_prompt):
            self._opts.system_prompt = system_prompt
        if utils.is_given(temperature):
            self._opts.temperature = temperature

    async def aclose(self) -> None:
        await self._client.aclose()


class DeepslateRealtimeSession(
    llm.RealtimeSession[
        Literal[
            "deepslate_server_event_received",
            "deepslate_client_event_sent",
            "user_transcription",
            "audio_transcript",
        ]
    ]
):
    """A session for the Deepslate Realtime API.

    Wraps ``DeepslateSession`` from deepslate-core and translates its
    callbacks into LiveKit agent events and channel writes.  All
    protobuf details are encapsulated in the core session; this class
    contains only LiveKit-specific logic.
    """

    def __init__(self, realtime_model: RealtimeModel):
        super().__init__(realtime_model)
        self._realtime_model = realtime_model
        self._opts = realtime_model._opts

        # LiveKit context
        self._tools = llm.ToolContext.empty()
        self._chat_ctx = llm.ChatContext.empty()
        self._instructions: str | None = None

        # Generation tracking
        self._current_generation: _ResponseGeneration | None = None
        self._response_created_futures: dict[str, asyncio.Future[GenerationCreatedEvent]] = {}
        self._pending_user_generation: bool = False
        self._pending_user_text: str | None = None

        # Conversation query tracking: query_id → Future[str]
        self._pending_queries: dict[str, asyncio.Future[str]] = {}

        # Core session — owns the WebSocket lifecycle
        self._session = DeepslateSession(
            client=realtime_model._client,
            options=realtime_model._opts,
            vad_config=realtime_model._vad_config,
            tts_config=realtime_model._tts_config,
            on_text_fragment=self._on_text_fragment,
            on_audio_chunk=self._on_audio_chunk,
            on_tool_call=self._on_tool_call,
            on_error=self._on_error,
            on_response_begin=self._on_response_begin,
            on_response_end=self._on_response_end,
            on_user_transcription=self._on_user_transcription,
            on_interruption=self._on_interruption,
            on_chat_history=self._on_chat_history,
            on_conversation_query_result=self._on_conversation_query_result,
            on_fatal_error=self._on_fatal_error,
        )
        asyncio.ensure_future(self._session.start())

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    @property
    def tools(self) -> ToolContext:
        return self._tools.copy()

    async def update_instructions(self, instructions: str) -> None:
        """Update system prompt for the next session initialization."""
        self._instructions = instructions
        self._opts.system_prompt = instructions
        logger.debug("instructions updated (will take effect on next session)")

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        """Capture new user messages and handle function call outputs."""
        existing_ids = {item.id for item in self._chat_ctx.items}

        for item in chat_ctx.items:
            if item.id not in existing_ids:
                if item.type == "message" and item.role == "user":
                    if text := item.text_content:
                        self._pending_user_text = text
                elif item.type == "function_call_output":
                    await self._session.send_tool_response(item.call_id, item.output)

        self._chat_ctx = chat_ctx.copy()

    async def update_tools(self, tools: list[FunctionTool | RawFunctionTool | Any]) -> None:
        """Sync tool definitions to the server."""
        tools_dicts = []
        for tool in tools:
            if is_function_tool(tool):
                schema = llm.utils.build_legacy_openai_schema(tool, internally_tagged=True)
                tools_dicts.append({
                    "type": "function",
                    "function": {
                        "name": schema["name"],
                        "description": schema.get("description", ""),
                        "parameters": schema.get("parameters", {}),
                    },
                })
            elif is_raw_function_tool(tool):
                info = get_raw_function_info(tool)
                tools_dicts.append({
                    "type": "function",
                    "function": {
                        "name": info.name,
                        "description": info.raw_schema.get("description", ""),
                        "parameters": info.raw_schema.get("parameters", {}),
                    },
                })

        await self._session.update_tools(tools_dicts)
        self._tools = llm.ToolContext(tools)
        logger.debug(f"updated tools: {[t.get('function', {}).get('name') for t in tools_dicts]}")

    def update_options(self, *, tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN) -> None:
        """Dynamic tool_choice updates not supported."""
        pass

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        """Push an audio frame to Deepslate."""
        asyncio.ensure_future(
            self._session.send_audio(
                frame.data.tobytes(),
                frame.sample_rate,
                frame.num_channels,
            )
        )

    def push_video(self, frame: rtc.VideoFrame) -> None:
        """Video input is not supported by Deepslate."""
        logger.warning("Deepslate does not support video input")

    def send_text(
        self,
        text: str,
        mode: InferenceTriggerMode = InferenceTriggerMode.NO_TRIGGER,
    ) -> None:
        """Send text input to Deepslate."""
        asyncio.ensure_future(self._session.initialize())
        asyncio.ensure_future(self._session.send_text(text, trigger=mode))

    def speak_direct(self, text: str, include_in_history: bool = True) -> None:
        """Bypass the LLM and speak text directly via TTS."""
        asyncio.ensure_future(self._session.initialize())
        asyncio.ensure_future(self._session.send_direct_speech(text, include_in_history))

    def query_conversation(
        self,
        prompt: str | None = None,
        instructions: str | None = None,
    ) -> asyncio.Future[str]:
        """Run a one-shot side-channel inference against the current conversation.

        Returns a ``Future`` that resolves to the model's complete text reply.
        """
        query_id = utils.shortuuid("query_")
        fut: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        self._pending_queries[query_id] = fut

        asyncio.ensure_future(self._session.initialize())
        asyncio.ensure_future(
            self._session.send_conversation_query(query_id, prompt, instructions)
        )
        return fut

    def export_chat_history(self, await_pending: bool = False) -> None:
        """Request the server to export the current chat history.

        The result is delivered via the ``chat_history_exported`` event.
        """
        asyncio.ensure_future(self._session.export_chat_history(await_pending))

    def generate_reply(
        self, *, instructions: NotGivenOr[str] = NOT_GIVEN
    ) -> asyncio.Future[GenerationCreatedEvent]:
        """Request the model to generate a reply."""
        fut: asyncio.Future[GenerationCreatedEvent] = asyncio.Future()
        request_id = utils.shortuuid("gen_")
        self._response_created_futures[request_id] = fut
        self._pending_user_generation = True

        if utils.is_given(instructions):
            self._instructions = instructions

        if self._pending_user_text:
            if utils.is_given(instructions):
                asyncio.ensure_future(
                    self._session.send_text(
                        self._pending_user_text,
                        trigger=InferenceTriggerMode.NO_TRIGGER,
                    )
                )
                asyncio.ensure_future(
                    self._session.trigger_inference(instructions=instructions)
                )
            else:
                asyncio.ensure_future(self._session.initialize())
                asyncio.ensure_future(
                    self._session.send_text(
                        self._pending_user_text,
                        trigger=InferenceTriggerMode.IMMEDIATE,
                    )
                )
            self._pending_user_text = None
        else:
            asyncio.ensure_future(self._session.initialize())
            asyncio.ensure_future(
                self._session.trigger_inference(
                    instructions=instructions if utils.is_given(instructions) else None
                )
            )

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
        """Deepslate uses server-side VAD for auto-commit."""
        pass

    def clear_audio(self) -> None:
        """Audio buffer clearing is not yet supported by the Deepslate backend."""
        logger.warning("clear_audio not yet supported by Deepslate backend")

    def interrupt(self) -> None:
        """Interrupt the current generation."""
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
        """Deepslate handles truncation server-side automatically."""
        pass

    async def aclose(self) -> None:
        """Close the session."""
        if self._current_generation:
            with contextlib.suppress(asyncio.InvalidStateError):
                self._current_generation.done_fut.set_result(None)
        await self._session.close()

    async def _on_text_fragment(self, text: str) -> None:
        if self._current_generation is None:
            self._create_generation()
        if self._current_generation is None:
            return
        self._current_generation.text_ch.send_nowait(text)
        self._current_generation.audio_transcript += text
        if self._current_generation.first_token_timestamp is None:
            self._current_generation.first_token_timestamp = time.time()

    async def _on_audio_chunk(
        self,
        pcm_bytes: bytes,
        sample_rate: int,
        num_channels: int,
        transcript: str | None,
    ) -> None:
        if self._current_generation is None:
            self._create_generation()
        if self._current_generation is None:
            return

        frame = rtc.AudioFrame(
            data=pcm_bytes,
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=len(pcm_bytes) // 2,
        )
        self._current_generation.audio_ch.send_nowait(frame)

        if self._current_generation.first_token_timestamp is None:
            self._current_generation.first_token_timestamp = time.time()

        if transcript:
            self._current_generation.audio_transcript += transcript
            self.emit("audio_transcript", transcript)

    async def _on_tool_call(
        self, call_id: str, function_name: str, params: dict
    ) -> None:
        if self._current_generation is None:
            self._create_generation()
        if self._current_generation is None:
            return
        self._current_generation.function_ch.send_nowait(
            FunctionCall(
                call_id=call_id,
                name=function_name,
                arguments=json.dumps(params),
            )
        )
        logger.debug(f"tool call request: {function_name}({call_id})")
        self._close_current_generation()

    async def _on_response_begin(self) -> None:
        if self._current_generation is None:
            self._create_generation()

    async def _on_response_end(self) -> None:
        self._close_current_generation()

    async def _on_interruption(self) -> None:
        if self._current_generation is not None:
            self.emit("input_speech_started", InputSpeechStartedEvent())
            self._close_current_generation()

    async def _on_user_transcription(
        self, text: str, language: str | None, turn_id: int = 0
    ) -> None:
        self.emit(
            "user_transcription",
            SimpleNamespace(text=text, language=language or ""),
        )

    async def _on_chat_history(self, messages) -> None:
        self.emit("chat_history_exported", messages)

    async def _on_conversation_query_result(self, query_id: str, text: str) -> None:
        fut = self._pending_queries.pop(query_id, None)
        if fut is not None and not fut.done():
            fut.set_result(text)
        else:
            logger.warning(
                f"received conversation_query_result for unknown query_id: '{query_id}'"
            )

    async def _on_error(
        self, category: str, message: str, trace_id: str | None
    ) -> None:
        trace_suffix = f" (trace_id={trace_id})" if trace_id else ""
        error_msg = f"[Deepslate] {category}: {message}{trace_suffix}"
        logger.error(error_msg)
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self._realtime_model.label,
                error=RuntimeError(error_msg),
                recoverable=False,
            ),
        )

    async def _on_fatal_error(self, e: Exception) -> None:
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self._realtime_model.label,
                error=e,
                recoverable=False,
            ),
        )

    def _create_generation(self) -> None:
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
        if self._current_generation is None:
            return
        self._current_generation.text_ch.close()
        self._current_generation.audio_ch.close()
        self._current_generation.function_ch.close()
        self._current_generation.message_ch.close()
        with contextlib.suppress(asyncio.InvalidStateError):
            self._current_generation.done_fut.set_result(None)
        self._current_generation = None