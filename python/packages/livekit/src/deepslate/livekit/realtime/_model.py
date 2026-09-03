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

import asyncio
import contextlib
import json
import os
import time
import warnings
from collections import OrderedDict, deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

import aiohttp
from livekit import rtc
from livekit.agents import llm, utils, FunctionTool, NOT_GIVEN, NotGivenOr
from livekit.agents.llm import (
    FunctionCall,
    GenerationCreatedEvent,
    InputSpeechStartedEvent,
    InputSpeechStoppedEvent,
    InputTranscriptionCompleted,
    MessageGeneration,
    RawFunctionTool,
    Tool,
    ToolChoice,
    ToolContext,
)
from livekit.agents.llm.chat_context import Instructions
from livekit.agents.llm.tool_context import (
    get_raw_function_info,
    is_function_tool,
    is_raw_function_tool,
)
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import TimedString

import importlib.metadata

try:
    __version__ = importlib.metadata.version("deepslate-livekit")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from deepslate.core import (
    BaseDeepslateClient,
    ChatMessageDict,
    DeepslateOptions,
    DeepslateSession,
    DeepslateSessionListener,
    ElevenLabsTtsConfig,
    HostedTtsConfig,
    HostedVoiceCloneConfig,
    FunctionToolDict,
    TriggerMode,
    VadConfig,
)

from .._log import logger

DEEPSLATE_BASE_URL = "https://app.deepslate.eu"

SETTLE_GRACE_PERIOD = 0.5

_BYTES_PER_SAMPLE = 2

_SETTLED_GENERATION_LIMIT = 16

ABANDONED_TOOL_RESULT = {
    "error": "tool_call_cancelled",
    "detail": (
        "The user interrupted before this tool call returned, so its result "
        "is unavailable. Respond to what the user just said instead."
    ),
}


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
    turn_id: int = 0
    first_token_timestamp: float | None = None
    audio_transcript: str = ""
    raw_text: str = ""
    spoken_text: str = ""
    last_audio_bytes_played: int = 0
    text_complete: bool = False
    response_ended: bool = False
    settle_watchdog: asyncio.Task[None] | None = None
    uninterruptable: bool = False
    audio_bytes: int = 0
    audio_sample_rate: int | None = None
    audio_channels: int = 1


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
        usage_heartbeat_interval_s: float = 20.0,
        # VAD configuration
        vad_confidence_threshold: float | None = None,
        vad_min_volume: float | None = None,
        vad_start_duration_ms: int | None = None,
        vad_stop_duration_ms: int | None = None,
        vad_backbuffer_duration_ms: int | None = None,
        vad_config: VadConfig | None = None,
        # TTS configuration
        tts_config: ElevenLabsTtsConfig | HostedTtsConfig | HostedVoiceCloneConfig | None = None,
        supports_playback_reporting: bool = False,
        http_session: aiohttp.ClientSession | None = None,
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
            usage_heartbeat_interval_s: Interval in seconds between connected-time
                                        usage reports while a session is active.
            vad_config: Voice activity detection tuning.
            tts_config: TTS configuration. When provided, audio output is enabled.
                        Use ``ElevenLabsTtsConfig`` for ElevenLabs-hosted synthesis,
                        ``HostedTtsConfig`` for Deepslate-hosted (already cloned/existing) voices,
                        or ``HostedVoiceCloneConfig`` to clone a voice on the fly by
                        supplying a raw audio sample. When None (default), only text
                        output is provided.
            supports_playback_reporting: When True, report how much of an
                        interrupted assistant turn the caller actually heard, so
                        the server truncates the model's context to match instead
                        of falling back to elapsed-time estimation which is less precise.
                        Off by default.
            http_session: Optional shared aiohttp session.
        """
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=True,
                turn_detection=True,
                user_transcription=True,
                # Corvidae auto-generates the reply after a tool call
                # (handle_tool_call_response -> dispatch_inference). Must be True,
                # otherwise livekit-agents interrupts and fires a competing reply
                # that truncates the server's auto-reply.
                auto_tool_reply_generation=True,
                audio_output=tts_config is not None,
                manual_function_calls=False,
                per_response_tool_choice=False,
            )
        )

        self._tts_config = tts_config
        self._usage_heartbeat_interval_s = usage_heartbeat_interval_s

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

            deepslate_organization_id = organization_id or os.environ.get(
                "DEEPSLATE_ORGANIZATION_ID"
            )
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
            supports_playback_reporting=supports_playback_reporting,
        )

        deprecated_vad_kwargs = {
            "vad_confidence_threshold": vad_confidence_threshold,
            "vad_min_volume": vad_min_volume,
            "vad_start_duration_ms": vad_start_duration_ms,
            "vad_stop_duration_ms": vad_stop_duration_ms,
            "vad_backbuffer_duration_ms": vad_backbuffer_duration_ms,
        }
        explicit_deprecated_vad_kwargs = {
            k: v for k, v in deprecated_vad_kwargs.items() if v is not None
        }

        if vad_config is not None and explicit_deprecated_vad_kwargs:
            warnings.warn(
                f"`vad_config` was provided along with deprecated flat kwargs "
                f"({', '.join(explicit_deprecated_vad_kwargs)}); `vad_config` takes "
                "precedence and the flat kwargs are ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
        elif explicit_deprecated_vad_kwargs:
            warnings.warn(
                f"{', '.join(explicit_deprecated_vad_kwargs)} are deprecated and will be "
                "removed in a future release; pass a `vad_config=VadConfig(...)` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        if vad_config is not None:
            self._vad_config = vad_config
        else:
            vad_defaults = VadConfig()

            def _coalesce(value, default):
                return default if value is None else value

            self._vad_config = VadConfig(
                confidence_threshold=_coalesce(
                    vad_confidence_threshold, vad_defaults.confidence_threshold
                ),
                min_volume=_coalesce(vad_min_volume, vad_defaults.min_volume),
                start_duration_ms=_coalesce(
                    vad_start_duration_ms, vad_defaults.start_duration_ms
                ),
                stop_duration_ms=_coalesce(vad_stop_duration_ms, vad_defaults.stop_duration_ms),
                backbuffer_duration_ms=_coalesce(
                    vad_backbuffer_duration_ms, vad_defaults.backbuffer_duration_ms
                ),
            )

        self._client = BaseDeepslateClient(
            opts=self._opts,
            user_agent=f"DeepslateLiveKit/{__version__}",
            http_session=http_session,
        )

    @property
    def provider(self) -> str:
        """Return the provider identifier for this model."""
        return "deepslate"

    @property
    def model(self) -> str:
        """Return the model identifier used in emitted usage/metrics metadata."""
        return "opal"

    def session(
        self, *, turn_detection_disabled: bool = False
    ) -> "DeepslateRealtimeSession":
        """Create a new Deepslate real-time session."""
        if turn_detection_disabled:
            logger.warning(
                "turn_detection_disabled is not supported and will be ignored"
            )
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
        """Close the model and release the underlying HTTP client."""
        await self._client.aclose()


class DeepslateRealtimeSession(
    llm.RealtimeSession[
        Literal[
            "deepslate_server_event_received",
            "deepslate_client_event_sent",
            "audio_transcript",
            "model_text_fragment",
            "session_initialized",
            "turn_snapshot",
        ]
    ],
    DeepslateSessionListener,
):
    """A session for the Deepslate Realtime API.

    Wraps ``DeepslateSession`` from deepslate-core and translates its
    callbacks into LiveKit agent events and channel writes.  All
    protobuf details are encapsulated in the core session; this class
    contains only LiveKit-specific logic.
    """

    def __init__(self, realtime_model: RealtimeModel):
        """Initialize the session and start the underlying core session."""
        super().__init__(realtime_model)
        self._realtime_model = realtime_model
        self._opts = realtime_model._opts

        self._session_start_time = time.monotonic()
        self._last_usage_report_time = self._session_start_time
        self._connection_attempt_started_at = self._session_start_time
        self._usage_heartbeat_task = asyncio.create_task(self._usage_heartbeat())

        self._audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        self._audio_task = asyncio.create_task(self._audio_worker())

        # LiveKit context
        self._tools = llm.ToolContext.empty()
        self._chat_ctx = llm.ChatContext.empty()
        self._instructions: str | None = None

        # Generation tracking
        self._generations: dict[int, _ResponseGeneration] = {}
        self._settled_turns: deque[int] = deque(maxlen=64)
        self._last_turn_id: int | None = None
        self._response_created_futures: dict[
            str, asyncio.Future[GenerationCreatedEvent]
        ] = {}
        self._pending_user_generation: bool = False
        self._pending_uninterruptable: bool = False
        self._pending_user_text: str | None = None

        self._settled_generations: OrderedDict[str, _ResponseGeneration] = (
            OrderedDict()
        )
        self._playback_report_tasks: set[asyncio.Task[None]] = set()

        # Conversation query tracking: query_id → Future[str]
        self._pending_queries: dict[str, asyncio.Future[str]] = {}

        # Tool state
        self._tools_dicts: list[FunctionToolDict] = []
        self._tool_choice: ToolChoice | None = None
        self._tool_tasks: set[asyncio.Task[None]] = set()
        self._tool_sync_lock = asyncio.Lock()
        self._outstanding_tool_calls: dict[str, tuple[str, int]] = {}

        # Core session — owns the WebSocket lifecycle
        self._session = DeepslateSession(
            client=realtime_model._client,
            options=realtime_model._opts,
            vad_config=realtime_model._vad_config,
            tts_config=realtime_model._tts_config,
            listener=self,
        )
        self._session.start()

    @property
    def chat_ctx(self) -> llm.ChatContext:
        """Return a copy of the current chat context."""
        return self._chat_ctx.copy()

    @property
    def tools(self) -> ToolContext:
        """Return a copy of the current tool context."""
        return self._tools.copy()

    async def update_instructions(self, instructions: str | Instructions) -> None:
        """Update system prompt for the next session initialization."""
        if isinstance(instructions, Instructions):
            modality: Literal["audio", "text"] = (
                "audio" if self._realtime_model.capabilities.audio_output else "text"
            )
            instructions = instructions.render(modality=modality)

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
                    if self._outstanding_tool_calls.pop(item.call_id, None) is None:
                        continue
                    await self._session.send_tool_response(item.call_id, item.output)

        self._chat_ctx = chat_ctx.copy()

    async def update_tools(
        self, tools: list[FunctionTool | RawFunctionTool | Any]
    ) -> None:
        """Sync tool definitions to the server."""
        tools_dicts = []
        for tool in tools:
            if is_function_tool(tool):
                schema = llm.utils.build_legacy_openai_schema(
                    tool, internally_tagged=True
                )
                tools_dicts.append(
                    {
                        "type": "function",
                        "function": {
                            "name": schema["name"],
                            "description": schema.get("description", ""),
                            "parameters": schema.get("parameters", {}),
                        },
                    }
                )
            elif is_raw_function_tool(tool):
                info = get_raw_function_info(tool)
                tools_dicts.append(
                    {
                        "type": "function",
                        "function": {
                            "name": info.name,
                            "description": info.raw_schema.get("description", ""),
                            "parameters": info.raw_schema.get("parameters", {}),
                        },
                    }
                )

        self._tools_dicts = tools_dicts
        self._tools = llm.ToolContext(tools)
        await self._sync_tool_choice()
        logger.debug(
            f"updated tools: {[t.get('function', {}).get('name') for t in tools_dicts]}"
        )

    def update_options(
        self, *, tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN
    ) -> None:
        """Apply a tool_choice constraint.

        The base ``RealtimeSession.update_options`` is synchronous and called
        fire-and-forget, so the server sync runs as a tracked background task
        rather than being awaited.
        """
        if not utils.is_given(tool_choice) or tool_choice == self._tool_choice:
            return
        self._tool_choice = tool_choice
        task = asyncio.create_task(self._sync_tool_choice())
        self._tool_tasks.add(task)
        task.add_done_callback(self._tool_tasks.discard)
        task.add_done_callback(self._on_tool_task_done)

    def _on_tool_task_done(self, task: asyncio.Task[None]) -> None:
        """Surface failures from a background tool-choice sync task.

        Tool syncs are scheduled fire-and-forget by :meth:`update_options`, so
        their exceptions would otherwise be swallowed. Logs any non-cancellation
        error; cancellations (e.g. during :meth:`aclose`) are ignored.
        """
        if not task.cancelled() and (exc := task.exception()) is not None:
            logger.error("tool_choice sync failed", exc_info=exc)

    def _effective_tools_dicts(self) -> list[FunctionToolDict]:
        """Return the tools list filtered by the current tool_choice."""
        tc = self._tool_choice
        if tc == "none":
            return []
        if isinstance(tc, dict):  # NamedToolChoice
            name = tc.get("function", {}).get("name")
            return [
                t
                for t in self._tools_dicts
                if t.get("function", {}).get("name") == name
            ]
        # "auto", "required", None → send all tools
        return self._tools_dicts

    async def _sync_tool_choice(self) -> None:
        """Push the effective tool list (after applying tool_choice) to the server.

        Holds ``_tool_sync_lock`` so overlapping syncs (a direct
        :meth:`update_tools` and a background :meth:`update_options` task) are
        serialized and the server's final tool list reflects the latest update.
        """
        async with self._tool_sync_lock:
            await self._session.update_tools(self._effective_tools_dicts())

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        """Push an audio frame to Deepslate."""
        if not self._audio_ch.closed:
            self._audio_ch.send_nowait(frame)

    async def _audio_worker(self) -> None:
        """Persistent background task that pushes queued frames to Deepslate."""
        try:
            async for frame in self._audio_ch:
                await self._session.send_audio(
                    frame.data.tobytes(),
                    frame.sample_rate,
                    frame.num_channels,
                )
        except Exception as e:
            logger.error(f"Deepslate audio dispatcher worker failed: {e}", exc_info=True)

    def push_video(self, frame: rtc.VideoFrame) -> None:
        """Video input is not supported by Deepslate."""
        logger.warning("Deepslate does not support video input")

    async def send_text(
        self,
        text: str,
        mode: TriggerMode = TriggerMode.NO_TRIGGER,
    ) -> None:
        """Send text input to Deepslate."""
        await self._session.initialize()
        await self._session.send_text(text, trigger=mode)

    async def speak_direct(
        self,
        text: str,
        include_in_history: bool = True,
        uninterruptable: bool = False,
    ) -> None:
        """Bypass the LLM and speak text directly via TTS."""
        await self._session.initialize()
        # Consumed by the next _create_generation() so on_vad_state_event can
        # tell this generation apart from a normal, barge-in-able one.
        self._pending_uninterruptable = uninterruptable
        await self._session.send_direct_speech(
            text, include_in_history, uninterruptable
        )

    async def query_conversation(
        self,
        prompt: str | None = None,
        instructions: str | None = None,
    ) -> str:
        """Run a one-shot side-channel inference against the current conversation.

        Returns the model's complete text reply.
        """
        query_id = utils.shortuuid("query_")
        fut: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        self._pending_queries[query_id] = fut

        await self._session.initialize()
        await self._session.send_conversation_query(query_id, prompt, instructions)
        return await fut

    async def export_chat_history(
        self, await_pending: bool = False, exclude_audio: bool = False
    ) -> list[ChatMessageDict]:
        """Request the server to export the current chat history and return it.

        Also emitted via the ``chat_history_exported`` event.
        """
        return await self._session.export_chat_history(await_pending, exclude_audio)

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[GenerationCreatedEvent]:
        """Request the model to generate a reply."""
        return asyncio.create_task(
            self._generate_reply(
                instructions=instructions,
                tool_choice=tool_choice,
                tools=tools,
            )
        )

    async def _generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> GenerationCreatedEvent:
        """Async implementation backing :meth:`generate_reply`."""
        if utils.is_given(tool_choice):
            logger.warning(
                "tool_choice is not supported in generate_reply and will be ignored"
            )

        if utils.is_given(tools):
            await self.update_tools(tools)

        fut: asyncio.Future[GenerationCreatedEvent] = asyncio.Future()
        request_id = utils.shortuuid("gen_")
        self._response_created_futures[request_id] = fut
        self._pending_user_generation = True

        if utils.is_given(instructions):
            self._instructions = instructions

        if self._pending_user_text:
            if utils.is_given(instructions):
                await self._session.send_text(
                    self._pending_user_text,
                    trigger=TriggerMode.NO_TRIGGER,
                )
                await self._session.trigger_inference(instructions=instructions)
            else:
                await self._session.initialize()
                await self._session.send_text(
                    self._pending_user_text,
                    trigger=TriggerMode.IMMEDIATE,
                )
            self._pending_user_text = None
        else:
            await self._session.initialize()
            await self._session.trigger_inference(
                instructions=instructions if utils.is_given(instructions) else None
            )

        timeout = self._opts.generate_reply_timeout

        if timeout > 0:
            try:
                return await asyncio.wait_for(fut, timeout=timeout)
            except asyncio.TimeoutError:
                raise TimeoutError(f"generate_reply timed out after {timeout}s")
        else:
            return await fut

    def commit_audio(self) -> None:
        """Deepslate uses server-side VAD for auto-commit."""
        pass

    def clear_audio(self) -> None:
        """Audio buffer clearing is not yet supported by the Deepslate backend."""
        logger.warning("clear_audio not yet supported by Deepslate backend")

    def interrupt(self) -> None:
        """Interrupt all currently open generations."""
        for gen in list(self._generations.values()):
            self._settle_generation(gen, cancelled=True)

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Report how much of an interrupted turn the user actually heard.

        Truncation still happens server-side without this; the report only makes
        it accurate.
        """
        if not self._realtime_model._opts.supports_playback_reporting:
            return

        if self._realtime_model._tts_config is None:
            return

        gen = self._find_generation(message_id)
        if gen is None:
            logger.debug(
                "playback position not reported: no generation for message",
                extra={"message_id": message_id},
            )
            return

        if gen.uninterruptable:
            logger.debug(
                "playback position not reported: turn is uninterruptable",
                extra={"message_id": message_id, "turn_id": gen.turn_id},
            )
            return

        bytes_played = self._playback_bytes(gen, audio_end_ms)
        self._spawn_playback_report(bytes_played, gen.turn_id)

    def _find_generation(self, message_id: str) -> _ResponseGeneration | None:
        """Resolve a livekit message id to its generation, open or just settled."""
        for gen in self._generations.values():
            if gen.response_id == message_id:
                return gen
        return self._settled_generations.get(message_id)

    def _retain_settled_generation(self, gen: _ResponseGeneration) -> None:
        """Keep a just-settled generation resolvable by a later truncate()."""
        self._settled_generations[gen.response_id] = gen
        self._settled_generations.move_to_end(gen.response_id)
        while len(self._settled_generations) > _SETTLED_GENERATION_LIMIT:
            self._settled_generations.popitem(last=False)

    @staticmethod
    def _playback_bytes(gen: _ResponseGeneration, audio_end_ms: int) -> int:
        """Convert a played duration to a byte offset into this turn's audio."""
        sample_rate = gen.audio_sample_rate or 24000
        channels = gen.audio_channels or 1
        frame_size = channels * _BYTES_PER_SAMPLE
        raw = int(max(audio_end_ms, 0) / 1000 * sample_rate * frame_size)
        aligned = raw - (raw % frame_size)
        return max(0, min(aligned, gen.audio_bytes))

    def _spawn_playback_report(self, bytes_played: int, turn_id: int) -> None:
        """Send a playback position report without blocking the caller."""
        task = asyncio.create_task(
            self._session.report_playback_position(bytes_played, turn_id)
        )
        self._playback_report_tasks.add(task)
        task.add_done_callback(self._playback_report_tasks.discard)
        task.add_done_callback(self._on_playback_report_done)

    @staticmethod
    def _on_playback_report_done(task: asyncio.Task[None]) -> None:
        """Surface failures from a fire-and-forget playback report."""
        if not task.cancelled() and (exc := task.exception()) is not None:
            logger.error("playback position report failed", exc_info=exc)

    async def aclose(self) -> None:
        """Close the session."""
        self._outstanding_tool_calls.clear()
        self._usage_heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._usage_heartbeat_task

        tool_tasks = tuple(self._tool_tasks)
        for task in tool_tasks:
            task.cancel()
        if tool_tasks:
            await asyncio.gather(*tool_tasks, return_exceptions=True)

        report_tasks = tuple(self._playback_report_tasks)
        if report_tasks:
            await asyncio.gather(*report_tasks, return_exceptions=True)

        for gen in list(self._generations.values()):
            self._settle_generation(gen, cancelled=True)

        # Close channel to break the async for-loop in the worker
        if not self._audio_ch.closed:
            self._audio_ch.close()

        # Await worker to ensure graceful shutdown
        with contextlib.suppress(asyncio.CancelledError):
            await self._audio_task

        self._report_session_duration()

        await self._session.close()

    async def on_connecting(self) -> None:
        """Reset per-connection state at the start of a connection attempt."""
        self._outstanding_tool_calls.clear()
        for gen in list(self._generations.values()):
            self._settle_generation(gen, cancelled=True)
        self._generations.clear()
        self._settled_turns.clear()
        self._last_turn_id = None
        self._settled_generations.clear()
        self._connection_attempt_started_at = time.monotonic()

    async def on_session_initialized(self) -> None:
        """Emit ``session_initialized`` once the core session is ready."""
        self._report_connection_acquired(
            time.monotonic() - self._connection_attempt_started_at
        )
        self.emit("session_initialized", None)

    async def on_text_fragment(
        self, text: str, turn_id: int | None = None
    ) -> None:
        """Accumulate a raw text fragment for its turn."""
        self.emit(
            "model_text_fragment",
            SimpleNamespace(text=text, turn_id=turn_id),
        )

        gen = self._get_or_create_generation(turn_id)
        if gen is None:
            return
        gen.raw_text += text
        if gen.first_token_timestamp is None:
            gen.first_token_timestamp = time.time()

        if self._realtime_model._tts_config is None:
            gen.text_ch.send_nowait(text)
            gen.audio_transcript += text

    async def on_audio_chunk(
        self,
        pcm_bytes: bytes,
        sample_rate: int,
        channels: int,
        transcript: str | None,
        turn_id: int | None = None,
    ) -> None:
        """Stream an audio chunk into its generation."""
        gen = self._get_or_create_generation(turn_id)
        if gen is None:
            return

        frame = rtc.AudioFrame(
            data=pcm_bytes,
            sample_rate=sample_rate,
            num_channels=channels,
            samples_per_channel=len(pcm_bytes) // 2 // max(channels, 1),
        )
        gen.audio_ch.send_nowait(frame)
        gen.audio_bytes += len(pcm_bytes)
        gen.audio_sample_rate = sample_rate
        gen.audio_channels = channels

        if gen.first_token_timestamp is None:
            gen.first_token_timestamp = time.time()

    async def on_model_speech_progress(
        self,
        turn_id: int,
        text: str,
        audio_bytes_played: int,
        exact: bool,
    ) -> None:
        """Feed newly-audible text into its generation, paced to playback."""
        gen = self._generations.get(turn_id)
        if gen is None:
            return

        previous_bytes_played = gen.last_audio_bytes_played
        if audio_bytes_played < previous_bytes_played:
            logger.warning(
                "Deepslate: ModelSpeechProgress.audio_bytes_played went "
                f"backward for turn_id={turn_id} "
                f"({audio_bytes_played} < {previous_bytes_played}); our playback "
                "reports have possibly de-synced"
            )

        bytes_per_second = (gen.audio_sample_rate or 24000) * (gen.audio_channels or 1) * 2
        start_time = min(previous_bytes_played, audio_bytes_played) / bytes_per_second
        end_time = audio_bytes_played / bytes_per_second
        gen.last_audio_bytes_played = audio_bytes_played

        if text:
            gen.text_ch.send_nowait(TimedString(text, start_time=start_time, end_time=end_time))
            gen.audio_transcript += text
            gen.spoken_text += text
            self.emit("audio_transcript", text)

        self._maybe_settle_after_playback(gen)

    async def on_inference_complete(self, turn_id: int) -> None:
        """Mark a turn's raw text as complete (no more ModelTextFragment)."""
        self.emit(
            "deepslate_server_event_received",
            SimpleNamespace(type="inference_complete", turn_id=turn_id),
        )
        gen = self._generations.get(turn_id)
        if gen is not None:
            gen.text_complete = True
            self._maybe_settle_after_playback(gen)

    async def on_turn_snapshot(self, message: ChatMessageDict, is_final: bool) -> None:
        """Surface the server's authoritative view of a turn's content."""
        self.emit(
            "turn_snapshot",
            SimpleNamespace(message=message, is_final=is_final),
        )

        if not is_final:
            return

        turn_id = message.get("turn_id")
        if turn_id is None:
            return
        gen = self._generations.get(turn_id)
        if gen is None:
            return

        gen.text_complete = True
        gen.response_ended = True
        if message.get("delivery_status") == "DELIVERY_INTERRUPTED":
            self._settle_generation(gen, cancelled=True)
            return
        self._maybe_settle_after_playback(gen)

    async def on_tool_call(
        self, call_id: str, name: str, params: dict, turn_id: int | None = None
    ) -> None:
        """Forward a server tool-call request into its generation."""
        if turn_id is None:
            turn_id = self._last_turn_id if self._last_turn_id is not None else 0

        gen = self._generations.get(turn_id)
        reopened = gen is None and turn_id in self._settled_turns
        if gen is None:
            if reopened:
                logger.warning(
                    "Deepslate: tool call %s(%s) arrived after turn_id=%s had "
                    "already settled; running it on a reopened generation",
                    name,
                    call_id,
                    turn_id,
                )
            gen = self._create_generation(turn_id)

        self._outstanding_tool_calls[call_id] = (name, turn_id)
        gen.function_ch.send_nowait(
            FunctionCall(
                call_id=call_id,
                name=name,
                arguments=json.dumps(params),
            )
        )
        logger.debug(f"tool call request: {name}({call_id})")

        if reopened:
            gen.response_ended = True
            self._settle_generation(gen)

    async def on_response_begin(self, turn_id: int = 0) -> None:
        """Start a new generation at the beginning of a server response."""
        existing = self._generations.get(turn_id)
        if existing is not None:
            logger.warning(
                "Deepslate: ResponseBegin for already-open turn_id=%s; "
                "recycling the stale generation",
                turn_id,
            )
            self._settle_generation(existing, cancelled=True)
        self._create_generation(turn_id)

    async def on_response_end(self, turn_id: int = 0) -> None:
        """Close the generation when the server response ends."""
        gen = self._generations.get(turn_id)
        if gen is None:
            return
        gen.response_ended = True
        if self._realtime_model._tts_config is None:
            self._settle_generation(gen)
            return
        self._maybe_settle_after_playback(gen)

    async def on_user_transcription(
        self, text: str, language: str | None, turn_id: int
    ) -> None:
        """Emit the input-transcription-completed event livekit-agents listens for."""
        self.emit(
            "input_audio_transcription_completed",
            InputTranscriptionCompleted(
                item_id=utils.shortuuid("item_"),
                transcript=text,
                is_final=True,
            ),
        )

    async def on_chat_history(self, messages) -> None:
        """Emit the exported chat history to listeners."""
        self.emit("chat_history_exported", messages)

    async def on_conversation_query_result(self, query_id: str, text: str) -> None:
        """Resolve the pending future for a conversation query result."""
        fut = self._pending_queries.pop(query_id, None)
        if fut is not None and not fut.done():
            fut.set_result(text)
        else:
            logger.warning(
                f"received conversation_query_result for unknown query_id: '{query_id}'"
            )

    async def on_error(self, category: str, message: str, trace_id: str | None) -> None:
        """Log a server error and emit a recoverable=False error event."""
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

    async def on_fatal_error(self, e: Exception) -> None:
        """Emit an error event for an unrecoverable session failure."""
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self._realtime_model.label,
                error=e,
                recoverable=False,
            ),
        )

    async def on_vad_state_event(
        self,
        from_state: str,
        to_state: str,
        session_time_ms: int,
        packet_id: int,
    ) -> None:
        """Handle a VAD state transition, interrupting on confirmed user speech."""
        open_gens = list(self._generations.values())
        if from_state == "SPEECH_STARTING" and to_state == "SPEECH":
            protected = any(
                gen.uninterruptable and not self._audio_drained(gen)
                for gen in open_gens
            )
            if not protected:
                self.emit("input_speech_started", InputSpeechStartedEvent())
                for gen in open_gens:
                    self._settle_generation(gen, cancelled=True)
        elif from_state == "SPEECH_ENDING" and to_state == "SILENCE":
            self.emit(
                "input_speech_stopped",
                InputSpeechStoppedEvent(
                    user_transcription_enabled=self._realtime_model.capabilities.user_transcription
                ),
            )

        self.emit(
            "deepslate_server_event_received",
            SimpleNamespace(
                type="vad_state_event",
                from_state=from_state,
                to_state=to_state,
                session_time_ms=session_time_ms,
                packet_id=packet_id,
            ),
        )

    async def on_context_truncated(
        self,
        truncated_turn_ids: list[int],
        response_turn_id: int,
    ) -> None:
        """Emit a context-truncation event reported by the server."""
        self.emit(
            "deepslate_server_event_received",
            SimpleNamespace(
                type="context_truncated",
                truncated_turn_ids=truncated_turn_ids,
                response_turn_id=response_turn_id,
            ),
        )

    def _get_or_create_generation(
        self, turn_id: int | None
    ) -> _ResponseGeneration | None:
        """Resolve the open generation for ``turn_id``, creating it if needed."""
        if turn_id is None:
            turn_id = self._last_turn_id if self._last_turn_id is not None else 0
        gen = self._generations.get(turn_id)
        if gen is not None:
            return gen
        if turn_id in self._settled_turns:
            return None
        return self._create_generation(turn_id)

    def _create_generation(self, turn_id: int) -> _ResponseGeneration:
        """Create a new response generation for ``turn_id`` and emit ``generation_created``."""
        is_user_initiated = self._pending_user_generation
        self._pending_user_generation = False
        is_uninterruptable = self._pending_uninterruptable
        self._pending_uninterruptable = False

        response_id = utils.shortuuid("resp_")
        gen = _ResponseGeneration(
            message_ch=utils.aio.Chan(),
            function_ch=utils.aio.Chan(),
            text_ch=utils.aio.Chan(),
            audio_ch=utils.aio.Chan(),
            done_fut=asyncio.Future(),
            response_id=response_id,
            created_timestamp=time.time(),
            turn_id=turn_id,
            uninterruptable=is_uninterruptable,
        )
        self._generations[turn_id] = gen
        self._last_turn_id = turn_id

        has_audio = self._realtime_model._tts_config is not None
        msg_modalities: asyncio.Future[list[Literal["text", "audio"]]] = (
            asyncio.Future()
        )
        if has_audio:
            msg_modalities.set_result(["audio"])
        else:
            msg_modalities.set_result(["text"])
            gen.audio_ch.close()

        gen.message_ch.send_nowait(
            MessageGeneration(
                message_id=response_id,
                text_stream=gen.text_ch,
                audio_stream=gen.audio_ch,
                modalities=msg_modalities,
            )
        )

        generation_ev = GenerationCreatedEvent(
            message_stream=gen.message_ch,
            function_stream=gen.function_ch,
            user_initiated=is_user_initiated,
            response_id=response_id,
        )

        self.emit("generation_created", generation_ev)

        for fut in list(self._response_created_futures.values()):
            if not fut.done():
                fut.set_result(generation_ev)
        self._response_created_futures.clear()

        return gen

    def _text_fully_spoken(self, gen: _ResponseGeneration) -> bool:
        """Whether every speakable character of ``gen`` appears to be audible."""
        if not gen.text_complete:
            return False
        if not gen.raw_text:
            return True
        return not gen.raw_text[len(gen.spoken_text) :].strip()

    def _audio_drained(self, gen: _ResponseGeneration) -> bool:
        """Whether every audio byte dispatched for ``gen`` has been played."""
        if not gen.response_ended:
            return False
        return gen.last_audio_bytes_played >= gen.audio_bytes

    def _delivery_complete(self, gen: _ResponseGeneration) -> bool:
        """Whether ``gen`` is finished: nothing left to play and nothing to say."""
        if not gen.response_ended:
            return False
        if self._audio_drained(gen):
            return True
        if not gen.raw_text:
            return False
        return self._text_fully_spoken(gen)

    def _maybe_settle_after_playback(self, gen: _ResponseGeneration) -> None:
        """Settle ``gen`` once every audio byte it produced has been played."""
        if not gen.response_ended:
            return
        if self._delivery_complete(gen):
            self._settle_generation(gen)
            return
        self._start_settle_watchdog(gen)

    def _start_settle_watchdog(self, gen: _ResponseGeneration) -> None:
        """Arm the backstop that settles ``gen`` if speech progress stalls."""
        if gen.settle_watchdog is not None:
            return
        gen.settle_watchdog = asyncio.create_task(
            self._settle_watchdog_task(gen),
            name=f"deepslate.settle_watchdog.turn_{gen.turn_id}",
        )

    async def _settle_watchdog_task(self, gen: _ResponseGeneration) -> None:
        """Settle ``gen`` once its audio can no longer be playing."""
        try:
            while True:
                bytes_per_second = (
                    (gen.audio_sample_rate or 24000) * (gen.audio_channels or 1) * 2
                )
                bytes_seen = gen.audio_bytes
                played_before = gen.last_audio_bytes_played
                unplayed = max(bytes_seen - played_before, 0)

                await asyncio.sleep(unplayed / bytes_per_second + SETTLE_GRACE_PERIOD)

                if self._generations.get(gen.turn_id) is not gen:
                    return
                if self._delivery_complete(gen):
                    logger.debug(
                        f"settling turn_id={gen.turn_id} after full playback"
                    )
                    self._settle_generation(gen)
                    return
                if (
                    gen.audio_bytes > bytes_seen
                    or gen.last_audio_bytes_played != played_before
                ):
                    continue

                if bool(gen.raw_text) and not self._text_fully_spoken(gen):
                    logger.warning(
                        "Deepslate: settling turn_id=%s with an incomplete live "
                        "transcript, speech progress stalled at %s/%s characters "
                        "(%s/%s audio bytes reported played)",
                        gen.turn_id,
                        len(gen.spoken_text),
                        len(gen.raw_text),
                        gen.last_audio_bytes_played,
                        gen.audio_bytes,
                    )
                else:
                    logger.debug(
                        f"settling turn_id={gen.turn_id} on playback watchdog "
                        f"({gen.last_audio_bytes_played}/{gen.audio_bytes} audio "
                        "bytes reported played)"
                    )
                self._settle_generation(gen)
                return
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception(
                f"settle watchdog failed for turn_id={gen.turn_id}; settling"
            )
            self._settle_generation(gen)

    def _settle_generation(
        self, gen: _ResponseGeneration, *, cancelled: bool = False
    ) -> None:
        """Close one generation's channels and mark it complete"""
        if (watchdog := gen.settle_watchdog) is not None:
            gen.settle_watchdog = None
            if watchdog is not asyncio.current_task():
                watchdog.cancel()
        if self._generations.get(gen.turn_id) is not gen:
            return
        if not gen.text_ch.closed:
            if gen.audio_transcript == "":
                gen.text_ch.send_nowait("")
            gen.text_ch.close()
        gen.audio_ch.close()
        gen.function_ch.close()
        gen.message_ch.close()
        with contextlib.suppress(asyncio.InvalidStateError):
            gen.done_fut.set_result(None)
        del self._generations[gen.turn_id]
        self._retain_settled_generation(gen)
        if gen.turn_id not in self._settled_turns:
            self._settled_turns.append(gen.turn_id)
        if cancelled:
            self._cancel_tool_calls_for(gen.turn_id)
        self._emit_generation_metrics(gen, cancelled=cancelled)

    def _cancel_tool_calls_for(self, turn_id: int) -> None:
        """Answer tool calls livekit-agents is about to throw away."""
        for call_id, (name, call_turn_id) in list(self._outstanding_tool_calls.items()):
            if call_turn_id != turn_id:
                continue
            del self._outstanding_tool_calls[call_id]
            task = asyncio.create_task(
                self._session.send_tool_response(call_id, ABANDONED_TOOL_RESULT)
            )
            self._tool_tasks.add(task)
            task.add_done_callback(self._tool_tasks.discard)
            task.add_done_callback(self._on_tool_task_done)

    def _emit_generation_metrics(self, gen: _ResponseGeneration, *, cancelled: bool) -> None:
        """Emit latency and usage metrics for a just-closed generation."""
        now = time.time()
        duration = now - gen.created_timestamp
        ttft = (
            gen.first_token_timestamp - gen.created_timestamp
            if gen.first_token_timestamp is not None
            else -1.0
        )

        audio_tokens = 0
        if gen.audio_bytes and gen.audio_sample_rate:
            audio_duration = (
                gen.audio_bytes / 2 / (gen.audio_channels or 1) / gen.audio_sample_rate
            )
            audio_tokens = round(audio_duration * 1000)

        self.emit(
            "metrics_collected",
            RealtimeModelMetrics(
                request_id=gen.response_id,
                timestamp=gen.created_timestamp,
                duration=duration,
                ttft=ttft,
                cancelled=cancelled,
                input_token_details=RealtimeModelMetrics.InputTokenDetails(),
                output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                    audio_tokens=audio_tokens
                ),
                metadata=self._metadata(),
            ),
        )

    async def _usage_heartbeat(self) -> None:
        """Periodically report connected time as billing usage."""
        try:
            while True:
                await asyncio.sleep(self._realtime_model._usage_heartbeat_interval_s)
                self._report_session_duration()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.error("usage heartbeat failed", exc_info=True)

    def _metadata(self) -> Metadata:
        return Metadata(
            model_name=self._realtime_model.model,
            model_provider=self._realtime_model.provider,
        )

    def _report_connection_acquired(self, acquire_time: float) -> None:
        """Report the one-time connection-acquire latency for this session."""
        self.emit(
            "metrics_collected",
            RealtimeModelMetrics(
                request_id="",
                timestamp=time.time(),
                acquire_time=acquire_time,
                input_token_details=RealtimeModelMetrics.InputTokenDetails(),
                output_token_details=RealtimeModelMetrics.OutputTokenDetails(),
                metadata=self._metadata(),
            ),
        )

    def _report_session_duration(self) -> None:
        """Report connected wall-clock time (since the last report) as billing usage."""
        now_monotonic = time.monotonic()
        delta = now_monotonic - self._last_usage_report_time
        if delta <= 0:
            return
        self._last_usage_report_time = now_monotonic

        self.emit(
            "metrics_collected",
            RealtimeModelMetrics(
                request_id="",
                timestamp=time.time(),
                session_duration=delta,
                input_token_details=RealtimeModelMetrics.InputTokenDetails(),
                output_token_details=RealtimeModelMetrics.OutputTokenDetails(),
                metadata=self._metadata(),
            ),
        )
