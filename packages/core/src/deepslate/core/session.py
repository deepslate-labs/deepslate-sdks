from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any, Optional

import aiohttp

from .client import BaseDeepslateClient
from .options import DeepslateOptions, ElevenLabsTtsConfig, VadConfig
from .proto import realtime_pb2 as proto
from ._types import ChatMessageDict, FunctionToolDict, TriggerMode
from ._utils import build_initialize_request, dict_to_struct, parse_chat_history, struct_to_dict

logger = logging.getLogger("deepslate.core")

OnTextFragment = Callable[[str], Awaitable[None]]

OnAudioChunk = Callable[[bytes, int, int, Optional[str]], Awaitable[None]]

OnToolCall = Callable[[str, str, dict], Awaitable[None]]

OnError = Callable[[str, str, Optional[str]], Awaitable[None]]

OnResponseBegin = Callable[[], Awaitable[None]]
OnResponseEnd = Callable[[], Awaitable[None]]

OnUserTranscription = Callable[[str, Optional[str], int], Awaitable[None]]

OnPlaybackBufferClear = Callable[[], Awaitable[None]]

OnChatHistory = Callable[[list[ChatMessageDict]], Awaitable[None]]

OnConversationQueryResult = Callable[[str, str], Awaitable[None]]

OnFatalError = Callable[[Exception], Awaitable[None]]


class DeepslateSession:
    """Manages a single logical Deepslate realtime connection.

    Wraps ``BaseDeepslateClient`` and owns the full WebSocket lifecycle
    """

    def __init__(
        self,
        client: BaseDeepslateClient,
        options: DeepslateOptions,
        vad_config: Optional[VadConfig] = None,
        tts_config: Optional[ElevenLabsTtsConfig] = None,
        # Callbacks
        on_text_fragment: Optional[OnTextFragment] = None,
        on_audio_chunk: Optional[OnAudioChunk] = None,
        on_tool_call: Optional[OnToolCall] = None,
        on_error: Optional[OnError] = None,
        on_response_begin: Optional[OnResponseBegin] = None,
        on_response_end: Optional[OnResponseEnd] = None,
        on_user_transcription: Optional[OnUserTranscription] = None,
        on_playback_buffer_clear: Optional[OnPlaybackBufferClear] = None,
        on_chat_history: Optional[OnChatHistory] = None,
        on_conversation_query_result: Optional[OnConversationQueryResult] = None,
        on_fatal_error: Optional[OnFatalError] = None,
    ) -> None:
        self._client = client
        self._owns_client = False  # set to True by DeepslateSession.create()
        self._options = options
        self._vad_config = vad_config or VadConfig()
        self._tts_config = tts_config
        self._current_tools: list[FunctionToolDict] = []
        self._should_stop = False

        # Callbacks
        self.on_text_fragment = on_text_fragment
        self.on_audio_chunk = on_audio_chunk
        self.on_tool_call = on_tool_call
        self.on_error = on_error
        self.on_response_begin = on_response_begin
        self.on_response_end = on_response_end
        self.on_user_transcription = on_user_transcription
        self.on_playback_buffer_clear = on_playback_buffer_clear
        self.on_chat_history = on_chat_history
        self.on_conversation_query_result = on_conversation_query_result
        self.on_fatal_error = on_fatal_error

        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session_initialized = False
        self._sample_rate: Optional[int] = None
        self._channels: Optional[int] = None
        self._packet_id_counter = 0
        self._pending_before_init: list[proto.ServiceBoundMessage] = []

        self._send_queue: asyncio.Queue[Optional[proto.ServiceBoundMessage]] = (
            asyncio.Queue()
        )
        self._pending_query_ids: deque[str] = deque()

        self._main_task: Optional[asyncio.Task] = None

    @property
    def session_initialized(self) -> bool:
        """True after the server has acknowledged ``InitializeSessionRequest``"""
        return self._session_initialized

    @property
    def sample_rate(self) -> Optional[int]:
        """Audio sample rate negotiated during initialization, or ``None``"""
        return self._sample_rate

    @property
    def channels(self) -> Optional[int]:
        """Channel count negotiated during initialization, or ``None``"""
        return self._channels

    def start(self) -> None:
        """Spawn the background task that drives ``client.run_with_retry()``.

        Idempotent — calling more than once is a no-op.
        """
        if self._main_task is not None:
            return
        self._should_stop = False
        self._main_task = asyncio.create_task(
            self._main(), name="DeepslateSession._main"
        )

    async def close(self) -> None:
        """Stop the session and cancel the background task.

        If this session was created via :meth:`create` it also closes the
        ``BaseDeepslateClient`` it owns.
        """
        self._should_stop = True
        if self._main_task is not None and not self._main_task.done():
            self._main_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._main_task
        self._main_task = None
        if self._owns_client:
            await self._client.aclose()

    @classmethod
    def create(
        cls,
        options: DeepslateOptions,
        *,
        vad_config: Optional[VadConfig] = None,
        tts_config: Optional[ElevenLabsTtsConfig] = None,
        user_agent: str = "DeepslateCore",
        http_session: Optional[Any] = None,
        on_text_fragment: Optional[OnTextFragment] = None,
        on_audio_chunk: Optional[OnAudioChunk] = None,
        on_tool_call: Optional[OnToolCall] = None,
        on_error: Optional[OnError] = None,
        on_response_begin: Optional[OnResponseBegin] = None,
        on_response_end: Optional[OnResponseEnd] = None,
        on_user_transcription: Optional[OnUserTranscription] = None,
        on_playback_buffer_clear: Optional[OnPlaybackBufferClear] = None,
        on_chat_history: Optional[OnChatHistory] = None,
        on_conversation_query_result: Optional[OnConversationQueryResult] = None,
        on_fatal_error: Optional[OnFatalError] = None,
    ) -> "DeepslateSession":
        """Create a session together with its own ``BaseDeepslateClient``.

        Use this when you don't need to share a ``BaseDeepslateClient``
        across multiple sessions.  The returned session owns the client
        and closes it automatically when :meth:`close` is called.
        """
        client = BaseDeepslateClient(
            opts=options, user_agent=user_agent, http_session=http_session
        )
        session = cls(
            client=client,
            options=options,
            vad_config=vad_config,
            tts_config=tts_config,
            on_text_fragment=on_text_fragment,
            on_audio_chunk=on_audio_chunk,
            on_tool_call=on_tool_call,
            on_error=on_error,
            on_response_begin=on_response_begin,
            on_response_end=on_response_end,
            on_user_transcription=on_user_transcription,
            on_playback_buffer_clear=on_playback_buffer_clear,
            on_chat_history=on_chat_history,
            on_conversation_query_result=on_conversation_query_result,
            on_fatal_error=on_fatal_error,
        )
        session._owns_client = True
        return session

    _TRIGGER_MODE_MAP = {
        TriggerMode.NO_TRIGGER: proto.InferenceTriggerMode.NO_TRIGGER,
        TriggerMode.QUEUE: proto.InferenceTriggerMode.QUEUE,
        TriggerMode.IMMEDIATE: proto.InferenceTriggerMode.IMMEDIATE,
    }

    async def send_audio(
        self,
        pcm_bytes: bytes,
        sample_rate: int,
        channels: int,
        trigger: TriggerMode = TriggerMode.IMMEDIATE,
    ) -> int:
        """Send raw PCM audio.

        Automatically initializes the session on the first call.  If the
        audio format changes after initialization a ``ReconfigureSessionRequest``
        is sent transparently.

        Returns the ``packet_id`` assigned to this frame.
        """
        await self._ensure_initialized(sample_rate, channels)

        if sample_rate != self._sample_rate or channels != self._channels:
            self._sample_rate = sample_rate
            self._channels = channels
            reconfig = proto.ReconfigureSessionRequest(
                input_audio_line=proto.AudioLineConfiguration(
                    sample_rate=sample_rate,
                    channel_count=channels,
                    sample_format=proto.SampleFormat.SIGNED_16_BIT,
                )
            )
            await self._enqueue_or_buffer(
                proto.ServiceBoundMessage(reconfigure_session_request=reconfig)
            )

        packet_id = self._next_packet_id()
        user_input = proto.UserInput(
            packet_id=packet_id,
            mode=self._TRIGGER_MODE_MAP[trigger],
            audio_data=proto.AudioData(data=pcm_bytes),
        )
        await self._enqueue_or_buffer(proto.ServiceBoundMessage(user_input=user_input))
        return packet_id

    async def send_text(
        self,
        text: str,
        trigger: TriggerMode = TriggerMode.IMMEDIATE,
    ) -> int:
        """Send a text ``UserInput``.

        Buffers the message if the session is not yet initialized.
        Returns the ``packet_id`` assigned.
        """
        packet_id = self._next_packet_id()
        user_input = proto.UserInput(
            packet_id=packet_id,
            mode=self._TRIGGER_MODE_MAP[trigger],
            text_data=proto.TextData(data=text),
        )
        await self._enqueue_or_buffer(proto.ServiceBoundMessage(user_input=user_input))
        return packet_id

    async def initialize(self, sample_rate: int = 24000, channels: int = 1) -> None:
        """Explicitly initialize the session without sending audio.

        Use this for text-only flows where ``send_audio()`` will never be
        called and the session still needs to be set up.
        """
        await self._ensure_initialized(sample_rate, channels)

    async def trigger_inference(self, instructions: Optional[str] = None) -> None:
        """Send a ``TriggerInference`` message to request a model reply."""
        trigger = proto.TriggerInference()
        if instructions is not None:
            trigger.extra_instructions = instructions
        await self._enqueue_or_buffer(proto.ServiceBoundMessage(trigger_inference=trigger))

    async def send_tool_response(self, call_id: str, result: Any) -> None:
        """Serialize ``result`` and send a ``ToolCallResponse``."""
        import json

        result_str = result if isinstance(result, str) else json.dumps(result)
        response = proto.ToolCallResponse(id=call_id, result=result_str)
        await self._enqueue_or_buffer(
            proto.ServiceBoundMessage(tool_call_response=response)
        )

    async def update_tools(self, tools: list[FunctionToolDict]) -> None:
        """Persist tool definitions and sync them to the server.

        If the session is not yet initialized the tools are buffered and sent
        immediately after ``InitializeSessionRequest``.  The full list is
        always re-sent on every reconnect so plugins do not need to replay it.
        """
        self._current_tools = tools
        msg = self._build_update_tools_msg(tools)
        if msg is not None:
            await self._enqueue_or_buffer(msg)

    async def reconfigure(
        self,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """Send a ``ReconfigureSessionRequest`` with the provided fields.

        Fields left as ``None`` are omitted from the request.
        """
        if system_prompt is None and temperature is None:
            return
        inference_config = proto.InferenceConfiguration()
        if system_prompt is not None:
            inference_config.system_prompt = system_prompt
        if temperature is not None:
            inference_config.temperature = temperature
        reconfig = proto.ReconfigureSessionRequest(
            inference_configuration=inference_config
        )
        await self._enqueue_or_buffer(
            proto.ServiceBoundMessage(reconfigure_session_request=reconfig)
        )

    async def send_direct_speech(
        self, text: str, include_in_history: bool = True
    ) -> None:
        """Bypass the LLM and speak ``text`` directly via TTS."""
        direct_speech = proto.DirectSpeech(text=text, include_in_history=include_in_history)
        await self._enqueue_or_buffer(
            proto.ServiceBoundMessage(direct_speech=direct_speech)
        )

    async def export_chat_history(self, await_pending: bool = False) -> None:
        """Request a chat history export.

        The result is delivered asynchronously via the ``on_chat_history``
        callback.
        """
        req = proto.ExportChatHistoryRequest(await_pending=await_pending)
        await self._enqueue_or_buffer(
            proto.ServiceBoundMessage(export_chat_history_request=req)
        )

    async def send_conversation_query(
        self,
        query_id: str,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
    ) -> None:
        """Send a side-channel ``ConversationQuery``"""
        if prompt is None and instructions is None:
            raise ValueError("At least one of 'prompt' or 'instructions' must be provided.")
        self._pending_query_ids.append(query_id)
        query = proto.ConversationQuery()
        if prompt is not None:
            query.prompt = prompt
        if instructions is not None:
            query.instructions = instructions
        await self._enqueue_or_buffer(proto.ServiceBoundMessage(conversation_query=query))

    async def report_playback_position(self, bytes_played: int) -> None:
        """Send a ``PlaybackPositionReport`` for server-side audio truncation."""
        report = proto.PlaybackPositionReport(bytes_played=bytes_played)
        await self._enqueue_or_buffer(
            proto.ServiceBoundMessage(playback_position_report=report)
        )

    def _next_packet_id(self) -> int:
        self._packet_id_counter += 1
        return self._packet_id_counter

    def _reset_state(self) -> None:
        """Reset all per-connection state before each reconnect attempt.

        Preserves ``_current_tools`` so they are re-synced after the new
        ``InitializeSessionRequest`` completes.
        """
        self._ws = None
        self._session_initialized = False
        self._sample_rate = None
        self._channels = None
        self._packet_id_counter = 0
        self._pending_before_init.clear()
        self._pending_query_ids.clear()
        # Replace with a fresh queue; the previous send loop has already been
        # cancelled before _run_ws is called again.
        self._send_queue = asyncio.Queue()

    async def _ensure_initialized(self, sample_rate: int, channels: int) -> None:
        """Idempotent session initialization."""
        if self._session_initialized:
            return

        self._sample_rate = sample_rate
        self._channels = channels

        init_request = build_initialize_request(
            sample_rate=sample_rate,
            num_channels=channels,
            vad_config=self._vad_config,
            system_prompt=self._options.system_prompt,
            tts_config=self._tts_config,
            temperature=self._options.temperature,
        )
        await self._send_queue.put(
            proto.ServiceBoundMessage(initialize_session_request=init_request)
        )
        logger.debug(
            f"DeepslateSession: initializing session ({sample_rate}Hz, {channels}ch)"
        )

        if self._current_tools:
            tools_msg = self._build_update_tools_msg(self._current_tools)
            if tools_msg is not None:
                await self._send_queue.put(tools_msg)

        for msg in self._pending_before_init:
            await self._send_queue.put(msg)
        self._pending_before_init.clear()

        self._session_initialized = True

    async def _enqueue_or_buffer(self, msg: proto.ServiceBoundMessage) -> None:
        """Route a message to the send queue or the pre-init buffer."""
        if self._session_initialized:
            await self._send_queue.put(msg)
        else:
            self._pending_before_init.append(msg)

    def _build_update_tools_msg(
        self, tools: list[FunctionToolDict]
    ) -> Optional[proto.ServiceBoundMessage]:
        if not tools:
            return None
        tool_definitions = []
        for tool in tools:
            func = tool.get("function", {})
            tool_definitions.append(
                proto.ToolDefinition(
                    name=func.get("name", ""),
                    description=func.get("description", ""),
                    parameters=dict_to_struct(func.get("parameters", {})),
                )
            )
        update_req = proto.UpdateToolDefinitionsRequest(tool_definitions=tool_definitions)
        return proto.ServiceBoundMessage(update_tool_definitions_request=update_req)

    async def _fire(self, cb: Optional[Callable], *args: Any) -> None:
        """Null-safe, exception-isolated async callback invocation.

        Exceptions raised inside a callback are logged and swallowed so that a
        misbehaving plugin callback cannot crash the receive loop.
        """
        if cb is None:
            return
        try:
            result = cb(*args)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception("DeepslateSession: unhandled exception in callback")

    async def _main(self) -> None:
        await self._client.run_with_retry(
            self._run_ws,
            should_continue=lambda: not self._should_stop,
            on_fatal_error=self._on_fatal_error,
        )

    async def _on_fatal_error(self, e: Exception) -> None:
        await self._fire(self.on_fatal_error, e)

    async def _run_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Run one WebSocket session.

        Called once per connection attempt by ``run_with_retry``.  Resets
        per-connection state, then runs the send loop and receive loop
        concurrently, exiting (and triggering a reconnect) as soon as either
        one completes or raises.
        """
        self._reset_state()
        self._ws = ws
        closing = False
        logger.info("DeepslateSession: connected to Deepslate Realtime API")

        async def _send_loop() -> None:
            nonlocal closing
            while True:
                msg = await self._send_queue.get()
                if msg is None:  # shutdown sentinel
                    closing = True
                    await ws.close()
                    return
                try:
                    await ws.send_bytes(msg.SerializeToString())
                except Exception as e:
                    logger.error(f"DeepslateSession: send error: {e}")
                    break

        async def _recv_loop() -> None:
            while True:
                raw = await ws.receive()

                if raw.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing:
                        return
                    close_code = ws.close_code
                    close_reason = raw.extra if raw.extra else "no reason provided"
                    logger.error(
                        f"DeepslateSession: WebSocket closed unexpectedly: "
                        f"code={close_code}, reason={close_reason}"
                    )
                    raise aiohttp.ClientError(
                        f"WebSocket closed unexpectedly: "
                        f"code={close_code}, reason={close_reason}"
                    )

                if raw.type == aiohttp.WSMsgType.ERROR:
                    raise aiohttp.ClientError(
                        f"WebSocket error: {ws.exception()}"
                    )

                if raw.type != aiohttp.WSMsgType.BINARY:
                    logger.warning(
                        f"DeepslateSession: unexpected message type: {raw.type}"
                    )
                    continue

                client_msg = proto.ClientBoundMessage()
                client_msg.ParseFromString(raw.data)
                await self._handle_server_message(client_msg)

        send_task = asyncio.create_task(_send_loop(), name="DeepslateSession._send_loop")
        recv_task = asyncio.create_task(_recv_loop(), name="DeepslateSession._recv_loop")
        try:
            done, _ = await asyncio.wait(
                [send_task, recv_task], return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()  # re-raise any exception to run_with_retry
        finally:
            send_task.cancel()
            recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await send_task
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await recv_task
            with contextlib.suppress(Exception):
                await ws.close()

    async def _handle_server_message(self, msg: proto.ClientBoundMessage) -> None:
        """Route a ``ClientBoundMessage`` to the appropriate callback."""
        payload_type = msg.WhichOneof("payload")

        if payload_type == "response_begin":
            await self._fire(self.on_response_begin)

        elif payload_type == "response_end":
            await self._fire(self.on_response_end)

        elif payload_type == "model_text_fragment":
            await self._fire(self.on_text_fragment, msg.model_text_fragment.text)

        elif payload_type == "model_audio_chunk":
            chunk = msg.model_audio_chunk
            if chunk.audio and chunk.audio.data:
                transcript: Optional[str] = chunk.transcript if chunk.transcript else None
                await self._fire(
                    self.on_audio_chunk,
                    chunk.audio.data,
                    self._sample_rate or 24000,
                    self._channels or 1,
                    transcript,
                )

        elif payload_type == "user_transcription_result":
            result = msg.user_transcription_result
            await self._fire(
                self.on_user_transcription,
                result.text,
                result.language or None,
                result.turn_id,
            )

        elif payload_type == "playback_clear_buffer":
            await self._fire(self.on_playback_buffer_clear)

        elif payload_type == "tool_call_request":
            req = msg.tool_call_request
            params = struct_to_dict(req.parameters) if req.HasField("parameters") else {}
            await self._fire(self.on_tool_call, req.id, req.name, params)

        elif payload_type == "conversation_query_result":
            result_text = msg.conversation_query_result.text
            query_id = self._pending_query_ids.popleft() if self._pending_query_ids else ""
            if not query_id:
                logger.warning(
                    "DeepslateSession: received conversation_query_result "
                    "with no pending query_id"
                )
            await self._fire(self.on_conversation_query_result, query_id, result_text)

        elif payload_type == "chat_history":
            messages = parse_chat_history(msg.chat_history)
            await self._fire(self.on_chat_history, messages)

        elif payload_type == "error":
            notification = msg.error
            category_name = proto.SessionErrorCategory.Name(notification.category)
            trace_id: Optional[str] = (
                notification.trace_id if notification.HasField("trace_id") else None
            )
            logger.error(
                f"DeepslateSession: server error [{category_name}]: "
                f"{notification.message}"
                + (f" (trace_id={trace_id})" if trace_id else "")
            )
            await self._fire(self.on_error, category_name, notification.message, trace_id)

        else:
            logger.debug(f"DeepslateSession: unhandled payload type: {payload_type}")