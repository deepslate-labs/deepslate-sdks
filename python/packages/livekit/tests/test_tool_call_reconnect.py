"""A tool call that is running when the websocket drops must still be answered.

``proto/realtime.proto`` says every ``ToolCallRequest`` "must be responded to
with a corresponding ToolCallResponse". The Node core and the Pipecat adapter
send a result that arrives after a reconnect on the new connection. These tests
check that the LiveKit adapter does the same.

The server is a local aiohttp stand-in that speaks the binary protobuf framing.
No network access outside 127.0.0.1 and no LiveKit room is needed.

Run from ``python/packages/livekit``::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import json
import unittest

from aiohttp import WSMsgType, web
from livekit import rtc
from livekit.agents import llm

from deepslate.core.proto import realtime_pb2 as pb
from deepslate.livekit import RealtimeModel


class _StubServer:
    """Sends one tool call on the first connection, then drops it."""

    def __init__(self, *, repeat_request: bool = False) -> None:
        self.repeat_request = repeat_request
        self.connections = 0
        # (connection number, payload kind, message)
        self.received: list[tuple[int, str, pb.ServiceBoundMessage]] = []
        self.port = 0
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/", self._handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    def tool_responses(self, call_id: str) -> list[tuple[int, str]]:
        return [
            (conn, msg.tool_call_response.result)
            for conn, kind, msg in self.received
            if kind == "tool_call_response" and msg.tool_call_response.id == call_id
        ]

    async def _send_tool_request(self, ws: web.WebSocketResponse) -> None:
        params = pb.ToolCallRequest().parameters
        params.update({"slot": "2026-12-01T10:00"})
        request = pb.ToolCallRequest(
            id="call-1", name="book_appointment", parameters=params, turn_id=1
        )
        await ws.send_bytes(
            pb.ClientBoundMessage(tool_call_request=request).SerializeToString()
        )

    async def _handler(self, request: web.Request) -> web.WebSocketResponse:
        self.connections += 1
        conn = self.connections
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for raw in ws:
            if raw.type != WSMsgType.BINARY:
                continue
            msg = pb.ServiceBoundMessage()
            msg.ParseFromString(raw.data)
            kind = msg.WhichOneof("payload") or ""
            if kind == "user_input":
                continue  # audio keep-alive frames
            self.received.append((conn, kind, msg))
            if kind != "initialize_session_request":
                continue
            await ws.send_bytes(
                pb.ClientBoundMessage(session_ready=pb.SessionReady()).SerializeToString()
            )
            if conn != 1:
                continue
            await asyncio.sleep(0.2)
            await ws.send_bytes(
                pb.ClientBoundMessage(
                    response_begin=pb.ResponseBegin(turn_id=1)
                ).SerializeToString()
            )
            await self._send_tool_request(ws)
            if self.repeat_request:
                await asyncio.sleep(0.05)
                await self._send_tool_request(ws)
            await asyncio.sleep(0.3)
            await ws.close(code=1011, message=b"test: drop")
        return ws


@llm.function_tool
async def book_appointment(slot: str) -> str:
    """Book an appointment in the given slot."""
    return "unused: the test runs the tool body itself"


class _Harness:
    """Plays the part livekit-agents plays around the realtime session."""

    def __init__(self, server: _StubServer, tool_seconds: float) -> None:
        self.server = server
        self.tool_seconds = tool_seconds
        self.tool_runs = 0
        self.function_calls = 0
        self._tasks: set[asyncio.Task] = set()

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _run_tool(self, session, fc) -> None:
        self.tool_runs += 1
        await asyncio.sleep(self.tool_seconds)
        output = json.dumps({"booked": True, "confirmation": f"CONF-{self.tool_runs}"})
        ctx = session.chat_ctx
        ctx.items.append(
            llm.FunctionCall(call_id=fc.call_id, name=fc.name, arguments=fc.arguments)
        )
        ctx.items.append(
            llm.FunctionCallOutput(
                call_id=fc.call_id, name=fc.name, output=output, is_error=False
            )
        )
        await session.update_chat_ctx(ctx)

    async def _consume(self, session, generation) -> None:
        async for fc in generation.function_stream:
            self.function_calls += 1
            self._spawn(self._run_tool(session, fc))

    async def run(self, run_seconds: float) -> None:
        model = RealtimeModel(
            vendor_id="v",
            organization_id="o",
            api_key="test-not-a-key",
            base_url="http://127.0.0.1:9",
            ws_url=f"ws://127.0.0.1:{self.server.port}/",
            tts_config=None,
        )
        session = model.session()
        session.on("generation_created", lambda g: self._spawn(self._consume(session, g)))
        await session.update_tools([book_appointment])

        async def keepalive() -> None:
            frame = rtc.AudioFrame(
                data=b"\x00" * 960,
                sample_rate=24000,
                num_channels=1,
                samples_per_channel=480,
            )
            while True:
                session.push_audio(frame)
                await asyncio.sleep(0.02)

        ka = self._spawn(keepalive())
        try:
            await asyncio.sleep(run_seconds)
        finally:
            ka.cancel()
            await asyncio.wait_for(session.aclose(), 5)
            for task in list(self._tasks):
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            await model.aclose()


class ToolCallAcrossReconnectTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.server = _StubServer()
        await self.server.start()

    async def asyncTearDown(self) -> None:
        await self.server.stop()

    async def test_result_after_reconnect_is_sent_on_the_new_connection(self) -> None:
        # The client waits 2 s before redialling, so a 3.5 s tool returns after
        # the second connection is ready.
        harness = _Harness(self.server, tool_seconds=3.5)
        await harness.run(run_seconds=7.0)

        self.assertGreaterEqual(self.server.connections, 2)
        self.assertEqual(harness.tool_runs, 1)
        responses = self.server.tool_responses("call-1")
        self.assertEqual(
            [conn for conn, _ in responses],
            [2],
            "the late result was not sent on the second connection",
        )
        self.assertIn("CONF-1", responses[0][1])

    async def test_result_during_backoff_is_sent_on_the_new_connection(self) -> None:
        # A 1 s tool returns while the client is waiting to redial.
        harness = _Harness(self.server, tool_seconds=1.0)
        await harness.run(run_seconds=6.0)

        self.assertGreaterEqual(self.server.connections, 2)
        self.assertEqual(harness.tool_runs, 1)
        responses = self.server.tool_responses("call-1")
        self.assertEqual(
            [conn for conn, _ in responses],
            [2],
            "the result produced during the backoff was not sent",
        )
        self.assertIn("CONF-1", responses[0][1])


class RepeatedToolCallRequestTest(unittest.IsolatedAsyncioTestCase):
    async def test_request_repeated_while_the_tool_runs_is_not_run_twice(self) -> None:
        # The server sends the same ToolCallRequest id twice, 50 ms apart.
        server = _StubServer(repeat_request=True)
        await server.start()
        try:
            harness = _Harness(server, tool_seconds=1.0)
            await harness.run(run_seconds=6.0)
        finally:
            await server.stop()

        self.assertEqual(harness.function_calls, 1)
        self.assertEqual(harness.tool_runs, 1)
        responses = server.tool_responses("call-1")
        self.assertEqual(len(responses), 1)
        self.assertIn("CONF-1", responses[0][1])


if __name__ == "__main__":
    unittest.main()
