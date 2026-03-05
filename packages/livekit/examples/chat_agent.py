"""
Deepslate + LiveKit Agents — Chat Agent Example
================================================
A voice AI assistant that joins a LiveKit room and responds to speech.
Includes two example function tools (weather lookup and location detection)
to demonstrate Deepslate's function-calling support.

Setup
-----
1. Copy .env.example to .env and fill in your credentials.
2. Start a local LiveKit server (or point LIVEKIT_URL at a hosted one).
3. Run:  python chat_agent.py dev

Requirements
------------
    pip install deepslate-livekit python-dotenv
"""

import os
import random
from typing import Any

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, RunContext, function_tool, room_io

from deepslate.livekit import ElevenLabsTtsConfig, RealtimeModel

# Load .env from the examples directory, then allow .env.local to override
# (useful for keeping secrets out of version control).
_script_dir = os.path.dirname(os.path.realpath(__file__))
load_dotenv(os.path.join(_script_dir, ".env"))
load_dotenv(os.path.join(_script_dir, ".env.local"), override=True)


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")


# ---------------------------------------------------------------------------
# Function tools
# ---------------------------------------------------------------------------

@function_tool()
async def lookup_weather(
    _context: RunContext,
    location: str,
) -> dict[str, Any]:
    """Get the current weather for a given location."""
    return {
        "location": location,
        "temperature_celsius": random.randint(10, 35),
        "precipitation": random.choice(["none", "light", "moderate", "heavy"]),
        "air_pressure_hpa": random.randint(900, 1100),
    }


@function_tool()
async def get_current_location(
    _context: RunContext,
) -> str:
    """Get the user's current location."""
    return "Berlin"


# ---------------------------------------------------------------------------
# Server + session
# ---------------------------------------------------------------------------

server = AgentServer()


@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    session = AgentSession(
        llm=RealtimeModel(
            # DEEPSLATE_WS_URL can be set to override the default endpoint,
            # which is useful for local development/testing.
            ws_url=os.environ.get("DEEPSLATE_WS_URL"),
            tts_config=ElevenLabsTtsConfig.from_env(),
        ),
        tools=[lookup_weather, get_current_location],
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)