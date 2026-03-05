"""
Deepslate + Pipecat — Daily.co Voice Bot Example
=================================================
A voice bot that joins a Daily.co WebRTC room and responds to speech.
Demonstrates ElevenLabs server-side TTS and two example function tools
(weather lookup and location detection).

Setup
-----
1. Copy .env.example to .env and fill in your credentials.
2. Create a Daily.co room and paste its URL into DAILY_ROOM_URL.
3. Run:  python simple_bot.py

Requirements
------------
    pip install deepslate-pipecat "pipecat-ai[daily]" aiohttp python-dotenv loguru
"""

import asyncio
import os
import random
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import LLMSetToolsFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.services.daily import DailyParams, DailyTransport

from deepslate.pipecat import DeepslateOptions, DeepslateRealtimeLLMService, ElevenLabsTtsConfig

load_dotenv(override=True)

logger.remove()
logger.add(sys.stderr, level="DEBUG")


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling JSON schema format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Get the current weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city or location to look up weather for.",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_location",
            "description": "Get the user's current location.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ---------------------------------------------------------------------------
# Function handlers
# ---------------------------------------------------------------------------

async def lookup_weather(params: FunctionCallParams):
    location = params.arguments.get("location", "unknown")
    result = {
        "location": location,
        "temperature_celsius": random.randint(10, 35),
        "precipitation": random.choice(["none", "light", "moderate", "heavy"]),
        "air_pressure_hpa": random.randint(900, 1100),
    }
    logger.info(f"lookup_weather({location}) → {result}")
    await params.result_callback(result)


async def get_current_location(params: FunctionCallParams):
    result = {"location": "Berlin"}
    logger.info(f"get_current_location() → {result}")
    await params.result_callback(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    daily_api_key = os.getenv("DAILY_API_KEY")
    daily_room_url = os.getenv("DAILY_ROOM_URL")

    if not daily_api_key or not daily_room_url:
        logger.error("Please set DAILY_API_KEY and DAILY_ROOM_URL in your .env file")
        return

    # Fetch a meeting token so the bot can join the Daily room as a trusted participant.
    async with aiohttp.ClientSession() as http:
        headers = {"Authorization": f"Bearer {daily_api_key}"}
        room_name = daily_room_url.split("/")[-1]
        async with http.post(
            "https://api.daily.co/v1/meeting-tokens",
            headers=headers,
            json={"properties": {"room_name": room_name}},
        ) as r:
            if r.status != 200:
                logger.error(f"Failed to get Daily token: {await r.text()}")
                return
            token = (await r.json())["token"]

    # 1. Transport — Daily.co WebRTC
    transport = DailyTransport(
        room_url=daily_room_url,
        token=token,
        bot_name="Deepslate Bot",
        params=DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            camera_out_enabled=False,
            vad_enabled=False,  # Deepslate handles VAD server-side
        ),
    )

    # 2. Deepslate LLM service
    try:
        opts = DeepslateOptions.from_env(
            system_prompt="You are a friendly and helpful AI assistant. Keep your answers concise."
        )
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return

    tts = ElevenLabsTtsConfig.from_env()
    llm = DeepslateRealtimeLLMService(options=opts, tts_config=tts)

    llm.register_function("lookup_weather", lookup_weather)
    llm.register_function("get_current_location", get_current_location)

    # 3. Pipeline
    pipeline = Pipeline([
        transport.input(),
        llm,
        transport.output(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # Sync tool definitions to Deepslate (queued before the pipeline starts).
    await task.queue_frame(LLMSetToolsFrame(tools=TOOLS))

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info(f"Participant {participant['id']} joined. Listening...")

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info(f"Participant {participant['id']} left.")
        await task.cancel()

    # 4. Run
    runner = PipelineRunner()
    logger.info("Starting pipeline runner...")
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())