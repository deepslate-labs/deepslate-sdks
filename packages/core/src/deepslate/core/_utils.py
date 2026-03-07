from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct

from .options import ElevenLabsLocation, ElevenLabsTtsConfig, VadConfig
from .proto import realtime_pb2 as proto


def duration_from_ms(ms: int) -> proto.Duration:
    """Convert milliseconds to protobuf Duration."""
    seconds = ms // 1000
    nanos = (ms % 1000) * 1_000_000
    return proto.Duration(seconds=seconds, nanos=nanos)


def struct_to_dict(struct: Struct) -> dict:
    """Convert protobuf Struct to Python dict."""
    return json_format.MessageToDict(struct)


def dict_to_struct(d: dict) -> Struct:
    """Convert Python dict to protobuf Struct."""
    struct = Struct()
    json_format.ParseDict(d, struct)
    return struct


def build_ws_url(base_url: str, vendor_id: str, organization_id: str) -> str:
    """Build WebSocket URL for Deepslate realtime endpoint.

    Args:
        base_url: Base URL (e.g., "https://app.deepslate.eu")
        vendor_id: Vendor ID
        organization_id: Organization ID

    Returns:
        WebSocket URL for the realtime endpoint
    """
    parsed = urlparse(base_url)

    if parsed.scheme == "https":
        scheme = "wss"
    elif parsed.scheme == "http":
        scheme = "ws"
    else:
        scheme = parsed.scheme

    host = parsed.netloc or parsed.path

    return f"{scheme}://{host}/api/v1/vendors/{vendor_id}/organizations/{organization_id}/realtime"


# Maps the Python ElevenLabsLocation enum to the proto ElevenLabsLocation enum value.
# Shared by deepslate-livekit and deepslate-pipecat when building InitializeSessionRequest.
ELEVENLABS_LOCATION_MAP: dict[ElevenLabsLocation, proto.ElevenLabsLocation] = {
    ElevenLabsLocation.US: proto.ElevenLabsLocation.US,
    ElevenLabsLocation.EU: proto.ElevenLabsLocation.EU,
    ElevenLabsLocation.INDIA: proto.ElevenLabsLocation.INDIA,
}


def build_initialize_request(
    sample_rate: int,
    num_channels: int,
    vad_config: VadConfig,
    system_prompt: str,
    tts_config: Optional[ElevenLabsTtsConfig] = None,
    temperature: float = 1.0,
) -> proto.InitializeSessionRequest:
    """Build a proto.InitializeSessionRequest from core configuration objects.

    Shared by deepslate-pipecat and deepslate-livekit to avoid duplicating
    the protobuf construction logic in each plugin.
    """
    tts_proto = None
    if tts_config is not None:
        el_config = proto.ElevenLabsTtsConfiguration(
            api_key=tts_config.api_key,
            voice_id=tts_config.voice_id,
            location=ELEVENLABS_LOCATION_MAP[tts_config.location],
        )
        if tts_config.model_id:
            el_config.model_id = tts_config.model_id
        if tts_config.voice_settings is not None:
            el_config.voice_settings.CopyFrom(tts_config.voice_settings.to_proto())
        tts_proto = proto.TtsConfiguration(eleven_labs=el_config)

    audio_line = proto.AudioLineConfiguration(
        sample_rate=sample_rate,
        channel_count=num_channels,
        sample_format=proto.SampleFormat.SIGNED_16_BIT,
    )

    return proto.InitializeSessionRequest(
        input_audio_line=audio_line,
        output_audio_line=audio_line,
        vad_configuration=proto.VadConfiguration(
            confidence_threshold=vad_config.confidence_threshold,
            min_volume=vad_config.min_volume,
            start_duration=duration_from_ms(vad_config.start_duration_ms),
            stop_duration=duration_from_ms(vad_config.stop_duration_ms),
            backbuffer_duration=duration_from_ms(vad_config.backbuffer_duration_ms),
        ),
        inference_configuration=proto.InferenceConfiguration(
            system_prompt=system_prompt,
            temperature=temperature,
        ),
        tts_configuration=tts_proto,
    )