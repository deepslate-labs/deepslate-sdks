from __future__ import annotations

from urllib.parse import urlparse

from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct

from .options import ElevenLabsLocation
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