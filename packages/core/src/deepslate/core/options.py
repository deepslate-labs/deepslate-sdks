from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional


@dataclass
class DeepslateOptions:
    """Core Deepslate connection and model options."""

    vendor_id: str
    """Deepslate vendor ID."""

    organization_id: str
    """Deepslate organization ID."""

    api_key: str
    """Deepslate API key."""

    base_url: str = "https://app.deepslate.eu"
    """Base URL for the Deepslate API."""

    system_prompt: str = "You are a helpful assistant."
    """System prompt dictating the behavior of the model."""

    temperature: float = 1.0
    """Sampling temperature for the model (0.0 to 2.0). Higher values produce more random output."""

    ws_url: Optional[str] = None
    """Optional direct WebSocket URL (bypasses standard auth URL construction)."""

    max_retries: int = 3
    """Maximum number of reconnection attempts before giving up."""

    generate_reply_timeout: float = 30.0
    """Timeout in seconds for generate_reply (0 = no timeout)."""

    @classmethod
    def from_env(
        cls,
        vendor_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> "DeepslateOptions":
        """Create options, falling back to DEEPSLATE_... environment variables."""
        resolved_vendor_id = vendor_id or os.environ.get("DEEPSLATE_VENDOR_ID")
        if not resolved_vendor_id:
            raise ValueError(
                "Deepslate vendor ID required. "
                "Provide vendor_id or set DEEPSLATE_VENDOR_ID env var."
            )

        resolved_org_id = organization_id or os.environ.get("DEEPSLATE_ORGANIZATION_ID")
        if not resolved_org_id:
            raise ValueError(
                "Deepslate organization ID required. "
                "Provide organization_id or set DEEPSLATE_ORGANIZATION_ID env var."
            )

        resolved_api_key = api_key or os.environ.get("DEEPSLATE_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Deepslate API key required. "
                "Provide api_key or set DEEPSLATE_API_KEY env var."
            )

        return cls(
            vendor_id=resolved_vendor_id,
            organization_id=resolved_org_id,
            api_key=resolved_api_key,
            **kwargs,
        )


@dataclass
class VadConfig:
    """Voice Activity Detection configuration handled server-side by Deepslate."""

    confidence_threshold: float = 0.5
    """Minimum confidence required to consider audio as speech (0.0 to 1.0)."""

    min_volume: float = 0.01
    """Minimum volume level to consider audio as speech (0.0 to 1.0)."""

    start_duration_ms: int = 200
    """Duration of speech to detect start of speech (milliseconds)."""

    stop_duration_ms: int = 500
    """Duration of silence to detect end of speech (milliseconds)."""

    backbuffer_duration_ms: int = 1000
    """Duration of audio to buffer before speech detection (milliseconds)."""


class ElevenLabsLocation(Enum):
    """ElevenLabs API endpoint region.

    See: https://elevenlabs.io/docs/overview/administration/data-residency
    """

    US = "US"
    """US endpoint (default): api.elevenlabs.io"""

    EU = "EU"
    """EU endpoint. Requires enterprise access to ElevenLabs."""

    INDIA = "INDIA"
    """India endpoint. Requires enterprise access to ElevenLabs."""


@dataclass
class ElevenLabsVoiceSettingsConfig:
    """ElevenLabs voice settings for fine-grained TTS control."""

    stability: Optional[float] = None
    """Voice stability (0.0 to 1.0). Lower values add expressiveness."""

    similarity_boost: Optional[float] = None
    """Voice clarity and similarity boost (0.0 to 1.0)."""

    style: Optional[float] = None
    """Style exaggeration (0.0 to 1.0). Not available for all models."""

    use_speaker_boost: Optional[bool] = None
    """Boost similarity to the original speaker."""

    speed: Optional[float] = None
    """Speaking speed multiplier."""

    def to_proto(self):
        """Convert to a ``proto.ElevenLabsVoiceSettings`` protobuf message."""
        from deepslate.core.proto import realtime_pb2 as proto

        kwargs = {}
        if self.stability is not None:
            kwargs["stability"] = self.stability
        if self.similarity_boost is not None:
            kwargs["similarity_boost"] = self.similarity_boost
        if self.style is not None:
            kwargs["style"] = self.style
        if self.use_speaker_boost is not None:
            kwargs["use_speaker_boost"] = self.use_speaker_boost
        if self.speed is not None:
            kwargs["speed"] = self.speed

        return proto.ElevenLabsVoiceSettings(**kwargs)


@dataclass
class ElevenLabsTtsConfig:
    """ElevenLabs TTS configuration for Deepslate-hosted TTS.

    When provided, audio output is enabled directly from Deepslate using
    ElevenLabs for text-to-speech synthesis.
    """

    api_key: str
    """ElevenLabs API key."""

    voice_id: str
    """Voice ID (e.g., '21m00Tcm4TlvDq8ikWAM' for Rachel)."""

    model_id: Optional[str] = None
    """Model ID (e.g., 'eleven_turbo_v2'). Uses ElevenLabs default if not set."""

    location: ElevenLabsLocation = ElevenLabsLocation.US
    """ElevenLabs API endpoint region. Defaults to US."""

    voice_settings: Optional[ElevenLabsVoiceSettingsConfig] = None
    """Optional fine-grained voice settings (stability, similarity boost, etc.)."""

    @classmethod
    def from_env(
        cls,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        location: Optional[ElevenLabsLocation] = None,
    ) -> "ElevenLabsTtsConfig":
        """Create config, falling back to ELEVENLABS_... environment variables."""
        resolved_api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "ElevenLabs API key required. "
                "Provide api_key or set ELEVENLABS_API_KEY env var."
            )

        resolved_voice_id = voice_id or os.environ.get("ELEVENLABS_VOICE_ID")
        if not resolved_voice_id:
            raise ValueError(
                "ElevenLabs voice ID required. "
                "Provide voice_id or set ELEVENLABS_VOICE_ID env var."
            )

        resolved_model_id = model_id or os.environ.get("ELEVENLABS_MODEL_ID")

        return cls(
            api_key=resolved_api_key,
            voice_id=resolved_voice_id,
            model_id=resolved_model_id,
            location=location or ElevenLabsLocation.US,
        )