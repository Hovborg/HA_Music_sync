"""Platform API wrappers for Music Sync."""
from .sonos_api import SonosAPI, SonosSpeaker
from .cast_api import GoogleCastAPI, CastSpeaker

__all__ = ["SonosAPI", "SonosSpeaker", "GoogleCastAPI", "CastSpeaker"]
