"""
Sonos API wrapper for Music Sync.

Provides direct control of Sonos speakers including Era 100
using the SoCo library (UPnP-based, works locally without cloud).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime

_LOGGER = logging.getLogger(__name__)

try:
    import soco
    from soco import SoCo
    SOCO_AVAILABLE = True
except ImportError:
    SOCO_AVAILABLE = False
    soco = None
    SoCo = None


@dataclass
class SonosSpeaker:
    """Represents a Sonos speaker."""

    name: str
    ip: str
    model: str = ""
    model_number: str = ""
    is_era: bool = False  # Era 100, Era 300, etc.
    device: Any = None  # SoCo instance

    # Sync-related
    offset_ms: float = 0.0
    last_position_ms: float = 0.0
    last_position_time: datetime = field(default_factory=datetime.now)


class SonosAPI:
    """
    Direct API wrapper for Sonos speakers.

    Uses SoCo library for local UPnP control.
    Works with all Sonos models including Era 100/300.
    """

    # Era models that don't support AirPlay streaming from 3rd party
    ERA_MODELS = ['Era 100', 'Era 300', 'Move 2', 'Roam 2', 'Arc Ultra']

    def __init__(self):
        """Initialize the Sonos API."""
        if not SOCO_AVAILABLE:
            raise RuntimeError("SoCo library not available. Install with: pip install soco")

        self.speakers: dict[str, SonosSpeaker] = {}
        self._discovery_done = False

    async def discover(self, timeout: int = 5) -> list[SonosSpeaker]:
        """
        Discover all Sonos speakers on the network.

        Returns:
            List of discovered SonosSpeaker objects
        """
        _LOGGER.info("Discovering Sonos speakers...")

        # Run discovery in executor (blocking call)
        loop = asyncio.get_event_loop()
        devices = await loop.run_in_executor(
            None,
            lambda: soco.discover(timeout=timeout)
        )

        if not devices:
            _LOGGER.warning("No Sonos speakers found")
            return []

        self.speakers.clear()

        for device in devices:
            try:
                info = device.get_speaker_info()
                model = info.get('model_name', 'Unknown')
                model_number = info.get('model_number', '')

                is_era = any(era in model for era in self.ERA_MODELS)

                speaker = SonosSpeaker(
                    name=device.player_name,
                    ip=device.ip_address,
                    model=model,
                    model_number=model_number,
                    is_era=is_era,
                    device=device,
                )

                self.speakers[device.player_name] = speaker
                _LOGGER.info(
                    "Found Sonos: %s (%s) at %s %s",
                    device.player_name, model, device.ip_address,
                    "[ERA - no RAOP]" if is_era else ""
                )

            except Exception as err:
                _LOGGER.error("Error getting speaker info: %s", err)

        self._discovery_done = True
        return list(self.speakers.values())

    def get_speaker(self, name: str) -> Optional[SonosSpeaker]:
        """Get speaker by name."""
        return self.speakers.get(name)

    def get_speaker_by_ip(self, ip: str) -> Optional[SonosSpeaker]:
        """Get speaker by IP address."""
        for speaker in self.speakers.values():
            if speaker.ip == ip:
                return speaker
        return None

    async def play_uri(
        self,
        speaker_name: str,
        uri: str,
        title: str = "Music Sync"
    ) -> bool:
        """
        Play an audio URI on a Sonos speaker.

        Args:
            speaker_name: Name of the speaker
            uri: HTTP URL to audio file/stream
            title: Display title

        Returns:
            True if successful
        """
        speaker = self.speakers.get(speaker_name)
        if not speaker or not speaker.device:
            _LOGGER.error("Speaker not found: %s", speaker_name)
            return False

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: speaker.device.play_uri(uri, title=title)
            )
            _LOGGER.info("Playing on %s: %s", speaker_name, uri)
            return True

        except Exception as err:
            _LOGGER.error("Failed to play on %s: %s", speaker_name, err)
            return False

    async def play(self, speaker_name: str) -> bool:
        """Resume playback."""
        speaker = self.speakers.get(speaker_name)
        if not speaker or not speaker.device:
            return False

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, speaker.device.play)
            return True
        except Exception as err:
            _LOGGER.error("Failed to play %s: %s", speaker_name, err)
            return False

    async def pause(self, speaker_name: str) -> bool:
        """Pause playback."""
        speaker = self.speakers.get(speaker_name)
        if not speaker or not speaker.device:
            return False

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, speaker.device.pause)
            return True
        except Exception as err:
            _LOGGER.error("Failed to pause %s: %s", speaker_name, err)
            return False

    async def stop(self, speaker_name: str) -> bool:
        """Stop playback."""
        speaker = self.speakers.get(speaker_name)
        if not speaker or not speaker.device:
            return False

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, speaker.device.stop)
            return True
        except Exception as err:
            _LOGGER.error("Failed to stop %s: %s", speaker_name, err)
            return False

    async def seek(self, speaker_name: str, position_ms: float) -> bool:
        """
        Seek to position in milliseconds.

        Args:
            speaker_name: Speaker name
            position_ms: Position in milliseconds

        Returns:
            True if successful
        """
        speaker = self.speakers.get(speaker_name)
        if not speaker or not speaker.device:
            return False

        # Convert ms to HH:MM:SS format
        total_seconds = int(position_ms / 1000)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        position_str = f"{hours}:{minutes:02d}:{seconds:02d}"

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: speaker.device.seek(position_str)
            )
            _LOGGER.debug("Seeked %s to %s", speaker_name, position_str)
            return True
        except Exception as err:
            _LOGGER.error("Failed to seek %s: %s", speaker_name, err)
            return False

    async def get_position_ms(self, speaker_name: str) -> Optional[float]:
        """
        Get current playback position in milliseconds.

        Returns:
            Position in ms, or None if unavailable
        """
        speaker = self.speakers.get(speaker_name)
        if not speaker or not speaker.device:
            return None

        try:
            loop = asyncio.get_event_loop()
            track_info = await loop.run_in_executor(
                None,
                speaker.device.get_current_track_info
            )

            position = track_info.get('position', '0:00:00')

            # Parse "H:MM:SS" to milliseconds
            parts = position.split(':')
            if len(parts) == 3:
                h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
                position_ms = (h * 3600 + m * 60 + s) * 1000

                # Update speaker state
                speaker.last_position_ms = position_ms
                speaker.last_position_time = datetime.now()

                return position_ms

            return None

        except Exception as err:
            _LOGGER.error("Failed to get position for %s: %s", speaker_name, err)
            return None

    async def get_transport_state(self, speaker_name: str) -> Optional[str]:
        """
        Get transport state (PLAYING, PAUSED_PLAYBACK, STOPPED, TRANSITIONING).
        """
        speaker = self.speakers.get(speaker_name)
        if not speaker or not speaker.device:
            return None

        try:
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(
                None,
                speaker.device.get_current_transport_info
            )
            return info.get('current_transport_state')
        except Exception as err:
            _LOGGER.error("Failed to get state for %s: %s", speaker_name, err)
            return None

    async def set_volume(self, speaker_name: str, volume: int) -> bool:
        """Set volume (0-100)."""
        speaker = self.speakers.get(speaker_name)
        if not speaker or not speaker.device:
            return False

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: setattr(speaker.device, 'volume', volume)
            )
            return True
        except Exception as err:
            _LOGGER.error("Failed to set volume on %s: %s", speaker_name, err)
            return False

    async def get_volume(self, speaker_name: str) -> Optional[int]:
        """Get volume (0-100)."""
        speaker = self.speakers.get(speaker_name)
        if not speaker or not speaker.device:
            return None

        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: speaker.device.volume
            )
        except Exception as err:
            _LOGGER.error("Failed to get volume from %s: %s", speaker_name, err)
            return None

    # =========================================================================
    # Grouping (Sonos-to-Sonos only)
    # =========================================================================

    async def join_group(self, speaker_name: str, master_name: str) -> bool:
        """
        Join a speaker to a master speaker's group.

        Note: This creates Sonos-internal sync, not cross-platform sync.
        """
        speaker = self.speakers.get(speaker_name)
        master = self.speakers.get(master_name)

        if not speaker or not master:
            return False

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: speaker.device.join(master.device)
            )
            _LOGGER.info("Joined %s to %s", speaker_name, master_name)
            return True
        except Exception as err:
            _LOGGER.error("Failed to join group: %s", err)
            return False

    async def unjoin(self, speaker_name: str) -> bool:
        """Remove speaker from its current group."""
        speaker = self.speakers.get(speaker_name)
        if not speaker or not speaker.device:
            return False

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, speaker.device.unjoin)
            return True
        except Exception as err:
            _LOGGER.error("Failed to unjoin %s: %s", speaker_name, err)
            return False
