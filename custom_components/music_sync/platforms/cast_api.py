"""
Google Cast API wrapper for Music Sync.

Provides direct control of Google Cast devices (Google Home, Nest, Chromecast)
using the pychromecast library.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime

_LOGGER = logging.getLogger(__name__)

try:
    import pychromecast
    from pychromecast.controllers.media import MediaController
    PYCHROMECAST_AVAILABLE = True
except ImportError:
    PYCHROMECAST_AVAILABLE = False
    pychromecast = None


@dataclass
class CastSpeaker:
    """Represents a Google Cast device."""

    name: str
    ip: str
    port: int = 8009
    model: str = ""
    cast_type: str = ""  # 'cast', 'audio', 'group'
    cast: Any = None  # Chromecast instance

    # Sync-related
    offset_ms: float = 0.0
    last_position_ms: float = 0.0
    last_position_time: datetime = field(default_factory=datetime.now)


class GoogleCastAPI:
    """
    Direct API wrapper for Google Cast devices.

    Uses pychromecast library for local mDNS-based control.
    """

    def __init__(self):
        """Initialize the Cast API."""
        if not PYCHROMECAST_AVAILABLE:
            raise RuntimeError(
                "pychromecast library not available. "
                "Install with: pip install pychromecast"
            )

        self.speakers: dict[str, CastSpeaker] = {}
        self._browser = None
        self._discovery_done = False

    async def discover(self, timeout: int = 10) -> list[CastSpeaker]:
        """
        Discover all Cast devices on the network.

        Returns:
            List of discovered CastSpeaker objects
        """
        _LOGGER.info("Discovering Google Cast devices...")

        loop = asyncio.get_event_loop()

        # Run blocking discovery in executor
        chromecasts, browser = await loop.run_in_executor(
            None,
            lambda: pychromecast.get_chromecasts(timeout=timeout)
        )

        self._browser = browser
        self.speakers.clear()

        for cc in chromecasts:
            try:
                # Wait for connection
                await loop.run_in_executor(None, cc.wait)

                speaker = CastSpeaker(
                    name=cc.name,
                    ip=cc.host,
                    port=cc.port,
                    model=cc.model_name,
                    cast_type=cc.cast_type,
                    cast=cc,
                )

                self.speakers[cc.name] = speaker
                _LOGGER.info(
                    "Found Cast: %s (%s) at %s [%s]",
                    cc.name, cc.model_name, cc.host, cc.cast_type
                )

            except Exception as err:
                _LOGGER.error("Error connecting to Cast device: %s", err)

        self._discovery_done = True
        return list(self.speakers.values())

    def stop_discovery(self):
        """Stop the discovery browser."""
        if self._browser:
            self._browser.stop_discovery()

    def get_speaker(self, name: str) -> Optional[CastSpeaker]:
        """Get speaker by name."""
        return self.speakers.get(name)

    def get_speaker_by_ip(self, ip: str) -> Optional[CastSpeaker]:
        """Get speaker by IP address."""
        for speaker in self.speakers.values():
            if speaker.ip == ip:
                return speaker
        return None

    async def play_url(
        self,
        speaker_name: str,
        url: str,
        content_type: str = 'audio/mp3',
        stream_type: str = 'BUFFERED',
        title: str = "Music Sync",
    ) -> bool:
        """
        Play an audio URL on a Cast device.

        Args:
            speaker_name: Name of the speaker
            url: HTTP URL to audio file/stream
            content_type: MIME type (audio/mp3, audio/flac, etc.)
            stream_type: 'BUFFERED' or 'LIVE'
            title: Display title

        Returns:
            True if successful
        """
        speaker = self.speakers.get(speaker_name)
        if not speaker or not speaker.cast:
            _LOGGER.error("Speaker not found: %s", speaker_name)
            return False

        try:
            mc = speaker.cast.media_controller

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: mc.play_media(
                    url,
                    content_type,
                    stream_type=stream_type,
                    title=title,
                )
            )

            # Wait for media to start
            await loop.run_in_executor(
                None,
                lambda: mc.block_until_active(timeout=30)
            )

            _LOGGER.info("Playing on %s: %s", speaker_name, url)
            return True

        except Exception as err:
            _LOGGER.error("Failed to play on %s: %s", speaker_name, err)
            return False

    async def play(self, speaker_name: str) -> bool:
        """Resume playback."""
        speaker = self.speakers.get(speaker_name)
        if not speaker or not speaker.cast:
            return False

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                speaker.cast.media_controller.play
            )
            return True
        except Exception as err:
            _LOGGER.error("Failed to play %s: %s", speaker_name, err)
            return False

    async def pause(self, speaker_name: str) -> bool:
        """Pause playback."""
        speaker = self.speakers.get(speaker_name)
        if not speaker or not speaker.cast:
            return False

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                speaker.cast.media_controller.pause
            )
            return True
        except Exception as err:
            _LOGGER.error("Failed to pause %s: %s", speaker_name, err)
            return False

    async def stop(self, speaker_name: str) -> bool:
        """Stop playback."""
        speaker = self.speakers.get(speaker_name)
        if not speaker or not speaker.cast:
            return False

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                speaker.cast.media_controller.stop
            )
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
        if not speaker or not speaker.cast:
            return False

        # Convert ms to seconds
        position_sec = position_ms / 1000

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: speaker.cast.media_controller.seek(position_sec)
            )
            _LOGGER.debug("Seeked %s to %.2f sec", speaker_name, position_sec)
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
        if not speaker or not speaker.cast:
            return None

        try:
            status = speaker.cast.media_controller.status

            if status and status.current_time is not None:
                position_ms = status.current_time * 1000

                # Update speaker state
                speaker.last_position_ms = position_ms
                speaker.last_position_time = datetime.now()

                return position_ms

            return None

        except Exception as err:
            _LOGGER.error("Failed to get position for %s: %s", speaker_name, err)
            return None

    async def get_player_state(self, speaker_name: str) -> Optional[str]:
        """
        Get player state (PLAYING, PAUSED, IDLE, BUFFERING, UNKNOWN).
        """
        speaker = self.speakers.get(speaker_name)
        if not speaker or not speaker.cast:
            return None

        try:
            status = speaker.cast.media_controller.status
            return status.player_state if status else None
        except Exception as err:
            _LOGGER.error("Failed to get state for %s: %s", speaker_name, err)
            return None

    async def set_volume(self, speaker_name: str, volume: float) -> bool:
        """Set volume (0.0 - 1.0)."""
        speaker = self.speakers.get(speaker_name)
        if not speaker or not speaker.cast:
            return False

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: speaker.cast.set_volume(volume)
            )
            return True
        except Exception as err:
            _LOGGER.error("Failed to set volume on %s: %s", speaker_name, err)
            return False

    async def get_volume(self, speaker_name: str) -> Optional[float]:
        """Get volume (0.0 - 1.0)."""
        speaker = self.speakers.get(speaker_name)
        if not speaker or not speaker.cast:
            return None

        try:
            status = speaker.cast.status
            return status.volume_level if status else None
        except Exception as err:
            _LOGGER.error("Failed to get volume from %s: %s", speaker_name, err)
            return None

    async def mute(self, speaker_name: str, muted: bool) -> bool:
        """Set mute state."""
        speaker = self.speakers.get(speaker_name)
        if not speaker or not speaker.cast:
            return False

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: speaker.cast.set_volume_muted(muted)
            )
            return True
        except Exception as err:
            _LOGGER.error("Failed to mute %s: %s", speaker_name, err)
            return False
