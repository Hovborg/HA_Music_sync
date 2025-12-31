"""Media player platform for Music Sync."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from homeassistant.components.media_player import (
    MediaPlayerEntity,
    MediaPlayerEntityFeature,
    MediaPlayerState,
    MediaType,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    DOMAIN,
    ATTR_SPEAKER_OFFSETS,
    ATTR_LAST_CALIBRATION,
)
from .coordinator import MusicSyncCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Music Sync media player from a config entry."""
    coordinator: MusicSyncCoordinator = hass.data[DOMAIN][entry.entry_id]

    async_add_entities([SyncedMediaPlayer(coordinator, entry)])


class SyncedMediaPlayer(CoordinatorEntity[MusicSyncCoordinator], MediaPlayerEntity):
    """
    A virtual media player that represents a synchronized speaker group.

    This entity controls multiple speakers as one, applying latency
    compensation to keep them in sync.
    """

    _attr_has_entity_name = True
    _attr_name = "Synced Speaker Group"
    _attr_media_content_type = MediaType.MUSIC

    def __init__(
        self,
        coordinator: MusicSyncCoordinator,
        entry: ConfigEntry,
    ) -> None:
        """Initialize the synced media player."""
        super().__init__(coordinator)

        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_synced_group"

        # State tracking
        self._state = MediaPlayerState.IDLE
        self._volume_level: float = 0.5
        self._is_volume_muted: bool = False
        self._media_title: str | None = None
        self._media_artist: str | None = None

        # Supported features
        self._attr_supported_features = (
            MediaPlayerEntityFeature.PLAY
            | MediaPlayerEntityFeature.PAUSE
            | MediaPlayerEntityFeature.STOP
            | MediaPlayerEntityFeature.VOLUME_SET
            | MediaPlayerEntityFeature.VOLUME_MUTE
            | MediaPlayerEntityFeature.PLAY_MEDIA
        )

    @property
    def state(self) -> MediaPlayerState:
        """Return the state of the player."""
        return self._state

    @property
    def volume_level(self) -> float | None:
        """Return the volume level (0..1)."""
        return self._volume_level

    @property
    def is_volume_muted(self) -> bool | None:
        """Return True if volume is muted."""
        return self._is_volume_muted

    @property
    def media_title(self) -> str | None:
        """Return the title of current playing media."""
        return self._media_title

    @property
    def media_artist(self) -> str | None:
        """Return the artist of current playing media."""
        return self._media_artist

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return extra state attributes."""
        attrs = {
            "speakers": self.coordinator.speaker_entities,
            "is_calibrated": self.coordinator.is_calibrated,
            "is_calibrating": self.coordinator.data.is_calibrating,
        }

        # Add speaker offsets
        offsets = {}
        for entity_id, speaker in self.coordinator.data.speakers.items():
            offsets[entity_id] = {
                "offset_ms": speaker.offset_ms,
                "confidence": speaker.confidence,
                "is_reference": speaker.is_reference,
            }
        attrs[ATTR_SPEAKER_OFFSETS] = offsets

        # Add last calibration time
        if self.coordinator.data.last_full_calibration:
            attrs[ATTR_LAST_CALIBRATION] = (
                self.coordinator.data.last_full_calibration.isoformat()
            )

        return attrs

    async def async_play_media(
        self,
        media_type: MediaType | str,
        media_id: str,
        **kwargs: Any,
    ) -> None:
        """
        Play media on all synchronized speakers.

        This method applies latency compensation to each speaker
        to keep them in sync.
        """
        _LOGGER.info("Playing media on synced group: %s", media_id)

        self._media_title = media_id
        self._state = MediaPlayerState.PLAYING

        # Get delays for each speaker
        speaker_delays = {}
        for entity_id in self.coordinator.speaker_entities:
            delay_ms = self.coordinator.get_speaker_delay(entity_id)
            speaker_delays[entity_id] = delay_ms

        _LOGGER.debug("Speaker delays: %s", speaker_delays)

        # Play on each speaker with appropriate delay
        # We use asyncio.gather to start them as close together as possible
        tasks = []
        for entity_id, delay_ms in speaker_delays.items():
            task = self._play_with_delay(
                entity_id, media_type, media_id, delay_ms, **kwargs
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

        self.async_write_ha_state()

    async def _play_with_delay(
        self,
        entity_id: str,
        media_type: MediaType | str,
        media_id: str,
        delay_ms: float,
        **kwargs: Any,
    ) -> None:
        """Play media on a speaker with a specific delay."""
        if delay_ms > 0:
            _LOGGER.debug("Delaying %s by %.1fms", entity_id, delay_ms)
            await asyncio.sleep(delay_ms / 1000)

        await self.hass.services.async_call(
            "media_player",
            "play_media",
            {
                "entity_id": entity_id,
                "media_content_id": media_id,
                "media_content_type": media_type,
            },
            blocking=False,
        )

    async def async_media_play(self) -> None:
        """Send play command to all speakers."""
        _LOGGER.debug("Play command to synced group")
        self._state = MediaPlayerState.PLAYING

        await self._send_to_all_speakers("media_play")
        self.async_write_ha_state()

    async def async_media_pause(self) -> None:
        """Send pause command to all speakers."""
        _LOGGER.debug("Pause command to synced group")
        self._state = MediaPlayerState.PAUSED

        await self._send_to_all_speakers("media_pause")
        self.async_write_ha_state()

    async def async_media_stop(self) -> None:
        """Send stop command to all speakers."""
        _LOGGER.debug("Stop command to synced group")
        self._state = MediaPlayerState.IDLE

        await self._send_to_all_speakers("media_stop")
        self.async_write_ha_state()

    async def async_set_volume_level(self, volume: float) -> None:
        """Set volume level on all speakers."""
        _LOGGER.debug("Setting volume to %.2f on synced group", volume)
        self._volume_level = volume

        tasks = []
        for entity_id in self.coordinator.speaker_entities:
            task = self.hass.services.async_call(
                "media_player",
                "volume_set",
                {
                    "entity_id": entity_id,
                    "volume_level": volume,
                },
                blocking=False,
            )
            tasks.append(task)

        await asyncio.gather(*tasks)
        self.async_write_ha_state()

    async def async_mute_volume(self, mute: bool) -> None:
        """Mute/unmute all speakers."""
        _LOGGER.debug("Setting mute to %s on synced group", mute)
        self._is_volume_muted = mute

        tasks = []
        for entity_id in self.coordinator.speaker_entities:
            task = self.hass.services.async_call(
                "media_player",
                "volume_mute",
                {
                    "entity_id": entity_id,
                    "is_volume_muted": mute,
                },
                blocking=False,
            )
            tasks.append(task)

        await asyncio.gather(*tasks)
        self.async_write_ha_state()

    async def _send_to_all_speakers(self, service: str) -> None:
        """Send a service call to all speakers."""
        tasks = []
        for entity_id in self.coordinator.speaker_entities:
            task = self.hass.services.async_call(
                "media_player",
                service,
                {"entity_id": entity_id},
                blocking=False,
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        self.async_write_ha_state()
