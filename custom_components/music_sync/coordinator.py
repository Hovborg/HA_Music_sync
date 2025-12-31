"""Coordinator for Music Sync integration."""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.network import get_url

from .const import (
    DOMAIN,
    CONF_SPEAKERS,
    CONF_MICROPHONE_DEVICE,
    CONF_SAMPLE_RATE,
    CONF_CALIBRATION_INTERVAL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_CALIBRATION_INTERVAL,
    MIN_CONFIDENCE_SCORE,
)
from .calibration.audio_analyzer import AudioAnalyzer

_LOGGER = logging.getLogger(__name__)


@dataclass
class SpeakerOffset:
    """Represents calibration data for a speaker."""

    entity_id: str
    offset_ms: float = 0.0
    confidence: float = 0.0
    last_calibration: datetime | None = None
    is_reference: bool = False


@dataclass
class MusicSyncData:
    """Data structure for Music Sync coordinator."""

    speakers: dict[str, SpeakerOffset] = field(default_factory=dict)
    is_calibrating: bool = False
    last_full_calibration: datetime | None = None
    reference_speaker: str | None = None


class MusicSyncCoordinator(DataUpdateCoordinator[MusicSyncData]):
    """Coordinator for managing Music Sync state and calibration."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=30),
        )
        self.entry = entry
        self.data = MusicSyncData()

        # Configuration
        self._speaker_entities: list[str] = entry.data.get(CONF_SPEAKERS, [])
        self._microphone_device: str | None = entry.data.get(CONF_MICROPHONE_DEVICE)
        self._sample_rate: int = entry.options.get(CONF_SAMPLE_RATE, DEFAULT_SAMPLE_RATE)
        self._calibration_interval: int = entry.options.get(
            CONF_CALIBRATION_INTERVAL, DEFAULT_CALIBRATION_INTERVAL
        )

        # Audio analyzer
        self._analyzer: AudioAnalyzer | None = None

        # Auto-calibration task
        self._calibration_unsub: callable | None = None

        # Home Assistant paths
        self._www_path: Path | None = None
        self._ha_base_url: str | None = None

    async def async_initialize(self) -> None:
        """Initialize the coordinator."""
        _LOGGER.info("Initializing Music Sync coordinator")

        # Setup www folder for serving calibration files
        self._www_path = Path(self.hass.config.path("www")) / "music_sync"
        self._www_path.mkdir(parents=True, exist_ok=True)
        _LOGGER.info("Calibration files will be saved to: %s", self._www_path)

        # Get Home Assistant base URL for serving files
        try:
            self._ha_base_url = get_url(self.hass, prefer_external=False)
            _LOGGER.info("Home Assistant URL: %s", self._ha_base_url)
        except Exception as err:
            _LOGGER.warning("Could not get HA URL, using default: %s", err)
            self._ha_base_url = "http://homeassistant.local:8123"

        # Initialize speaker data
        for entity_id in self._speaker_entities:
            self.data.speakers[entity_id] = SpeakerOffset(entity_id=entity_id)

        # Set first speaker as reference
        if self._speaker_entities:
            first = self._speaker_entities[0]
            self.data.speakers[first].is_reference = True
            self.data.reference_speaker = first

        # Initialize audio analyzer with HA's www path
        try:
            self._analyzer = AudioAnalyzer(
                sample_rate=self._sample_rate,
                device=self._microphone_device,
                output_path=self._www_path,
            )
            await self._analyzer.async_initialize()
        except Exception as err:
            _LOGGER.error("Failed to initialize audio analyzer: %s", err)

        # Set up auto-calibration
        if self._calibration_interval > 0:
            self._calibration_unsub = async_track_time_interval(
                self.hass,
                self._async_auto_calibrate,
                timedelta(seconds=self._calibration_interval),
            )

    def _get_media_url(self, filename: str) -> str:
        """Get the full URL for a file in the www/music_sync folder."""
        # Files in /config/www/music_sync/ are served at /local/music_sync/
        return f"{self._ha_base_url}/local/music_sync/{filename}"

    async def async_shutdown(self) -> None:
        """Shutdown the coordinator."""
        if self._calibration_unsub:
            self._calibration_unsub()

        if self._analyzer:
            await self._analyzer.async_close()

    async def _async_update_data(self) -> MusicSyncData:
        """Fetch data from speakers."""
        # In a real implementation, this would check speaker states
        return self.data

    async def _async_auto_calibrate(self, _now: datetime) -> None:
        """Run automatic calibration."""
        if self.data.is_calibrating:
            _LOGGER.debug("Skipping auto-calibration, already in progress")
            return

        _LOGGER.info("Running automatic calibration")
        await self.async_calibrate_all()

    async def async_calibrate_all(self) -> dict[str, Any]:
        """Calibrate all speakers."""
        if self.data.is_calibrating:
            return {"error": "Calibration already in progress"}

        self.data.is_calibrating = True
        results = {}

        try:
            _LOGGER.info("Starting full calibration for %d speakers", len(self._speaker_entities))

            if not self._analyzer:
                raise RuntimeError("Audio analyzer not initialized")

            # Calibrate each speaker individually
            for entity_id in self._speaker_entities:
                result = await self._calibrate_single_speaker(entity_id)
                results[entity_id] = result

                # Update stored offset
                if result.get("success"):
                    speaker = self.data.speakers[entity_id]
                    speaker.offset_ms = result["offset_ms"]
                    speaker.confidence = result["confidence"]
                    speaker.last_calibration = datetime.now()

            # Normalize offsets relative to reference speaker
            await self._normalize_offsets()

            self.data.last_full_calibration = datetime.now()
            _LOGGER.info("Full calibration complete: %s", results)

        except Exception as err:
            _LOGGER.error("Calibration failed: %s", err)
            results["error"] = str(err)

        finally:
            self.data.is_calibrating = False
            self.async_set_updated_data(self.data)

        return results

    async def async_calibrate_speaker(self, entity_id: str) -> dict[str, Any]:
        """Calibrate a single speaker."""
        if entity_id not in self.data.speakers:
            return {"error": f"Speaker {entity_id} not configured"}

        if self.data.is_calibrating:
            return {"error": "Calibration already in progress"}

        self.data.is_calibrating = True

        try:
            result = await self._calibrate_single_speaker(entity_id)

            if result.get("success"):
                speaker = self.data.speakers[entity_id]
                speaker.offset_ms = result["offset_ms"]
                speaker.confidence = result["confidence"]
                speaker.last_calibration = datetime.now()

                await self._normalize_offsets()

            return result

        finally:
            self.data.is_calibrating = False
            self.async_set_updated_data(self.data)

    async def _calibrate_single_speaker(self, entity_id: str) -> dict[str, Any]:
        """Calibrate a single speaker by playing test tone and measuring delay."""
        _LOGGER.debug("Calibrating speaker: %s", entity_id)

        if not self._analyzer:
            return {"success": False, "error": "Analyzer not available"}

        try:
            # Generate test signal
            test_signal = self._analyzer.generate_test_signal()

            # Save test signal to www folder (returns just the filename)
            test_filename = await self._analyzer.save_test_signal(test_signal)

            # Get full URL for the speaker to access
            test_url = self._get_media_url(test_filename)
            _LOGGER.debug("Calibration audio URL: %s", test_url)

            # Start recording
            recording_task = asyncio.create_task(
                self._analyzer.record_audio(duration=3.0)
            )

            # Wait a moment for recording to start
            await asyncio.sleep(0.2)

            # Play test tone on speaker via Home Assistant
            await self.hass.services.async_call(
                "media_player",
                "play_media",
                {
                    "entity_id": entity_id,
                    "media_content_id": test_url,
                    "media_content_type": "music",
                },
                blocking=True,
            )

            # Wait for recording to complete
            recorded_audio = await recording_task

            # Stop playback
            await self.hass.services.async_call(
                "media_player",
                "media_stop",
                {"entity_id": entity_id},
                blocking=True,
            )

            # Analyze offset
            offset_ms, confidence = self._analyzer.find_offset(
                test_signal, recorded_audio
            )

            _LOGGER.info(
                "Speaker %s: offset=%.2fms, confidence=%.2f",
                entity_id,
                offset_ms,
                confidence,
            )

            return {
                "success": True,
                "offset_ms": offset_ms,
                "confidence": confidence,
                "reliable": confidence >= MIN_CONFIDENCE_SCORE,
            }

        except Exception as err:
            _LOGGER.error("Failed to calibrate %s: %s", entity_id, err)
            return {"success": False, "error": str(err)}

    async def _normalize_offsets(self) -> None:
        """Normalize all offsets relative to the reference speaker."""
        if not self.data.reference_speaker:
            return

        ref_offset = self.data.speakers[self.data.reference_speaker].offset_ms

        for speaker in self.data.speakers.values():
            # Subtract reference offset so reference becomes 0
            speaker.offset_ms -= ref_offset

    async def async_set_offset(self, entity_id: str, offset_ms: float) -> None:
        """Manually set offset for a speaker."""
        if entity_id in self.data.speakers:
            self.data.speakers[entity_id].offset_ms = offset_ms
            self.data.speakers[entity_id].last_calibration = datetime.now()
            self.async_set_updated_data(self.data)

    async def async_clear_offsets(self) -> None:
        """Clear all stored offsets."""
        for speaker in self.data.speakers.values():
            speaker.offset_ms = 0.0
            speaker.confidence = 0.0
            speaker.last_calibration = None

        self.data.last_full_calibration = None
        self.async_set_updated_data(self.data)

    def get_speaker_delay(self, entity_id: str) -> float:
        """Get the delay to apply for a speaker (inverse of offset)."""
        if entity_id not in self.data.speakers:
            return 0.0

        offset = self.data.speakers[entity_id].offset_ms

        # If speaker is ahead (negative offset), we need to delay it
        # If speaker is behind (positive offset), we need to advance others
        # For simplicity, we delay all speakers relative to the slowest
        max_offset = max(s.offset_ms for s in self.data.speakers.values())

        return max_offset - offset

    @property
    def speaker_entities(self) -> list[str]:
        """Return list of configured speaker entities."""
        return self._speaker_entities

    @property
    def is_calibrated(self) -> bool:
        """Return True if calibration has been performed."""
        return self.data.last_full_calibration is not None
