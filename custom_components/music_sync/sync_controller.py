"""
Sync Controller for Music Sync.

Handles synchronization between Sonos Era 100 and Google speakers
using microphone-based latency detection.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime
from enum import Enum

from .platforms.sonos_api import SonosAPI, SonosSpeaker
from .platforms.cast_api import GoogleCastAPI, CastSpeaker
from .calibration.audio_analyzer import AudioAnalyzer

_LOGGER = logging.getLogger(__name__)


class SpeakerType(Enum):
    """Speaker platform type."""
    SONOS = "sonos"
    GOOGLE_CAST = "google_cast"
    UNKNOWN = "unknown"


@dataclass
class SyncedSpeaker:
    """Unified speaker representation for sync operations."""

    name: str
    speaker_type: SpeakerType
    ip: str
    model: str = ""

    # Calibration data
    offset_ms: float = 0.0  # Positive = speaker is behind, needs less delay
    confidence: float = 0.0
    last_calibration: datetime = None

    # Reference to underlying API object
    api_speaker: Any = None


@dataclass
class SyncGroup:
    """A group of speakers to be synchronized."""

    name: str
    speakers: list[SyncedSpeaker] = field(default_factory=list)
    reference_speaker: Optional[SyncedSpeaker] = None  # Speaker with 0 offset


class SyncController:
    """
    Main controller for synchronizing Sonos and Google speakers.

    Flow:
    1. Discover all speakers (Sonos + Google Cast)
    2. Create a sync group with selected speakers
    3. Calibrate using microphone (measures latency differences)
    4. Play audio with compensated timing
    """

    def __init__(
        self,
        microphone_device: int | str | None = None,
        sample_rate: int = 44100,
    ):
        """
        Initialize the sync controller.

        Args:
            microphone_device: Audio input device ID or name
            sample_rate: Sample rate for audio analysis
        """
        self.sonos_api = SonosAPI()
        self.cast_api = GoogleCastAPI()
        self.analyzer = AudioAnalyzer(
            sample_rate=sample_rate,
            device=microphone_device
        )

        self.all_speakers: dict[str, SyncedSpeaker] = {}
        self.sync_groups: dict[str, SyncGroup] = {}

        self._audio_server_url: Optional[str] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize APIs and analyzer."""
        _LOGGER.info("Initializing Sync Controller...")

        await self.analyzer.async_initialize()
        self._initialized = True

    async def shutdown(self) -> None:
        """Clean up resources."""
        await self.analyzer.async_close()
        self.cast_api.stop_discovery()

    async def discover_all_speakers(self) -> dict[str, SyncedSpeaker]:
        """
        Discover all Sonos and Google Cast speakers.

        Returns:
            Dict of speaker name -> SyncedSpeaker
        """
        self.all_speakers.clear()

        # Discover Sonos
        _LOGGER.info("Discovering Sonos speakers...")
        sonos_speakers = await self.sonos_api.discover()

        for sp in sonos_speakers:
            synced = SyncedSpeaker(
                name=sp.name,
                speaker_type=SpeakerType.SONOS,
                ip=sp.ip,
                model=sp.model,
                api_speaker=sp,
            )
            self.all_speakers[sp.name] = synced
            _LOGGER.info("  Sonos: %s (%s)%s",
                        sp.name, sp.model,
                        " [Era - no RAOP]" if sp.is_era else "")

        # Discover Google Cast
        _LOGGER.info("Discovering Google Cast speakers...")
        cast_speakers = await self.cast_api.discover()

        for sp in cast_speakers:
            synced = SyncedSpeaker(
                name=sp.name,
                speaker_type=SpeakerType.GOOGLE_CAST,
                ip=sp.ip,
                model=sp.model,
                api_speaker=sp,
            )
            self.all_speakers[sp.name] = synced
            _LOGGER.info("  Cast: %s (%s) [%s]", sp.name, sp.model, sp.cast_type)

        _LOGGER.info("Found %d speakers total", len(self.all_speakers))
        return self.all_speakers

    def create_sync_group(
        self,
        group_name: str,
        speaker_names: list[str],
        reference_speaker_name: Optional[str] = None,
    ) -> SyncGroup:
        """
        Create a sync group from selected speakers.

        Args:
            group_name: Name for the group
            speaker_names: List of speaker names to include
            reference_speaker_name: Speaker to use as reference (0 offset)

        Returns:
            SyncGroup object
        """
        speakers = []
        reference = None

        for name in speaker_names:
            speaker = self.all_speakers.get(name)
            if speaker:
                speakers.append(speaker)
                if name == reference_speaker_name:
                    reference = speaker
            else:
                _LOGGER.warning("Speaker not found: %s", name)

        # Default to first speaker as reference
        if not reference and speakers:
            reference = speakers[0]

        group = SyncGroup(
            name=group_name,
            speakers=speakers,
            reference_speaker=reference,
        )

        self.sync_groups[group_name] = group
        _LOGGER.info(
            "Created sync group '%s' with %d speakers, reference: %s",
            group_name, len(speakers),
            reference.name if reference else "None"
        )

        return group

    # =========================================================================
    # Microphone-Based Calibration
    # =========================================================================

    async def calibrate_group(
        self,
        group_name: str,
        audio_url: str,
    ) -> dict[str, float]:
        """
        Calibrate all speakers in a group using microphone.

        This plays a test signal on each speaker individually and
        uses cross-correlation to measure the latency.

        Args:
            group_name: Name of sync group
            audio_url: URL to calibration audio file

        Returns:
            Dict of speaker name -> offset in ms
        """
        group = self.sync_groups.get(group_name)
        if not group:
            raise ValueError(f"Group not found: {group_name}")

        _LOGGER.info("Starting calibration for group '%s'", group_name)

        # Generate test signal
        test_signal = self.analyzer.generate_test_signal(duration_ms=200)
        test_file = await self.analyzer.save_test_signal(test_signal)

        results = {}

        for speaker in group.speakers:
            _LOGGER.info("Calibrating: %s", speaker.name)

            offset_ms, confidence = await self._calibrate_single_speaker(
                speaker, test_file, test_signal
            )

            speaker.offset_ms = offset_ms
            speaker.confidence = confidence
            speaker.last_calibration = datetime.now()

            results[speaker.name] = offset_ms

            _LOGGER.info(
                "  %s: offset=%.1fms, confidence=%.1f",
                speaker.name, offset_ms, confidence
            )

            # Brief pause between speakers
            await asyncio.sleep(0.5)

        # Normalize offsets relative to reference speaker
        if group.reference_speaker:
            ref_offset = group.reference_speaker.offset_ms
            for speaker in group.speakers:
                speaker.offset_ms -= ref_offset
                results[speaker.name] -= ref_offset

            _LOGGER.info("Normalized offsets (reference: %s = 0ms)",
                        group.reference_speaker.name)

        return results

    async def _calibrate_single_speaker(
        self,
        speaker: SyncedSpeaker,
        test_file: str,
        test_signal,
    ) -> tuple[float, float]:
        """
        Calibrate a single speaker.

        Returns:
            Tuple of (offset_ms, confidence)
        """
        # Start recording before playback
        record_task = asyncio.create_task(
            self.analyzer.record_audio(duration=3.0)
        )

        # Wait for recording to start
        await asyncio.sleep(0.2)

        # Play test signal on the speaker
        if speaker.speaker_type == SpeakerType.SONOS:
            await self.sonos_api.play_uri(speaker.name, test_file)
        elif speaker.speaker_type == SpeakerType.GOOGLE_CAST:
            await self.cast_api.play_url(speaker.name, test_file)

        # Wait for recording to complete
        recording = await record_task

        # Stop playback
        if speaker.speaker_type == SpeakerType.SONOS:
            await self.sonos_api.stop(speaker.name)
        elif speaker.speaker_type == SpeakerType.GOOGLE_CAST:
            await self.cast_api.stop(speaker.name)

        # Analyze offset
        offset_ms, confidence = self.analyzer.find_offset(test_signal, recording)

        return offset_ms, confidence

    async def calibrate_pair_simultaneous(
        self,
        speaker1_name: str,
        speaker2_name: str,
        audio_url: str,
    ) -> float:
        """
        Calibrate two speakers by playing simultaneously and
        measuring the time difference with microphone.

        This is better for real-world sync because it measures
        actual acoustic arrival times.

        Args:
            speaker1_name: First speaker
            speaker2_name: Second speaker
            audio_url: Audio to play

        Returns:
            Offset in ms (positive = speaker2 is behind speaker1)
        """
        speaker1 = self.all_speakers.get(speaker1_name)
        speaker2 = self.all_speakers.get(speaker2_name)

        if not speaker1 or not speaker2:
            raise ValueError("Speaker not found")

        _LOGGER.info("Simultaneous calibration: %s vs %s", speaker1_name, speaker2_name)

        # Generate unique tones for each speaker
        tones = self.analyzer.generate_unique_tones(
            num_speakers=2,
            duration_ms=500,
            base_freq=1000,
            freq_step=500
        )

        # Save tones to files
        tone1_file = await self.analyzer.save_test_signal(tones[0])
        tone2_file = await self.analyzer.save_test_signal(tones[1])

        # Rename with unique names
        # (In real implementation, save to different files)

        # Start recording
        record_task = asyncio.create_task(
            self.analyzer.record_audio(duration=3.0)
        )

        await asyncio.sleep(0.2)

        # Play both simultaneously
        play_tasks = []

        if speaker1.speaker_type == SpeakerType.SONOS:
            play_tasks.append(self.sonos_api.play_uri(speaker1_name, tone1_file))
        else:
            play_tasks.append(self.cast_api.play_url(speaker1_name, tone1_file))

        if speaker2.speaker_type == SpeakerType.SONOS:
            play_tasks.append(self.sonos_api.play_uri(speaker2_name, tone2_file))
        else:
            play_tasks.append(self.cast_api.play_url(speaker2_name, tone2_file))

        await asyncio.gather(*play_tasks)

        # Wait for recording
        recording = await record_task

        # Analyze multi-tone recording
        frequencies = [1000, 1500]  # Hz for speaker1 and speaker2
        onsets = self.analyzer.analyze_multi_speaker_recording(
            recording, frequencies, bandwidth=100
        )

        # Stop playback
        stop_tasks = []
        if speaker1.speaker_type == SpeakerType.SONOS:
            stop_tasks.append(self.sonos_api.stop(speaker1_name))
        else:
            stop_tasks.append(self.cast_api.stop(speaker1_name))

        if speaker2.speaker_type == SpeakerType.SONOS:
            stop_tasks.append(self.sonos_api.stop(speaker2_name))
        else:
            stop_tasks.append(self.cast_api.stop(speaker2_name))

        await asyncio.gather(*stop_tasks)

        # Calculate offset
        onset1 = onsets.get(1000, 0)
        onset2 = onsets.get(1500, 0)
        offset_ms = onset2 - onset1

        _LOGGER.info(
            "Simultaneous calibration result: %s=%.1fms, %s=%.1fms, diff=%.1fms",
            speaker1_name, onset1, speaker2_name, onset2, offset_ms
        )

        return offset_ms

    # =========================================================================
    # Synchronized Playback
    # =========================================================================

    async def play_synced(
        self,
        group_name: str,
        audio_url: str,
        content_type: str = 'audio/mp3',
    ) -> bool:
        """
        Play audio on all speakers in a group with sync compensation.

        Uses calculated offsets to delay speakers appropriately.

        Args:
            group_name: Sync group name
            audio_url: URL to audio
            content_type: MIME type

        Returns:
            True if successful
        """
        group = self.sync_groups.get(group_name)
        if not group:
            _LOGGER.error("Group not found: %s", group_name)
            return False

        _LOGGER.info("Starting synced playback for group '%s'", group_name)

        # Find the maximum offset (slowest speaker)
        max_offset = max(sp.offset_ms for sp in group.speakers)

        # Calculate delay for each speaker
        # Speakers with lower offset need more delay
        speaker_delays = {}
        for speaker in group.speakers:
            delay = max_offset - speaker.offset_ms
            speaker_delays[speaker.name] = delay
            _LOGGER.debug("  %s: delay=%.1fms", speaker.name, delay)

        # Start playback with delays
        tasks = []
        for speaker in group.speakers:
            delay_ms = speaker_delays[speaker.name]
            task = self._play_with_delay(speaker, audio_url, content_type, delay_ms)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        success = all(r is True for r in results if not isinstance(r, Exception))
        return success

    async def _play_with_delay(
        self,
        speaker: SyncedSpeaker,
        audio_url: str,
        content_type: str,
        delay_ms: float,
    ) -> bool:
        """Play on a speaker with a specific delay."""

        if delay_ms > 0:
            _LOGGER.debug("Delaying %s by %.1fms", speaker.name, delay_ms)
            await asyncio.sleep(delay_ms / 1000)

        if speaker.speaker_type == SpeakerType.SONOS:
            return await self.sonos_api.play_uri(speaker.name, audio_url)
        elif speaker.speaker_type == SpeakerType.GOOGLE_CAST:
            return await self.cast_api.play_url(
                speaker.name, audio_url, content_type
            )
        return False

    async def pause_all(self, group_name: str) -> bool:
        """Pause all speakers in a group."""
        group = self.sync_groups.get(group_name)
        if not group:
            return False

        tasks = []
        for speaker in group.speakers:
            if speaker.speaker_type == SpeakerType.SONOS:
                tasks.append(self.sonos_api.pause(speaker.name))
            elif speaker.speaker_type == SpeakerType.GOOGLE_CAST:
                tasks.append(self.cast_api.pause(speaker.name))

        await asyncio.gather(*tasks)
        return True

    async def stop_all(self, group_name: str) -> bool:
        """Stop all speakers in a group."""
        group = self.sync_groups.get(group_name)
        if not group:
            return False

        tasks = []
        for speaker in group.speakers:
            if speaker.speaker_type == SpeakerType.SONOS:
                tasks.append(self.sonos_api.stop(speaker.name))
            elif speaker.speaker_type == SpeakerType.GOOGLE_CAST:
                tasks.append(self.cast_api.stop(speaker.name))

        await asyncio.gather(*tasks)
        return True

    async def set_volume_all(self, group_name: str, volume: float) -> bool:
        """
        Set volume on all speakers in a group.

        Args:
            group_name: Group name
            volume: 0.0 - 1.0 (will be converted appropriately per platform)
        """
        group = self.sync_groups.get(group_name)
        if not group:
            return False

        tasks = []
        for speaker in group.speakers:
            if speaker.speaker_type == SpeakerType.SONOS:
                # Sonos uses 0-100
                tasks.append(self.sonos_api.set_volume(
                    speaker.name, int(volume * 100)
                ))
            elif speaker.speaker_type == SpeakerType.GOOGLE_CAST:
                # Cast uses 0.0-1.0
                tasks.append(self.cast_api.set_volume(speaker.name, volume))

        await asyncio.gather(*tasks)
        return True
