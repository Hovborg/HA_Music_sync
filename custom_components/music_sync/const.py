"""Constants for the Music Sync integration."""
from typing import Final

from homeassistant.const import Platform

DOMAIN: Final = "music_sync"

# Configuration keys
CONF_SPEAKERS: Final = "speakers"
CONF_MICROPHONE_DEVICE: Final = "microphone_device"
CONF_SAMPLE_RATE: Final = "sample_rate"
CONF_CALIBRATION_INTERVAL: Final = "calibration_interval"

# Default values
DEFAULT_SAMPLE_RATE: Final = 44100
DEFAULT_CALIBRATION_INTERVAL: Final = 3600  # 1 hour in seconds

# Calibration settings
CALIBRATION_DURATION: Final = 3.0  # seconds to record
TEST_SIGNAL_DURATION: Final = 0.2  # 200ms chirp
MIN_CONFIDENCE_SCORE: Final = 10.0  # standard score threshold

# Frequency assignments for multi-speaker calibration
SPEAKER_FREQUENCIES: Final = {
    0: 1000,   # Hz
    1: 1500,
    2: 2000,
    3: 2500,
    4: 3000,
    5: 3500,
    6: 4000,
    7: 4500,
}

# Services
SERVICE_CALIBRATE: Final = "calibrate"
SERVICE_CALIBRATE_SPEAKER: Final = "calibrate_speaker"
SERVICE_SET_OFFSET: Final = "set_offset"
SERVICE_CLEAR_OFFSETS: Final = "clear_offsets"

# Attributes
ATTR_OFFSET_MS: Final = "offset_ms"
ATTR_CONFIDENCE: Final = "confidence"
ATTR_LAST_CALIBRATION: Final = "last_calibration"
ATTR_SPEAKER_OFFSETS: Final = "speaker_offsets"

# Platforms
PLATFORMS: Final = [Platform.MEDIA_PLAYER]
