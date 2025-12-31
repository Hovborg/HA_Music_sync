"""Constants for the Music Sync integration."""
DOMAIN = "music_sync"

# Configuration keys
CONF_SPEAKERS = "speakers"
CONF_MICROPHONE_DEVICE = "microphone_device"
CONF_SAMPLE_RATE = "sample_rate"
CONF_CALIBRATION_INTERVAL = "calibration_interval"

# Default values
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CALIBRATION_INTERVAL = 3600  # 1 hour in seconds

# Calibration settings
CALIBRATION_DURATION = 3.0  # seconds to record
TEST_SIGNAL_DURATION = 0.2  # 200ms chirp
MIN_CONFIDENCE_SCORE = 10.0  # standard score threshold

# Frequency assignments for multi-speaker calibration
SPEAKER_FREQUENCIES = {
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
SERVICE_CALIBRATE = "calibrate"
SERVICE_CALIBRATE_SPEAKER = "calibrate_speaker"
SERVICE_SET_OFFSET = "set_offset"
SERVICE_CLEAR_OFFSETS = "clear_offsets"

# Attributes
ATTR_OFFSET_MS = "offset_ms"
ATTR_CONFIDENCE = "confidence"
ATTR_LAST_CALIBRATION = "last_calibration"
ATTR_SPEAKER_OFFSETS = "speaker_offsets"

# Platforms
PLATFORMS = ["media_player"]
