# Music Sync

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/hacs/integration)
[![Validate](https://github.com/hovborg/ha-music-sync/actions/workflows/validate.yml/badge.svg)](https://github.com/hovborg/ha-music-sync/actions/workflows/validate.yml)
[![Hassfest](https://github.com/hovborg/ha-music-sync/actions/workflows/hassfest.yml/badge.svg)](https://github.com/hovborg/ha-music-sync/actions/workflows/hassfest.yml)

A Home Assistant custom integration for synchronizing audio playback across different speaker platforms (Sonos, Google Cast, etc.) using microphone-based latency detection.

## Features

- **Cross-platform speaker sync**: Synchronize Sonos speakers with Google Cast devices
- **Microphone-based calibration**: Automatically detect and compensate for latency differences between speakers
- **Works with Sonos Era 100**: Uses direct UPnP API (SoCo) - no AirPlay required
- **Cross-correlation algorithm**: Accurately detects audio delays using chirp signals

## How It Works

1. **Calibration Phase**: The integration plays a test chirp signal on each speaker individually
2. **Recording**: A microphone (e.g., USB microphone connected to Home Assistant host) records the audio
3. **Analysis**: Cross-correlation is used to measure the exact latency for each speaker
4. **Compensation**: When playing audio, speakers are started with compensated timing to achieve sync

## Requirements

### Hardware
- USB microphone connected to the machine running Home Assistant
- Sonos speakers (including Era 100/300)
- Google Cast speakers (Google Home, Nest, Chromecast Audio)

### Software Dependencies
- numpy >= 1.24.0
- scipy >= 1.10.0
- sounddevice >= 0.4.6
- soco (for Sonos control)
- pychromecast (for Google Cast control)

## Installation

### HACS (Recommended)

1. Open HACS in Home Assistant
2. Click the three dots in the top right corner
3. Select "Custom repositories"
4. Add this repository URL: `https://github.com/hovborg/ha-music-sync`
5. Select "Integration" as the category
6. Click "Add"
7. Search for "Music Sync" and install it
8. Restart Home Assistant

### Manual Installation

1. Download the `custom_components/music_sync` folder
2. Copy it to your Home Assistant's `custom_components` directory
3. Restart Home Assistant

## Configuration

1. Go to **Settings** > **Devices & Services**
2. Click **+ Add Integration**
3. Search for "Music Sync"
4. Follow the configuration flow:
   - Select speakers to synchronize
   - Configure microphone device (optional, uses default if not specified)
   - Set sample rate (default: 44100 Hz)

## Usage

### Services

#### `music_sync.calibrate_all`
Calibrate all configured speakers.

```yaml
service: music_sync.calibrate_all
```

#### `music_sync.calibrate_speaker`
Calibrate a specific speaker.

```yaml
service: music_sync.calibrate_speaker
data:
  entity_id: media_player.living_room_speaker
```

#### `music_sync.set_offset`
Manually set the offset for a speaker (in milliseconds).

```yaml
service: music_sync.set_offset
data:
  entity_id: media_player.living_room_speaker
  offset_ms: 50
```

#### `music_sync.clear_offsets`
Clear all calibration data.

```yaml
service: music_sync.clear_offsets
```

## Sensors

The integration creates sensors for each configured speaker showing:
- Current offset (ms)
- Calibration confidence score
- Last calibration timestamp

## Technical Details

### Calibration Process

1. A chirp signal (1000-4000 Hz sweep, 200ms) is generated
2. The signal is saved to `/config/www/music_sync/` and served via Home Assistant's built-in web server
3. Each speaker plays the signal while the microphone records
4. Cross-correlation is used to find the peak offset between the original and recorded signal
5. The confidence score indicates how reliable the measurement is (>10 is considered good)

### Cross-Correlation Algorithm

The algorithm uses FFT-based cross-correlation for efficient detection:

```python
correlation = signal.correlate(recorded, reference, mode="full", method="fft")
peak_index = np.argmax(np.abs(correlation))
offset_samples = peak_index - len(reference) + 1
offset_ms = (offset_samples / sample_rate) * 1000
```

### Known Limitations

- **Sonos Era 100/300**: AirPlay (RAOP) has been removed from firmware, but UPnP control works
- **Google Cast**: Initial buffering can add 5-20 seconds delay; calibration accounts for this
- **Alexa**: Very limited API access; not currently supported

## Troubleshooting

### Microphone not detected
- Ensure the microphone is connected and recognized by the OS
- Check `sounddevice.query_devices()` output
- Try specifying the device ID explicitly in configuration

### Low confidence scores
- Move the microphone closer to the speakers
- Reduce background noise during calibration
- Increase the calibration signal duration

### Speakers not staying in sync
- Re-run calibration
- Check network latency to speakers
- Ensure all speakers are on the same network segment

## Development

### Testing without hardware

```bash
# Run the test script
python test_sonos_google_sync.py --discover
python test_sonos_google_sync.py --list-audio
python test_sonos_google_sync.py --test-mic
```

### Project Structure

```
custom_components/music_sync/
├── __init__.py           # Integration setup
├── manifest.json         # Integration manifest
├── config_flow.py        # Configuration UI
├── coordinator.py        # Main sync coordinator
├── media_player.py       # Media player entity
├── services.yaml         # Service definitions
├── const.py              # Constants
├── calibration/
│   └── audio_analyzer.py # Cross-correlation analysis
└── platforms/
    ├── sonos_api.py      # Sonos (SoCo) wrapper
    └── cast_api.py       # Google Cast wrapper
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Credits

- [SoCo](https://github.com/SoCo/SoCo) - Sonos Controller library
- [pychromecast](https://github.com/balloob/pychromecast) - Google Cast library
- [Home Assistant](https://www.home-assistant.io/) - Home automation platform
