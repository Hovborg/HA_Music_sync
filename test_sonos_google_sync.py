#!/usr/bin/env python3
"""
Test script for Sonos Era 100 + Google Speaker synchronization.

This script demonstrates:
1. Discovery of Sonos and Google Cast speakers
2. Microphone-based latency calibration
3. Synchronized playback across platforms

Usage:
    python test_sonos_google_sync.py --discover
    python test_sonos_google_sync.py --calibrate --sonos "Era 100" --google "Living Room"
    python test_sonos_google_sync.py --play --url "http://example.com/audio.mp3"

Requirements:
    pip install soco pychromecast numpy scipy sounddevice
"""

import argparse
import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from custom_components.music_sync.platforms.sonos_api import SonosAPI, SOCO_AVAILABLE
from custom_components.music_sync.platforms.cast_api import GoogleCastAPI, PYCHROMECAST_AVAILABLE
from custom_components.music_sync.calibration.audio_analyzer import AudioAnalyzer, SCIPY_AVAILABLE, SOUNDDEVICE_AVAILABLE


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_status(label: str, available: bool):
    """Print library status."""
    status = "✅ OK" if available else "❌ Missing"
    print(f"  {label:<25} {status}")


async def check_dependencies():
    """Check if all required libraries are available."""
    print_header("DEPENDENCY CHECK")

    print_status("SoCo (Sonos)", SOCO_AVAILABLE)
    print_status("PyChromecast (Google)", PYCHROMECAST_AVAILABLE)
    print_status("SciPy (Audio analysis)", SCIPY_AVAILABLE)
    print_status("SoundDevice (Microphone)", SOUNDDEVICE_AVAILABLE)

    all_ok = all([SOCO_AVAILABLE, PYCHROMECAST_AVAILABLE,
                  SCIPY_AVAILABLE, SOUNDDEVICE_AVAILABLE])

    if not all_ok:
        print("\n⚠️  Missing dependencies. Install with:")
        print("   pip install soco pychromecast numpy scipy sounddevice")
        return False

    print("\n✅ All dependencies OK")
    return True


async def discover_speakers():
    """Discover all Sonos and Google Cast speakers."""
    print_header("SPEAKER DISCOVERY")

    # Sonos
    print("\n--- Sonos Speakers ---")
    if SOCO_AVAILABLE:
        sonos_api = SonosAPI()
        speakers = await sonos_api.discover(timeout=5)

        if speakers:
            for sp in speakers:
                era_tag = " [ERA - no RAOP]" if sp.is_era else ""
                print(f"  • {sp.name}")
                print(f"    Model: {sp.model}{era_tag}")
                print(f"    IP: {sp.ip}")
        else:
            print("  No Sonos speakers found")
    else:
        print("  SoCo not available")

    # Google Cast
    print("\n--- Google Cast Speakers ---")
    if PYCHROMECAST_AVAILABLE:
        cast_api = GoogleCastAPI()
        speakers = await cast_api.discover(timeout=10)

        if speakers:
            for sp in speakers:
                print(f"  • {sp.name}")
                print(f"    Model: {sp.model}")
                print(f"    IP: {sp.ip}")
                print(f"    Type: {sp.cast_type}")
        else:
            print("  No Cast speakers found")

        cast_api.stop_discovery()
    else:
        print("  PyChromecast not available")


async def list_audio_devices():
    """List available audio input devices."""
    print_header("AUDIO INPUT DEVICES (Microphones)")

    if not SOUNDDEVICE_AVAILABLE:
        print("  SoundDevice not available")
        return

    import sounddevice as sd

    devices = sd.query_devices()
    print("\n  Available microphones:\n")

    for idx, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            is_default = device == sd.query_devices(kind="input")
            marker = " ← DEFAULT" if is_default else ""
            print(f"  [{idx}] {device['name']}{marker}")
            print(f"       Channels: {device['max_input_channels']}, "
                  f"Rate: {device['default_samplerate']:.0f} Hz")


async def test_microphone(device_id: int | None = None):
    """Test microphone recording."""
    print_header("MICROPHONE TEST")

    if not SOUNDDEVICE_AVAILABLE:
        print("  SoundDevice not available")
        return

    import sounddevice as sd
    import numpy as np

    sample_rate = 44100
    duration = 2.0

    if device_id is not None:
        device_info = sd.query_devices(device_id)
        print(f"\n  Using device [{device_id}]: {device_info['name']}")
    else:
        device_info = sd.query_devices(kind="input")
        print(f"\n  Using default: {device_info['name']}")

    print(f"\n  Recording for {duration} seconds... Make some noise!")

    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32',
        device=device_id
    )
    sd.wait()

    recording = recording.flatten()
    peak = np.max(np.abs(recording))
    rms = np.sqrt(np.mean(recording ** 2))

    print(f"\n  Peak amplitude: {peak:.4f}")
    print(f"  RMS level:      {rms:.4f}")

    if peak < 0.01:
        print("\n  ⚠️  Very low signal - check microphone connection")
    elif peak > 0.95:
        print("\n  ⚠️  Signal clipping - reduce input gain")
    else:
        print("\n  ✅ Signal level OK")


async def calibrate_speakers(
    sonos_name: str,
    google_name: str,
    mic_device: int | None = None,
):
    """
    Calibrate sync between Sonos and Google speaker.

    Plays a test tone on each speaker and measures
    the time difference using the microphone.
    """
    print_header("SPEAKER CALIBRATION")

    if not all([SOCO_AVAILABLE, PYCHROMECAST_AVAILABLE,
                SCIPY_AVAILABLE, SOUNDDEVICE_AVAILABLE]):
        print("  Missing dependencies")
        return None

    from scipy import signal
    from scipy.io import wavfile
    import sounddevice as sd
    import numpy as np
    import tempfile
    import time

    sample_rate = 44100

    # Initialize APIs
    print("\n1. Discovering speakers...")

    sonos_api = SonosAPI()
    await sonos_api.discover()
    sonos_speaker = sonos_api.get_speaker(sonos_name)

    if not sonos_speaker:
        print(f"   ❌ Sonos speaker not found: {sonos_name}")
        return None
    print(f"   ✅ Found Sonos: {sonos_speaker.name} ({sonos_speaker.model})")

    cast_api = GoogleCastAPI()
    await cast_api.discover()
    google_speaker = cast_api.get_speaker(google_name)

    if not google_speaker:
        print(f"   ❌ Google speaker not found: {google_name}")
        cast_api.stop_discovery()
        return None
    print(f"   ✅ Found Google: {google_speaker.name} ({google_speaker.model})")

    # Generate test signals (different frequencies)
    print("\n2. Generating test signals...")

    duration_sec = 0.3
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec))

    # Sonos gets 1000 Hz
    tone_sonos = np.sin(2 * np.pi * 1000 * t) * np.hanning(len(t))
    tone_sonos = (tone_sonos * 32767).astype(np.int16)

    # Google gets 1500 Hz
    tone_google = np.sin(2 * np.pi * 1500 * t) * np.hanning(len(t))
    tone_google = (tone_google * 32767).astype(np.int16)

    # Save to temp files
    temp_dir = tempfile.mkdtemp()
    sonos_file = os.path.join(temp_dir, "tone_sonos.wav")
    google_file = os.path.join(temp_dir, "tone_google.wav")

    wavfile.write(sonos_file, sample_rate, tone_sonos)
    wavfile.write(google_file, sample_rate, tone_google)

    print(f"   Sonos tone: 1000 Hz")
    print(f"   Google tone: 1500 Hz")

    # Calibrate each speaker individually
    results = {}

    for name, speaker_api, speaker, tone_file, freq in [
        ("Sonos", sonos_api, sonos_speaker, sonos_file, 1000),
        ("Google", cast_api, google_speaker, google_file, 1500),
    ]:
        print(f"\n3. Calibrating {name}...")

        # Start recording
        record_duration = 3.0
        print(f"   Recording {record_duration}s...")

        recording = sd.rec(
            int(record_duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            device=mic_device
        )

        # Wait for recording to start
        await asyncio.sleep(0.3)

        # Play tone
        print(f"   Playing {freq} Hz tone...")

        if name == "Sonos":
            # For Sonos we need to serve the file via HTTP
            # For now, just show what would happen
            print(f"   ⚠️  Note: Sonos needs HTTP URL, using file:// for test")
            await sonos_api.play_uri(sonos_speaker.name, f"file://{tone_file}")
        else:
            # Google Cast also needs HTTP URL
            print(f"   ⚠️  Note: Cast needs HTTP URL, using file:// for test")
            await cast_api.play_url(google_speaker.name, f"file://{tone_file}")

        # Wait for recording
        sd.wait()
        recording = recording.flatten()

        # Stop playback
        if name == "Sonos":
            await sonos_api.stop(sonos_speaker.name)
        else:
            await cast_api.stop(google_speaker.name)

        # Analyze: bandpass filter and find onset
        low, high = freq - 100, freq + 100
        sos = signal.butter(4, [low, high], btype='band', fs=sample_rate, output='sos')
        filtered = signal.sosfilt(sos, recording)

        # Envelope
        envelope = np.abs(signal.hilbert(filtered))

        # Find onset
        threshold = np.max(envelope) * 0.1
        onset_indices = np.where(envelope > threshold)[0]

        if len(onset_indices) > 0:
            onset_sample = onset_indices[0]
            onset_ms = (onset_sample / sample_rate) * 1000
            results[name] = onset_ms
            print(f"   ✅ Detected onset: {onset_ms:.1f} ms")
        else:
            print(f"   ❌ Could not detect tone")
            results[name] = None

        await asyncio.sleep(0.5)

    # Calculate offset
    print_header("CALIBRATION RESULTS")

    if results.get("Sonos") is not None and results.get("Google") is not None:
        sonos_onset = results["Sonos"]
        google_onset = results["Google"]
        offset = google_onset - sonos_onset

        print(f"\n  Sonos onset:    {sonos_onset:.1f} ms")
        print(f"  Google onset:   {google_onset:.1f} ms")
        print(f"  -" * 20)
        print(f"  Offset:         {offset:+.1f} ms")

        if offset > 0:
            print(f"\n  → Google is {abs(offset):.1f}ms BEHIND Sonos")
            print(f"  → To sync: Delay Sonos by {abs(offset):.1f}ms")
        elif offset < 0:
            print(f"\n  → Sonos is {abs(offset):.1f}ms BEHIND Google")
            print(f"  → To sync: Delay Google by {abs(offset):.1f}ms")
        else:
            print(f"\n  → Speakers are in sync!")

        return offset
    else:
        print("\n  ❌ Calibration failed - could not detect both tones")
        return None

    # Cleanup
    cast_api.stop_discovery()
    os.remove(sonos_file)
    os.remove(google_file)
    os.rmdir(temp_dir)


async def play_synced(
    sonos_name: str,
    google_name: str,
    audio_url: str,
    offset_ms: float = 0,
):
    """
    Play audio on both speakers with sync compensation.

    Args:
        sonos_name: Sonos speaker name
        google_name: Google speaker name
        audio_url: URL to audio file
        offset_ms: Offset from calibration (positive = Google behind)
    """
    print_header("SYNCHRONIZED PLAYBACK")

    if not all([SOCO_AVAILABLE, PYCHROMECAST_AVAILABLE]):
        print("  Missing dependencies")
        return

    # Initialize
    print("\n1. Connecting to speakers...")

    sonos_api = SonosAPI()
    await sonos_api.discover()
    sonos = sonos_api.get_speaker(sonos_name)

    cast_api = GoogleCastAPI()
    await cast_api.discover()
    google = cast_api.get_speaker(google_name)

    if not sonos or not google:
        print("   ❌ Speakers not found")
        return

    print(f"   ✅ Sonos: {sonos.name}")
    print(f"   ✅ Google: {google.name}")

    # Calculate delays
    print("\n2. Calculating sync delays...")

    if offset_ms > 0:
        # Google is behind, delay Sonos
        sonos_delay = abs(offset_ms)
        google_delay = 0
    else:
        # Sonos is behind, delay Google
        sonos_delay = 0
        google_delay = abs(offset_ms)

    print(f"   Sonos delay:  {sonos_delay:.1f} ms")
    print(f"   Google delay: {google_delay:.1f} ms")

    # Play with delays
    print(f"\n3. Starting playback: {audio_url}")

    async def play_sonos():
        if sonos_delay > 0:
            await asyncio.sleep(sonos_delay / 1000)
        await sonos_api.play_uri(sonos_name, audio_url)

    async def play_google():
        if google_delay > 0:
            await asyncio.sleep(google_delay / 1000)
        await cast_api.play_url(google_name, audio_url)

    await asyncio.gather(play_sonos(), play_google())

    print("\n   ✅ Playback started on both speakers")
    print("\n   Press Ctrl+C to stop...")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n\n4. Stopping playback...")
        await sonos_api.stop(sonos_name)
        await cast_api.stop(google_name)

    cast_api.stop_discovery()
    print("   Done")


async def main():
    parser = argparse.ArgumentParser(
        description="Test Sonos Era 100 + Google Speaker synchronization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--discover", "-d",
        action="store_true",
        help="Discover all speakers"
    )
    parser.add_argument(
        "--list-audio", "-l",
        action="store_true",
        help="List audio input devices"
    )
    parser.add_argument(
        "--test-mic", "-t",
        action="store_true",
        help="Test microphone recording"
    )
    parser.add_argument(
        "--calibrate", "-c",
        action="store_true",
        help="Calibrate speaker sync"
    )
    parser.add_argument(
        "--play", "-p",
        action="store_true",
        help="Play synced audio"
    )
    parser.add_argument(
        "--sonos", "-s",
        type=str,
        help="Sonos speaker name"
    )
    parser.add_argument(
        "--google", "-g",
        type=str,
        help="Google speaker name"
    )
    parser.add_argument(
        "--mic", "-m",
        type=int,
        default=None,
        help="Microphone device ID"
    )
    parser.add_argument(
        "--url", "-u",
        type=str,
        help="Audio URL to play"
    )
    parser.add_argument(
        "--offset", "-o",
        type=float,
        default=0,
        help="Sync offset in ms (from calibration)"
    )

    args = parser.parse_args()

    # Default action
    if not any([args.discover, args.list_audio, args.test_mic,
                args.calibrate, args.play]):
        await check_dependencies()
        parser.print_help()
        return

    print_header("SONOS + GOOGLE SYNC TEST")

    if not await check_dependencies():
        return

    if args.discover:
        await discover_speakers()

    if args.list_audio:
        await list_audio_devices()

    if args.test_mic:
        await test_microphone(args.mic)

    if args.calibrate:
        if not args.sonos or not args.google:
            print("\n⚠️  Specify --sonos and --google speaker names")
            return
        await calibrate_speakers(args.sonos, args.google, args.mic)

    if args.play:
        if not args.sonos or not args.google or not args.url:
            print("\n⚠️  Specify --sonos, --google, and --url")
            return
        await play_synced(args.sonos, args.google, args.url, args.offset)


if __name__ == "__main__":
    asyncio.run(main())
