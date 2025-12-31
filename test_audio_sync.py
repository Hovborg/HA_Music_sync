#!/usr/bin/env python3
"""
Test script for Music Sync audio analyzer.

This script allows you to test the cross-correlation based
latency detection without Home Assistant.

Usage:
    python test_audio_sync.py --list-devices      # List audio devices
    python test_audio_sync.py --test-mic          # Test microphone
    python test_audio_sync.py --calibrate         # Full calibration test
    python test_audio_sync.py --simulate          # Simulate with fake delay
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
    from scipy import signal
    from scipy.io import wavfile
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("\nInstall with:")
    print("  pip install numpy scipy sounddevice")
    sys.exit(1)


# ============================================================================
# Audio Analysis Functions (same as in the integration)
# ============================================================================

def generate_chirp(
    sample_rate: int = 44100,
    duration_ms: int = 200,
    freq_start: int = 1000,
    freq_end: int = 4000,
) -> np.ndarray:
    """Generate a chirp signal for calibration."""
    duration_sec = duration_ms / 1000
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, dtype=np.float32)

    # Generate chirp
    chirp = signal.chirp(
        t, f0=freq_start, f1=freq_end, t1=duration_sec, method="linear"
    )

    # Apply Hanning window
    window = np.hanning(num_samples).astype(np.float32)
    chirp = (chirp * window).astype(np.float32)

    # Normalize
    chirp = chirp / np.max(np.abs(chirp))

    return chirp


def find_offset(
    reference: np.ndarray,
    recorded: np.ndarray,
    sample_rate: int = 44100,
) -> tuple[float, float]:
    """
    Find the time offset between reference signal and recording.

    Returns:
        Tuple of (offset_ms, confidence_score)
    """
    # Normalize signals
    ref_norm = reference / (np.max(np.abs(reference)) + 1e-10)
    rec_norm = recorded / (np.max(np.abs(recorded)) + 1e-10)

    # Compute cross-correlation using FFT
    correlation = signal.correlate(rec_norm, ref_norm, mode="full", method="fft")

    # Find the peak
    peak_index = np.argmax(np.abs(correlation))

    # Calculate offset in samples
    zero_lag_index = len(reference) - 1
    offset_samples = peak_index - zero_lag_index

    # Convert to milliseconds
    offset_ms = (offset_samples / sample_rate) * 1000

    # Calculate confidence (standard score of peak)
    correlation_abs = np.abs(correlation)
    peak_value = correlation_abs[peak_index]
    mean_value = np.mean(correlation_abs)
    std_value = np.std(correlation_abs)

    if std_value > 0:
        confidence = (peak_value - mean_value) / std_value
    else:
        confidence = 0.0

    return offset_ms, confidence


# ============================================================================
# Test Functions
# ============================================================================

def list_devices():
    """List all available audio devices."""
    print("\n" + "=" * 60)
    print("AVAILABLE AUDIO DEVICES")
    print("=" * 60)

    devices = sd.query_devices()

    print("\n--- INPUT DEVICES (Microphones) ---")
    for idx, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            marker = " *" if device == sd.query_devices(kind="input") else ""
            print(f"  [{idx}] {device['name']}")
            print(f"       Channels: {device['max_input_channels']}, "
                  f"Sample Rate: {device['default_samplerate']:.0f} Hz{marker}")

    print("\n--- OUTPUT DEVICES (Speakers) ---")
    for idx, device in enumerate(devices):
        if device["max_output_channels"] > 0:
            marker = " *" if device == sd.query_devices(kind="output") else ""
            print(f"  [{idx}] {device['name']}")
            print(f"       Channels: {device['max_output_channels']}, "
                  f"Sample Rate: {device['default_samplerate']:.0f} Hz{marker}")

    print("\n(* = system default)")
    print()


def test_microphone(device_id: int | None = None, duration: float = 3.0):
    """Test microphone recording."""
    sample_rate = 44100

    print("\n" + "=" * 60)
    print("MICROPHONE TEST")
    print("=" * 60)

    if device_id is not None:
        device_info = sd.query_devices(device_id)
        print(f"\nUsing device [{device_id}]: {device_info['name']}")
    else:
        device_info = sd.query_devices(kind="input")
        print(f"\nUsing default input: {device_info['name']}")

    print(f"\nRecording for {duration} seconds...")
    print("Make some noise!\n")

    # Record
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32,
        device=device_id,
    )
    sd.wait()

    recording = recording.flatten()

    # Analyze
    peak_amplitude = np.max(np.abs(recording))
    rms = np.sqrt(np.mean(recording ** 2))

    print("Recording complete!\n")
    print(f"  Peak amplitude: {peak_amplitude:.4f}")
    print(f"  RMS level:      {rms:.4f}")
    print(f"  Samples:        {len(recording)}")

    if peak_amplitude < 0.01:
        print("\n  WARNING: Very low signal level. Check microphone connection.")
    elif peak_amplitude > 0.95:
        print("\n  WARNING: Signal may be clipping. Reduce input gain.")
    else:
        print("\n  Signal level looks good!")

    # Save recording
    output_file = Path("test_recording.wav")
    wavfile.write(str(output_file), sample_rate, (recording * 32767).astype(np.int16))
    print(f"\n  Saved to: {output_file.absolute()}")


def test_playback_and_record(
    input_device: int | None = None,
    output_device: int | None = None,
):
    """Test by playing a chirp and recording it."""
    sample_rate = 44100

    print("\n" + "=" * 60)
    print("PLAYBACK + RECORD TEST")
    print("=" * 60)
    print("\nThis test will:")
    print("  1. Generate a test chirp signal")
    print("  2. Play it through your speakers")
    print("  3. Record with your microphone")
    print("  4. Measure the round-trip latency\n")

    # Generate chirp
    chirp = generate_chirp(sample_rate, duration_ms=200)
    print(f"Generated chirp: {len(chirp)} samples ({len(chirp)/sample_rate*1000:.1f}ms)")

    # Prepare recording buffer (3 seconds)
    record_duration = 3.0
    record_samples = int(record_duration * sample_rate)

    print(f"\nStarting recording ({record_duration}s)...")
    print("Playing chirp in 0.5 seconds...\n")

    # Start recording
    recording = sd.rec(
        record_samples,
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32,
        device=input_device,
    )

    # Wait a bit, then play
    time.sleep(0.5)
    play_time = time.time()
    sd.play(chirp, sample_rate, device=output_device)
    sd.wait()

    # Wait for recording to finish
    while sd.get_stream().active:
        time.sleep(0.1)

    recording = recording.flatten()

    # Find offset
    print("Analyzing recording...")
    offset_ms, confidence = find_offset(chirp, recording, sample_rate)

    print("\n" + "-" * 40)
    print("RESULTS")
    print("-" * 40)
    print(f"  Detected offset:    {offset_ms:+.2f} ms")
    print(f"  Confidence score:   {confidence:.2f}")

    if confidence >= 10:
        print(f"  Status:             GOOD (reliable detection)")
    elif confidence >= 5:
        print(f"  Status:             OK (detection may be accurate)")
    else:
        print(f"  Status:             POOR (detection unreliable)")

    print("\n  Note: The offset includes speaker + microphone latency,")
    print("        plus the acoustic travel time.")

    # Save files for manual inspection
    wavfile.write("chirp_reference.wav", sample_rate, (chirp * 32767).astype(np.int16))
    wavfile.write("chirp_recorded.wav", sample_rate, (recording * 32767).astype(np.int16))
    print("\n  Saved: chirp_reference.wav, chirp_recorded.wav")


def simulate_latency():
    """Simulate latency detection with a known offset."""
    sample_rate = 44100

    print("\n" + "=" * 60)
    print("SIMULATION TEST")
    print("=" * 60)
    print("\nThis test simulates speaker latency by adding a known")
    print("delay to the signal and verifying detection accuracy.\n")

    # Generate chirp
    chirp = generate_chirp(sample_rate, duration_ms=200)

    # Test various delays
    test_delays_ms = [0, 25, 50, 100, 150, 200, 300, 500]

    print(f"{'Actual Delay':>15} | {'Detected':>12} | {'Error':>10} | {'Confidence':>10}")
    print("-" * 60)

    for actual_delay_ms in test_delays_ms:
        # Create simulated recording
        delay_samples = int(actual_delay_ms * sample_rate / 1000)

        # Add silence before chirp, then some noise after
        silence = np.zeros(delay_samples, dtype=np.float32)
        noise = np.random.randn(sample_rate).astype(np.float32) * 0.01
        simulated = np.concatenate([silence, chirp * 0.5, noise])

        # Find offset
        detected_ms, confidence = find_offset(chirp, simulated, sample_rate)

        error = detected_ms - actual_delay_ms

        print(f"{actual_delay_ms:>12} ms | {detected_ms:>+10.2f} ms | "
              f"{error:>+8.2f} ms | {confidence:>10.1f}")

    print("\n  Errors should be very small (< 1ms) for reliable detection.")


def run_full_calibration(
    input_device: int | None = None,
    output_device: int | None = None,
    num_runs: int = 3,
):
    """Run multiple calibration passes and average results."""
    sample_rate = 44100

    print("\n" + "=" * 60)
    print("FULL CALIBRATION TEST")
    print("=" * 60)
    print(f"\nRunning {num_runs} calibration passes...\n")

    results = []

    for i in range(num_runs):
        print(f"Pass {i + 1}/{num_runs}...")

        # Generate chirp
        chirp = generate_chirp(sample_rate, duration_ms=200)

        # Record
        record_samples = int(2.0 * sample_rate)
        recording = sd.rec(
            record_samples,
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
            device=input_device,
        )

        time.sleep(0.3)
        sd.play(chirp, sample_rate, device=output_device)
        sd.wait()

        recording = recording.flatten()

        offset_ms, confidence = find_offset(chirp, recording, sample_rate)
        results.append((offset_ms, confidence))

        print(f"  Offset: {offset_ms:+.2f}ms, Confidence: {confidence:.1f}")

        time.sleep(0.5)

    # Calculate statistics
    offsets = [r[0] for r in results]
    confidences = [r[1] for r in results]

    mean_offset = np.mean(offsets)
    std_offset = np.std(offsets)
    mean_confidence = np.mean(confidences)

    print("\n" + "-" * 40)
    print("CALIBRATION RESULTS")
    print("-" * 40)
    print(f"  Mean offset:        {mean_offset:+.2f} ms")
    print(f"  Std deviation:      {std_offset:.2f} ms")
    print(f"  Mean confidence:    {mean_confidence:.1f}")

    if std_offset < 5:
        print(f"\n  Consistency:        EXCELLENT")
    elif std_offset < 10:
        print(f"\n  Consistency:        GOOD")
    else:
        print(f"\n  Consistency:        POOR (high variance)")

    return mean_offset


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test audio sync calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_audio_sync.py --list-devices
  python test_audio_sync.py --test-mic --input 2
  python test_audio_sync.py --calibrate --input 2 --output 5
  python test_audio_sync.py --simulate
        """,
    )

    parser.add_argument(
        "--list-devices", "-l",
        action="store_true",
        help="List all audio devices",
    )
    parser.add_argument(
        "--test-mic", "-m",
        action="store_true",
        help="Test microphone recording",
    )
    parser.add_argument(
        "--calibrate", "-c",
        action="store_true",
        help="Run playback + record calibration test",
    )
    parser.add_argument(
        "--full-calibrate", "-f",
        action="store_true",
        help="Run multiple calibration passes",
    )
    parser.add_argument(
        "--simulate", "-s",
        action="store_true",
        help="Run simulation with known delays",
    )
    parser.add_argument(
        "--input", "-i",
        type=int,
        default=None,
        help="Input device ID (microphone)",
    )
    parser.add_argument(
        "--output", "-o",
        type=int,
        default=None,
        help="Output device ID (speaker)",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=3.0,
        help="Recording duration in seconds (default: 3.0)",
    )

    args = parser.parse_args()

    # Default action
    if not any([args.list_devices, args.test_mic, args.calibrate,
                args.full_calibrate, args.simulate]):
        args.list_devices = True

    print("\n" + "=" * 60)
    print("  MUSIC SYNC - Audio Calibration Test Tool")
    print("=" * 60)

    if args.list_devices:
        list_devices()

    if args.test_mic:
        test_microphone(args.input, args.duration)

    if args.calibrate:
        test_playback_and_record(args.input, args.output)

    if args.full_calibrate:
        run_full_calibration(args.input, args.output)

    if args.simulate:
        simulate_latency()


if __name__ == "__main__":
    main()
