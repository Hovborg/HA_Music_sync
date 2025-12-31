#!/usr/bin/env python3
"""
Visualization demo for cross-correlation based audio sync.

This script creates visual plots showing how the algorithm
detects latency between speakers.

Usage:
    python demo_visualization.py
"""

import numpy as np
from scipy import signal

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib not installed. Install with: pip install matplotlib")


def generate_chirp(sample_rate: int, duration_ms: int) -> np.ndarray:
    """Generate a chirp signal."""
    duration_sec = duration_ms / 1000
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, dtype=np.float32)

    chirp = signal.chirp(t, f0=1000, f1=4000, t1=duration_sec, method="linear")
    window = np.hanning(num_samples)

    return (chirp * window).astype(np.float32)


def simulate_speaker_recording(
    reference: np.ndarray,
    delay_ms: float,
    sample_rate: int,
    noise_level: float = 0.05,
    attenuation: float = 0.3,
) -> np.ndarray:
    """
    Simulate what a microphone would record from a speaker.

    Adds delay, noise, and attenuation to simulate real conditions.
    """
    delay_samples = int(delay_ms * sample_rate / 1000)

    # Create recording with delay
    total_samples = len(reference) + delay_samples + sample_rate  # +1s padding
    recording = np.zeros(total_samples, dtype=np.float32)

    # Add delayed and attenuated signal
    recording[delay_samples:delay_samples + len(reference)] = reference * attenuation

    # Add noise
    noise = np.random.randn(total_samples).astype(np.float32) * noise_level
    recording += noise

    return recording


def visualize_sync_detection():
    """Create visualization of the sync detection algorithm."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib required for visualization")
        return

    sample_rate = 44100

    # Generate reference chirp
    chirp = generate_chirp(sample_rate, duration_ms=200)

    # Simulate recordings from 3 speakers with different delays
    speakers = {
        "Google Home": {"delay": 150, "color": "#4285F4"},
        "Sonos": {"delay": 75, "color": "#000000"},
        "Alexa": {"delay": 220, "color": "#00CAFF"},
    }

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Music Sync - Cross-Correlation Latency Detection", fontsize=14, fontweight="bold")

    # Time axis for reference signal
    t_ref = np.arange(len(chirp)) / sample_rate * 1000

    # Plot reference signal
    axes[0, 0].plot(t_ref, chirp, color="#333333", linewidth=0.8)
    axes[0, 0].set_title("Reference Signal (Chirp)")
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].set_xlim([0, max(t_ref)])
    axes[0, 0].grid(True, alpha=0.3)

    # Results table data
    results = []

    # Process each speaker
    for idx, (name, config) in enumerate(speakers.items()):
        actual_delay = config["delay"]
        color = config["color"]

        # Simulate recording
        recording = simulate_speaker_recording(
            chirp, actual_delay, sample_rate,
            noise_level=0.03, attenuation=0.4
        )

        # Cross-correlation
        correlation = signal.correlate(recording, chirp, mode="full", method="fft")
        correlation_norm = correlation / np.max(np.abs(correlation))

        # Find peak
        peak_idx = np.argmax(np.abs(correlation))
        zero_lag = len(chirp) - 1
        detected_samples = peak_idx - zero_lag
        detected_ms = detected_samples / sample_rate * 1000

        # Confidence
        corr_abs = np.abs(correlation_norm)
        confidence = (corr_abs[peak_idx] - np.mean(corr_abs)) / np.std(corr_abs)

        results.append((name, actual_delay, detected_ms, confidence))

        # Plot recording
        t_rec = np.arange(len(recording)) / sample_rate * 1000
        ax_rec = axes[idx, 0] if idx > 0 else axes[0, 0]

        if idx > 0:
            ax_rec.plot(t_rec, recording, color=color, linewidth=0.5, alpha=0.8)
            ax_rec.axvline(x=actual_delay, color="red", linestyle="--",
                          label=f"Actual delay: {actual_delay}ms", linewidth=1.5)
            ax_rec.set_title(f"{name} - Recorded Signal")
            ax_rec.set_xlabel("Time (ms)")
            ax_rec.set_ylabel("Amplitude")
            ax_rec.set_xlim([0, 500])
            ax_rec.legend(loc="upper right")
            ax_rec.grid(True, alpha=0.3)

        # Plot correlation
        ax_corr = axes[idx, 1]
        lag_ms = (np.arange(len(correlation)) - zero_lag) / sample_rate * 1000

        ax_corr.plot(lag_ms, correlation_norm, color=color, linewidth=0.8)
        ax_corr.axvline(x=detected_ms, color="green", linestyle="-",
                       label=f"Detected: {detected_ms:.1f}ms", linewidth=2)
        ax_corr.axvline(x=actual_delay, color="red", linestyle="--",
                       label=f"Actual: {actual_delay}ms", linewidth=1.5)

        ax_corr.set_title(f"{name} - Cross-Correlation (Confidence: {confidence:.1f})")
        ax_corr.set_xlabel("Lag (ms)")
        ax_corr.set_ylabel("Correlation")
        ax_corr.set_xlim([-50, 400])
        ax_corr.legend(loc="upper right")
        ax_corr.grid(True, alpha=0.3)

        # Mark peak
        ax_corr.plot(detected_ms, correlation_norm[peak_idx], "go", markersize=10)

    plt.tight_layout()

    # Print results
    print("\n" + "=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)
    print(f"{'Speaker':<15} {'Actual':>10} {'Detected':>12} {'Error':>10} {'Confidence':>12}")
    print("-" * 60)

    for name, actual, detected, conf in results:
        error = detected - actual
        status = "OK" if abs(error) < 5 else "CHECK"
        print(f"{name:<15} {actual:>8}ms {detected:>10.1f}ms {error:>+8.1f}ms {conf:>10.1f} [{status}]")

    # Calculate sync corrections
    print("\n" + "=" * 60)
    print("SYNC CORRECTIONS")
    print("=" * 60)

    # Find the slowest speaker (highest delay)
    max_delay = max(r[2] for r in results)

    print(f"\nReference point: {max_delay:.1f}ms (slowest speaker)")
    print("\nTo synchronize, add these delays:\n")

    for name, actual, detected, conf in results:
        correction = max_delay - detected
        print(f"  {name:<15}: +{correction:>6.1f}ms")

    print("\n" + "-" * 60)
    print("After correction, all speakers should play in sync!")

    # Save figure
    plt.savefig("sync_detection_demo.png", dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: sync_detection_demo.png")

    plt.show()


def visualize_multi_tone_detection():
    """
    Demonstrate multi-speaker simultaneous detection using different frequencies.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib required for visualization")
        return

    sample_rate = 44100
    duration = 0.5  # seconds

    # Each speaker gets a unique frequency
    speakers = {
        "Google (1000Hz)": {"freq": 1000, "delay": 150, "color": "#4285F4"},
        "Sonos (1500Hz)": {"freq": 1500, "delay": 75, "color": "#000000"},
        "Alexa (2000Hz)": {"freq": 2000, "delay": 220, "color": "#00CAFF"},
    }

    # Generate individual tones
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Create combined recording (what microphone would hear)
    combined = np.zeros(int(sample_rate * 1.5))  # 1.5s buffer

    for name, config in speakers.items():
        # Generate tone
        tone = np.sin(2 * np.pi * config["freq"] * t) * np.hanning(len(t))

        # Add to combined with delay
        delay_samples = int(config["delay"] * sample_rate / 1000)
        combined[delay_samples:delay_samples + len(tone)] += tone * 0.3

    # Add noise
    combined += np.random.randn(len(combined)) * 0.02

    # Analyze
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Multi-Tone Speaker Detection", fontsize=14, fontweight="bold")

    # Plot combined signal
    t_combined = np.arange(len(combined)) / sample_rate * 1000
    axes[0, 0].plot(t_combined, combined, linewidth=0.5, color="#333")
    axes[0, 0].set_title("Combined Recording (all speakers)")
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].set_xlim([0, 500])
    axes[0, 0].grid(True, alpha=0.3)

    # Plot spectrogram
    f, t_spec, Sxx = signal.spectrogram(combined, sample_rate, nperseg=1024)
    axes[0, 1].pcolormesh(t_spec * 1000, f, 10 * np.log10(Sxx + 1e-10),
                          shading="gouraud", cmap="viridis")
    axes[0, 1].set_title("Spectrogram")
    axes[0, 1].set_xlabel("Time (ms)")
    axes[0, 1].set_ylabel("Frequency (Hz)")
    axes[0, 1].set_ylim([0, 3000])

    # Add frequency markers
    for name, config in speakers.items():
        axes[0, 1].axhline(y=config["freq"], color=config["color"],
                          linestyle="--", alpha=0.7, linewidth=2)

    # Bandpass filter and detect each tone
    detected_onsets = {}

    for name, config in speakers.items():
        freq = config["freq"]
        color = config["color"]

        # Design bandpass filter
        low = freq - 100
        high = freq + 100
        sos = signal.butter(4, [low, high], btype="band", fs=sample_rate, output="sos")

        # Apply filter
        filtered = signal.sosfilt(sos, combined)

        # Get envelope
        analytic = signal.hilbert(filtered)
        envelope = np.abs(analytic)

        # Find onset
        threshold = np.max(envelope) * 0.1
        onset_idx = np.argmax(envelope > threshold)
        onset_ms = onset_idx / sample_rate * 1000

        detected_onsets[name] = onset_ms

    # Plot filtered signals with onset detection
    ax_filtered = axes[1, 0]

    for name, config in speakers.items():
        freq = config["freq"]
        color = config["color"]

        # Filter
        low = freq - 100
        high = freq + 100
        sos = signal.butter(4, [low, high], btype="band", fs=sample_rate, output="sos")
        filtered = signal.sosfilt(sos, combined)

        # Envelope
        envelope = np.abs(signal.hilbert(filtered))

        t_ms = np.arange(len(envelope)) / sample_rate * 1000
        ax_filtered.plot(t_ms, envelope, color=color, label=name, linewidth=1.5)

        # Mark onset
        onset_ms = detected_onsets[name]
        ax_filtered.axvline(x=onset_ms, color=color, linestyle="--", alpha=0.7)

    ax_filtered.set_title("Bandpass Filtered Envelopes")
    ax_filtered.set_xlabel("Time (ms)")
    ax_filtered.set_ylabel("Envelope")
    ax_filtered.set_xlim([0, 500])
    ax_filtered.legend()
    ax_filtered.grid(True, alpha=0.3)

    # Results bar chart
    ax_bar = axes[1, 1]
    names = list(speakers.keys())
    actual_delays = [speakers[n]["delay"] for n in names]
    detected_delays = [detected_onsets[n] for n in names]
    colors = [speakers[n]["color"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax_bar.bar(x - width/2, actual_delays, width, label="Actual", color="lightgray", edgecolor="black")
    bars2 = ax_bar.bar(x + width/2, detected_delays, width, label="Detected", color=colors, edgecolor="black")

    ax_bar.set_title("Delay Comparison")
    ax_bar.set_ylabel("Delay (ms)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([n.split()[0] for n in names])
    ax_bar.legend()
    ax_bar.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("multi_tone_demo.png", dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: multi_tone_demo.png")

    plt.show()


def main():
    print("\n" + "=" * 60)
    print("  MUSIC SYNC - Algorithm Visualization Demo")
    print("=" * 60)

    print("\n1. Cross-correlation based detection (sequential)")
    visualize_sync_detection()

    print("\n\n2. Multi-tone detection (simultaneous)")
    visualize_multi_tone_detection()


if __name__ == "__main__":
    main()
