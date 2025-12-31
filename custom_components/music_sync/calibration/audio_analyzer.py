"""Audio analysis for latency detection using cross-correlation."""
from __future__ import annotations

import asyncio
import logging
import tempfile
import os
from pathlib import Path
from typing import Any

import numpy as np

_LOGGER = logging.getLogger(__name__)

# Type hints for optional dependencies
try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    sd = None

try:
    from scipy import signal
    from scipy.io import wavfile

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    signal = None
    wavfile = None


class AudioAnalyzer:
    """
    Analyzes audio to detect latency between speakers using cross-correlation.

    The analyzer works by:
    1. Generating a test signal (chirp/beep)
    2. Playing it on a speaker via Home Assistant
    3. Recording the audio with a microphone
    4. Using cross-correlation to find the time offset
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        device: str | int | None = None,
        output_path: Path | None = None,
    ) -> None:
        """
        Initialize the audio analyzer.

        Args:
            sample_rate: Audio sample rate (default 44100)
            device: Microphone device ID or name
            output_path: Directory to save calibration files (e.g., HA's www folder)
                        If None, uses a temp directory
        """
        self._sample_rate = sample_rate
        self._device = self._parse_device(device)
        self._output_path: Path | None = output_path
        self._temp_dir: Path | None = None
        self._initialized = False

    def _parse_device(self, device: str | int | None) -> int | None:
        """Parse device identifier."""
        if device is None or device == "default":
            return None
        try:
            return int(device)
        except (ValueError, TypeError):
            return None

    async def async_initialize(self) -> None:
        """Initialize the analyzer."""
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError(
                "sounddevice library not available. "
                "Install with: pip install sounddevice"
            )

        if not SCIPY_AVAILABLE:
            raise RuntimeError(
                "scipy library not available. " "Install with: pip install scipy"
            )

        # Use provided output path or create temp directory
        if self._output_path:
            self._output_path.mkdir(parents=True, exist_ok=True)
            _LOGGER.info("Using output path: %s", self._output_path)
        else:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="music_sync_"))
            _LOGGER.info("Using temp directory: %s", self._temp_dir)

        # Verify microphone access
        try:
            devices = sd.query_devices()
            if self._device is not None:
                device_info = devices[self._device]
                if device_info.get("max_input_channels", 0) == 0:
                    raise RuntimeError(f"Device {self._device} has no input channels")
                _LOGGER.info("Using microphone: %s", device_info["name"])
            else:
                default = sd.query_devices(kind="input")
                _LOGGER.info("Using default microphone: %s", default["name"])
        except Exception as err:
            raise RuntimeError(f"Failed to access microphone: {err}") from err

        self._initialized = True

    async def async_close(self) -> None:
        """Clean up resources."""
        if self._temp_dir and self._temp_dir.exists():
            for file in self._temp_dir.iterdir():
                file.unlink()
            self._temp_dir.rmdir()

    def generate_test_signal(
        self,
        duration_ms: int = 200,
        freq_start: int = 1000,
        freq_end: int = 4000,
    ) -> np.ndarray:
        """
        Generate a chirp signal for calibration.

        A chirp signal sweeps from freq_start to freq_end Hz,
        making it easy to detect via cross-correlation even
        in noisy environments.
        """
        duration_sec = duration_ms / 1000
        num_samples = int(self._sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, num_samples, dtype=np.float32)

        # Generate chirp
        chirp = signal.chirp(
            t, f0=freq_start, f1=freq_end, t1=duration_sec, method="linear"
        )

        # Apply Hanning window to avoid clicks
        window = np.hanning(num_samples).astype(np.float32)
        chirp = (chirp * window).astype(np.float32)

        # Normalize
        chirp = chirp / np.max(np.abs(chirp))

        return chirp

    async def save_test_signal(self, signal_data: np.ndarray) -> str:
        """
        Save test signal to a WAV file for playback.

        Returns:
            If output_path was provided (HA www folder): returns just the filename
            If using temp dir: returns full path
        """
        # Determine output directory
        if self._output_path:
            output_dir = self._output_path
            return_filename_only = True
        elif self._temp_dir:
            output_dir = self._temp_dir
            return_filename_only = False
        else:
            raise RuntimeError("Analyzer not initialized")

        filename = "calibration_chirp.wav"
        file_path = output_dir / filename

        # Convert to 16-bit PCM
        signal_int16 = (signal_data * 32767).astype(np.int16)

        # Write WAV file
        wavfile.write(str(file_path), self._sample_rate, signal_int16)

        _LOGGER.debug("Saved test signal to %s", file_path)

        # Return just filename for HA URL construction, or full path for temp files
        return filename if return_filename_only else str(file_path)

    async def record_audio(self, duration: float = 3.0) -> np.ndarray:
        """
        Record audio from the microphone.

        Args:
            duration: Recording duration in seconds

        Returns:
            Numpy array of recorded audio samples
        """
        if not self._initialized:
            raise RuntimeError("Analyzer not initialized")

        num_samples = int(duration * self._sample_rate)

        _LOGGER.debug(
            "Recording %0.1f seconds (%d samples)", duration, num_samples
        )

        # Record in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        recording = await loop.run_in_executor(
            None,
            lambda: sd.rec(
                num_samples,
                samplerate=self._sample_rate,
                channels=1,
                dtype=np.float32,
                device=self._device,
            ),
        )

        # Wait for recording to complete
        await loop.run_in_executor(None, sd.wait)

        # Flatten to 1D
        recording = recording.flatten()

        _LOGGER.debug("Recorded %d samples", len(recording))
        return recording

    def find_offset(
        self,
        reference: np.ndarray,
        recorded: np.ndarray,
    ) -> tuple[float, float]:
        """
        Find the time offset between reference signal and recording.

        Uses cross-correlation to find where the reference signal
        appears in the recording.

        Args:
            reference: The original test signal
            recorded: The recorded audio from microphone

        Returns:
            Tuple of (offset_ms, confidence_score)
            - offset_ms: Time offset in milliseconds
              Positive = recorded is delayed
              Negative = recorded is early
            - confidence_score: Standard score of correlation peak
              Higher is more reliable (>10 is good)
        """
        # Normalize signals
        ref_norm = reference / (np.max(np.abs(reference)) + 1e-10)
        rec_norm = recorded / (np.max(np.abs(recorded)) + 1e-10)

        # Compute cross-correlation using FFT (fast)
        correlation = signal.correlate(rec_norm, ref_norm, mode="full", method="fft")

        # Find the peak
        peak_index = np.argmax(np.abs(correlation))

        # Calculate offset in samples
        # The correlation array has length (len(rec) + len(ref) - 1)
        # Peak at index len(ref)-1 means zero offset
        zero_lag_index = len(reference) - 1
        offset_samples = peak_index - zero_lag_index

        # Convert to milliseconds
        offset_ms = (offset_samples / self._sample_rate) * 1000

        # Calculate confidence (standard score of peak)
        correlation_abs = np.abs(correlation)
        peak_value = correlation_abs[peak_index]
        mean_value = np.mean(correlation_abs)
        std_value = np.std(correlation_abs)

        if std_value > 0:
            confidence = (peak_value - mean_value) / std_value
        else:
            confidence = 0.0

        _LOGGER.debug(
            "Cross-correlation: offset=%d samples (%.2fms), confidence=%.2f",
            offset_samples,
            offset_ms,
            confidence,
        )

        return offset_ms, confidence

    def generate_unique_tones(
        self,
        num_speakers: int,
        duration_ms: int = 500,
        base_freq: int = 1000,
        freq_step: int = 500,
    ) -> list[np.ndarray]:
        """
        Generate unique tones for multi-speaker simultaneous calibration.

        Each speaker gets a different frequency so they can be
        distinguished in a single recording.
        """
        tones = []
        duration_sec = duration_ms / 1000
        num_samples = int(self._sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, num_samples, dtype=np.float32)

        for i in range(num_speakers):
            freq = base_freq + (i * freq_step)

            # Generate sine wave
            tone = np.sin(2 * np.pi * freq * t)

            # Apply window
            window = np.hanning(num_samples).astype(np.float32)
            tone = (tone * window).astype(np.float32)

            tones.append(tone)

        return tones

    def analyze_multi_speaker_recording(
        self,
        recording: np.ndarray,
        frequencies: list[int],
        bandwidth: int = 100,
    ) -> dict[int, float]:
        """
        Analyze a recording containing multiple speaker tones.

        Uses bandpass filters to isolate each frequency and
        detect when each tone starts.

        Args:
            recording: The recorded audio
            frequencies: List of frequencies to detect
            bandwidth: Bandwidth for bandpass filter in Hz

        Returns:
            Dict mapping frequency -> onset time in ms
        """
        onsets = {}

        for freq in frequencies:
            # Design bandpass filter
            low = freq - bandwidth
            high = freq + bandwidth
            sos = signal.butter(
                4, [low, high], btype="band", fs=self._sample_rate, output="sos"
            )

            # Apply filter
            filtered = signal.sosfilt(sos, recording)

            # Compute envelope using Hilbert transform
            analytic = signal.hilbert(filtered)
            envelope = np.abs(analytic)

            # Find onset (when envelope exceeds threshold)
            threshold = np.max(envelope) * 0.1
            onset_indices = np.where(envelope > threshold)[0]

            if len(onset_indices) > 0:
                onset_sample = onset_indices[0]
                onset_ms = (onset_sample / self._sample_rate) * 1000
                onsets[freq] = onset_ms
            else:
                onsets[freq] = -1  # Not detected

        return onsets
