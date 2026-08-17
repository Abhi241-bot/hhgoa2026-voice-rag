"""
Audio recording utility using sounddevice.

Provides a simple function to record microphone input and return raw WAV bytes.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import sounddevice as sd
import soundfile as sf

from src.logger import get_logger
from src.models import AudioInput

log = get_logger("stt.recorder")

DEFAULT_SAMPLE_RATE = 16000  # 16kHz — standard for STT
DEFAULT_CHANNELS = 1         # mono


def record_audio(
    duration_seconds: float = 5.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
) -> AudioInput:
    """
    Record audio from the default microphone.

    Args:
        duration_seconds: How long to record in seconds.
        sample_rate: Sample rate in Hz.
        channels: Number of audio channels (1=mono, 2=stereo).

    Returns:
        AudioInput with raw WAV bytes.
    """
    log.info("recording_start", duration_s=duration_seconds)

    audio_data = sd.rec(
        int(duration_seconds * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
    )
    sd.wait()  # Block until recording is done

    log.info("recording_complete", samples=len(audio_data))

    # Convert numpy array to WAV bytes
    wav_bytes = _numpy_to_wav_bytes(audio_data, sample_rate)
    return AudioInput(audio_bytes=wav_bytes, format="wav")


def audio_file_to_input(file_path: str) -> AudioInput:
    """
    Load an audio file and return an AudioInput.

    Supports WAV, MP3, OGG, FLAC via soundfile/librosa.
    """
    return AudioInput(file_path=file_path, format=file_path.rsplit(".", 1)[-1].lower())


def _numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert a numpy int16 array to WAV bytes."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 2 bytes = int16
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    buffer.seek(0)
    return buffer.read()
