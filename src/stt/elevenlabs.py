"""
ElevenLabs Speech-to-Text integration.

Provides the ElevenLabsSTT tool for transcribing audio input (microphone or file).
Falls back to a MockSTT when the API key is not set (useful for testing).

ElevenLabs STT API: https://elevenlabs.io/docs/api-reference/speech-to-text
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Optional

from src.config import settings
from src.logger import get_logger
from src.models import AudioInput, TranscriptionResult

log = get_logger("stt.elevenlabs")


class ElevenLabsSTT:
    """
    Transcribes audio using the ElevenLabs Speech-to-Text API.

    Args:
        api_key: ElevenLabs API key (defaults to ELEVENLABS_API_KEY env var).
        model: ElevenLabs STT model (default: 'scribe_v1').
        language_code: ISO 639-1 language code hint (default: 'en').
    """

    # ElevenLabs current STT model
    DEFAULT_MODEL = "scribe_v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        language_code: str = "en",
    ) -> None:
        self._api_key = api_key or settings.elevenlabs_api_key
        self.model = model
        self.language_code = language_code

        if not self._api_key:
            raise ValueError(
                "ElevenLabs API key not set. "
                "Set ELEVENLABS_API_KEY in .env or pass api_key=..."
            )

        from elevenlabs.client import ElevenLabs
        self._client = ElevenLabs(api_key=self._api_key)
        log.info("elevenlabs_stt_init", model=self.model)

    def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        """
        Transcribe an AudioInput object.

        Args:
            audio: AudioInput with either audio_bytes or file_path set.

        Returns:
            TranscriptionResult with transcript, confidence, and latency.
        """
        start = time.perf_counter()

        # Load audio data
        if audio.audio_bytes:
            audio_data = io.BytesIO(audio.audio_bytes)
            audio_data.name = f"audio.{audio.format}"
        elif audio.file_path:
            file_path = Path(audio.file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio.file_path}")
            audio_data = open(file_path, "rb")
        else:
            raise ValueError("AudioInput must have either audio_bytes or file_path")

        try:
            result = self._client.speech_to_text.convert(
                file=audio_data,
                model_id=self.model,
                language_code=self.language_code,
            )
        finally:
            if hasattr(audio_data, "close") and not isinstance(audio_data, io.BytesIO):
                audio_data.close()

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        transcript = result.text if hasattr(result, "text") else str(result)

        log.info(
            "transcription_complete",
            transcript=transcript[:100],
            latency_ms=latency_ms,
        )

        return TranscriptionResult(
            transcript=transcript,
            confidence=1.0,  # ElevenLabs doesn't expose confidence; default to 1.0
            language=self.language_code,
            latency_ms=latency_ms,
        )

    def transcribe_file(self, file_path: str) -> TranscriptionResult:
        """Convenience method for file-path input."""
        return self.transcribe(AudioInput(file_path=file_path))

    def __repr__(self) -> str:
        return f"ElevenLabsSTT(model='{self.model}', language='{self.language_code}')"


class MockSTT:
    """
    Mock STT for testing without an API key.
    Returns a fixed or configurable transcript.
    """

    def __init__(self, mock_transcript: str = "What is the capital of France?") -> None:
        self.mock_transcript = mock_transcript
        log.warning("mock_stt_active", message="Using MockSTT — no real transcription")

    def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        time.sleep(0.01)  # Simulate minimal latency
        return TranscriptionResult(
            transcript=self.mock_transcript,
            confidence=1.0,
            language="en",
            latency_ms=10.0,
        )

    def transcribe_file(self, file_path: str) -> TranscriptionResult:
        return self.transcribe(AudioInput(file_path=file_path))


def get_stt_tool(mock: bool = False, mock_transcript: str = "") -> ElevenLabsSTT | MockSTT:
    """
    Factory function to get the appropriate STT tool.

    Args:
        mock: If True, return MockSTT regardless of API key.
        mock_transcript: If mock=True, use this transcript.

    Returns:
        ElevenLabsSTT if API key is available, else MockSTT.
    """
    if mock:
        return MockSTT(mock_transcript=mock_transcript or "What is the capital of France?")

    if not settings.elevenlabs_api_key:
        log.warning("no_elevenlabs_key", fallback="Using MockSTT")
        return MockSTT()

    return ElevenLabsSTT()
