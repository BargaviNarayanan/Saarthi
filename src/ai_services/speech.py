"""Local mock for speech-to-text transcription."""

from __future__ import annotations

from typing import Any


class SpeechClient:
    """Provide credential-free mock transcription results."""

    def transcribe(
        self,
        audio_content: bytes,
        language_code: str = "en-IN",
        sample_rate_hertz: int = 16000,
        encoding: str = "LINEAR16",
    ) -> dict[str, Any]:
        """Return a predictable transcript for non-empty audio input."""
        return {
            "transcript": "Mock transcription" if audio_content else "",
            "confidence": 0.0,
            "language_code": language_code,
            "sample_rate_hertz": sample_rate_hertz,
            "encoding": encoding,
        }
