"""Local mock for text-to-speech synthesis."""

from __future__ import annotations


class TextToSpeechClient:
    """Synthesize deterministic placeholder audio without cloud credentials."""

    def synthesize(
        self,
        text: str,
        language_code: str = "en-IN",
        voice_name: str | None = None,
    ) -> bytes:
        """Return placeholder audio bytes describing the synthesis request."""
        voice = voice_name or f"{language_code}-Standard-A"
        return f"mock audio ({language_code}, {voice}): {text}".encode("utf-8")
