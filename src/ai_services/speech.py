"""Google Cloud Speech-to-Text and Text-to-Speech integration."""

from __future__ import annotations

import os
from typing import Any

from google.cloud import speech_v1, texttospeech_v1


class SpeechClient:
    """Transcribe voice queries and synthesize localized responses."""

    def __init__(self, project_id: str | None = None) -> None:
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.stt = speech_v1.SpeechClient()
        self.tts = texttospeech_v1.TextToSpeechClient()

    def transcribe(
        self,
        audio_content: bytes,
        language_code: str = "en-IN",
        sample_rate_hertz: int = 16000,
        encoding: speech_v1.RecognitionConfig.AudioEncoding = speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
    ) -> dict[str, Any]:
        response = self.stt.recognize(
            config=speech_v1.RecognitionConfig(
                encoding=encoding,
                sample_rate_hertz=sample_rate_hertz,
                language_code=language_code,
                enable_automatic_punctuation=True,
            ),
            audio=speech_v1.RecognitionAudio(content=audio_content),
        )
        alternatives = [
            result.alternatives[0]
            for result in response.results
            if result.alternatives
        ]
        return {
            "transcript": " ".join(item.transcript for item in alternatives),
            "confidence": alternatives[0].confidence if alternatives else 0.0,
            "language_code": language_code,
        }

    def synthesize(
        self,
        text: str,
        language_code: str = "ta-IN",
        voice_name: str | None = None,
    ) -> bytes:
        voice = texttospeech_v1.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name or f"{language_code}-Standard-A",
        )
        response = self.tts.synthesize_speech(
            input=texttospeech_v1.SynthesisInput(text=text),
            voice=voice,
            audio_config=texttospeech_v1.AudioConfig(
                audio_encoding=texttospeech_v1.AudioEncoding.MP3,
            ),
        )
        return response.audio_content
