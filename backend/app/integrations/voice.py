"""Backend voice adapters backed by local speech mocks."""

from __future__ import annotations

import base64
from pathlib import Path

from src.ai_services.speech import SpeechClient
from src.ai_services.texttospeech import TextToSpeechClient


class SpeechToTextManager:
    """Read audio and return local mock transcription results."""

    language_codes = {
        "english": "en-IN",
        "hindi": "hi-IN",
        "tamil": "ta-IN",
        "telugu": "te-IN",
        "kannada": "kn-IN",
        "malayalam": "ml-IN",
        "marathi": "mr-IN",
        "bengali": "bn-IN",
        "punjabi": "pa-IN",
        "gujarati": "gu-IN",
        "urdu": "ur-IN",
        "assamese": "as-IN",
        "odia": "or-IN",
    }

    def __init__(
        self,
        project_id: str | None = None,
        credentials_path: str | None = None,
    ) -> None:
        self.client = SpeechClient()

    def _normalize_language(self, language: str) -> str:
        return self.language_codes.get(language.casefold(), language)

    def transcribe_audio_file(
        self,
        audio_file_path: str,
        language: str = "English",
        enable_automatic_punctuation: bool = True,
        enable_profanity_filter: bool = True,
    ) -> dict:
        try:
            audio_content = Path(audio_file_path).read_bytes()
        except FileNotFoundError:
            return {
                "success": False,
                "error": "Audio file not found",
                "transcription": "",
            }
        return self.transcribe_audio_bytes(audio_content, language)

    def transcribe_audio_bytes(
        self,
        audio_content: bytes,
        language: str = "English",
        sample_rate_hertz: int = 16000,
        audio_encoding: str = "LINEAR16",
        enable_automatic_punctuation: bool = True,
    ) -> dict:
        language_code = self._normalize_language(language)
        result = self.client.transcribe(
            audio_content,
            language_code=language_code,
            sample_rate_hertz=sample_rate_hertz,
            encoding=audio_encoding,
        )
        transcript = result["transcript"]
        return {
            "success": bool(transcript),
            "transcription": transcript,
            "confidence": result["confidence"],
            "language": language,
            "language_code": language_code,
            "alternatives": [],
            **({"message": "No speech detected"} if not transcript else {}),
        }

    def streaming_transcribe(self, audio_chunks, language: str = "English") -> dict:
        return self.transcribe_audio_bytes(b"".join(audio_chunks), language)

    def get_supported_languages(self) -> dict[str, str]:
        return dict(self.language_codes)


class TextToSpeechManager:
    """Return local placeholder audio in the backend's existing response shape."""

    language_codes = SpeechToTextManager.language_codes

    def __init__(
        self,
        project_id: str | None = None,
        credentials_path: str | None = None,
    ) -> None:
        self.client = TextToSpeechClient()

    def _normalize_language(self, language: str) -> str:
        return self.language_codes.get(language.casefold(), language)

    def synthesize_speech(
        self,
        text: str,
        language: str = "English",
        gender: str = "FEMALE",
        speaking_rate: float = 1.0,
    ) -> dict:
        language_code = self._normalize_language(language)
        audio = self.client.synthesize(text, language_code=language_code)
        return {
            "success": True,
            "audio_content": audio,
            "audio_content_base64": base64.b64encode(audio).decode("ascii"),
            "language": language,
            "language_code": language_code,
            "voice": f"{language_code}-Standard-A",
        }

    def synthesize_speech_to_file(
        self,
        text: str,
        output_file_path: str,
        language: str = "English",
        gender: str = "FEMALE",
        speaking_rate: float = 1.0,
    ) -> dict:
        result = self.synthesize_speech(text, language, gender, speaking_rate)
        Path(output_file_path).write_bytes(result["audio_content"])
        return {"success": True, "file_path": output_file_path}

    def get_supported_languages(self) -> dict[str, str]:
        return dict(self.language_codes)

    def get_available_voices(self) -> dict[str, list[dict[str, str]]]:
        return {
            code: [{"name": f"{code}-Standard-A", "gender": "FEMALE"}]
            for code in self.language_codes.values()
        }
