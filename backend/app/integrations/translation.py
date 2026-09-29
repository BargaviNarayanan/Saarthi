"""Backend translation adapter backed by the local translation mock."""

from __future__ import annotations

from src.ai_services.translate import TranslateClient


class TranslationManager:
    """Keep the backend's translation API while using a local mock client."""

    language_codes = {
        "english": "en",
        "hindi": "hi",
        "tamil": "ta",
        "telugu": "te",
        "kannada": "kn",
        "malayalam": "ml",
        "marathi": "mr",
        "bengali": "bn",
        "punjabi": "pa",
        "gujarati": "gu",
        "urdu": "ur",
        "assamese": "as",
        "odia": "or",
        "sindhi": "sd",
    }

    def __init__(
        self,
        project_id: str | None = None,
        credentials_path: str | None = None,
        api_version: str = "mock",
        cache_enabled: bool = True,
    ) -> None:
        self.client = TranslateClient()
        self.cache_enabled = cache_enabled
        self.translation_cache: dict[tuple[str, str, str], dict] = {}

    def _normalize_language_code(self, language: str) -> str:
        return self.language_codes.get(language.casefold(), language)

    def detect_language(self, text: str) -> dict:
        result = self.client.translate(text, target_language="en")
        language_code = result["detected_source_language"]
        language_name = next(
            (name.title() for name, code in self.language_codes.items() if code == language_code),
            "English" if language_code == "en" else "Unknown",
        )
        return {
            "language_code": language_code,
            "language_name": language_name,
            "confidence": 1.0,
            "success": True,
        }

    def translate_text(
        self,
        text: str,
        source_language: str = "auto",
        target_language: str = "en",
    ) -> dict:
        source_code = (
            None if source_language == "auto" else self._normalize_language_code(source_language)
        )
        target_code = self._normalize_language_code(target_language)
        cache_key = (text, source_code or "auto", target_code)
        if self.cache_enabled and cache_key in self.translation_cache:
            return {**self.translation_cache[cache_key], "from_cache": True}

        translation = self.client.translate(
            text,
            target_language=target_code,
            source_language=source_code,
        )
        result = {
            "success": True,
            "original_text": text,
            "translated_text": translation["translated_text"],
            "source_language": translation["detected_source_language"],
            "target_language": target_code,
            "from_cache": False,
        }
        if self.cache_enabled:
            self.translation_cache[cache_key] = result
        return result

    def translate_to_english(self, text: str, source_language: str = "auto") -> dict:
        return self.translate_text(text, source_language, "en")

    def translate_to_regional_language(
        self,
        text: str,
        target_language: str,
    ) -> dict:
        return self.translate_text(text, "en", target_language)

    def batch_translate(self, texts: list[str], target_language: str) -> list[dict]:
        return [self.translate_text(text, target_language=target_language) for text in texts]

    def get_supported_languages(self) -> dict[str, str]:
        return {code: name.title() for name, code in self.language_codes.items()}

    def get_cache_stats(self) -> dict[str, int | bool]:
        return {"entries": len(self.translation_cache), "enabled": self.cache_enabled}

    def clear_cache(self) -> None:
        self.translation_cache.clear()

    def process_multilingual_query(self, query: str) -> dict:
        detected = self.detect_language(query)
        translated = self.translate_to_english(query, detected["language_code"])
        return {"language": detected, "translation": translated}
