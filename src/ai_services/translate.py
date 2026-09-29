"""Local mock for text translation."""

from __future__ import annotations

import re


class TranslateClient:
    """Return deterministic mock translations without cloud credentials."""

    def translate(
        self,
        text: str,
        target_language: str,
        source_language: str | None = None,
    ) -> dict[str, str]:
        """Return a clearly marked local translation result."""
        detected_language = source_language or self._detect_language(text)
        return {
            "original_text": text,
            "translated_text": f"translated to {target_language}: {text}",
            "detected_source_language": detected_language,
            "target_language": target_language,
        }

    @staticmethod
    def _detect_language(text: str) -> str:
        if re.search(r"[\u0B80-\u0BFF]", text):
            return "ta"
        if re.search(r"[\u0900-\u097F]", text):
            return "hi"
        return "en"


TranslationClient = TranslateClient
