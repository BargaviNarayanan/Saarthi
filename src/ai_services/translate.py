"""Google Cloud Translation integration."""

from __future__ import annotations

import os

from google.cloud import translate_v2


class TranslationClient:
    """Translate citizen queries and chatbot responses."""

    def __init__(self, project_id: str | None = None) -> None:
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.client = translate_v2.Client(project=self.project_id)

    def translate(
        self,
        text: str,
        target_language: str,
        source_language: str | None = None,
    ) -> dict:
        result = self.client.translate(
            text,
            target_language=target_language,
            source_language=source_language,
            format_="text",
        )
        return {
            "original_text": text,
            "translated_text": result["translatedText"],
            "detected_source_language": result.get("detectedSourceLanguage"),
            "target_language": target_language,
        }
