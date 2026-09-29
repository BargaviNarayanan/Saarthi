"""Local mock for Vertex AI predictions."""

from __future__ import annotations

from typing import Any


class VertexAIClient:
    """Provide deterministic service predictions without a cloud connection."""

    _categories = {
        "vital_records": ("birth certificate", "death certificate", "marriage certificate"),
        "transport": ("vehicle", "driving", "license", "registration"),
        "social_welfare": ("ration", "pension", "welfare", "subsidy"),
        "travel": ("passport", "visa", "travel"),
        "grievance": ("complaint", "grievance", "problem", "issue"),
        "tracking": ("track", "status", "application"),
    }

    def predict(self, query: str, language: str = "en") -> dict[str, Any]:
        """Classify a query using local keyword matching."""
        normalized_query = query.casefold()
        category = next(
            (
                category
                for category, keywords in self._categories.items()
                if any(keyword in normalized_query for keyword in keywords)
            ),
            "general",
        )
        return {
            "predictions": [
                {
                    "category": category,
                    "confidence": 0.8 if category != "general" else 0.5,
                }
            ],
            "deployed_model_id": "local-mock",
            "language": language,
        }
