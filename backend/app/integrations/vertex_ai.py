"""Backend prediction adapters backed by the local Vertex AI mock."""

from __future__ import annotations

from src.ai_services.vertex import VertexAIClient


class VertexAIServicePredictor:
    """Preserve backend prediction responses using local keyword classification."""

    _category_services = {
        "vital_records": ["birth_certificate", "death_certificate", "marriage_certificate"],
        "transport": ["vehicle_registration", "driving_license"],
        "social_welfare": ["ration_card", "pension"],
        "travel": ["passport"],
        "grievance": ["complaint_registration"],
        "tracking": ["application_tracking"],
        "general": [],
    }
    _category_names = {
        "vital_records": "Vital Records",
        "transport": "Transport & Vehicles",
        "social_welfare": "Social Welfare",
        "travel": "Travel & Immigration",
        "grievance": "Grievance & Complaints",
        "tracking": "Application Tracking",
        "general": "General Services",
    }

    def __init__(
        self,
        project_id: str | None = None,
        location: str = "us-central1",
        endpoint_id: str | None = None,
        model_id: str | None = None,
        credentials_path: str | None = None,
    ) -> None:
        self.client = VertexAIClient()

    def predict_service_category(
        self,
        query: str,
        language: str = "en",
        confidence_threshold: float = 0.5,
    ) -> dict:
        prediction = self.client.predict(query, language)["predictions"][0]
        category = prediction["category"]
        confidence = prediction["confidence"]
        return {
            "success": confidence >= confidence_threshold,
            "predicted_category": category,
            "category_name": self._category_names[category],
            "confidence": confidence,
            "recommended_services": self._category_services[category],
            "reasoning": "Matched locally using service-related keywords.",
        }

    def batch_predict(self, queries: list[str], language: str = "en") -> list[dict]:
        return [self.predict_service_category(query, language) for query in queries]

    def get_category_details(self, category: str) -> dict:
        return {
            "category": category,
            "name": self._category_names.get(category, "Unknown"),
            "services": self._category_services.get(category, []),
        }

    def get_all_categories(self) -> dict[str, dict]:
        return {
            category: self.get_category_details(category)
            for category in self._category_names
        }

    def evaluate_model_performance(self, test_queries: list[tuple[str, str]]) -> dict:
        correct = sum(
            self.predict_service_category(query)["predicted_category"] == expected
            for query, expected in test_queries
        )
        total = len(test_queries)
        return {"accuracy": correct / total if total else 0.0, "total": total}


class VertexAITextClassifier:
    """Compatibility helper for callers expecting a simple classifier."""

    def __init__(self, *args, **kwargs) -> None:
        self.predictor = VertexAIServicePredictor(*args, **kwargs)

    def classify(self, text: str, language: str = "en") -> dict:
        return self.predictor.predict_service_category(text, language)
