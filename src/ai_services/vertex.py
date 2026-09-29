"""Vertex AI endpoint integration."""

from __future__ import annotations

import os
from typing import Any

from google.cloud import aiplatform


class VertexAIClient:
    """Client for invoking a deployed Vertex AI endpoint."""

    def __init__(
        self,
        project_id: str | None = None,
        location: str | None = None,
        endpoint_id: str | None = None,
    ) -> None:
        self.project_id = project_id or os.environ["GCP_PROJECT_ID"]
        self.location = location or os.getenv("GCP_LOCATION", "us-central1")
        self.endpoint_id = endpoint_id or os.environ["VERTEX_ENDPOINT_ID"]
        aiplatform.init(project=self.project_id, location=self.location)
        self.endpoint = aiplatform.Endpoint(self.endpoint_id)

    def predict(self, query: str, language: str = "en") -> dict[str, Any]:
        """Return the deployed model's prediction for a citizen query."""
        response = self.endpoint.predict(
            instances=[{"text": query, "language": language}]
        )
        return {
            "predictions": list(response.predictions),
            "deployed_model_id": response.deployed_model_id,
        }
