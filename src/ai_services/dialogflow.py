"""Local mock for Dialogflow intent detection."""

from __future__ import annotations

from typing import Any


class DialogflowClient:
    """Detect common government-service intents with local keyword matching."""

    _intents = {
        "apply_birth_certificate": ("birth certificate", "birth", "born"),
        "apply_driving_license": ("driving license", "driver license", "driving licence"),
        "apply_passport": ("passport", "visa"),
        "file_complaint": ("complaint", "grievance", "report an issue"),
        "track_application": ("track", "application status", "status of"),
    }

    def detect_intent(
        self,
        session_id: str,
        text: str,
        language_code: str = "en-US",
    ) -> dict[str, Any]:
        """Return a local intent result with the same shape expected by the API."""
        normalized_text = text.casefold()
        intent = next(
            (
                name
                for name, keywords in self._intents.items()
                if any(keyword in normalized_text for keyword in keywords)
            ),
            "unknown",
        )
        confidence = 0.85 if intent != "unknown" else 0.0
        responses = {
            "apply_birth_certificate": "I can help you apply for a birth certificate.",
            "apply_driving_license": "I can help you with a driving license.",
            "apply_passport": "I can help you with passport and travel services.",
            "file_complaint": "I can help you register a complaint.",
            "track_application": "I can help you track your application.",
            "unknown": "How can I help with a government service?",
        }
        return {
            "intent": intent,
            "confidence": confidence,
            "response": responses[intent],
            "session_id": session_id,
            "language_code": language_code,
        }


DialogflowCXClient = DialogflowClient


def send_to_dialogflow(query: str) -> dict[str, str]:
    """Return a local Dialogflow-style response."""
    result = DialogflowClient().detect_intent("default", query)
    return {"intent": result["intent"], "response": result["response"]}
