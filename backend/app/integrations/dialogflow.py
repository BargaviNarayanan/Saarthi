"""Backend Dialogflow adapters backed by the local intent mock."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from src.ai_services.dialogflow import DialogflowClient


class DialogflowCXManager:
    """Provide the existing backend interface over the local Dialogflow mock."""

    def __init__(
        self,
        project_id: str | None = None,
        location: str = "us-central1",
        agent_id: str | None = None,
        credentials_path: str | None = None,
    ) -> None:
        self.client = DialogflowClient()

    def detect_intent(
        self,
        session_id: str,
        text_input: str,
        language_code: str = "en-US",
    ) -> dict[str, Any]:
        result = self.client.detect_intent(session_id, text_input, language_code)
        return {
            "intent_display_name": result["intent"],
            "confidence": result["confidence"],
            "text": text_input,
            "language": language_code,
            "fulfillment_text": result["response"],
            "session_id": session_id,
            "matched": result["intent"] != "unknown",
        }

    def send_message_with_context(
        self,
        session_id: str,
        text_input: str,
        language_code: str = "en-US",
    ) -> dict[str, Any]:
        result = self.client.detect_intent(session_id, text_input, language_code)
        return {
            "intent": result["intent"],
            "confidence": result["confidence"],
            "response": result["response"],
            "session_id": session_id,
        }


class DialogflowWebhookHandler:
    """Adapt webhook payloads to local mock intent responses."""

    def __init__(self, db=None) -> None:
        self.db = db
        self.client = DialogflowClient()

    def process_webhook_request(self, request_payload: dict) -> dict:
        query = request_payload.get("queryResult", {}).get("queryText", "")
        session_id = request_payload.get("session", "webhook")
        result = self.client.detect_intent(session_id, query)
        return {
            "fulfillmentText": result["response"],
            "intent": result["intent"],
            "confidence": result["confidence"],
        }


class DialogflowSessionManager:
    """Store lightweight local session state for the backend."""

    def __init__(self) -> None:
        self.sessions: dict[str, dict] = {}

    def create_session(self, user_id: str, language: str = "en-US") -> str:
        session_id = str(uuid4())
        self.sessions[session_id] = {
            "user_id": user_id,
            "language": language,
            "context": {},
            "last_activity": datetime.utcnow(),
        }
        return session_id

    def get_session(self, session_id: str) -> dict | None:
        return self.sessions.get(session_id)

    def update_session_context(self, session_id: str, context: dict) -> bool:
        session = self.sessions.get(session_id)
        if session is None:
            return False
        session["context"].update(context)
        session["last_activity"] = datetime.utcnow()
        return True

    def end_session(self, session_id: str) -> bool:
        return self.sessions.pop(session_id, None) is not None

    def cleanup_inactive_sessions(self, timeout_minutes: int = 30) -> int:
        cutoff = datetime.utcnow() - timedelta(minutes=timeout_minutes)
        inactive = [
            session_id
            for session_id, session in self.sessions.items()
            if session["last_activity"] < cutoff
        ]
        for session_id in inactive:
            del self.sessions[session_id]
        return len(inactive)
