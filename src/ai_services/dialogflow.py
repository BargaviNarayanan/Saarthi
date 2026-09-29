"""Dialogflow CX integration helpers."""

from __future__ import annotations

import os
from typing import Any

from google.cloud import dialogflowcx_v3


def send_to_dialogflow(query: str) -> dict:
    """Send a citizen query to Dialogflow.

    This placeholder intentionally has no implementation yet.
    """
    pass


class DialogflowCXClient:
    """Small wrapper for sending text queries to a Dialogflow CX agent."""

    def __init__(
        self,
        project_id: str | None = None,
        location: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        self.project_id = project_id or os.environ["DIALOGFLOW_PROJECT_ID"]
        self.location = location or os.getenv("DIALOGFLOW_LOCATION", "global")
        self.agent_id = agent_id or os.environ["DIALOGFLOW_AGENT_ID"]
        self.client = dialogflowcx_v3.SessionsClient()

    def detect_intent(
        self,
        session_id: str,
        text: str,
        language_code: str = "en",
    ) -> dict[str, Any]:
        session = self.client.session_path(
            project=self.project_id,
            location=self.location,
            agent=self.agent_id,
            session=session_id,
        )
        response = self.client.detect_intent(
            request={
                "session": session,
                "query_input": {
                    "text": {"text": text},
                    "language_code": language_code,
                },
            }
        )
        result = response.query_result
        messages = [
            message.text.text[0]
            for message in result.response_messages
            if message.text and message.text.text
        ]
        return {
            "intent": result.intent.display_name if result.intent else None,
            "confidence": result.intent_detection_confidence,
            "response": "\n".join(messages),
        }
