"""Dialogflow CX integration helpers."""

from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

from google.cloud import dialogflowcx_v3


def send_to_dialogflow(query: str) -> dict:
    """Send a citizen query to Dialogflow CX and return its response.

    Dialogflow configuration is read from environment variables:

    * ``DIALOGFLOW_PROJECT_ID``
    * ``DIALOGFLOW_AGENT_ID``
    * ``DIALOGFLOW_LOCATION`` (optional, defaults to ``global``)
    * ``DIALOGFLOW_LANGUAGE_CODE`` (optional, defaults to ``en``)

    A new session is created for each call. For a multi-turn conversation,
    use :class:`DialogflowCXClient` directly and reuse a session ID.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    query = query.strip()
    project_id = os.getenv("DIALOGFLOW_PROJECT_ID")
    agent_id = os.getenv("DIALOGFLOW_AGENT_ID")
    location = os.getenv("DIALOGFLOW_LOCATION", "global")
    language_code = os.getenv("DIALOGFLOW_LANGUAGE_CODE", "en")

    missing = [
        name
        for name, value in (
            ("DIALOGFLOW_PROJECT_ID", project_id),
            ("DIALOGFLOW_AGENT_ID", agent_id),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing Dialogflow configuration: " + ", ".join(missing)
        )

    client = dialogflowcx_v3.SessionsClient()
    session_id = uuid4().hex
    session = client.session_path(
        project=project_id,
        location=location,
        agent=agent_id,
        session=session_id,
    )

    response = client.detect_intent(
        request={
            "session": session,
            "query_input": {
                "text": {"text": query},
                "language_code": language_code,
            },
        }
    )

    result = response.query_result
    messages = [
        text
        for message in result.response_messages
        if message.text
        for text in message.text.text
    ]

    return {
        "success": True,
        "session_id": session_id,
        "query": query,
        "intent": result.intent.display_name if result.intent else None,
        "confidence": float(result.intent_detection_confidence),
        "response": "\n".join(messages),
        "parameters": dict(result.parameters) if result.parameters else {},
    }


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
