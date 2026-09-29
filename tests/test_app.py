import sys
from types import ModuleType
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest


class StubManager:
    def __init__(self, *args, **kwargs):
        pass


def _integration_stub(module_name, **classes):
    module = ModuleType(module_name)
    for name, implementation in classes.items():
        setattr(module, name, implementation)
    return module


_integration_stubs = {
    "backend.app.integrations.translation": _integration_stub(
        "backend.app.integrations.translation",
        TranslationManager=StubManager,
    ),
    "backend.app.integrations.voice": _integration_stub(
        "backend.app.integrations.voice",
        SpeechToTextManager=StubManager,
        TextToSpeechManager=StubManager,
    ),
    "backend.app.integrations.vertex_ai": _integration_stub(
        "backend.app.integrations.vertex_ai",
        VertexAIServicePredictor=StubManager,
    ),
    "backend.app.integrations.dialogflow": _integration_stub(
        "backend.app.integrations.dialogflow",
        DialogflowCXManager=StubManager,
        DialogflowSessionManager=StubManager,
    ),
    "backend.app.database.repository": _integration_stub(
        "backend.app.database.repository",
        SaarthiDatabase=StubManager,
    ),
    "backend.app.nlp.processor": _integration_stub(
        "backend.app.nlp.processor",
        MultilingualNLPProcessor=StubManager,
    ),
}

with patch.dict(sys.modules, _integration_stubs):
    from backend.app import main


class FakeTranslator:
    def translate_text(self, *, text, source_language, target_language):
        return {
            "success": True,
            "original_text": text,
            "translated_text": f"translated {text}",
            "source_language": source_language,
            "target_language": target_language,
            "from_cache": False,
        }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setitem(main.components, "translator", FakeTranslator())
    monkeypatch.setitem(main.components, "stt", None)
    monkeypatch.setitem(main.components, "tts", None)
    monkeypatch.setitem(main.components, "vertex_ai", None)
    monkeypatch.setitem(main.components, "dialogflow", None)
    monkeypatch.setitem(main.components, "db", None)
    with TestClient(main.app) as test_client:
        yield test_client


def test_root_endpoint_returns_success(client):
    response = client.get("/")

    assert response.status_code == 200


def test_query_endpoint_returns_intent_and_response(client):
    response = client.post(
        "/api/query",
        json={"query": "How do I apply for a birth certificate?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "intent" in body
    assert "response_text" in body


def test_translate_endpoint_returns_translated_text(client):
    response = client.post(
        "/api/translate",
        json={
            "text": "Hello",
            "source_language": "en",
            "target_language": "hi",
        },
    )

    assert response.status_code == 200
    assert "translated" in response.json()["translated_text"]
