import json

from backend.app.integrations.dialogflow import DialogflowCXManager
from backend.app.integrations.translation import TranslationManager
from backend.app.integrations.vertex_ai import VertexAIServicePredictor
from backend.app.integrations.voice import TextToSpeechManager
from src.ai_services import (
    DialogflowClient,
    SpeechClient,
    TextToSpeechClient,
    TranslateClient,
    VertexAIClient,
)


def test_local_clients_return_json_compatible_results():
    intent = DialogflowClient().detect_intent("session", "I need a birth certificate")
    translation = TranslateClient().translate("Hello", target_language="hi")
    transcription = SpeechClient().transcribe(b"mock audio")
    prediction = VertexAIClient().predict("I need a passport")

    assert intent["intent"] == "apply_birth_certificate"
    assert translation["translated_text"].startswith("translated to hi:")
    assert transcription["transcript"] == "Mock transcription"
    assert prediction["predictions"][0]["category"] == "travel"


def test_local_text_to_speech_returns_bytes():
    audio = TextToSpeechClient().synthesize("Hello", language_code="en-IN")

    assert isinstance(audio, bytes)
    assert audio.startswith(b"mock audio")


def test_backend_mock_results_are_json_compatible():
    results = {
        "intent": DialogflowCXManager().send_message_with_context(
            "session", "track my application"
        ),
        "translation": TranslationManager().translate_text(
            "Hello", source_language="en", target_language="hi"
        ),
        "prediction": VertexAIServicePredictor().predict_service_category(
            "I need a passport"
        ),
        "audio": TextToSpeechManager().synthesize_speech("Hello")[
            "audio_content_base64"
        ],
    }

    assert json.loads(json.dumps(results)) == results
