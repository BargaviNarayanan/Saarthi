from fastapi import APIRouter
from src.ai_services import (
    DialogflowClient,
    SpeechClient,
    TextToSpeechClient,
    TranslateClient,
    VertexAIClient,
)

router = APIRouter()

@router.post("/predict")
def predict(query: str):
    client = VertexAIClient()
    return client.predict(query)

@router.post("/translate")
def translate(text: str, target_language: str = "fr"):
    client = TranslateClient()
    return client.translate(text, target_language)

@router.post("/dialogflow")
def dialogflow(text: str):
    client = DialogflowClient()
    return client.detect_intent(text)

@router.post("/speech")
def speech(audio: bytes):
    client = SpeechClient()
    return client.transcribe(audio)

@router.post("/tts")
def tts(text: str, language: str = "en"):
    client = TextToSpeechClient()
    return client.synthesize(text, language)
