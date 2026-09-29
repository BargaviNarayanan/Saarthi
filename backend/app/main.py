"""
Complete FastAPI Backend for Saarthi Government Services
Integrates all components: Translation, Voice, Vertex AI, Dialogflow CX, NLP
Provides unified REST API endpoints for frontend chatbot
"""

import logging
import os
import json
from typing import Dict, List, Optional
from datetime import datetime
import hashlib
import asyncio

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Import custom modules
from .integrations.translation import TranslationManager
from .integrations.voice import SpeechToTextManager, TextToSpeechManager
from .integrations.vertex_ai import VertexAIServicePredictor
from .integrations.dialogflow import DialogflowCXManager, DialogflowSessionManager
from .database.repository import SaarthiDatabase
from .nlp.processor import MultilingualNLPProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('saarthi_backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Saarthi Government Services API",
    description="Multilingual chatbot for government services using Dialogflow CX, Vertex AI, and Translation API",
    version="2.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Initialize Components
# ==========================================

logger.info("="*80)
logger.info("SAARTHI BACKEND INITIALIZATION")
logger.info("="*80)

# Get configuration from environment
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-project-id")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
VERTEX_ENDPOINT_ID = os.getenv("VERTEX_ENDPOINT_ID", "")
DIALOGFLOW_AGENT_ID = os.getenv("DIALOGFLOW_AGENT_ID", "")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# Initialize components
components = {}

try:
    logger.info("Initializing Translation Manager...")
    components['translator'] = TranslationManager(
        project_id=PROJECT_ID,
        credentials_path=CREDENTIALS_PATH if CREDENTIALS_PATH else None,
        api_version="v3",
        cache_enabled=True
    )
    logger.info("✓ Translation Manager initialized")
except Exception as e:
    logger.error(f"⚠ Translation Manager failed: {str(e)}")
    components['translator'] = None

try:
    logger.info("Initializing Speech-to-Text Manager...")
    components['stt'] = SpeechToTextManager(
        project_id=PROJECT_ID,
        credentials_path=CREDENTIALS_PATH if CREDENTIALS_PATH else None
    )
    logger.info("✓ Speech-to-Text Manager initialized")
except Exception as e:
    logger.error(f"⚠ Speech-to-Text Manager failed: {str(e)}")
    components['stt'] = None

try:
    logger.info("Initializing Text-to-Speech Manager...")
    components['tts'] = TextToSpeechManager(
        project_id=PROJECT_ID,
        credentials_path=CREDENTIALS_PATH if CREDENTIALS_PATH else None
    )
    logger.info("✓ Text-to-Speech Manager initialized")
except Exception as e:
    logger.error(f"⚠ Text-to-Speech Manager failed: {str(e)}")
    components['tts'] = None

try:
    logger.info("Initializing Vertex AI Predictor...")
    if VERTEX_ENDPOINT_ID:
        components['vertex_ai'] = VertexAIServicePredictor(
            project_id=PROJECT_ID,
            location=LOCATION,
            endpoint_id=VERTEX_ENDPOINT_ID,
            credentials_path=CREDENTIALS_PATH if CREDENTIALS_PATH else None
        )
        logger.info("✓ Vertex AI Predictor initialized")
    else:
        logger.info("⚠ Vertex AI Endpoint ID not configured, will use fallback")
        components['vertex_ai'] = None
except Exception as e:
    logger.error(f"⚠ Vertex AI Predictor failed: {str(e)}")
    components['vertex_ai'] = None

try:
    logger.info("Initializing Dialogflow CX Manager...")
    if DIALOGFLOW_AGENT_ID:
        components['dialogflow'] = DialogflowCXManager(
            project_id=PROJECT_ID,
            location=LOCATION,
            agent_id=DIALOGFLOW_AGENT_ID,
            credentials_path=CREDENTIALS_PATH if CREDENTIALS_PATH else None
        )
        logger.info("✓ Dialogflow CX Manager initialized")
    else:
        logger.info("⚠ Dialogflow Agent ID not configured")
        components['dialogflow'] = None
except Exception as e:
    logger.error(f"⚠ Dialogflow CX Manager failed: {str(e)}")
    components['dialogflow'] = None

try:
    logger.info("Initializing Database...")
    components['db'] = SaarthiDatabase("saarthi.db")
    logger.info("✓ Database initialized")
except Exception as e:
    logger.error(f"⚠ Database initialization failed: {str(e)}")
    components['db'] = None

try:
    logger.info("Initializing NLP Processor...")
    components['nlp'] = MultilingualNLPProcessor(device="cpu")
    logger.info("✓ NLP Processor initialized")
except Exception as e:
    logger.error(f"⚠ NLP Processor failed: {str(e)}")
    components['nlp'] = None

# Initialize session manager
components['session_manager'] = DialogflowSessionManager()
logger.info("✓ Session Manager initialized")

logger.info("="*80)
logger.info("INITIALIZATION COMPLETE")
logger.info("="*80 + "\n")


# ==========================================
# Pydantic Models
# ==========================================

class QueryRequest(BaseModel):
    """Request model for text query"""
    query: str = Field(..., description="Citizen's query in any language")
    language: str = Field(default="English", description="Language of query")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    use_voice: bool = Field(default=False, description="Include voice in response")


class TranslateRequest(BaseModel):
    """Request model for translation"""
    text: str = Field(..., description="Text to translate")
    source_language: str = Field(default="auto", description="Source language")
    target_language: str = Field(..., description="Target language")


class VoiceQueryRequest(BaseModel):
    """Request model for voice query"""
    audio_base64: str = Field(..., description="Base64 encoded audio")
    language: str = Field(default="English", description="Language of audio")
    audio_encoding: str = Field(default="LINEAR16", description="Audio encoding")
    sample_rate: int = Field(default=16000, description="Sample rate in Hz")


class PredictRequest(BaseModel):
    """Request model for service category prediction"""
    query: str = Field(..., description="Query for category prediction")
    language: str = Field(default="English", description="Language of query")


class ComplaintRequest(BaseModel):
    """Request model for complaint filing"""
    citizen_name: str
    citizen_email: Optional[str] = None
    citizen_phone: Optional[str] = None
    issue_category: str
    issue_description: str
    priority: str = "normal"
    language: str = "English"


class FeedbackRequest(BaseModel):
    """Request model for user feedback"""
    query_hash: str
    service_id: str
    rating: int  # 1-5
    feedback_text: Optional[str] = None
    is_helpful: bool


# ==========================================
# Response Models
# ==========================================

class ServiceInfo(BaseModel):
    """Service information model"""
    id: str
    name: str
    category: str
    description: str
    processing_time: str
    cost: str


class ChatResponse(BaseModel):
    """Response model for chat queries"""
    success: bool
    query: str
    detected_language: str
    intent: Dict
    predicted_category: Optional[str] = None
    matched_services: List[Dict]
    response_text: str
    confidence: float
    timestamp: str


# ==========================================
# Health & Status Endpoints
# ==========================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "status": "running",
        "application": "Saarthi Government Services API",
        "version": "2.0.0",
        "description": "Multilingual chatbot for government services",
        "components": {
            "translation": "enabled" if components['translator'] else "disabled",
            "speech_to_text": "enabled" if components['stt'] else "disabled",
            "text_to_speech": "enabled" if components['tts'] else "disabled",
            "vertex_ai": "enabled" if components['vertex_ai'] else "disabled",
            "dialogflow_cx": "enabled" if components['dialogflow'] else "disabled",
            "database": "enabled" if components['db'] else "disabled"
        },
        "endpoints": {
            "health": "/health",
            "query": "/api/query",
            "voice_query": "/api/voice/query",
            "translate": "/api/translate",
            "predict": "/api/predict",
            "services": "/api/services",
            "complaints": "/api/complaints",
            "feedback": "/api/feedback"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "translation": "ready" if components['translator'] else "unavailable",
            "stt": "ready" if components['stt'] else "unavailable",
            "tts": "ready" if components['tts'] else "unavailable",
            "vertex_ai": "ready" if components['vertex_ai'] else "unavailable",
            "dialogflow": "ready" if components['dialogflow'] else "unavailable",
            "database": "ready" if components['db'] else "unavailable",
            "nlp": "ready" if components['nlp'] else "unavailable"
        }
    }


# ==========================================
# Main Query Endpoint
# ==========================================

@app.post("/api/query", response_model=ChatResponse)
async def process_query(request: QueryRequest, http_request: Request):
    """
    Main endpoint for processing citizen queries
    Handles: Translation → Intent Detection → Service Matching → Response Generation
    """
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Validate input
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        user_hash = hashlib.sha256(
            (request.user_id or "anonymous").encode()
        ).hexdigest()[:16]
        
        # Step 1: Translate query to English
        translator = components.get('translator')
        if translator and request.language != "English":
            translation_result = translator.translate_to_english(
                request.query,
                request.language
            )
            english_query = translation_result.get("translated_text", request.query)
            detected_language = translation_result.get("source_language", request.language)
        else:
            english_query = request.query
            detected_language = request.language
        
        logger.info(f"Translated query: {english_query}")
        
        # Step 2: Try Dialogflow CX first
        dialogflow = components.get('dialogflow')
        if dialogflow:
            try:
                session_id = f"user_{user_hash}_{datetime.utcnow().timestamp()}"
                
                df_result = dialogflow.send_message_with_context(
                    session_id=session_id,
                    text_input=english_query,
                    language_code="en-US"
                )
                
                logger.info(f"Dialogflow intent: {df_result.get('intent')}")
                
                intent = df_result.get('intent', 'unknown')
                confidence = df_result.get('confidence', 0.0)
                
            except Exception as e:
                logger.warning(f"Dialogflow processing failed: {str(e)}")
                df_result = None
                intent = "unknown"
                confidence = 0.0
        else:
            df_result = None
            intent = "unknown"
            confidence = 0.0
        
        # Step 3: Predict service category with Vertex AI
        vertex_ai = components.get('vertex_ai')
        predicted_category = None
        matched_services = []
        
        if vertex_ai:
            try:
                prediction = vertex_ai.predict_service_category(
                    query=english_query,
                    language="en"
                )
                
                if prediction.get('success'):
                    predicted_category = prediction.get('predicted_category')
                    recommended_services = prediction.get('recommended_services', [])
                    
                    # Get service details from database
                    db = components.get('db')
                    if db:
                        for service_id in recommended_services:
                            service = db.get_service_by_id(service_id)
                            if service:
                                matched_services.append({
                                    "id": service.get('id'),
                                    "name": service.get('name'),
                                    "category": service.get('category'),
                                    "description": service.get('description'),
                                    "processing_time": service.get('processing_time'),
                                    "cost": service.get('cost'),
                                    "match_confidence": prediction.get('confidence')
                                })
                    
                    logger.info(f"Predicted category: {predicted_category}")
                    
            except Exception as e:
                logger.warning(f"Vertex AI prediction failed: {str(e)}")
        
        # Step 4: Fallback to keyword-based search if no matches
        if not matched_services:
            db = components.get('db')
            if db:
                matched_services_from_db = db.search_services_by_keyword(request.query)
                for service in matched_services_from_db[:3]:  # Top 3 results
                    matched_services.append({
                        "id": service.get('id'),
                        "name": service.get('name'),
                        "category": service.get('category'),
                        "description": service.get('description'),
                        "processing_time": service.get('processing_time'),
                        "cost": service.get('cost'),
                        "match_confidence": 0.7
                    })
        
        # Step 5: Generate response text
        if matched_services:
            service = matched_services[0]
            response_text = (
                f"🔍 **{service['name']}**\n\n"
                f"📋 **Category:** {service['category']}\n"
                f"⏱️ **Processing Time:** {service['processing_time']}\n"
                f"💰 **Cost:** {service['cost']}\n\n"
                f"📝 **Description:** {service['description']}"
            )
        else:
            response_text = (
                "I couldn't find a matching government service for your query. "
                "Please try describing your need differently. "
                "You can ask about: Birth Certificate, Vehicle Registration, Driving License, "
                "Ration Card, Passport, or file a complaint."
            )
        
        # Step 6: Translate response to user's language if needed
        translator = components.get('translator')
        if translator and request.language != "English":
            translation_result = translator.translate_to_regional_language(
                response_text,
                request.language
            )
            response_text_localized = translation_result.get("translated_text", response_text)
        else:
            response_text_localized = response_text
        
        # Step 7: Create audit log
        if db := components.get('db'):
            audit_data = {
                'user_hash': user_hash,
                'action_type': 'query_processing',
                'service_accessed': matched_services[0].get('id') if matched_services else None,
                'query_hash': hashlib.sha256(request.query.encode()).hexdigest()[:16],
                'result': 'success' if matched_services else 'no_match',
                'ip_address': http_request.client.host if http_request.client else 'unknown',
                'data_sensitivity': matched_services[0].get('data_sensitivity', 'low') if matched_services else 'low'
            }
            db.add_audit_log(audit_data)
        
        return ChatResponse(
            success=True,
            query=request.query,
            detected_language=detected_language,
            intent={"intent": intent, "confidence": confidence},
            predicted_category=predicted_category,
            matched_services=matched_services,
            response_text=response_text_localized,
            confidence=confidence,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# ==========================================
# Translation Endpoint
# ==========================================

@app.post("/api/translate")
async def translate_text(request: TranslateRequest):
    """Translate text between languages"""
    try:
        translator = components.get('translator')
        if not translator:
            raise HTTPException(status_code=503, detail="Translation service unavailable")
        
        result = translator.translate_text(
            text=request.text,
            source_language=request.source_language,
            target_language=request.target_language
        )
        
        return {
            "success": result.get("success", False),
            "original_text": result.get("original_text"),
            "translated_text": result.get("translated_text"),
            "source_language": result.get("source_language"),
            "target_language": result.get("target_language"),
            "from_cache": result.get("from_cache", False)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Translation failed")


# ==========================================
# Vertex AI Prediction Endpoint
# ==========================================

@app.post("/api/predict")
async def predict_service_category(request: PredictRequest):
    """Predict service category using Vertex AI"""
    try:
        vertex_ai = components.get('vertex_ai')
        if not vertex_ai:
            raise HTTPException(status_code=503, detail="Prediction service unavailable")
        
        # Translate query to English first
        translator = components.get('translator')
        if translator and request.language != "English":
            trans_result = translator.translate_to_english(request.query, request.language)
            english_query = trans_result.get("translated_text", request.query)
        else:
            english_query = request.query
        
        # Get prediction
        prediction = vertex_ai.predict_service_category(
            query=english_query,
            language="en"
        )
        
        # Get service details
        matched_services = []
        db = components.get('db')
        if db and prediction.get('recommended_services'):
            for service_id in prediction['recommended_services'][:3]:
                service = db.get_service_by_id(service_id)
                if service:
                    matched_services.append({
                        "id": service.get('id'),
                        "name": service.get('name'),
                        "category": service.get('category')
                    })
        
        return {
            "success": prediction.get("success"),
            "predicted_category": prediction.get("predicted_category"),
            "category_name": prediction.get("category_name"),
            "confidence": prediction.get("confidence"),
            "matched_services": matched_services,
            "reasoning": prediction.get("reasoning")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")


# ==========================================
# Voice Query Endpoint
# ==========================================

@app.post("/api/voice/query")
async def process_voice_query(
    audio_file: UploadFile = File(...),
    language: str = Form("English")
):
    """Process voice query: STT → Processing → TTS"""
    try:
        logger.info(f"Processing voice query in {language}")
        
        stt = components.get('stt')
        if not stt:
            raise HTTPException(status_code=503, detail="Speech-to-Text unavailable")
        
        # Save uploaded audio
        audio_path = f"/tmp/voice_query_{datetime.utcnow().timestamp()}.wav"
        with open(audio_path, "wb") as f:
            f.write(await audio_file.read())
        
        # Step 1: Transcribe audio
        transcription = stt.transcribe_audio_file(audio_path, language)
        
        if not transcription.get("success"):
            raise HTTPException(status_code=400, detail="Could not transcribe audio")
        
        citizen_query = transcription.get("transcription")
        logger.info(f"Transcribed: {citizen_query}")
        
        # Step 2: Process query (reuse main query processing)
        query_request = QueryRequest(
            query=citizen_query,
            language=language,
            use_voice=True
        )
        
        response = await process_query(query_request, http_request=None)
        
        # Step 3: Convert response to speech
        tts = components.get('tts')
        if tts:
            tts_result = tts.synthesize_speech(
                text=response.response_text,
                language=language,
                gender="FEMALE",
                speaking_rate=0.9
            )
            
            audio_base64 = tts_result.get("audio_content_base64")
        else:
            audio_base64 = None
        
        return {
            "success": True,
            "transcription": citizen_query,
            "response": response.dict(),
            "audio_response_base64": audio_base64,
            "has_audio": audio_base64 is not None
        }
        
    except Exception as e:
        logger.error(f"Voice query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")


# ==========================================
# Services Endpoints
# ==========================================

@app.get("/api/services")
async def get_all_services():
    """Get all available services"""
    db = components.get('db')
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        conn = db.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM services ORDER BY category")
        rows = cursor.fetchall()
        conn.close()
        
        services = [dict(row) for row in rows]
        
        return {
            "success": True,
            "services": services,
            "count": len(services)
        }
    except Exception as e:
        logger.error(f"Error retrieving services: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving services")


@app.get("/api/services/{service_id}")
async def get_service(service_id: str):
    """Get specific service details"""
    db = components.get('db')
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        service = db.get_service_by_id(service_id)
        
        if not service:
            raise HTTPException(status_code=404, detail="Service not found")
        
        return {
            "success": True,
            "service": dict(service)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving service: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving service")


@app.get("/api/services/category/{category}")
async def get_services_by_category(category: str):
    """Get services by category"""
    db = components.get('db')
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        services = db.get_services_by_category(category)
        
        return {
            "success": True if services else False,
            "services": services,
            "category": category,
            "count": len(services)
        }
    except Exception as e:
        logger.error(f"Error retrieving services by category: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving services")


@app.get("/api/search")
async def search_services(keyword: str):
    """Search services by keyword"""
    db = components.get('db')
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    if not keyword or len(keyword.strip()) == 0:
        raise HTTPException(status_code=400, detail="Keyword cannot be empty")
    
    try:
        services = db.search_services_by_keyword(keyword)
        
        return {
            "success": True,
            "services": services,
            "keyword": keyword,
            "count": len(services)
        }
    except Exception as e:
        logger.error(f"Error searching services: {str(e)}")
        raise HTTPException(status_code=500, detail="Error searching services")


# ==========================================
# Complaints Endpoint
# ==========================================

@app.post("/api/complaints")
async def register_complaint(request: ComplaintRequest):
    """Register a public complaint"""
    db = components.get('db')
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        logger.info(f"Registering complaint from {request.citizen_name}")
        
        data_hash = hashlib.sha256(
            f"{request.citizen_name}_{request.issue_description}".encode()
        ).hexdigest()[:16]
        
        complaint_data = {
            'citizen_name': request.citizen_name,
            'citizen_email': request.citizen_email,
            'citizen_phone': request.citizen_phone,
            'issue_category': request.issue_category,
            'issue_description': request.issue_description,
            'priority': request.priority,
            'data_hash': data_hash
        }
        
        success, complaint_id = db.add_complaint(complaint_data)
        
        if success:
            # Translate acknowledgement if needed
            translator = components.get('translator')
            if translator and request.language != "English":
                acknowledgement = "Your complaint has been successfully registered."
                trans_result = translator.translate_to_regional_language(
                    acknowledgement,
                    request.language
                )
                message = trans_result.get("translated_text", acknowledgement)
            else:
                message = "Your complaint has been successfully registered."
            
            return {
                "success": True,
                "complaint_id": complaint_id,
                "message": message,
                "status": "registered"
            }
        else:
            raise HTTPException(status_code=500, detail="Error registering complaint")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering complaint: {str(e)}")
        raise HTTPException(status_code=500, detail="Error registering complaint")


@app.get("/api/complaints/{complaint_id}")
async def get_complaint_status(complaint_id: str):
    """Get complaint status"""
    db = components.get('db')
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        complaint = db.get_complaint_by_id(complaint_id)
        
        if not complaint:
            raise HTTPException(status_code=404, detail="Complaint not found")
        
        return {
            "success": True,
            "complaint_id": complaint_id,
            "status": complaint['status'],
            "created_at": complaint['created_at'],
            "updated_at": complaint['updated_at']
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving complaint: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving complaint")


# ==========================================
# Feedback Endpoint
# ==========================================

@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback"""
    db = components.get('db')
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        if request.rating < 1 or request.rating > 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
        
        feedback_data = {
            'user_hash': "anonymous",
            'query_hash': request.query_hash,
            'service_id': request.service_id,
            'rating': request.rating,
            'feedback_text': request.feedback_text,
            'is_helpful': request.is_helpful
        }
        
        success = db.add_user_feedback(feedback_data)
        
        if success:
            return {
                "success": True,
                "message": "Thank you for your feedback!"
            }
        else:
            raise HTTPException(status_code=500, detail="Error submitting feedback")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Error submitting feedback")


# ==========================================
# Utility Endpoints
# ==========================================

@app.get("/api/languages")
async def get_supported_languages():
    """Get all supported languages"""
    translator = components.get('translator')
    
    languages = {}
    
    if translator:
        languages = translator.get_supported_languages()
    
    return {
        "success": True,
        "supported_languages": languages,
        "count": len(languages)
    }


@app.get("/api/categories")
async def get_service_categories():
    """Get all service categories"""
    db = components.get('db')
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        conn = db.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT category FROM services ORDER BY category")
        rows = cursor.fetchall()
        conn.close()
        
        categories = [row[0] for row in rows]
        
        return {
            "success": True,
            "categories": categories,
            "count": len(categories)
        }
    except Exception as e:
        logger.error(f"Error retrieving categories: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving categories")


@app.get("/api/cache/stats")
async def get_cache_statistics():
    """Get translation cache statistics"""
    translator = components.get('translator')
    
    if translator:
        stats = translator.get_cache_stats()
    else:
        stats = {}
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "cache_stats": stats
    }


@app.post("/api/cache/clear")
async def clear_cache():
    """Clear translation cache"""
    translator = components.get('translator')
    
    if translator:
        translator.clear_cache()
        return {"success": True, "message": "Cache cleared"}
    else:
        raise HTTPException(status_code=503, detail="Translation service unavailable")


# ==========================================
# Error Handlers
# ==========================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ==========================================
# Startup and Shutdown
# ==========================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Backend startup complete. Ready to accept requests.")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Backend shutting down...")
    
    if db := components.get('db'):
        db.disconnect()


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
