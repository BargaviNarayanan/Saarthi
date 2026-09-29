"""
FastAPI Backend for Saarthi Government Services Chatbot
Integrates NLP processor, database, and query mapper
"""

import logging
from typing import Dict, Optional
from datetime import datetime
import hashlib
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import custom modules
from nlp_processor import MultilingualNLPProcessor
from database import SaarthiDatabase
from query_mapper import QueryAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Saarthi Government Services API",
    description="Multilingual chatbot for government services assistance",
    version="1.0.0"
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
try:
    logger.info("Initializing NLP Processor...")
    nlp_processor = MultilingualNLPProcessor(device="cpu")
    logger.info("✓ NLP Processor initialized")
except Exception as e:
    logger.error(f"Failed to initialize NLP Processor: {str(e)}")
    nlp_processor = None

try:
    logger.info("Initializing Database...")
    db = SaarthiDatabase("saarthi.db")
    logger.info("✓ Database initialized")
except Exception as e:
    logger.error(f"Failed to initialize Database: {str(e)}")
    db = None

try:
    logger.info("Initializing Query Analyzer...")
    query_analyzer = QueryAnalyzer()
    logger.info("✓ Query Analyzer initialized")
except Exception as e:
    logger.error(f"Failed to initialize Query Analyzer: {str(e)}")
    query_analyzer = None


# ==========================================
# Pydantic Models for Request/Response
# ==========================================

class ChatMessage(BaseModel):
    """Request model for chat messages"""
    query: str
    language: str = "English"
    user_id: Optional[str] = None


class ServiceResponse(BaseModel):
    """Response model for service information"""
    id: str
    name: str
    description: str
    category: str
    processing_time: str
    cost: str
    eligibility: str
    authority: str
    required_documents: list
    steps: list
    match_confidence: float
    data_sensitivity: str


class ChatResponse(BaseModel):
    """Response model for chat messages"""
    success: bool
    original_query: str
    detected_language: str
    intent: Dict
    matched_services: list
    chatbot_response: str
    confidence: float
    requires_human_review: bool = False


class ComplaintRequest(BaseModel):
    """Request model for complaints"""
    citizen_name: str
    citizen_email: Optional[str] = None
    citizen_phone: Optional[str] = None
    issue_category: str
    issue_description: str
    priority: str = "normal"


class ComplaintResponse(BaseModel):
    """Response model for complaint registration"""
    success: bool
    complaint_id: str
    message: str
    status: str


class FeedbackRequest(BaseModel):
    """Request model for user feedback"""
    query_hash: str
    service_id: str
    rating: int  # 1-5
    feedback_text: Optional[str] = None
    is_helpful: bool


# ==========================================
# Utility Functions
# ==========================================

def generate_query_hash(query: str) -> str:
    """Generate hash of query for caching and tracking"""
    return hashlib.sha256(query.encode()).hexdigest()[:16]


def generate_user_hash(user_id: Optional[str]) -> str:
    """Generate anonymized user hash for privacy"""
    if user_id:
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    return "anonymous"


def format_chatbot_response(nlp_result: Dict, matched_services: list) -> str:
    """Format NLP results into user-friendly chatbot response"""
    
    if not matched_services:
        return (
            f"I understand you're looking for: {nlp_result.get('top_intent', 'unknown')}. "
            f"However, I couldn't find a matching government service. "
            f"Please try describing your need differently or browse our services by category."
        )
    
    service = matched_services[0]  # Top match
    response = (
        f"🔍 I found a matching service!\n\n"
        f"**Service:** {service['name']}\n"
        f"**Category:** {service['category']}\n"
        f"**Authority:** {service['authority']}\n"
        f"**Processing Time:** {service['processing_time']}\n"
        f"**Cost:** {service['cost']}\n\n"
        f"**Description:**\n{service['description']}\n\n"
    )
    
    if len(matched_services) > 1:
        response += f"\n*I also found {len(matched_services) - 1} other relevant service(s). Would you like to know more?*"
    
    if service.get('data_sensitivity') == 'high':
        response += "\n\n⚠️ **Note:** This service involves sensitive personal information. Ensure you're on a secure device."
    
    return response


def create_audit_log(request: Request, query: str, result: Dict, user_hash: str):
    """Create audit log entry for compliance"""
    if not db:
        return
    
    try:
        log_data = {
            'user_hash': user_hash,
            'action_type': 'query_processing',
            'service_accessed': result.get('matched_services', [{}])[0].get('id'),
            'query_hash': generate_query_hash(query),
            'result': 'success' if result.get('success') else 'no_match',
            'ip_address': request.client.host if request.client else 'unknown',
            'user_agent': request.headers.get('user-agent', 'unknown'),
            'data_sensitivity': result.get('matched_services', [{}])[0].get('data_sensitivity', 'low'),
            'details': {
                'confidence': result.get('confidence'),
                'intent': result.get('intent', {}).get('top_intent')
            }
        }
        db.add_audit_log(log_data)
    except Exception as e:
        logger.error(f"Error creating audit log: {str(e)}")


# ==========================================
# API Endpoints
# ==========================================

@app.get("/")
async def root():
    """API information and available endpoints"""
    return {
        "status": "running",
        "application": "Saarthi Government Services Chatbot",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/api/chat",
            "services": "/api/services",
            "services_by_id": "/api/services/{service_id}",
            "services_by_category": "/api/services/category/{category}",
            "search": "/api/search?keyword=term",
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
            "nlp_processor": "ready" if nlp_processor else "unavailable",
            "database": "ready" if db else "unavailable",
            "query_analyzer": "ready" if query_analyzer else "unavailable"
        }
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage, request: Request):
    """
    Main chatbot endpoint - processes user queries and returns matched services
    """
    try:
        logger.info(f"Processing chat query: {message.query}")
        
        # Validate input
        if not message.query or not message.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        user_hash = generate_user_hash(message.user_id)
        query_hash = generate_query_hash(message.query)
        
        # Check cache first
        cached_result = None
        if db:
            cached_result = db.get_cached_query(query_hash)
            if cached_result:
                logger.info(f"Cache hit for query: {message.query}")
        
        # Process with NLP if not cached
        if cached_result is None:
            # Step 1: Multilingual NLP processing
            if nlp_processor and message.language != "English":
                nlp_result = nlp_processor.process_multilingual_query(
                    message.query,
                    message.language
                )
                english_query = nlp_result.get('english_query', message.query)
                detected_language = nlp_result.get('language_detection', {}).get('language', message.language)
                intent_result = nlp_result.get('intent_classification', {})
            else:
                english_query = message.query
                detected_language = message.language
                intent_result = {}
            
            # Step 2: Query mapping to services
            if query_analyzer:
                mapping_result = query_analyzer.map_query_to_service(
                    english_query,
                    message.language,
                    user_hash
                )
            else:
                mapping_result = {"success": False, "matched_services": []}
            
            # Format response
            matched_services = mapping_result.get('matched_services', [])
            confidence = mapping_result.get('matched_services', [{}])[0].get('match_confidence', 0.0) if matched_services else 0.0
            
            nlp_response = format_chatbot_response(intent_result, matched_services)
            
            response_data = {
                "success": mapping_result.get('success', False),
                "original_query": message.query,
                "detected_language": detected_language,
                "intent": intent_result,
                "matched_services": matched_services,
                "chatbot_response": nlp_response,
                "confidence": confidence,
                "requires_human_review": any(
                    s.get('data_sensitivity') == 'high' 
                    for s in matched_services
                )
            }
            
            # Cache the result
            if db:
                cache_data = {
                    'original_query': message.query,
                    'source_language': message.language,
                    'matched_services': [s['id'] for s in matched_services],
                    'intent': intent_result.get('top_intent', 'unknown'),
                    'confidence': confidence
                }
                db.cache_query_result(query_hash, cache_data)
        else:
            # Use cached result
            response_data = json.loads(cached_result['matched_services']) if isinstance(cached_result['matched_services'], str) else cached_result['matched_services']
            response_data['confidence'] = cached_result['confidence']
        
        # Create audit log
        create_audit_log(request, message.query, response_data, user_hash)
        
        logger.info(f"Query processed successfully. Found {len(response_data['matched_services'])} services")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/api/services")
async def get_all_services():
    """Get all available services"""
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        conn = db.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM services ORDER BY category")
        rows = cursor.fetchall()
        conn.close()
        
        services = [dict(row) for row in rows]
        logger.info(f"Retrieved {len(services)} services")
        
        return {
            "success": True,
            "services": services,
            "count": len(services)
        }
    except Exception as e:
        logger.error(f"Error retrieving services: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving services")


@app.get("/api/services/{service_id}")
async def get_service_by_id(service_id: str):
    """Get specific service details"""
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
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        services = db.get_services_by_category(category)
        
        if not services:
            return {
                "success": False,
                "services": [],
                "message": f"No services found in category: {category}"
            }
        
        return {
            "success": True,
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


@app.post("/api/complaints", response_model=ComplaintResponse)
async def register_complaint(complaint: ComplaintRequest, request: Request):
    """Register a public complaint"""
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        logger.info(f"Registering complaint from {complaint.citizen_name}")
        
        # Generate data hash for privacy-preserving logging
        data_hash = hashlib.sha256(
            f"{complaint.citizen_name}_{complaint.issue_description}".encode()
        ).hexdigest()[:16]
        
        complaint_data = {
            'citizen_name': complaint.citizen_name,
            'citizen_email': complaint.citizen_email,
            'citizen_phone': complaint.citizen_phone,
            'issue_category': complaint.issue_category,
            'issue_description': complaint.issue_description,
            'priority': complaint.priority,
            'data_hash': data_hash
        }
        
        success, complaint_id = db.add_complaint(complaint_data)
        
        if success:
            # Log to audit trail
            create_audit_log(
                request,
                f"Complaint: {complaint.issue_category}",
                {"matched_services": []},
                "complaint_system"
            )
            
            return {
                "success": True,
                "complaint_id": complaint_id,
                "message": f"Your complaint has been registered with ID: {complaint_id}. We will review it shortly.",
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


@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback for model improvement"""
    if not db:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        # Validate rating
        if feedback.rating < 1 or feedback.rating > 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
        
        feedback_data = {
            'user_hash': "anonymous",
            'query_hash': feedback.query_hash,
            'service_id': feedback.service_id,
            'rating': feedback.rating,
            'feedback_text': feedback.feedback_text,
            'is_helpful': feedback.is_helpful
        }
        
        success = db.add_user_feedback(feedback_data)
        
        if success:
            return {
                "success": True,
                "message": "Thank you for your feedback! It helps us improve."
            }
        else:
            raise HTTPException(status_code=500, detail="Error submitting feedback")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Error submitting feedback")


@app.get("/api/cache-stats")
async def get_cache_statistics():
    """Get query cache statistics (for monitoring)"""
    if not nlp_processor:
        return {"error": "NLP processor unavailable"}
    
    stats = nlp_processor.get_embedding_cache_stats()
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "cache_stats": stats
    }


@app.post("/api/cache/clear")
async def clear_cache():
    """Clear query cache (admin endpoint)"""
    if not nlp_processor:
        raise HTTPException(status_code=503, detail="NLP processor unavailable")
    
    try:
        nlp_processor.clear_embedding_cache()
        if db:
            db.clear_embedding_cache()
        
        logger.info("Cache cleared")
        return {
            "success": True,
            "message": "Cache cleared successfully"
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail="Error clearing cache")


# ==========================================
# Error Handlers
# ==========================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return {
        "success": False,
        "error": exc.detail,
        "status_code": exc.status_code
    }


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return {
        "success": False,
        "error": "Internal server error",
        "status_code": 500
    }


# ==========================================
# Startup and Shutdown Events
# ==========================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("="*80)
    logger.info("SAARTHI BACKEND STARTING UP")
    logger.info("="*80)
    
    if db:
        # Populate sample data if empty
        conn = db.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM services")
        count = cursor.fetchone()[0]
        conn.close()
        
        if count == 0:
            logger.info("Populating sample data...")
            db.populate_sample_data()
    
    logger.info("✓ Backend ready for requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down...")
    if db:
        db.disconnect()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
