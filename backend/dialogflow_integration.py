"""
Dialogflow CX Integration Module for Saarthi Government Services
Connects FastAPI backend with Google Dialogflow CX for intent recognition and dialogue management
Handles intents: Apply for certificate, File complaint, Track application status
"""

import logging
import os
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import asyncio

from google.cloud import dialogflow_cx_v3
from google.api_core.gapic_v1 import client_info as grpc_client_info
from google.protobuf.struct_pb2 import Struct, Value

logger = logging.getLogger(__name__)


class DialogflowCXManager:
    """
    Manages interaction with Google Dialogflow CX
    Handles session management, intent detection, and response fulfillment
    """
    
    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        agent_id: Optional[str] = None,
        credentials_path: Optional[str] = None
    ):
        """
        Initialize Dialogflow CX Manager
        
        Args:
            project_id: Google Cloud project ID
            location: Location of the Dialogflow agent (default: us-central1)
            agent_id: Dialogflow CX agent ID
            credentials_path: Path to service account JSON key file
        """
        self.project_id = project_id
        self.location = location
        self.agent_id = agent_id
        
        # Set up credentials
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        logger.info(f"Initializing Dialogflow CX Manager for project: {project_id}")
        
        try:
            # Initialize Dialogflow clients
            self.sessions_client = dialogflow_cx_v3.SessionsClient()
            self.agent_client = dialogflow_cx_v3.AgentsClient()
            logger.info("✓ Dialogflow CX clients initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Dialogflow CX clients: {str(e)}")
            raise RuntimeError(f"Dialogflow initialization failed: {str(e)}")
        
        # Intent mapping for government services
        self.intent_mapping = {
            "apply_birth_certificate": {
                "service_id": "birth_certificate",
                "service_name": "Birth Certificate",
                "category": "vital_records",
                "handler": self._handle_certificate_application
            },
            "apply_driving_license": {
                "service_id": "driving_license",
                "service_name": "Driving License",
                "category": "transport",
                "handler": self._handle_certificate_application
            },
            "apply_passport": {
                "service_id": "passport",
                "service_name": "Passport",
                "category": "travel",
                "handler": self._handle_certificate_application
            },
            "file_complaint": {
                "service_id": "complaint_registration",
                "service_name": "Complaint Registration",
                "category": "grievance",
                "handler": self._handle_complaint_filing
            },
            "track_application": {
                "service_id": "application_tracking",
                "service_name": "Application Tracking",
                "category": "tracking",
                "handler": self._handle_status_tracking
            },
            "apply_ration_card": {
                "service_id": "ration_card",
                "service_name": "Ration Card",
                "category": "social_welfare",
                "handler": self._handle_certificate_application
            },
            "vehicle_registration": {
                "service_id": "vehicle_registration",
                "service_name": "Vehicle Registration",
                "category": "transport",
                "handler": self._handle_certificate_application
            }
        }
    
    def _build_session_path(self, session_id: str) -> str:
        """Build Dialogflow CX session path"""
        return self.sessions_client.session_path(
            project=self.project_id,
            location=self.location,
            agent=self.agent_id,
            session=session_id
        )
    
    def detect_intent(
        self,
        session_id: str,
        text_input: str,
        language_code: str = "en-US"
    ) -> Dict:
        """
        Send user text to Dialogflow CX and detect intent
        
        Args:
            session_id: Unique session identifier
            text_input: User's natural language input
            language_code: Language code (e.g., 'en-US', 'ta-IN' for Tamil)
            
        Returns:
            Dictionary with detected intent and response
        """
        try:
            logger.info(f"Detecting intent for session: {session_id}, text: {text_input}")
            
            # Build session path
            session_path = self._build_session_path(session_id)
            
            # Create text input
            text_input_obj = dialogflow_cx_v3.TextInput(text=text_input)
            query_input = dialogflow_cx_v3.QueryInput(
                text=text_input_obj,
                language_code=language_code
            )
            
            # Send request to Dialogflow
            response = self.sessions_client.detect_intent(
                request={
                    "session": session_path,
                    "query_input": query_input
                }
            )
            
            # Extract results
            result = response.query_result
            
            detected_intent = {
                "intent_display_name": result.intent.display_name if result.intent else "None",
                "confidence": result.intent_detection_confidence if result.intent else 0.0,
                "text": text_input,
                "language": language_code,
                "fulfillment_text": result.response_messages[0].text.text if result.response_messages else "",
                "session_id": session_id,
                "matched": result.intent is not None and result.intent_detection_confidence > 0.5
            }
            
            logger.info(f"Detected intent: {detected_intent['intent_display_name']} "
                       f"(confidence: {detected_intent['confidence']:.2f})")
            
            return detected_intent
            
        except Exception as e:
            logger.error(f"Error detecting intent: {str(e)}", exc_info=True)
            return {
                "intent_display_name": "None",
                "confidence": 0.0,
                "text": text_input,
                "matched": False,
                "error": str(e)
            }
    
    def send_message_with_context(
        self,
        session_id: str,
        text_input: str,
        context_data: Optional[Dict] = None,
        language_code: str = "en-US"
    ) -> Dict:
        """
        Send message with context parameters to Dialogflow
        
        Args:
            session_id: Session identifier
            text_input: User input
            context_data: Additional context (citizen name, email, etc.)
            language_code: Language code
            
        Returns:
            Full dialogue response with context
        """
        try:
            session_path = self._build_session_path(session_id)
            
            # Create text input
            text_input_obj = dialogflow_cx_v3.TextInput(text=text_input)
            query_input = dialogflow_cx_v3.QueryInput(
                text=text_input_obj,
                language_code=language_code
            )
            
            # Build request with context
            request = {
                "session": session_path,
                "query_input": query_input
            }
            
            # Add context if provided
            if context_data:
                context_params = Struct()
                for key, value in context_data.items():
                    if isinstance(value, str):
                        context_params[key] = Value(string_value=value)
                    elif isinstance(value, (int, float)):
                        context_params[key] = Value(number_value=value)
                    elif isinstance(value, bool):
                        context_params[key] = Value(bool_value=value)
                
                request["query_parameters"] = {"payload": context_params}
            
            # Detect intent
            response = self.sessions_client.detect_intent(request=request)
            result = response.query_result
            
            # Extract webhook data if available
            webhook_data = {}
            if result.webhook_payloads:
                try:
                    webhook_data = json.loads(result.webhook_payloads[0].get_fields().get('data', {}))
                except Exception as e:
                    logger.warning(f"Could not parse webhook data: {str(e)}")
            
            response_data = {
                "session_id": session_id,
                "text": text_input,
                "intent": result.intent.display_name if result.intent else None,
                "confidence": float(result.intent_detection_confidence) if result.intent else 0.0,
                "response": result.response_messages[0].text.text if result.response_messages else "",
                "parameters": dict(result.parameters) if result.parameters else {},
                "webhook_data": webhook_data,
                "matched": result.intent is not None and result.intent_detection_confidence > 0.5
            }
            
            logger.info(f"Message processed. Intent: {response_data['intent']}")
            return response_data
            
        except Exception as e:
            logger.error(f"Error sending message with context: {str(e)}", exc_info=True)
            return {
                "session_id": session_id,
                "text": text_input,
                "matched": False,
                "error": str(e)
            }
    
    async def handle_fulfillment(
        self,
        intent_name: str,
        parameters: Dict,
        session_id: str,
        db = None
    ) -> Dict:
        """
        Handle intent fulfillment based on detected intent
        
        Args:
            intent_name: Dialogflow intent name
            parameters: Extracted parameters from intent
            session_id: Current session ID
            db: Database instance for storing data
            
        Returns:
            Fulfillment result with appropriate action
        """
        try:
            logger.info(f"Handling fulfillment for intent: {intent_name}")
            
            # Map intent to handler
            intent_config = self.intent_mapping.get(intent_name.lower())
            
            if not intent_config:
                logger.warning(f"No handler found for intent: {intent_name}")
                return {
                    "success": False,
                    "message": "I'm not sure how to help with that. Please try rephrasing.",
                    "intent": intent_name
                }
            
            # Call appropriate handler
            handler = intent_config.get("handler")
            if handler:
                result = await handler(parameters, session_id, db)
                return result
            
            return {
                "success": False,
                "message": "Error processing your request",
                "intent": intent_name
            }
            
        except Exception as e:
            logger.error(f"Error in fulfillment handling: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"An error occurred: {str(e)}",
                "intent": intent_name
            }
    
    async def _handle_certificate_application(
        self,
        parameters: Dict,
        session_id: str,
        db = None
    ) -> Dict:
        """Handle certificate/service application intent"""
        try:
            logger.info("Handling certificate application")
            
            # Extract parameters
            service_type = parameters.get("service_type", "document")
            citizen_name = parameters.get("citizen_name", "")
            citizen_email = parameters.get("citizen_email", "")
            citizen_phone = parameters.get("citizen_phone", "")
            
            # Validate required information
            if not citizen_name or not citizen_phone:
                return {
                    "success": False,
                    "message": "I need your name and phone number to proceed. Could you provide those?",
                    "requires_info": ["citizen_name", "citizen_phone"],
                    "intent": "apply_certificate"
                }
            
            # Store application in database
            if db:
                application_data = {
                    "citizen_name": citizen_name,
                    "citizen_email": citizen_email,
                    "citizen_phone": citizen_phone,
                    "service_type": service_type,
                    "session_id": session_id,
                    "created_at": datetime.utcnow().isoformat()
                }
                
                # Add to database (implementation depends on your DB)
                logger.info(f"Application stored: {application_data}")
            
            return {
                "success": True,
                "message": f"Great! I'm processing your application for {service_type}. "
                          f"You'll receive updates at {citizen_phone}. "
                          f"Your reference ID is: REF-{session_id[:8].upper()}",
                "reference_id": f"REF-{session_id[:8].upper()}",
                "service_type": service_type,
                "intent": "apply_certificate"
            }
            
        except Exception as e:
            logger.error(f"Error handling certificate application: {str(e)}")
            return {
                "success": False,
                "message": "Error processing your application",
                "intent": "apply_certificate"
            }
    
    async def _handle_complaint_filing(
        self,
        parameters: Dict,
        session_id: str,
        db = None
    ) -> Dict:
        """Handle complaint filing intent"""
        try:
            logger.info("Handling complaint filing")
            
            # Extract parameters
            citizen_name = parameters.get("citizen_name", "")
            citizen_email = parameters.get("citizen_email", "")
            citizen_phone = parameters.get("citizen_phone", "")
            issue_category = parameters.get("issue_category", "general")
            issue_description = parameters.get("issue_description", "")
            priority = parameters.get("priority", "normal")
            
            # Validate required information
            if not citizen_name or not issue_description:
                return {
                    "success": False,
                    "message": "To file a complaint, I need your name and a description of the issue. Could you provide those?",
                    "requires_info": ["citizen_name", "issue_description"],
                    "intent": "file_complaint"
                }
            
            # Store complaint in database
            complaint_id = f"COMP-{session_id[:8].upper()}"
            
            if db:
                complaint_data = {
                    'citizen_name': citizen_name,
                    'citizen_email': citizen_email,
                    'citizen_phone': citizen_phone,
                    'issue_category': issue_category,
                    'issue_description': issue_description,
                    'priority': priority,
                    'data_hash': hashlib.sha256(
                        f"{citizen_name}_{issue_description}".encode()
                    ).hexdigest()[:16]
                }
                
                success, complaint_id_db = db.add_complaint(complaint_data)
                complaint_id = complaint_id_db if success else complaint_id
            
            return {
                "success": True,
                "complaint_id": complaint_id,
                "message": f"Your complaint has been registered successfully! "
                          f"Complaint ID: {complaint_id}. "
                          f"We'll investigate and contact you at {citizen_phone} with updates.",
                "issue_category": issue_category,
                "priority": priority,
                "intent": "file_complaint"
            }
            
        except Exception as e:
            logger.error(f"Error handling complaint filing: {str(e)}")
            return {
                "success": False,
                "message": "Error filing your complaint",
                "intent": "file_complaint"
            }
    
    async def _handle_status_tracking(
        self,
        parameters: Dict,
        session_id: str,
        db = None
    ) -> Dict:
        """Handle application/complaint status tracking intent"""
        try:
            logger.info("Handling status tracking")
            
            # Extract parameters
            reference_id = parameters.get("reference_id", "")
            citizen_phone = parameters.get("citizen_phone", "")
            
            if not reference_id:
                return {
                    "success": False,
                    "message": "I need your reference ID or complaint ID to track your application. Could you provide it?",
                    "requires_info": ["reference_id"],
                    "intent": "track_application"
                }
            
            # Look up status in database
            status_info = {
                "reference_id": reference_id,
                "status": "under_review",
                "last_updated": datetime.utcnow().isoformat(),
                "estimated_completion": "5-7 days",
                "next_action": "Verification in progress"
            }
            
            if db:
                # Query database for actual status
                if reference_id.startswith("COMP"):
                    complaint = db.get_complaint_by_id(reference_id)
                    if complaint:
                        status_info = {
                            "reference_id": reference_id,
                            "status": complaint.get('status', 'registered'),
                            "last_updated": complaint.get('updated_at', datetime.utcnow().isoformat()),
                            "priority": complaint.get('priority', 'normal')
                        }
                # Add more tracking logic for applications
            
            return {
                "success": True,
                "message": f"Your application status (ID: {reference_id}): "
                          f"{status_info['status']}. "
                          f"Last updated: {status_info['last_updated']}. "
                          f"Estimated completion: {status_info.get('estimated_completion', 'Coming soon')}",
                "status_info": status_info,
                "intent": "track_application"
            }
            
        except Exception as e:
            logger.error(f"Error handling status tracking: {str(e)}")
            return {
                "success": False,
                "message": "Error retrieving your application status",
                "intent": "track_application"
            }


class DialogflowWebhookHandler:
    """
    Handles webhook requests from Dialogflow CX
    Processes fulfillment and sends responses back to Dialogflow
    """
    
    def __init__(self, db = None):
        """Initialize webhook handler"""
        self.db = db
        logger.info("Dialogflow Webhook Handler initialized")
    
    def process_webhook_request(self, request_payload: Dict) -> Dict:
        """
        Process incoming webhook request from Dialogflow
        
        Args:
            request_payload: Webhook request from Dialogflow
            
        Returns:
            Response payload for Dialogflow
        """
        try:
            logger.info(f"Processing webhook request: {json.dumps(request_payload, indent=2)}")
            
            # Extract request details
            fulfillment_info = request_payload.get("fulfillmentInfo", {})
            intent_info = request_payload.get("intentInfo", {})
            parameters = fulfillment_info.get("tag", "")
            session_id = request_payload.get("sessionInfo", {}).get("session", "")
            
            # Get current intent name
            current_intent = intent_info.get("displayName", "")
            
            # Extract parameters
            intent_params = request_payload.get("intentInfo", {}).get("parameters", {})
            
            # Process based on intent
            response_text = self._build_response_text(current_intent, intent_params)
            
            # Build response for Dialogflow
            response = {
                "fulfillmentResponse": {
                    "messages": [
                        {
                            "text": {
                                "text": [response_text]
                            }
                        }
                    ],
                    "outputAudioConfig": {
                        "audioEncoding": "OUTPUT_AUDIO_ENCODING_LINEAR_16"
                    }
                }
            }
            
            logger.info(f"Webhook response: {json.dumps(response, indent=2)}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing webhook: {str(e)}", exc_info=True)
            return {
                "fulfillmentResponse": {
                    "messages": [
                        {
                            "text": {
                                "text": ["I encountered an error processing your request. Please try again."]
                            }
                        }
                    ]
                }
            }
    
    def _build_response_text(self, intent: str, parameters: Dict) -> str:
        """Build response text based on intent and parameters"""
        
        intent_lower = intent.lower()
        
        if "apply" in intent_lower and "certificate" in intent_lower:
            citizen_name = parameters.get("person", {}).get("name", "")
            return f"Thank you {citizen_name}! Your application has been received. We'll process it within 7-15 days."
        
        elif "complaint" in intent_lower:
            issue = parameters.get("issue", "")
            return f"I've registered your complaint about {issue}. A reference ID has been generated for tracking."
        
        elif "track" in intent_lower or "status" in intent_lower:
            ref_id = parameters.get("reference_id", "")
            return f"Tracking your application {ref_id}. Current status: Under Review. We'll update you soon."
        
        else:
            return "I'm here to help with government services. You can apply for certificates, file complaints, or track your applications."


class DialogflowSessionManager:
    """
    Manages user sessions with Dialogflow CX
    Maintains conversation context and state
    """
    
    def __init__(self):
        """Initialize session manager"""
        self.active_sessions = {}
        logger.info("Session Manager initialized")
    
    def create_session(self, user_id: str, language: str = "en-US") -> str:
        """Create new session for user"""
        session_id = f"user_{user_id}_{datetime.utcnow().timestamp()}"
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "language": language,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "message_count": 0,
            "context": {}
        }
        logger.info(f"Session created: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session details"""
        return self.active_sessions.get(session_id)
    
    def update_session_context(self, session_id: str, context: Dict) -> bool:
        """Update session context"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["context"].update(context)
            self.active_sessions[session_id]["last_activity"] = datetime.utcnow().isoformat()
            return True
        return False
    
    def end_session(self, session_id: str) -> bool:
        """End session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Session ended: {session_id}")
            return True
        return False
    
    def cleanup_inactive_sessions(self, timeout_minutes: int = 30):
        """Remove inactive sessions"""
        from datetime import timedelta
        
        now = datetime.utcnow()
        inactive_sessions = []
        
        for session_id, session_data in self.active_sessions.items():
            last_activity = datetime.fromisoformat(session_data["last_activity"])
            if now - last_activity > timedelta(minutes=timeout_minutes):
                inactive_sessions.append(session_id)
        
        for session_id in inactive_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Inactive session removed: {session_id}")
        
        return len(inactive_sessions)


# Demonstration and testing
def demonstrate_dialogflow_integration():
    """Demonstrate Dialogflow CX integration"""
    
    print("\n" + "="*80)
    print("SAARTHI DIALOGFLOW CX INTEGRATION DEMONSTRATION")
    print("="*80 + "\n")
    
    # Note: Requires valid Google Cloud credentials and project setup
    print("This demonstration requires:")
    print("1. Google Cloud project with Dialogflow CX enabled")
    print("2. Service account JSON key file")
    print("3. Trained Dialogflow CX agent with intents")
    print("\nConfiguration:")
    print("- Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
    print("- Update project_id and agent_id below")
    
    # Example usage (when configured):
    # manager = DialogflowCXManager(
    #     project_id="your-project-id",
    #     agent_id="your-agent-id",
    #     credentials_path="/path/to/credentials.json"
    # )
    #
    # result = manager.detect_intent(
    #     session_id="user_123_session",
    #     text_input="I want to apply for a birth certificate"
    # )
    # print(result)
    
    print("\nFor full integration, configure credentials and uncomment the example above.")
    print("="*80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demonstrate_dialogflow_integration()
