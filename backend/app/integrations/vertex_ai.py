"""
Vertex AI Integration Module for Saarthi Government Services
Uses Google Cloud Vertex AI for intelligent service category prediction
Handles model inference, caching, and fallback strategies
"""

import logging
import os
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib
from functools import lru_cache

from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.v1.types import (
    predict_service,
    content,
)
from google.protobuf.json_format import MessageToDict

logger = logging.getLogger(__name__)


class VertexAIServicePredictor:
    """
    Vertex AI model for predicting government service categories
    Handles model inference, response parsing, and fallback strategies
    """
    
    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        endpoint_id: Optional[str] = None,
        model_id: Optional[str] = None,
        credentials_path: Optional[str] = None
    ):
        """
        Initialize Vertex AI Service Predictor
        
        Args:
            project_id: Google Cloud project ID
            location: GCP region (default: us-central1)
            endpoint_id: Deployed endpoint ID for the model
            model_id: Model ID if using direct model inference
            credentials_path: Path to service account JSON key
        """
        self.project_id = project_id
        self.location = location
        self.endpoint_id = endpoint_id
        self.model_id = model_id
        
        # Set credentials if provided
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        logger.info(f"Initializing Vertex AI Predictor for project: {project_id}")
        
        try:
            # Initialize Vertex AI
            aiplatform.init(project=project_id, location=location)
            logger.info("✓ Vertex AI initialized successfully")
            
            # Initialize prediction client
            self.prediction_client = aiplatform.gapic.PredictionServiceClient(
                client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
            )
            logger.info("✓ Prediction client initialized")
            
            # Get endpoint if endpoint_id provided
            if endpoint_id:
                self.endpoint = aiplatform.Endpoint(endpoint_id)
                logger.info(f"✓ Endpoint connected: {endpoint_id}")
            else:
                self.endpoint = None
                
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {str(e)}")
            raise RuntimeError(f"Vertex AI initialization failed: {str(e)}")
        
        # Define service categories and keywords for fallback
        self.service_categories = {
            "vital_records": {
                "name": "Vital Records",
                "services": ["birth_certificate", "death_certificate", "marriage_certificate"],
                "keywords": ["birth", "certificate", "born", "registration", "vital", "death", "marriage"]
            },
            "transport": {
                "name": "Transport & Vehicles",
                "services": ["vehicle_registration", "driving_license", "commercial_license"],
                "keywords": ["vehicle", "car", "bike", "driving", "license", "registration", "motor", "automobile", "transport"]
            },
            "social_welfare": {
                "name": "Social Welfare",
                "services": ["ration_card", "pension", "unemployment_benefit"],
                "keywords": ["ration", "card", "food", "subsidy", "grain", "welfare", "benefit", "pension"]
            },
            "travel": {
                "name": "Travel & Immigration",
                "services": ["passport", "visa", "travel_certificate"],
                "keywords": ["passport", "travel", "international", "visa", "foreign", "immigration"]
            },
            "grievance": {
                "name": "Grievance & Complaints",
                "services": ["complaint_registration", "issue_reporting"],
                "keywords": ["complaint", "issue", "problem", "grievance", "report", "concern", "water", "electricity", "road"]
            },
            "property": {
                "name": "Property & Land",
                "services": ["land_registration", "property_tax", "mutation"],
                "keywords": ["property", "land", "registration", "tax", "deed", "mutation", "ownership"]
            },
            "education": {
                "name": "Education & Scholarships",
                "services": ["admission", "scholarship", "certificate_verification"],
                "keywords": ["education", "school", "college", "scholarship", "admission", "certificate", "course"]
            },
            "healthcare": {
                "name": "Healthcare & Licenses",
                "services": ["health_certificate", "medical_license"],
                "keywords": ["health", "medical", "license", "certificate", "doctor", "hospital", "disease"]
            }
        }
        
        # Cache for model predictions
        self.prediction_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        logger.info("✓ Service categories and cache initialized")
    
    def predict_service_category(
        self,
        query: str,
        language: str = "en",
        confidence_threshold: float = 0.5
    ) -> Dict:
        """
        Predict service category for a citizen query using Vertex AI model
        
        Args:
            query: Citizen's natural language query
            language: Language code (e.g., 'en', 'ta' for Tamil)
            confidence_threshold: Minimum confidence for prediction acceptance
            
        Returns:
            Dictionary with predicted category, confidence, and service recommendations
        """
        try:
            logger.info(f"Predicting service category for query: {query}")
            
            # Check cache first
            cache_key = self._generate_cache_key(query)
            cached_result = self._get_cached_prediction(cache_key)
            if cached_result:
                logger.info(f"Cache hit for query: {query}")
                return cached_result
            
            # Prepare input for the model
            input_text = self._preprocess_query(query, language)
            
            # Call Vertex AI model for inference
            prediction = self._call_vertex_ai_model(input_text, language)
            
            if prediction.get("success"):
                # Parse model response
                result = self._parse_model_response(
                    prediction,
                    query,
                    confidence_threshold
                )
            else:
                # Fallback to keyword-based classification
                logger.warning("Vertex AI inference failed, using fallback strategy")
                result = self._fallback_classification(query, language)
            
            # Cache the result
            self._cache_prediction(cache_key, result)
            
            logger.info(f"Prediction complete. Category: {result.get('predicted_category')}, "
                       f"Confidence: {result.get('confidence'):.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting service category: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "predicted_category": "unknown",
                "confidence": 0.0,
                "message": "Unable to predict service category"
            }
    
    def _call_vertex_ai_model(self, input_text: str, language: str) -> Dict:
        """
        Call the deployed Vertex AI model for inference
        
        Args:
            input_text: Preprocessed query text
            language: Language code
            
        Returns:
            Model prediction response
        """
        try:
            if not self.endpoint:
                raise ValueError("No endpoint configured for model inference")
            
            logger.info("Calling Vertex AI endpoint...")
            
            # Prepare request for text-based model
            instances = [
                {
                    "content": input_text,
                    "mimeType": "text/plain"
                }
            ]
            
            parameters = {
                "temperature": 0.1,  # Low temperature for more deterministic results
                "maxOutputTokens": 256,
                "topP": 0.8,
                "topK": 40
            }
            
            # Make prediction using endpoint
            response = self.endpoint.predict(
                instances=instances,
                parameters=parameters
            )
            
            logger.info(f"Vertex AI response received")
            
            # Parse response
            predictions = response.predictions
            
            if predictions:
                prediction_text = predictions[0] if isinstance(predictions[0], str) else str(predictions[0])
                
                return {
                    "success": True,
                    "raw_response": prediction_text,
                    "predictions": predictions
                }
            else:
                return {
                    "success": False,
                    "error": "No predictions returned"
                }
                
        except Exception as e:
            logger.error(f"Error calling Vertex AI model: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _parse_model_response(
        self,
        prediction: Dict,
        original_query: str,
        confidence_threshold: float
    ) -> Dict:
        """
        Parse Vertex AI model response and extract category prediction
        
        Args:
            prediction: Raw model response
            original_query: Original user query
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Structured prediction result
        """
        try:
            raw_response = prediction.get("raw_response", "")
            
            # Try to parse JSON response from model
            try:
                response_json = json.loads(raw_response)
                predicted_category = response_json.get("category", "unknown")
                confidence = float(response_json.get("confidence", 0.0))
                reasoning = response_json.get("reasoning", "")
            except (json.JSONDecodeError, ValueError):
                # Parse text-based response
                predicted_category, confidence = self._extract_category_from_text(raw_response)
                reasoning = raw_response
            
            # Validate confidence threshold
            if confidence < confidence_threshold:
                logger.info(f"Confidence below threshold ({confidence:.2f} < {confidence_threshold}). Using fallback.")
                return self._fallback_classification(original_query, "en")
            
            # Get service details for predicted category
            category_details = self.service_categories.get(predicted_category)
            
            if not category_details:
                logger.warning(f"Unknown category predicted: {predicted_category}")
                return self._fallback_classification(original_query, "en")
            
            return {
                "success": True,
                "predicted_category": predicted_category,
                "category_name": category_details.get("name"),
                "confidence": confidence,
                "recommended_services": category_details.get("services", []),
                "reasoning": reasoning,
                "model_used": "vertex_ai_llm",
                "query": original_query
            }
            
        except Exception as e:
            logger.error(f"Error parsing model response: {str(e)}")
            return self._fallback_classification(original_query, "en")
    
    def _extract_category_from_text(self, text: str) -> Tuple[str, float]:
        """
        Extract category and confidence from text response
        
        Args:
            text: Model response text
            
        Returns:
            Tuple of (category, confidence_score)
        """
        text_lower = text.lower()
        
        # Map keywords to categories
        category_keywords = {
            "vital_records": ["vital", "birth", "death", "marriage", "certificate"],
            "transport": ["vehicle", "driving", "license", "car", "bike", "motor"],
            "social_welfare": ["ration", "welfare", "pension", "benefit", "subsidy"],
            "travel": ["passport", "travel", "visa", "international"],
            "grievance": ["complaint", "grievance", "issue", "problem", "report"],
            "property": ["property", "land", "registration", "deed"],
            "education": ["education", "school", "scholarship", "admission"],
            "healthcare": ["health", "medical", "doctor", "disease"]
        }
        
        # Find matching category
        best_category = "unknown"
        best_score = 0.0
        
        for category, keywords in category_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower) / len(keywords)
            if score > best_score:
                best_score = score
                best_category = category
        
        # Convert score to confidence (0-1)
        confidence = max(0.0, min(1.0, best_score))
        
        return best_category, confidence
    
    def _fallback_classification(self, query: str, language: str) -> Dict:
        """
        Fallback keyword-based classification when model fails
        
        Args:
            query: User query
            language: Language code
            
        Returns:
            Classification result
        """
        logger.info(f"Using fallback classification for query: {query}")
        
        query_lower = query.lower()
        scores = {}
        
        # Score each category based on keyword matches
        for category, details in self.service_categories.items():
            keywords = details.get("keywords", [])
            matches = sum(1 for kw in keywords if kw in query_lower)
            score = matches / len(keywords) if keywords else 0.0
            scores[category] = score
        
        # Find best match
        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]
        
        # Ensure minimum confidence
        if best_score == 0.0:
            best_category = "grievance"  # Default to grievance for unknown queries
            best_score = 0.3
        
        category_details = self.service_categories.get(best_category, {})
        
        return {
            "success": True,
            "predicted_category": best_category,
            "category_name": category_details.get("name"),
            "confidence": best_score,
            "recommended_services": category_details.get("services", []),
            "reasoning": "Keyword-based fallback classification",
            "model_used": "keyword_fallback",
            "query": query
        }
    
    def _preprocess_query(self, query: str, language: str) -> str:
        """
        Preprocess query for model input
        
        Args:
            query: Raw query text
            language: Language code
            
        Returns:
            Preprocessed query
        """
        # Remove special characters
        import re
        preprocessed = re.sub(r'[^\w\s]', '', query)
        
        # Add context about being a government service query
        if language in ["ta", "ta-IN"]:
            context = "Government service query in Tamil: "
        elif language in ["hi", "hi-IN"]:
            context = "Government service query in Hindi: "
        else:
            context = "Government service query in English: "
        
        return context + preprocessed
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        return hashlib.sha256(query.encode()).hexdigest()[:16]
    
    def _get_cached_prediction(self, cache_key: str) -> Optional[Dict]:
        """Get prediction from cache if available"""
        if cache_key in self.prediction_cache:
            cached_data = self.prediction_cache[cache_key]
            if (datetime.utcnow() - cached_data['timestamp']).total_seconds() < self.cache_ttl:
                return cached_data['result']
            else:
                del self.prediction_cache[cache_key]
        return None
    
    def _cache_prediction(self, cache_key: str, result: Dict):
        """Cache prediction result"""
        self.prediction_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.utcnow()
        }
    
    def batch_predict(
        self,
        queries: List[str],
        language: str = "en"
    ) -> List[Dict]:
        """
        Predict service categories for multiple queries
        
        Args:
            queries: List of citizen queries
            language: Language code
            
        Returns:
            List of prediction results
        """
        logger.info(f"Batch predicting for {len(queries)} queries")
        
        results = []
        for query in queries:
            result = self.predict_service_category(query, language)
            results.append(result)
        
        return results
    
    def get_category_details(self, category: str) -> Dict:
        """Get detailed information about a service category"""
        return self.service_categories.get(category, {})
    
    def get_all_categories(self) -> Dict:
        """Get all available service categories"""
        return self.service_categories
    
    def evaluate_model_performance(self, test_queries: List[Tuple[str, str]]) -> Dict:
        """
        Evaluate model performance on test queries
        
        Args:
            test_queries: List of (query, expected_category) tuples
            
        Returns:
            Performance metrics
        """
        logger.info(f"Evaluating model performance on {len(test_queries)} test queries")
        
        correct = 0
        total = len(test_queries)
        
        predictions = []
        
        for query, expected_category in test_queries:
            result = self.predict_service_category(query)
            predicted_category = result.get("predicted_category")
            
            predictions.append({
                "query": query,
                "expected": expected_category,
                "predicted": predicted_category,
                "correct": predicted_category == expected_category,
                "confidence": result.get("confidence")
            })
            
            if predicted_category == expected_category:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions) if predictions else 0.0
        
        return {
            "accuracy": accuracy,
            "total_queries": total,
            "correct_predictions": correct,
            "average_confidence": avg_confidence,
            "predictions": predictions
        }


class VertexAITextClassifier:
    """
    Specialized Vertex AI classifier for text-to-category mapping
    Uses pre-trained or custom classification models
    """
    
    def __init__(
        self,
        project_id: str,
        model_id: str,
        location: str = "us-central1"
    ):
        """Initialize text classifier"""
        self.project_id = project_id
        self.model_id = model_id
        self.location = location
        
        try:
            aiplatform.init(project=project_id, location=location)
            self.model = aiplatform.TextClassificationModel(self.model_id)
            logger.info(f"✓ Text classification model loaded: {model_id}")
        except Exception as e:
            logger.error(f"Failed to load text classification model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")
    
    def classify(
        self,
        texts: List[str],
        confidence_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Classify texts using Vertex AI text classification model
        
        Args:
            texts: List of texts to classify
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of classification results
        """
        try:
            logger.info(f"Classifying {len(texts)} texts")
            
            results = self.model.predict(texts)
            
            processed_results = []
            for i, result in enumerate(results):
                prediction = {
                    "text": texts[i] if i < len(texts) else "",
                    "categories": [],
                    "top_category": None
                }
                
                # Extract categories and confidences
                if hasattr(result, 'predictions'):
                    predictions = result.predictions
                    for pred in predictions:
                        if pred.confidence >= confidence_threshold:
                            prediction["categories"].append({
                                "category": pred.display_name,
                                "confidence": float(pred.confidence)
                            })
                    
                    if prediction["categories"]:
                        prediction["top_category"] = prediction["categories"][0]["category"]
                
                processed_results.append(prediction)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error classifying texts: {str(e)}")
            return []


# Demonstration and testing
def demonstrate_vertex_ai_prediction():
    """Demonstrate Vertex AI service category prediction"""
    
    print("\n" + "="*80)
    print("SAARTHI VERTEX AI SERVICE CATEGORY PREDICTION DEMONSTRATION")
    print("="*80 + "\n")
    
    print("Requirements:")
    print("1. Google Cloud project with Vertex AI enabled")
    print("2. Deployed model endpoint OR pre-trained classification model")
    print("3. Service account credentials configured")
    print("\nSetup:")
    print("- Export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json")
    print("- Set project ID and endpoint/model ID below")
    
    # Example usage (when configured)
    example_code = """
    predictor = VertexAIServicePredictor(
        project_id="your-project-id",
        location="us-central1",
        endpoint_id="your-endpoint-id",
        credentials_path="/path/to/credentials.json"
    )
    
    # Single query prediction
    result = predictor.predict_service_category(
        query="I need to apply for a birth certificate",
        language="en"
    )
    print(result)
    
    # Batch prediction
    queries = [
        "apply for driving license",
        "file a complaint about road damage",
        "track my passport application"
    ]
    batch_results = predictor.batch_predict(queries)
    for result in batch_results:
        print(result)
    
    # Model evaluation
    test_queries = [
        ("I want a birth certificate", "vital_records"),
        ("Vehicle registration needed", "transport"),
        ("File complaint about water", "grievance")
    ]
    metrics = predictor.evaluate_model_performance(test_queries)
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    """
    
    print("\nExample usage:")
    print(example_code)
    
    print("\n" + "="*80)
    print("For full integration, configure credentials and endpoint details.")
    print("="*80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demonstrate_vertex_ai_prediction()
