"""
NLP Processor for Saarthi - Multilingual Query Understanding
Integrates Hugging Face transformers for semantic understanding in English, Tamil, Telugu, Kannada
Includes translation, semantic similarity, and intent classification
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import hashlib
from functools import lru_cache
import warnings

# Suppress transformer warnings
warnings.filterwarnings('ignore')

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    pipeline,
    TranslationPipeline,
    ZeroShotClassificationPipeline
)

# Configure logging for audit trail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nlp_audit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MultilingualNLPProcessor:
    """
    Processes multilingual queries using Hugging Face transformers
    Supports: English, Tamil, Telugu, Kannada, Hindi
    Features: Translation, semantic similarity, intent classification, language detection
    """
    
    def __init__(self, device: str = "cpu", cache_embeddings: bool = True):
        """
        Initialize NLP processor with multilingual models
        
        Args:
            device: "cuda" or "cpu" for model inference
            cache_embeddings: Enable caching for performance optimization
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cache_embeddings = cache_embeddings
        self.embedding_cache = {}
        
        logger.info(f"Initializing NLP processor on device: {self.device}")
        
        try:
            # 1. Multilingual BERT for semantic embeddings
            logger.info("Loading multilingual BERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/multilingual-MiniLM-L12-v2"
            )
            self.embedding_model = AutoModel.from_pretrained(
                "sentence-transformers/multilingual-MiniLM-L12-v2"
            ).to(self.device)
            self.embedding_model.eval()
            logger.info("✓ Multilingual BERT loaded successfully")
            
            # 2. Translation pipeline (many-to-many)
            logger.info("Loading translation model...")
            self.translator = pipeline(
                "translation",
                model="facebook/m2m100_418M",
                device=0 if self.device == "cuda" else -1
            )
            logger.info("✓ Translation model loaded successfully")
            
            # 3. Zero-shot classification for intent detection
            logger.info("Loading zero-shot classification model...")
            self.intent_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1
            )
            logger.info("✓ Intent classifier loaded successfully")
            
            # 4. Language detection model
            logger.info("Loading language detection model...")
            self.language_detector = pipeline(
                "text-classification",
                model="papluca/xlm-roberta-base-language-detection",
                device=0 if self.device == "cuda" else -1
            )
            logger.info("✓ Language detector loaded successfully")
            
            # Language mapping for m2m100 model
            self.language_codes = {
                "english": "en_XX",
                "tamil": "tam_Taml",
                "telugu": "tel_Telu",
                "kannada": "kan_Knda",
                "hindi": "hin_Deva",
                "marathi": "mar_Deva",
                "bengali": "ben_Beng",
                "malayalam": "mal_Mlym",
                "punjabi": "pan_Guru",
                "gujarati": "guj_Gujr"
            }
            
            # Predefined government service intents
            self.service_intents = [
                "apply for birth certificate",
                "register vehicle",
                "apply for driving license",
                "request ration card",
                "apply for passport",
                "file complaint",
                "check service status",
                "get eligibility information",
                "track application",
                "payment inquiry"
            ]
            
            logger.info("✓ NLP Processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NLP processor: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize NLP components: {str(e)}")
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get semantic embeddings for texts using multilingual BERT
        Includes caching for performance
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of embeddings (shape: [len(texts), embedding_dim])
        """
        embeddings = []
        texts_to_encode = []
        cache_hits = []
        
        # Check cache
        for text in texts:
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            if self.cache_embeddings and text_hash in self.embedding_cache:
                embeddings.append(self.embedding_cache[text_hash])
                cache_hits.append(True)
            else:
                texts_to_encode.append((text, text_hash))
                cache_hits.append(False)
        
        # Encode texts not in cache
        if texts_to_encode:
            input_texts = [t[0] for t in texts_to_encode]
            
            with torch.no_grad():
                encoded = self.tokenizer(
                    input_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                model_output = self.embedding_model(**encoded)
                sentence_embeddings = self._mean_pooling(
                    model_output,
                    encoded["attention_mask"]
                )
            
            # Normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(
                sentence_embeddings,
                p=2,
                dim=1
            ).cpu().numpy()
            
            # Store in cache
            for (text, text_hash), embedding in zip(texts_to_encode, sentence_embeddings):
                if self.cache_embeddings:
                    self.embedding_cache[text_hash] = embedding
                embeddings.append(embedding)
        
        # Reconstruct in original order
        result_embeddings = []
        embed_idx = 0
        cache_idx = 0
        
        for hit in cache_hits:
            if hit:
                result_embeddings.append(embeddings[embed_idx])
            else:
                result_embeddings.append(embeddings[embed_idx])
            embed_idx += 1
        
        logger.info(f"Embeddings generated: {len(texts)} texts, Cache hits: {sum(cache_hits)}")
        return np.array(embeddings)
    
    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        """Mean pooling for embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def detect_language(self, text: str) -> Dict[str, str]:
        """
        Detect language of input text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with detected language and confidence score
        """
        try:
            result = self.language_detector(text[:512])[0]  # Limit to 512 chars
            language_map = {
                'en': 'English',
                'ta': 'Tamil',
                'te': 'Telugu',
                'kn': 'Kannada',
                'hi': 'Hindi',
                'mr': 'Marathi',
                'bn': 'Bengali',
                'ml': 'Malayalam',
                'pa': 'Punjabi',
                'gu': 'Gujarati'
            }
            
            detected_lang = language_map.get(result['label'], result['label'])
            logger.info(f"Detected language: {detected_lang} (confidence: {result['score']:.3f})")
            
            return {
                "language": detected_lang,
                "language_code": result['label'],
                "confidence": round(result['score'], 3)
            }
        except Exception as e:
            logger.error(f"Error in language detection: {str(e)}")
            return {
                "language": "Unknown",
                "language_code": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def translate_to_english(self, text: str, source_language: str) -> Dict[str, str]:
        """
        Translate text to English using Facebook's M2M100 model
        
        Args:
            text: Text to translate
            source_language: Source language name
            
        Returns:
            Dictionary with translated text and metadata
        """
        try:
            source_lang_code = self.language_codes.get(source_language.lower(), "en_XX")
            
            if source_lang_code == "en_XX":
                return {
                    "original_text": text,
                    "translated_text": text,
                    "source_language": source_language,
                    "target_language": "English",
                    "translation_needed": False
                }
            
            logger.info(f"Translating from {source_language} to English...")
            
            self.translator.src_lang = source_lang_code
            result = self.translator(text[:512], tgt_lang="en_XX")  # Limit to 512 chars
            
            translated_text = result[0]["translation_text"]
            logger.info(f"Translation successful: '{text}' -> '{translated_text}'")
            
            return {
                "original_text": text,
                "translated_text": translated_text,
                "source_language": source_language,
                "target_language": "English",
                "translation_needed": True
            }
        except Exception as e:
            logger.error(f"Error in translation: {str(e)}")
            return {
                "original_text": text,
                "translated_text": text,
                "source_language": source_language,
                "target_language": "English",
                "error": str(e),
                "translation_needed": False
            }
    
    def translate_to_regional(self, text: str, target_language: str) -> Dict[str, str]:
        """
        Translate English text to regional language
        
        Args:
            text: English text to translate
            target_language: Target regional language
            
        Returns:
            Dictionary with translated text and metadata
        """
        try:
            target_lang_code = self.language_codes.get(target_language.lower(), "en_XX")
            
            if target_lang_code == "en_XX":
                return {
                    "original_text": text,
                    "translated_text": text,
                    "source_language": "English",
                    "target_language": target_language,
                    "translation_needed": False
                }
            
            logger.info(f"Translating from English to {target_language}...")
            
            self.translator.src_lang = "en_XX"
            result = self.translator(text[:512], tgt_lang=target_lang_code)
            
            translated_text = result[0]["translation_text"]
            logger.info(f"Translation successful to {target_language}")
            
            return {
                "original_text": text,
                "translated_text": translated_text,
                "source_language": "English",
                "target_language": target_language,
                "translation_needed": True
            }
        except Exception as e:
            logger.error(f"Error in translation to {target_language}: {str(e)}")
            return {
                "original_text": text,
                "translated_text": text,
                "source_language": "English",
                "target_language": target_language,
                "error": str(e),
                "translation_needed": False
            }
    
    def classify_intent(self, query: str) -> Dict:
        """
        Classify user query intent using zero-shot classification
        
        Args:
            query: User query text
            
        Returns:
            Dictionary with classified intent and confidence scores
        """
        try:
            logger.info(f"Classifying intent for: {query}")
            
            result = self.intent_classifier(
                query,
                self.service_intents,
                multi_class=False
            )
            
            intent_scores = {}
            for label, score in zip(result['labels'], result['scores']):
                intent_scores[label] = round(score, 3)
            
            top_intent = result['labels'][0]
            top_score = result['scores'][0]
            
            logger.info(f"Top intent: {top_intent} (confidence: {top_score:.3f})")
            
            return {
                "query": query,
                "top_intent": top_intent,
                "confidence": round(top_score, 3),
                "all_intents": intent_scores,
                "detected": top_score > 0.3  # Threshold for detection
            }
        except Exception as e:
            logger.error(f"Error in intent classification: {str(e)}")
            return {
                "query": query,
                "top_intent": "unknown",
                "confidence": 0.0,
                "all_intents": {},
                "detected": False,
                "error": str(e)
            }
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts (0-1)
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            embeddings = self._get_embeddings([text1, text2])
            
            # Cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return max(0.0, min(1.0, float(similarity)))  # Clamp to [0, 1]
        except Exception as e:
            logger.error(f"Error in semantic similarity: {str(e)}")
            return 0.0
    
    def extract_semantic_meaning(self, query: str, context_words: List[str]) -> Dict:
        """
        Extract semantic meaning from query using embeddings
        
        Args:
            query: User query
            context_words: List of context-related words/phrases
            
        Returns:
            Dictionary with semantic analysis
        """
        try:
            logger.info(f"Extracting semantic meaning: {query}")
            
            all_texts = [query] + context_words
            embeddings = self._get_embeddings(all_texts)
            
            query_embedding = embeddings[0]
            context_embeddings = embeddings[1:]
            
            # Calculate similarity with context
            similarities = {}
            for word, embedding in zip(context_words, context_embeddings):
                sim = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                similarities[word] = round(float(sim), 3)
            
            # Sort by similarity
            sorted_similarities = dict(
                sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            )
            
            return {
                "query": query,
                "context_similarities": sorted_similarities,
                "most_relevant": list(sorted_similarities.keys())[0] if sorted_similarities else None
            }
        except Exception as e:
            logger.error(f"Error in semantic meaning extraction: {str(e)}")
            return {
                "query": query,
                "context_similarities": {},
                "error": str(e)
            }
    
    def process_multilingual_query(self, 
                                  query: str,
                                  preferred_language: Optional[str] = None) -> Dict:
        """
        Complete multilingual query processing pipeline
        
        Args:
            query: User query in any supported language
            preferred_language: User's preferred response language
            
        Returns:
            Dictionary with complete analysis
        """
        try:
            logger.info(f"Processing multilingual query: {query}")
            
            # Step 1: Detect language
            lang_detection = self.detect_language(query)
            detected_language = lang_detection.get("language", "English")
            
            # Step 2: Translate to English if needed
            if detected_language != "English":
                translation = self.translate_to_english(query, detected_language)
                english_query = translation.get("translated_text", query)
            else:
                english_query = query
                translation = {
                    "original_text": query,
                    "translated_text": query,
                    "translation_needed": False
                }
            
            # Step 3: Classify intent
            intent = self.classify_intent(english_query)
            
            # Step 4: Extract semantic meaning
            semantic_analysis = self.extract_semantic_meaning(
                english_query,
                self.service_intents
            )
            
            # Step 5: Translate response back to preferred language if needed
            response_language = preferred_language or detected_language
            response_translation = None
            
            if response_language != "English":
                # We'll prepare translated response for service descriptions
                response_translation = {
                    "target_language": response_language,
                    "target_language_code": self.language_codes.get(response_language.lower(), "en_XX")
                }
            
            # Compile results
            result = {
                "original_query": query,
                "english_query": english_query,
                "language_detection": lang_detection,
                "translation": translation,
                "intent_classification": intent,
                "semantic_analysis": semantic_analysis,
                "response_language": response_language,
                "response_translation": response_translation,
                "processing_metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "device": self.device,
                    "models_used": [
                        "sentence-transformers/multilingual-MiniLM-L12-v2",
                        "facebook/m2m100_418M",
                        "facebook/bart-large-mnli",
                        "papluca/xlm-roberta-base-language-detection"
                    ]
                }
            }
            
            logger.info(f"Query processing complete. Intent: {intent.get('top_intent')}, Confidence: {intent.get('confidence')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in multilingual query processing: {str(e)}", exc_info=True)
            return {
                "original_query": query,
                "error": str(e),
                "success": False
            }
    
    def get_embedding_cache_stats(self) -> Dict:
        """Get statistics about embedding cache"""
        return {
            "cache_enabled": self.cache_embeddings,
            "cached_embeddings": len(self.embedding_cache),
            "cache_size_estimate_mb": len(self.embedding_cache) * 0.05  # Approx 50KB per embedding
        }
    
    def clear_embedding_cache(self):
        """Clear the embedding cache"""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")


# Demonstration and testing
def demonstrate_nlp_processor():
    """
    Demonstrate the multilingual NLP processor
    """
    print("\n" + "="*80)
    print("SAARTHI MULTILINGUAL NLP PROCESSOR DEMONSTRATION")
    print("="*80 + "\n")
    
    # Initialize processor
    processor = MultilingualNLPProcessor(device="cpu")
    
    # Test queries in different languages
    test_queries = [
        "I need to apply for a birth certificate",
        "பிறப்பு சான்றிதழுக்கு விண்ணப்பிக்க வேண்டும்",  # Tamil
        "నేను జన్మ ప్రమాణపత్రానికి దరఖాస్తు చేయాలి",  # Telugu
        "ನನಗೆ ಜನನ ಪ್ರಮಾಣಪತ್ರಕ್ಕೆ ಅರ್ಜಿ ಸಲ್ಲಿಸಬೇಕು",  # Kannada
        "मुझे ड्राइविंग लाइसेंस के लिए आवेदन करना है",  # Hindi
        "vehicle registration के लिए क्या दस्तावेज चाहिए",
    ]
    
    for query in test_queries:
        print(f"\n{'─'*80}")
        print(f"Query: {query}")
        print(f"{'─'*80}")
        
        result = processor.process_multilingual_query(query)
        
        if result.get("success") is not False:
            print(f"✓ Language Detected: {result['language_detection']['language']} "
                  f"(confidence: {result['language_detection']['confidence']})")
            
            if result['translation']['translation_needed']:
                print(f"✓ Translated to English: {result['english_query']}")
            
            intent = result['intent_classification']
            print(f"✓ Intent: {intent['top_intent']} (confidence: {intent['confidence']})")
            
            semantic = result['semantic_analysis']
            if semantic.get('most_relevant'):
                print(f"✓ Most Relevant Service: {semantic['most_relevant']}")
        else:
            print(f"✗ Error: {result.get('error')}")
    
    # Cache statistics
    print(f"\n{'─'*80}")
    print("Cache Statistics:")
    print(f"{'─'*80}")
    cache_stats = processor.get_embedding_cache_stats()
    for key, value in cache_stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*80)
    print("For NLP audit logs, check: nlp_audit.log")
    print("="*80 + "\n")


if __name__ == "__main__":
    demonstrate_nlp_processor()
