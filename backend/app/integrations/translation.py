"""
Google Cloud Translation API Integration for Saarthi
Handles multilingual query translation and response localization
Supports 10+ Indian languages with caching and fallback strategies
"""

import logging
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib
import json

from google.cloud import translate_v2
from google.cloud import translate_v3
from functools import lru_cache

logger = logging.getLogger(__name__)


class TranslationManager:
    """
    Manages translations using Google Cloud Translation API
    Handles query translation to English and response translation back to user language
    Supports V2 (basic) and V3 (advanced) APIs with caching
    """
    
    def __init__(
        self,
        project_id: str,
        credentials_path: Optional[str] = None,
        api_version: str = "v3",
        cache_enabled: bool = True
    ):
        """
        Initialize Translation Manager
        
        Args:
            project_id: Google Cloud project ID
            credentials_path: Path to service account JSON key
            api_version: 'v2' for basic, 'v3' for advanced translation
            cache_enabled: Enable translation caching for performance
        """
        self.project_id = project_id
        self.api_version = api_version
        self.cache_enabled = cache_enabled
        self.translation_cache = {}
        
        # Set credentials if provided
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        logger.info(f"Initializing Translation Manager (API v{api_version})")
        
        try:
            if api_version == "v3":
                self.translate_client = translate_v3.TranslationServiceClient()
                self.parent = f"projects/{project_id}/locations/global"
                logger.info("✓ Translation API v3 client initialized")
            else:
                self.translate_client = translate_v2.Client()
                logger.info("✓ Translation API v2 client initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize Translation client: {str(e)}")
            raise RuntimeError(f"Translation API initialization failed: {str(e)}")
        
        # Language code mapping for Indian languages
        self.language_codes = {
            "english": "en",
            "hindi": "hi",
            "tamil": "ta",
            "telugu": "te",
            "kannada": "kn",
            "malayalam": "ml",
            "marathi": "mr",
            "bengali": "bn",
            "punjabi": "pa",
            "gujarati": "gu",
            "urdu": "ur",
            "assamese": "as",
            "odia": "or",
            "sindhi": "sd"
        }
        
        # Language names for display
        self.language_names = {v: k.title() for k, v in self.language_codes.items()}
        
        # Supported languages for this region
        self.supported_languages = list(self.language_codes.values())
        
        logger.info(f"✓ Translation Manager initialized with {len(self.supported_languages)} languages")
    
    def _generate_cache_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Generate cache key for translation"""
        key_string = f"{text}_{source_lang}_{target_lang}"
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _get_cached_translation(self, cache_key: str) -> Optional[str]:
        """Retrieve translation from cache"""
        if self.cache_enabled and cache_key in self.translation_cache:
            cached_data = self.translation_cache[cache_key]
            logger.info(f"Cache hit for translation")
            return cached_data['translated_text']
        return None
    
    def _cache_translation(self, cache_key: str, translated_text: str):
        """Store translation in cache"""
        if self.cache_enabled:
            self.translation_cache[cache_key] = {
                'translated_text': translated_text,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def detect_language(self, text: str) -> Dict[str, str]:
        """
        Detect the language of input text
        
        Args:
            text: Text to detect language from
            
        Returns:
            Dictionary with language code, name, and confidence
        """
        try:
            logger.info(f"Detecting language for text: {text[:50]}...")
            
            if self.api_version == "v3":
                response = self.translate_client.detect_language(
                    parent=self.parent,
                    content=text
                )
                
                if response.languages:
                    detected = response.languages[0]
                    language_code = detected.language_code
                    confidence = float(detected.confidence)
                else:
                    language_code = "en"
                    confidence = 1.0
                    
            else:  # v2
                result = self.translate_client.detect_language(text)
                language_code = result.get('language', 'en')
                confidence = result.get('confidence', 1.0)
            
            language_name = self.language_names.get(language_code, "Unknown")
            
            logger.info(f"Detected language: {language_name} ({language_code}) - Confidence: {confidence:.2f}")
            
            return {
                "language_code": language_code,
                "language_name": language_name,
                "confidence": confidence,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            return {
                "language_code": "en",
                "language_name": "English",
                "confidence": 0.0,
                "success": False,
                "error": str(e)
            }
    
    def translate_to_english(
        self,
        text: str,
        source_language: str = "auto"
    ) -> Dict[str, str]:
        """
        Translate text to English
        
        Args:
            text: Text to translate
            source_language: Source language code or name (default: auto-detect)
            
        Returns:
            Dictionary with original and translated text
        """
        try:
            # Normalize source language
            source_lang_code = self._normalize_language_code(source_language)
            
            if source_lang_code == "en":
                logger.info("Source language is English, returning original text")
                return {
                    "original_text": text,
                    "translated_text": text,
                    "source_language": "English",
                    "target_language": "English",
                    "translation_needed": False,
                    "success": True
                }
            
            # Check cache
            cache_key = self._generate_cache_key(text, source_lang_code, "en")
            cached_result = self._get_cached_translation(cache_key)
            
            if cached_result:
                return {
                    "original_text": text,
                    "translated_text": cached_result,
                    "source_language": self.language_names.get(source_lang_code, source_language),
                    "target_language": "English",
                    "translation_needed": True,
                    "from_cache": True,
                    "success": True
                }
            
            logger.info(f"Translating from {self.language_names.get(source_lang_code)} to English")
            
            if self.api_version == "v3":
                response = self.translate_client.translate_text(
                    request={
                        "parent": self.parent,
                        "source_language_code": source_lang_code,
                        "target_language_code": "en",
                        "contents": [text],
                        "mime_type": "text/plain"
                    }
                )
                translated_text = response.translations[0].translated_text if response.translations else text
                
            else:  # v2
                result = self.translate_client.translate_text(
                    text,
                    source_language=source_lang_code if source_lang_code != "auto" else None,
                    target_language="en"
                )
                translated_text = result.get('translatedText', text)
            
            # Cache the result
            self._cache_translation(cache_key, translated_text)
            
            logger.info(f"Translation complete: {text[:50]}... -> {translated_text[:50]}...")
            
            return {
                "original_text": text,
                "translated_text": translated_text,
                "source_language": self.language_names.get(source_lang_code, source_language),
                "target_language": "English",
                "translation_needed": True,
                "from_cache": False,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error translating to English: {str(e)}")
            return {
                "original_text": text,
                "translated_text": text,
                "source_language": source_language,
                "target_language": "English",
                "translation_needed": False,
                "success": False,
                "error": str(e)
            }
    
    def translate_to_regional_language(
        self,
        text: str,
        target_language: str
    ) -> Dict[str, str]:
        """
        Translate English text to regional language (e.g., Tamil, Telugu)
        
        Args:
            text: English text to translate
            target_language: Target language code or name
            
        Returns:
            Dictionary with original and translated text
        """
        try:
            # Normalize target language
            target_lang_code = self._normalize_language_code(target_language)
            
            if target_lang_code == "en":
                logger.info("Target language is English, returning original text")
                return {
                    "original_text": text,
                    "translated_text": text,
                    "source_language": "English",
                    "target_language": "English",
                    "translation_needed": False,
                    "success": True
                }
            
            # Check cache
            cache_key = self._generate_cache_key(text, "en", target_lang_code)
            cached_result = self._get_cached_translation(cache_key)
            
            if cached_result:
                return {
                    "original_text": text,
                    "translated_text": cached_result,
                    "source_language": "English",
                    "target_language": self.language_names.get(target_lang_code, target_language),
                    "translation_needed": True,
                    "from_cache": True,
                    "success": True
                }
            
            logger.info(f"Translating from English to {self.language_names.get(target_lang_code)}")
            
            if self.api_version == "v3":
                response = self.translate_client.translate_text(
                    request={
                        "parent": self.parent,
                        "source_language_code": "en",
                        "target_language_code": target_lang_code,
                        "contents": [text],
                        "mime_type": "text/plain"
                    }
                )
                translated_text = response.translations[0].translated_text if response.translations else text
                
            else:  # v2
                result = self.translate_client.translate_text(
                    text,
                    source_language="en",
                    target_language=target_lang_code
                )
                translated_text = result.get('translatedText', text)
            
            # Cache the result
            self._cache_translation(cache_key, translated_text)
            
            logger.info(f"Translation complete to {self.language_names.get(target_lang_code)}")
            
            return {
                "original_text": text,
                "translated_text": translated_text,
                "source_language": "English",
                "target_language": self.language_names.get(target_lang_code, target_language),
                "translation_needed": True,
                "from_cache": False,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error translating to {target_language}: {str(e)}")
            return {
                "original_text": text,
                "translated_text": text,
                "source_language": "English",
                "target_language": target_language,
                "translation_needed": False,
                "success": False,
                "error": str(e)
            }
    
    def translate_text(
        self,
        text: str,
        source_language: str,
        target_language: str
    ) -> Dict[str, str]:
        """
        General translation function for any source/target language pair
        
        Args:
            text: Text to translate
            source_language: Source language code or name
            target_language: Target language code or name
            
        Returns:
            Dictionary with translation result
        """
        try:
            source_lang_code = self._normalize_language_code(source_language)
            target_lang_code = self._normalize_language_code(target_language)
            
            if source_lang_code == target_lang_code:
                return {
                    "original_text": text,
                    "translated_text": text,
                    "source_language": self.language_names.get(source_lang_code),
                    "target_language": self.language_names.get(target_lang_code),
                    "translation_needed": False,
                    "success": True
                }
            
            # Check cache
            cache_key = self._generate_cache_key(text, source_lang_code, target_lang_code)
            cached_result = self._get_cached_translation(cache_key)
            
            if cached_result:
                return {
                    "original_text": text,
                    "translated_text": cached_result,
                    "source_language": self.language_names.get(source_lang_code),
                    "target_language": self.language_names.get(target_lang_code),
                    "translation_needed": True,
                    "from_cache": True,
                    "success": True
                }
            
            logger.info(f"Translating from {source_lang_code} to {target_lang_code}")
            
            if self.api_version == "v3":
                response = self.translate_client.translate_text(
                    request={
                        "parent": self.parent,
                        "source_language_code": source_lang_code,
                        "target_language_code": target_lang_code,
                        "contents": [text],
                        "mime_type": "text/plain"
                    }
                )
                translated_text = response.translations[0].translated_text if response.translations else text
                
            else:  # v2
                result = self.translate_client.translate_text(
                    text,
                    source_language=source_lang_code if source_lang_code != "auto" else None,
                    target_language=target_lang_code
                )
                translated_text = result.get('translatedText', text)
            
            # Cache the result
            self._cache_translation(cache_key, translated_text)
            
            return {
                "original_text": text,
                "translated_text": translated_text,
                "source_language": self.language_names.get(source_lang_code),
                "target_language": self.language_names.get(target_lang_code),
                "translation_needed": True,
                "from_cache": False,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            return {
                "original_text": text,
                "translated_text": text,
                "source_language": source_language,
                "target_language": target_language,
                "translation_needed": False,
                "success": False,
                "error": str(e)
            }
    
    def batch_translate(
        self,
        texts: List[str],
        source_language: str,
        target_language: str
    ) -> List[Dict[str, str]]:
        """
        Translate multiple texts efficiently
        
        Args:
            texts: List of texts to translate
            source_language: Source language code or name
            target_language: Target language code or name
            
        Returns:
            List of translation results
        """
        logger.info(f"Batch translating {len(texts)} texts from {source_language} to {target_language}")
        
        try:
            source_lang_code = self._normalize_language_code(source_language)
            target_lang_code = self._normalize_language_code(target_language)
            
            if source_lang_code == target_lang_code:
                return [
                    {
                        "original_text": text,
                        "translated_text": text,
                        "translation_needed": False,
                        "success": True
                    }
                    for text in texts
                ]
            
            if self.api_version == "v3":
                response = self.translate_client.translate_text(
                    request={
                        "parent": self.parent,
                        "source_language_code": source_lang_code,
                        "target_language_code": target_lang_code,
                        "contents": texts,
                        "mime_type": "text/plain"
                    }
                )
                
                results = []
                for i, translation in enumerate(response.translations):
                    results.append({
                        "original_text": texts[i],
                        "translated_text": translation.translated_text,
                        "translation_needed": True,
                        "success": True
                    })
                return results
                
            else:  # v2
                results = []
                for text in texts:
                    result = self.translate_client.translate_text(
                        text,
                        source_language=source_lang_code if source_lang_code != "auto" else None,
                        target_language=target_lang_code
                    )
                    results.append({
                        "original_text": text,
                        "translated_text": result.get('translatedText', text),
                        "translation_needed": True,
                        "success": True
                    })
                return results
                
        except Exception as e:
            logger.error(f"Error in batch translation: {str(e)}")
            return [
                {
                    "original_text": text,
                    "translated_text": text,
                    "translation_needed": False,
                    "success": False,
                    "error": str(e)
                }
                for text in texts
            ]
    
    def _normalize_language_code(self, language: str) -> str:
        """
        Normalize language name or code to standard code
        
        Args:
            language: Language name or code (e.g., 'Tamil', 'ta', 'tamil')
            
        Returns:
            Standard language code (e.g., 'ta')
        """
        language_lower = language.lower().strip()
        
        # Check if already a code
        if language_lower in self.language_codes.values():
            return language_lower
        
        # Check if it's a language name
        if language_lower in self.language_codes:
            return self.language_codes[language_lower]
        
        # Default to English
        logger.warning(f"Unknown language: {language}, defaulting to English")
        return "en"
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get all supported languages"""
        return {code: self.language_names.get(code, code) for code in self.supported_languages}
    
    def get_cache_stats(self) -> Dict:
        """Get translation cache statistics"""
        return {
            "cache_enabled": self.cache_enabled,
            "cached_translations": len(self.translation_cache),
            "cache_size_estimate_kb": len(json.dumps(self.translation_cache)) / 1024
        }
    
    def clear_cache(self):
        """Clear translation cache"""
        self.translation_cache.clear()
        logger.info("Translation cache cleared")
    
    def process_multilingual_query(
        self,
        query: str,
        user_language: str = "English"
    ) -> Dict:
        """
        Complete multilingual query processing pipeline
        
        Args:
            query: User query in any language
            user_language: User's preferred language
            
        Returns:
            Dictionary with language detection and English translation
        """
        try:
            logger.info(f"Processing multilingual query: {query}")
            
            # Step 1: Detect language
            lang_detection = self.detect_language(query)
            detected_language = lang_detection.get("language_code", "en")
            
            # Step 2: Translate to English if needed
            english_translation = self.translate_to_english(query, detected_language)
            english_query = english_translation.get("translated_text", query)
            
            # Store user language for response translation
            user_lang_code = self._normalize_language_code(user_language)
            
            return {
                "success": True,
                "original_query": query,
                "detected_language": lang_detection,
                "english_query": english_query,
                "user_language": user_language,
                "user_language_code": user_lang_code,
                "translation_pipeline": english_translation
            }
            
        except Exception as e:
            logger.error(f"Error processing multilingual query: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "original_query": query
            }


# Demonstration
def demonstrate_translation_integration():
    """Demonstrate translation API integration"""
    
    print("\n" + "="*80)
    print("SAARTHI GOOGLE TRANSLATION API INTEGRATION DEMONSTRATION")
    print("="*80 + "\n")
    
    print("Requirements:")
    print("1. Google Cloud project with Translation API enabled")
    print("2. Service account credentials configured")
    print("\nSetup:")
    print("- pip install google-cloud-translate")
    print("- export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json")
    
    example_code = '''
    # Initialize translation manager
    translator = TranslationManager(
        project_id="your-project-id",
        api_version="v3",
        cache_enabled=True
    )
    
    # Detect language
    detection = translator.detect_language("பிறப்பு சான்றிதழ் அதிகாரம்")
    print(f"Detected: {detection['language_name']}")
    
    # Translate Tamil query to English
    tamil_query = "நான் பிறப்பு சான்றிதழ் விண்ணப்பிக்க வேண்டும்"
    en_result = translator.translate_to_english(tamil_query, "tamil")
    print(f"Tamil: {tamil_query}")
    print(f"English: {en_result['translated_text']}")
    
    # Translate English response back to Tamil
    english_response = "Your birth certificate application has been received."
    ta_result = translator.translate_to_regional_language(english_response, "tamil")
    print(f"English: {english_response}")
    print(f"Tamil: {ta_result['translated_text']}")
    
    # Batch translation
    queries = [
        "I want a birth certificate",
        "I need to register my vehicle",
        "File a complaint"
    ]
    batch_results = translator.batch_translate(queries, "English", "tamil")
    for result in batch_results:
        print(f"{result['original_text']} -> {result['translated_text']}")
    '''
    
    print("\nExample usage:")
    print(example_code)
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demonstrate_translation_integration()
