"""
Google Cloud Speech-to-Text and Text-to-Speech Integration for Saarthi
Enables voice input/output for citizen queries in local languages
"""

import logging
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import io
import base64
from pathlib import Path

from google.cloud import speech_v1
from google.cloud import texttospeech_v1
from google.cloud.speech_v1 import SpeechClient, RecognitionConfig, RecognitionAudio
from google.cloud.texttospeech_v1 import TextToSpeechClient, SynthesisInput, VoiceSelectionParams, AudioConfig

logger = logging.getLogger(__name__)


class SpeechToTextManager:
    """
    Manages speech-to-text transcription using Google Cloud Speech-to-Text API
    Supports 10+ Indian languages with real-time and batch processing
    """
    
    def __init__(
        self,
        project_id: str,
        credentials_path: Optional[str] = None
    ):
        """
        Initialize Speech-to-Text Manager
        
        Args:
            project_id: Google Cloud project ID
            credentials_path: Path to service account JSON key
        """
        self.project_id = project_id
        
        # Set credentials if provided
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        logger.info(f"Initializing Speech-to-Text Manager")
        
        try:
            self.client = SpeechClient()
            logger.info("✓ Speech-to-Text client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Speech-to-Text client: {str(e)}")
            raise RuntimeError(f"Speech-to-Text initialization failed: {str(e)}")
        
        # Language code mapping for Indian languages
        self.language_codes = {
            "english": "en-IN",
            "hindi": "hi-IN",
            "tamil": "ta-IN",
            "telugu": "te-IN",
            "kannada": "kn-IN",
            "malayalam": "ml-IN",
            "marathi": "mr-IN",
            "bengali": "bn-IN",
            "punjabi": "pa-IN",
            "gujarati": "gu-IN",
            "urdu": "ur-IN",
            "assamese": "as-IN",
            "odia": "or-IN"
        }
        
        self.language_names = {v: k.title() for k, v in self.language_codes.items()}
    
    def transcribe_audio_file(
        self,
        audio_file_path: str,
        language: str = "English",
        enable_automatic_punctuation: bool = True,
        enable_profanity_filter: bool = True
    ) -> Dict:
        """
        Transcribe audio file to text
        
        Args:
            audio_file_path: Path to audio file (WAV, FLAC, OGG, MP3)
            language: Language of audio (English, Tamil, Telugu, etc.)
            enable_automatic_punctuation: Add automatic punctuation
            enable_profanity_filter: Filter profanity
            
        Returns:
            Dictionary with transcription results
        """
        try:
            logger.info(f"Transcribing audio file: {audio_file_path}")
            
            # Normalize language
            language_code = self._normalize_language(language)
            
            # Read audio file
            with open(audio_file_path, "rb") as audio_file:
                content = audio_file.read()
            
            # Detect audio encoding from file extension
            audio_encoding = self._detect_audio_encoding(audio_file_path)
            
            audio = RecognitionAudio(content=content)
            
            config = RecognitionConfig(
                encoding=audio_encoding,
                sample_rate_hertz=16000,
                language_code=language_code,
                enable_automatic_punctuation=enable_automatic_punctuation,
                use_enhanced=True,  # Use enhanced model for better accuracy
                model="latest_long"  # Use latest model
            )
            
            # Perform speech recognition
            response = self.client.recognize(config=config, audio=audio)
            
            if not response.results:
                logger.warning("No speech detected in audio")
                return {
                    "success": False,
                    "transcription": "",
                    "confidence": 0.0,
                    "message": "No speech detected"
                }
            
            # Extract best result
            result = response.results[0]
            if result.alternatives:
                transcript = result.alternatives[0].transcript
                confidence = result.alternatives[0].confidence if hasattr(result.alternatives[0], 'confidence') else 0.0
                
                logger.info(f"Transcription complete: {transcript}")
                
                return {
                    "success": True,
                    "transcription": transcript,
                    "confidence": float(confidence),
                    "language": self.language_names.get(language_code, language),
                    "language_code": language_code,
                    "alternatives": [alt.transcript for alt in result.alternatives[1:3]]  # Top 3 alternatives
                }
            
            return {
                "success": False,
                "transcription": "",
                "confidence": 0.0,
                "message": "No alternatives found"
            }
            
        except FileNotFoundError:
            logger.error(f"Audio file not found: {audio_file_path}")
            return {
                "success": False,
                "error": "Audio file not found",
                "transcription": ""
            }
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "transcription": ""
            }
    
    def transcribe_audio_bytes(
        self,
        audio_content: bytes,
        language: str = "English",
        sample_rate_hertz: int = 16000,
        audio_encoding: str = "LINEAR16",
        enable_automatic_punctuation: bool = True
    ) -> Dict:
        """
        Transcribe audio from bytes (e.g., from API upload)
        
        Args:
            audio_content: Audio data as bytes
            language: Language of audio
            sample_rate_hertz: Sample rate of audio
            audio_encoding: Audio encoding (LINEAR16, MULAW, AMR, etc.)
            enable_automatic_punctuation: Add automatic punctuation
            
        Returns:
            Dictionary with transcription results
        """
        try:
            logger.info(f"Transcribing audio bytes ({len(audio_content)} bytes)")
            
            # Normalize language
            language_code = self._normalize_language(language)
            
            # Convert encoding string to enum
            encoding_map = {
                "LINEAR16": speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
                "MULAW": speech_v1.RecognitionConfig.AudioEncoding.MULAW,
                "AMR": speech_v1.RecognitionConfig.AudioEncoding.AMR,
                "AMR_WB": speech_v1.RecognitionConfig.AudioEncoding.AMR_WB,
                "OGG_OPUS": speech_v1.RecognitionConfig.AudioEncoding.OGG_OPUS,
                "FLAC": speech_v1.RecognitionConfig.AudioEncoding.FLAC,
                "MP3": speech_v1.RecognitionConfig.AudioEncoding.MP3,
                "WEBM_OPUS": speech_v1.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            }
            
            encoding = encoding_map.get(audio_encoding, speech_v1.RecognitionConfig.AudioEncoding.LINEAR16)
            
            audio = RecognitionAudio(content=audio_content)
            
            config = RecognitionConfig(
                encoding=encoding,
                sample_rate_hertz=sample_rate_hertz,
                language_code=language_code,
                enable_automatic_punctuation=enable_automatic_punctuation,
                use_enhanced=True,
                model="latest_long"
            )
            
            # Perform speech recognition
            response = self.client.recognize(config=config, audio=audio)
            
            if not response.results:
                logger.warning("No speech detected in audio bytes")
                return {
                    "success": False,
                    "transcription": "",
                    "confidence": 0.0,
                    "message": "No speech detected"
                }
            
            result = response.results[0]
            if result.alternatives:
                transcript = result.alternatives[0].transcript
                confidence = getattr(result.alternatives[0], 'confidence', 0.0)
                
                logger.info(f"Transcription complete: {transcript}")
                
                return {
                    "success": True,
                    "transcription": transcript,
                    "confidence": float(confidence),
                    "language": self.language_names.get(language_code, language),
                    "language_code": language_code
                }
            
            return {
                "success": False,
                "transcription": "",
                "message": "No alternatives found"
            }
            
        except Exception as e:
            logger.error(f"Error transcribing audio bytes: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "transcription": ""
            }
    
    def streaming_transcribe(
        self,
        audio_stream,
        language: str = "English",
        interim_results: bool = True
    ) -> Dict:
        """
        Transcribe audio from a streaming source (for real-time transcription)
        
        Args:
            audio_stream: Audio stream or generator yielding audio chunks
            language: Language of audio
            interim_results: Include interim results
            
        Returns:
            Dictionary with transcription results
        """
        try:
            logger.info("Starting streaming transcription")
            
            language_code = self._normalize_language(language)
            
            config = RecognitionConfig(
                encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=language_code,
                enable_automatic_punctuation=True,
                use_enhanced=True,
                interim_results=interim_results
            )
            
            # Create streaming requests
            def request_generator():
                # First request with config
                yield speech_v1.StreamingRecognizeRequest(config=config)
                
                # Subsequent requests with audio content
                for audio_content in audio_stream:
                    yield speech_v1.StreamingRecognizeRequest(audio_content=audio_content)
            
            # Perform streaming speech recognition
            responses = self.client.streaming_recognize(request_generator())
            
            transcripts = []
            final_transcript = ""
            
            for response in responses:
                if not response.results:
                    continue
                
                result = response.results[0]
                
                if result.alternatives:
                    transcript = result.alternatives[0].transcript
                    is_final = result.is_final
                    
                    if is_final:
                        final_transcript = transcript
                        transcripts.append(transcript)
                        logger.info(f"Final transcript: {transcript}")
                    elif interim_results:
                        logger.info(f"Interim: {transcript}")
            
            return {
                "success": True,
                "transcription": " ".join(transcripts) or final_transcript,
                "language": self.language_names.get(language_code, language),
                "language_code": language_code
            }
            
        except Exception as e:
            logger.error(f"Error in streaming transcription: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "transcription": ""
            }
    
    def _normalize_language(self, language: str) -> str:
        """Normalize language name to code"""
        language_lower = language.lower().strip()
        
        if language_lower in self.language_codes:
            return self.language_codes[language_lower]
        
        if language_lower in self.language_codes.values():
            return language_lower
        
        logger.warning(f"Unknown language: {language}, defaulting to English")
        return self.language_codes.get("english", "en-IN")
    
    def _detect_audio_encoding(self, file_path: str) -> int:
        """Detect audio encoding from file extension"""
        extension = Path(file_path).suffix.lower()
        
        encoding_map = {
            ".wav": speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
            ".flac": speech_v1.RecognitionConfig.AudioEncoding.FLAC,
            ".ogg": speech_v1.RecognitionConfig.AudioEncoding.OGG_OPUS,
            ".mp3": speech_v1.RecognitionConfig.AudioEncoding.MP3,
            ".amr": speech_v1.RecognitionConfig.AudioEncoding.AMR,
        }
        
        return encoding_map.get(extension, speech_v1.RecognitionConfig.AudioEncoding.LINEAR16)
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get all supported languages for transcription"""
        return dict(sorted(self.language_codes.items()))


class TextToSpeechManager:
    """
    Manages text-to-speech synthesis using Google Cloud Text-to-Speech API
    Generates natural-sounding speech in 10+ Indian languages
    """
    
    def __init__(
        self,
        project_id: str,
        credentials_path: Optional[str] = None
    ):
        """
        Initialize Text-to-Speech Manager
        
        Args:
            project_id: Google Cloud project ID
            credentials_path: Path to service account JSON key
        """
        self.project_id = project_id
        
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        logger.info("Initializing Text-to-Speech Manager")
        
        try:
            self.client = TextToSpeechClient()
            logger.info("✓ Text-to-Speech client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Text-to-Speech client: {str(e)}")
            raise RuntimeError(f"Text-to-Speech initialization failed: {str(e)}")
        
        # Language code mapping
        self.language_codes = {
            "english": "en-IN",
            "hindi": "hi-IN",
            "tamil": "ta-IN",
            "telugu": "te-IN",
            "kannada": "kn-IN",
            "malayalam": "ml-IN",
            "marathi": "mr-IN",
            "bengali": "bn-IN",
            "punjabi": "pa-IN",
            "gujarati": "gu-IN",
            "urdu": "ur-IN",
        }
        
        # Available voices for each language
        self.voices = {
            "en-IN": [
                {"name": "en-IN-Standard-A", "ssml_gender": texttospeech_v1.SsmlVoiceGender.MALE},
                {"name": "en-IN-Standard-B", "ssml_gender": texttospeech_v1.SsmlVoiceGender.FEMALE},
                {"name": "en-IN-Standard-C", "ssml_gender": texttospeech_v1.SsmlVoiceGender.MALE},
            ],
            "hi-IN": [
                {"name": "hi-IN-Standard-A", "ssml_gender": texttospeech_v1.SsmlVoiceGender.FEMALE},
                {"name": "hi-IN-Standard-B", "ssml_gender": texttospeech_v1.SsmlVoiceGender.MALE},
            ],
            "ta-IN": [
                {"name": "ta-IN-Standard-A", "ssml_gender": texttospeech_v1.SsmlVoiceGender.FEMALE},
                {"name": "ta-IN-Standard-B", "ssml_gender": texttospeech_v1.SsmlVoiceGender.MALE},
            ],
            "te-IN": [
                {"name": "te-IN-Standard-A", "ssml_gender": texttospeech_v1.SsmlVoiceGender.FEMALE},
            ],
            "kn-IN": [
                {"name": "kn-IN-Standard-A", "ssml_gender": texttospeech_v1.SsmlVoiceGender.FEMALE},
            ],
            "ml-IN": [
                {"name": "ml-IN-Standard-A", "ssml_gender": texttospeech_v1.SsmlVoiceGender.FEMALE},
            ],
            "mr-IN": [
                {"name": "mr-IN-Standard-A", "ssml_gender": texttospeech_v1.SsmlVoiceGender.FEMALE},
            ],
            "bn-IN": [
                {"name": "bn-IN-Standard-A", "ssml_gender": texttospeech_v1.SsmlVoiceGender.FEMALE},
            ],
            "pa-IN": [
                {"name": "pa-IN-Standard-A", "ssml_gender": texttospeech_v1.SsmlVoiceGender.FEMALE},
            ],
            "gu-IN": [
                {"name": "gu-IN-Standard-A", "ssml_gender": texttospeech_v1.SsmlVoiceGender.FEMALE},
            ],
            "ur-IN": [
                {"name": "ur-IN-Standard-A", "ssml_gender": texttospeech_v1.SsmlVoiceGender.FEMALE},
            ],
        }
    
    def synthesize_speech(
        self,
        text: str,
        language: str = "English",
        gender: str = "FEMALE",
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
        output_format: str = "LINEAR16"
    ) -> Dict:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            language: Language of text (English, Tamil, Telugu, etc.)
            gender: Voice gender (MALE, FEMALE, NEUTRAL)
            speaking_rate: Speech rate (0.25 to 4.0)
            pitch: Pitch adjustment (-20.0 to 20.0)
            output_format: Output audio format (LINEAR16, OGG_OPUS, MULAW)
            
        Returns:
            Dictionary with audio content and metadata
        """
        try:
            logger.info(f"Synthesizing speech: '{text[:50]}...'")
            
            # Normalize language
            language_code = self._normalize_language(language)
            
            # Validate input
            if not text or len(text.strip()) == 0:
                raise ValueError("Text cannot be empty")
            
            if len(text) > 5000:
                logger.warning("Text exceeds 5000 characters, truncating")
                text = text[:5000]
            
            # Select voice
            voice = self._select_voice(language_code, gender)
            
            # Create synthesis input
            synthesis_input = SynthesisInput(text=text)
            
            # Select voice parameters
            voice_params = VoiceSelectionParams(
                language_code=language_code,
                name=voice["name"],
                ssml_gender=voice["ssml_gender"]
            )
            
            # Map output format
            audio_format_map = {
                "LINEAR16": texttospeech_v1.AudioEncoding.LINEAR16,
                "OGG_OPUS": texttospeech_v1.AudioEncoding.OGG_OPUS,
                "MULAW": texttospeech_v1.AudioEncoding.MULAW,
                "MP3": texttospeech_v1.AudioEncoding.MP3,
            }
            
            audio_encoding = audio_format_map.get(output_format, texttospeech_v1.AudioEncoding.LINEAR16)
            
            # Create audio config
            audio_config = AudioConfig(
                audio_encoding=audio_encoding,
                speaking_rate=speaking_rate,
                pitch=pitch
            )
            
            # Perform text-to-speech
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config
            )
            
            audio_content = response.audio_content
            
            logger.info(f"Speech synthesized successfully ({len(audio_content)} bytes)")
            
            return {
                "success": True,
                "audio_content": audio_content,
                "audio_content_base64": base64.b64encode(audio_content).decode('utf-8'),
                "text": text,
                "language": language,
                "language_code": language_code,
                "voice": voice["name"],
                "gender": gender,
                "audio_length_bytes": len(audio_content)
            }
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "audio_content": None
            }
    
    def synthesize_speech_to_file(
        self,
        text: str,
        output_file_path: str,
        language: str = "English",
        gender: str = "FEMALE",
        speaking_rate: float = 1.0,
        pitch: float = 0.0
    ) -> Dict:
        """
        Synthesize speech and save to file
        
        Args:
            text: Text to synthesize
            output_file_path: Path to save audio file
            language: Language of text
            gender: Voice gender
            speaking_rate: Speech rate
            pitch: Pitch adjustment
            
        Returns:
            Dictionary with file path and metadata
        """
        try:
            # Determine audio format from file extension
            extension = Path(output_file_path).suffix.lower()
            
            format_map = {
                ".wav": "LINEAR16",
                ".opus": "OGG_OPUS",
                ".mp3": "MP3",
                ".raw": "MULAW"
            }
            
            audio_format = format_map.get(extension, "LINEAR16")
            
            # Synthesize speech
            result = self.synthesize_speech(
                text=text,
                language=language,
                gender=gender,
                speaking_rate=speaking_rate,
                pitch=pitch,
                output_format=audio_format
            )
            
            if result.get("success"):
                # Write to file
                with open(output_file_path, "wb") as audio_file:
                    audio_file.write(result["audio_content"])
                
                logger.info(f"Audio saved to: {output_file_path}")
                
                result["output_file_path"] = output_file_path
                return result
            
            return result
            
        except Exception as e:
            logger.error(f"Error saving speech to file: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _normalize_language(self, language: str) -> str:
        """Normalize language name to code"""
        language_lower = language.lower().strip()
        
        if language_lower in self.language_codes:
            return self.language_codes[language_lower]
        
        if language_lower in self.language_codes.values():
            return language_lower
        
        logger.warning(f"Unknown language: {language}, defaulting to English")
        return self.language_codes.get("english", "en-IN")
    
    def _select_voice(self, language_code: str, gender: str = "FEMALE") -> Dict:
        """Select voice for language and gender"""
        voices = self.voices.get(language_code, [])
        
        if not voices:
            logger.warning(f"No voices available for {language_code}, using default")
            return {"name": f"{language_code}-Standard-A", "ssml_gender": texttospeech_v1.SsmlVoiceGender.FEMALE}
        
        # Find voice matching gender preference
        gender_map = {
            "MALE": texttospeech_v1.SsmlVoiceGender.MALE,
            "FEMALE": texttospeech_v1.SsmlVoiceGender.FEMALE,
            "NEUTRAL": texttospeech_v1.SsmlVoiceGender.NEUTRAL
        }
        
        preferred_gender = gender_map.get(gender, texttospeech_v1.SsmlVoiceGender.FEMALE)
        
        # Try to find matching gender
        for voice in voices:
            if voice["ssml_gender"] == preferred_gender:
                return voice
        
        # Return first available voice if gender not found
        return voices[0]
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get all supported languages for TTS"""
        return dict(sorted(self.language_codes.items()))
    
    def get_available_voices(self) -> Dict[str, List]:
        """Get all available voices by language"""
        return self.voices


def demonstrate_voice_integration():
    """Demonstrate voice integration"""
    
    print("\n" + "="*80)
    print("SAARTHI GOOGLE SPEECH & TEXT-TO-SPEECH INTEGRATION DEMONSTRATION")
    print("="*80 + "\n")
    
    print("Requirements:")
    print("1. Google Cloud project with Speech-to-Text and Text-to-Speech enabled")
    print("2. Service account credentials configured")
    print("\nSetup:")
    print("- pip install google-cloud-speech google-cloud-texttospeech")
    print("- export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json")
    
    example_code = '''
    # Initialize managers
    stt = SpeechToTextManager(project_id="your-project-id")
    tts = TextToSpeechManager(project_id="your-project-id")
    
    # Transcribe audio file
    result = stt.transcribe_audio_file(
        "citizen_query.wav",
        language="Tamil"
    )
    print(f"Transcription: {result['transcription']}")
    
    # Synthesize response
    response_text = "Your application has been received successfully."
    tts_result = tts.synthesize_speech_to_file(
        text=response_text,
        output_file_path="response.wav",
        language="Tamil",
        gender="FEMALE",
        speaking_rate=1.0
    )
    print(f"Audio saved to: {tts_result['output_file_path']}")
    
    # Complete voice conversation flow
    # 1. Citizen records voice query
    # 2. STT transcribes to text
    # 3. Process text with NLP/Vertex AI
    # 4. TTS converts response back to speech
    # 5. Return audio response to citizen
    '''
    
    print("\nExample usage:")
    print(example_code)
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demonstrate_voice_integration()
