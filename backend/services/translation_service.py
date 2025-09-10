import asyncio
from typing import Optional, Tuple
import requests
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import logging

# Set seed for consistent language detection
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

class TranslationService:
    def __init__(self):
        self.google_translator = GoogleTranslator()
        
        # Language code mappings
        self.language_mappings = {
            'en': 'English', 'hi': 'Hindi', 'ta': 'Tamil', 'pa': 'Punjabi',
            'bn': 'Bengali', 'te': 'Telugu', 'mr': 'Marathi', 'gu': 'Gujarati',
            'kn': 'Kannada', 'ml': 'Malayalam', 'ur': 'Urdu', 'es': 'Spanish',
            'fr': 'French', 'de': 'German', 'zh': 'Chinese', 'ja': 'Japanese',
            'ko': 'Korean', 'ar': 'Arabic'
        }
        
        # Reverse mapping
        self.reverse_language_mappings = {v: k for k, v in self.language_mappings.items()}
    
    async def detect_language(self, text: str) -> str:
        """Detect language of input text"""
        try:
            # Remove extra whitespace and check if text is meaningful
            text = text.strip()
            if len(text) < 3:
                return "English"  # Default to English for very short texts
            
            detected_code = detect(text)
            return self.language_mappings.get(detected_code, "English")
        except (LangDetectException, Exception) as e:
            logger.warning(f"Language detection failed: {e}")
            return "English"  # Default fallback
    
    async def translate_with_google(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using Google Translate via deep-translator"""
        try:
            # Convert language names to codes
            source_code = self.reverse_language_mappings.get(source_lang, source_lang.lower()[:2])
            target_code = self.reverse_language_mappings.get(target_lang, target_lang.lower()[:2])
            
            # Skip translation if same language
            if source_code == target_code:
                return text
            
            # Use deep-translator GoogleTranslator
            translator = GoogleTranslator(source=source_code, target=target_code)
            result = translator.translate(text)
            return result
        except Exception as e:
            logger.error(f"Google Translate failed: {e}")
            raise e
    
    async def translate_with_azure(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using Azure Translator (fallback)"""
        try:
            from backend.utils.config import config
            
            if not config.AZURE_TRANSLATOR_KEY:
                raise Exception("Azure Translator key not configured")
            
            # Convert to language codes
            source_code = self.reverse_language_mappings.get(source_lang, source_lang.lower()[:2])
            target_code = self.reverse_language_mappings.get(target_lang, target_lang.lower()[:2])
            
            # Skip translation if same language
            if source_code == target_code:
                return text
            
            url = "https://api.cognitive.microsofttranslator.com/translate"
            params = {
                'api-version': '3.0',
                'from': source_code,
                'to': target_code
            }
            headers = {
                'Ocp-Apim-Subscription-Key': config.AZURE_TRANSLATOR_KEY,
                'Ocp-Apim-Subscription-Region': config.AZURE_TRANSLATOR_REGION,
                'Content-type': 'application/json'
            }
            body = [{'text': text}]
            
            response = requests.post(url, params=params, headers=headers, json=body)
            response.raise_for_status()
            
            result = response.json()
            return result[0]['translations'][0]['text']
        except Exception as e:
            logger.error(f"Azure Translator failed: {e}")
            raise e
    
    async def translate(self, text: str, source_lang: str = None, target_lang: str = "English") -> Tuple[str, str]:
        """
        Main translation function with fallback mechanism
        Returns: (translated_text, detected_language)
        """
        try:
            # Detect language if not provided
            if not source_lang:
                source_lang = await self.detect_language(text)
            
            # Skip translation if already in target language
            if source_lang.lower() == target_lang.lower():
                return text, source_lang
            
            # Try Google Translate first
            try:
                translated = await self.translate_with_google(text, source_lang, target_lang)
                logger.info(f"Translation successful with Google Translate")
                return translated, source_lang
            except Exception as e:
                logger.warning(f"Google Translate failed, trying Azure: {e}")
                
                # Fallback to Azure Translator
                try:
                    translated = await self.translate_with_azure(text, source_lang, target_lang)
                    logger.info(f"Translation successful with Azure Translator")
                    return translated, source_lang
                except Exception as e:
                    logger.error(f"Azure Translator also failed: {e}")
                    # Return original text as last resort
                    return text, source_lang
                    
        except Exception as e:
            logger.error(f"Translation service failed completely: {e}")
            return text, source_lang or "English"
    
    async def batch_translate(self, texts: list, source_lang: str = None, target_lang: str = "English") -> list:
        """Translate multiple texts"""
        tasks = []
        for text in texts:
            task = self.translate(text, source_lang, target_lang)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        translated_texts = []
        for result in results:
            if isinstance(result, tuple):
                translated_texts.append(result[0])  # Get translated text
            else:
                translated_texts.append(str(result))  # Error case
        
        return translated_texts

# Global instance
translation_service = TranslationService()