import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
    
    # Azure Translator
    AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
    AZURE_TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION")
    
    # Application Settings
    BACKEND_PORT = int(os.getenv("BACKEND_PORT", 8000))
    FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", 8501))
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vectors")
    PDF_UPLOAD_PATH = os.getenv("PDF_UPLOAD_PATH", "./data/pdfs")
    
    # Model Settings
    DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 4000))
    
    # Language Settings
    SUPPORTED_LANGUAGES = {
        "English": "en",
        "Hindi": "hi", 
        "Tamil": "ta",
        "Punjabi": "pa",
        "Bengali": "bn",
        "Telugu": "te",
        "Marathi": "mr",
        "Gujarati": "gu",
        "Kannada": "kn",
        "Malayalam": "ml",
        "Urdu": "ur",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Chinese": "zh",
        "Japanese": "ja",
        "Korean": "ko",
        "Arabic": "ar"
    }

config = Config()