from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class ChatMessage(BaseModel):
    message: str
    pdf_id: Optional[str] = None
    output_language: str = "English"
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    detected_language: str
    output_language: str
    source_chunks: Optional[List[str]] = None
    pdf_references: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = datetime.now()

class PDFUploadResponse(BaseModel):
    pdf_id: str
    filename: str
    pages_count: int
    chunks_count: int
    language: str
    message: str
    timestamp: datetime = datetime.now()

class HealthResponse(BaseModel):
    status: str
    message: str
    services: Dict[str, str]
    timestamp: datetime = datetime.now()

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: datetime = datetime.now()

class TranslationRequest(BaseModel):
    text: str
    source_language: Optional[str] = None
    target_language: str = "English"

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    detected_language: str
    target_language: str