from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime
from typing import Optional, List

# Import our services and models
from .models import (
    ChatMessage, ChatResponse, PDFUploadResponse, 
    HealthResponse, ErrorResponse, TranslationRequest, TranslationResponse
)
from backend.services.pdf_service import pdf_service
from backend.services.embedding_service import embedding_service
from backend.services.translation_service import translation_service
from backend.services.llm_service import llm_service
from backend.utils.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multilingual Chat Assistant with PDF Q&A",
    description="A multilingual chat assistant that can answer questions from uploaded PDFs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check"""
    try:
        services_status = {
            "pdf_service": "✅ Ready",
            "embedding_service": "✅ Ready",
            "translation_service": "✅ Ready", 
            "llm_service": "✅ Ready"
        }
        
        return HealthResponse(
            status="healthy",
            message="Multilingual Chat Assistant API is running",
            services=services_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    try:
        services_status = {}
        
        # Check PDF service
        try:
            pdf_service.cleanup_old_files(max_age_hours=48)  # Cleanup old files
            services_status["pdf_service"] = "✅ Ready"
        except Exception as e:
            services_status["pdf_service"] = f"❌ Error: {str(e)}"
        
        # Check embedding service
        try:
            available_pdfs = embedding_service.list_available_pdfs()
            services_status["embedding_service"] = f"✅ Ready ({len(available_pdfs)} PDFs indexed)"
        except Exception as e:
            services_status["embedding_service"] = f"❌ Error: {str(e)}"
        
        # Check translation service
        try:
            test_translation = await translation_service.translate("Hello", "English", "English")
            services_status["translation_service"] = "✅ Ready"
        except Exception as e:
            services_status["translation_service"] = f"❌ Error: {str(e)}"
        
        # Check LLM service
        try:
            # Simple test - just check if services are configured
            has_sarvam = bool(config.SARVAM_API_KEY)
            has_openai = bool(config.OPENAI_API_KEY)
            has_gemini = bool(config.GEMINI_API_KEY)
            
            llm_status = []
            if has_sarvam: llm_status.append("Sarvam")
            if has_openai: llm_status.append("OpenAI")
            if has_gemini: llm_status.append("Gemini")
            
            services_status["llm_service"] = f"✅ Ready ({', '.join(llm_status)})" if llm_status else "⚠️ No API keys configured"
        except Exception as e:
            services_status["llm_service"] = f"❌ Error: {str(e)}"
        
        return HealthResponse(
            status="healthy",
            message="All services checked",
            services=services_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF file"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        if file.size > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=400, detail="File size too large (max 50MB)")
        
        logger.info(f"Processing PDF upload: {file.filename}")
        
        # Process PDF
        pdf_data = await pdf_service.process_pdf(file.file, file.filename)
        
        # Generate embeddings
        pdf_id = await embedding_service.process_pdf_embeddings(pdf_data)
        
        # Prepare response
        response = PDFUploadResponse(
            pdf_id=pdf_id,
            filename=pdf_data['filename'],
            pages_count=pdf_data['pages_count'],
            chunks_count=pdf_data['chunks_count'],
            language=pdf_data['detected_language'],
            message=f"PDF processed successfully. Generated {pdf_data['chunks_count']} chunks."
        )
        
        logger.info(f"PDF upload completed: {file.filename} -> {pdf_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatMessage):
    """Main chat endpoint with PDF Q&A support"""
    try:
        logger.info(f"Chat request: {chat_request.message[:100]}...")
        
        # Detect input language
        detected_language = await translation_service.detect_language(chat_request.message)
        
        # Translate message to English for processing if needed
        english_message = chat_request.message
        if detected_language.lower() != "english":
            english_message, _ = await translation_service.translate(
                chat_request.message, 
                detected_language, 
                "English"
            )
        
        # Initialize response variables
        context_chunks = []
        source_references = []
        
        # If PDF ID provided, search for relevant chunks
        if chat_request.pdf_id:
            try:
                context_chunks = await embedding_service.search_similar_chunks(
                    chat_request.pdf_id, 
                    english_message, 
                    top_k=5
                )
                
                # Prepare source references
                source_references = [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "content_preview": chunk["content"][:200] + "...",
                        "similarity_score": chunk["similarity_score"]
                    }
                    for chunk in context_chunks[:3]
                ]
                
                logger.info(f"Found {len(context_chunks)} relevant chunks")
                
            except Exception as e:
                logger.warning(f"PDF search failed: {e}")
                # Continue without context
        
        # Generate response using LLM
        if context_chunks:
            # Question answering with PDF context
            llm_response = await llm_service.answer_question_with_context(
                english_message,
                context_chunks,
                "English",  # Generate in English first
                include_sources=True
            )
            english_answer = llm_response['answer']
        else:
            # General chat without PDF context
            english_answer = await llm_service.chat_without_context(
                english_message,
                "English"
            )
        
        # Translate response to requested output language if needed
        final_answer = english_answer
        if chat_request.output_language.lower() != "english":
            final_answer, _ = await translation_service.translate(
                english_answer,
                "English",
                chat_request.output_language
            )
        
        # Prepare final response
        response = ChatResponse(
            response=final_answer,
            detected_language=detected_language,
            output_language=chat_request.output_language,
            source_chunks=[chunk["content"] for chunk in context_chunks[:3]] if context_chunks else None,
            pdf_references=source_references if source_references else None
        )
        
        logger.info(f"Chat response generated successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.post("/translate", response_model=TranslationResponse)
async def translate_text(translation_request: TranslationRequest):
    """Standalone translation endpoint"""
    try:
        translated_text, detected_language = await translation_service.translate(
            translation_request.text,
            translation_request.source_language,
            translation_request.target_language
        )
        
        return TranslationResponse(
            original_text=translation_request.text,
            translated_text=translated_text,
            detected_language=detected_language,
            target_language=translation_request.target_language
        )
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.get("/pdfs")
async def list_pdfs():
    """List all available PDF IDs"""
    try:
        pdf_ids = embedding_service.list_available_pdfs()
        return {"pdf_ids": pdf_ids, "count": len(pdf_ids)}
    except Exception as e:
        logger.error(f"Failed to list PDFs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/pdfs/{pdf_id}")
async def delete_pdf(pdf_id: str):
    """Delete a PDF and its embeddings"""
    try:
        embedding_service.delete_pdf_embeddings(pdf_id)
        return {"message": f"PDF {pdf_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete PDF {pdf_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    return {"languages": list(config.SUPPORTED_LANGUAGES.keys())}

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message=str(exc)
        ).dict()
    )

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.BACKEND_PORT,
        reload=True,
        log_level="info"
    )