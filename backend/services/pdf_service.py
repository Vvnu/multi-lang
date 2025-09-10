import os
import uuid
import hashlib
from typing import List, Dict, Any, Tuple
import PyPDF2
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from backend.utils.config import config
from backend.services.translation_service import translation_service
import logging
import asyncio

logger = logging.getLogger(__name__)

class PDFService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Ensure upload directory exists
        os.makedirs(config.PDF_UPLOAD_PATH, exist_ok=True)
    
    async def extract_text_pypdf2(self, file_path: str) -> Tuple[str, int]:
        """Extract text using PyPDF2 (fallback method)"""
        try:
            text = ""
            pages_count = 0
            
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                pages_count = len(reader.pages)
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num + 1}: {e}")
            
            return text.strip(), pages_count
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            raise e
    
    async def extract_text_pdfplumber(self, file_path: str) -> Tuple[str, int]:
        """Extract text using pdfplumber (primary method)"""
        try:
            text = ""
            pages_count = 0
            
            with pdfplumber.open(file_path) as pdf:
                pages_count = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num + 1}: {e}")
            
            return text.strip(), pages_count
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            raise e
    
    async def extract_text_from_pdf(self, file_path: str) -> Tuple[str, int]:
        """
        Extract text from PDF with fallback mechanism
        Returns: (extracted_text, pages_count)
        """
        try:
            # Try pdfplumber first (better for complex layouts)
            text, pages_count = await self.extract_text_pdfplumber(file_path)
            if text and len(text.strip()) > 50:  # Reasonable amount of text
                logger.info("Text extraction successful with pdfplumber")
                return text, pages_count
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")
        
        try:
            # Fallback to PyPDF2
            text, pages_count = await self.extract_text_pypdf2(file_path)
            if text and len(text.strip()) > 0:
                logger.info("Text extraction successful with PyPDF2")
                return text, pages_count
        except Exception as e:
            logger.error(f"PyPDF2 also failed: {e}")
        
        raise Exception("Failed to extract text from PDF with all methods")
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks for embeddings"""
        try:
            chunks = self.text_splitter.split_text(text)
            # Filter out very short chunks
            filtered_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 20]
            return filtered_chunks
        except Exception as e:
            logger.error(f"Text splitting failed: {e}")
            return [text]  # Return original text as single chunk
    
    def generate_pdf_id(self, filename: str, content_hash: str) -> str:
        """Generate unique PDF ID"""
        return hashlib.md5(f"{filename}_{content_hash}".encode()).hexdigest()
    
    def calculate_content_hash(self, content: str) -> str:
        """Calculate hash of PDF content"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def save_uploaded_file(self, uploaded_file, filename: str) -> str:
        """Save uploaded file to disk"""
        try:
            # Generate unique filename to avoid conflicts
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(config.PDF_UPLOAD_PATH, unique_filename)
            
            # Save file
            with open(file_path, "wb") as f:
                content = await uploaded_file.read()
                f.write(content)
            
            return file_path
        except Exception as e:
            logger.error(f"File save failed: {e}")
            raise e
    
    async def process_pdf(self, uploaded_file, filename: str) -> Dict[str, Any]:
        """
        Main PDF processing function
        Returns: Dictionary with PDF processing results
        """
        file_path = None
        try:
            # Save uploaded file
            file_path = await self.save_uploaded_file(uploaded_file, filename)
            
            # Extract text from PDF
            text, pages_count = await self.extract_text_from_pdf(file_path)
            
            if not text or len(text.strip()) < 10:
                raise Exception("PDF appears to be empty or contains no extractable text")
            
            # Calculate content hash
            content_hash = self.calculate_content_hash(text)
            
            # Generate PDF ID
            pdf_id = self.generate_pdf_id(filename, content_hash)
            
            # Detect language
            detected_language = await translation_service.detect_language(text[:1000])  # First 1000 chars
            
            # Split text into chunks
            chunks = self.split_text_into_chunks(text)
            
            # Prepare response
            result = {
                'pdf_id': pdf_id,
                'filename': filename,
                'file_path': file_path,
                'pages_count': pages_count,
                'chunks_count': len(chunks),
                'detected_language': detected_language,
                'text': text,
                'chunks': chunks,
                'content_hash': content_hash
            }
            
            logger.info(f"PDF processing completed: {filename} -> {pdf_id}")
            return result
            
        except Exception as e:
            # Cleanup file if processing failed
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
            
            logger.error(f"PDF processing failed for {filename}: {e}")
            raise e
    
    def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old PDF files (optional maintenance function)"""
        try:
            import time
            current_time = time.time()
            
            for filename in os.listdir(config.PDF_UPLOAD_PATH):
                file_path = os.path.join(config.PDF_UPLOAD_PATH, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > (max_age_hours * 3600):  # Convert hours to seconds
                        os.remove(file_path)
                        logger.info(f"Cleaned up old file: {filename}")
        except Exception as e:
            logger.error(f"File cleanup failed: {e}")

# Global instance
pdf_service = PDFService()