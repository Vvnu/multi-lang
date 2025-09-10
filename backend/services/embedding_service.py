import os
import numpy as np
import faiss
import pickle
import requests
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import openai
from backend.utils.config import config
import logging
import asyncio
import json

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.vector_db_path = config.VECTOR_DB_PATH
        self.local_model = None
        self.indexes = {}  # Store FAISS indexes for different PDFs
        self.metadata = {}  # Store metadata for chunks
        
        # Ensure vector DB directory exists
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        # Initialize OpenAI client
        if config.OPENAI_API_KEY:
            openai.api_key = config.OPENAI_API_KEY
    
    def load_local_model(self):
        """Load local sentence transformer model (fallback)"""
        try:
            if self.local_model is None:
                self.local_model = SentenceTransformer(config.DEFAULT_EMBEDDING_MODEL)
                logger.info(f"Loaded local embedding model: {config.DEFAULT_EMBEDDING_MODEL}")
            return self.local_model
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise e
    
    async def get_sarvam_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from Sarvam API (primary)"""
        try:
            if not config.SARVAM_API_KEY:
                raise Exception("Sarvam API key not configured")
            
            url = "https://api.sarvam.ai/text-embedding"
            headers = {
                "Authorization": f"Bearer {config.SARVAM_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Sarvam API might have batch limits, so process in chunks
            all_embeddings = []
            batch_size = 10  # Adjust based on API limits
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                data = {
                    "model": "sarvam-embed-1",  # Check Sarvam docs for exact model name
                    "texts": batch
                }
                
                response = requests.post(url, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                batch_embeddings = result.get("embeddings", [])
                all_embeddings.extend(batch_embeddings)
            
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            logger.info(f"Sarvam embeddings generated for {len(texts)} texts")
            return embeddings_array
            
        except Exception as e:
            logger.error(f"Sarvam embeddings failed: {e}")
            raise e
    
    async def get_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from OpenAI API (fallback)"""
        try:
            if not config.OPENAI_API_KEY:
                raise Exception("OpenAI API key not configured")
            
            # OpenAI has batch limits
            all_embeddings = []
            batch_size = 100  # OpenAI allows larger batches
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = openai.embeddings.create(
                    model="text-embedding-3-small",  # or text-embedding-ada-002
                    input=batch
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            logger.info(f"OpenAI embeddings generated for {len(texts)} texts")
            return embeddings_array
            
        except Exception as e:
            logger.error(f"OpenAI embeddings failed: {e}")
            raise e
    
    async def get_local_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from local model (backup)"""
        try:
            model = self.load_local_model()
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            logger.info(f"Local embeddings generated for {len(texts)} texts")
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Local embeddings failed: {e}")
            raise e
    
    async def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Main embedding generation with fallback mechanism
        Sarvam -> OpenAI -> Local Model
        """
        if not texts:
            return np.array([], dtype=np.float32)
        
        # Try Sarvam first
        try:
            embeddings = await self.get_sarvam_embeddings(texts)
            logger.info("Embeddings generated successfully with Sarvam")
            return embeddings
        except Exception as e:
            logger.warning(f"Sarvam embeddings failed, trying OpenAI: {e}")
        
        # Try OpenAI as fallback
        try:
            embeddings = await self.get_openai_embeddings(texts)
            logger.info("Embeddings generated successfully with OpenAI")
            return embeddings
        except Exception as e:
            logger.warning(f"OpenAI embeddings failed, trying local model: {e}")
        
        # Use local model as backup
        try:
            embeddings = await self.get_local_embeddings(texts)
            logger.info("Embeddings generated successfully with local model")
            return embeddings
        except Exception as e:
            logger.error(f"All embedding methods failed: {e}")
            raise Exception("Failed to generate embeddings with all available methods")
    
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create FAISS index for similarity search"""
        try:
            dimension = embeddings.shape[1]
            
            # Use IndexFlatIP for inner product (cosine similarity)
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
            
            logger.info(f"FAISS index created with {embeddings.shape[0]} vectors")
            return index
        except Exception as e:
            logger.error(f"FAISS index creation failed: {e}")
            raise e
    
    def save_embeddings(self, pdf_id: str, embeddings: np.ndarray, chunks: List[str], metadata: Dict):
        """Save embeddings and metadata to disk"""
        try:
            # Create FAISS index
            index = self.create_faiss_index(embeddings)
            
            # Save FAISS index
            index_path = os.path.join(self.vector_db_path, f"{pdf_id}.index")
            faiss.write_index(index, index_path)
            
            # Save metadata
            metadata_path = os.path.join(self.vector_db_path, f"{pdf_id}.metadata")
            full_metadata = {
                'chunks': chunks,
                'metadata': metadata,
                'embedding_count': len(embeddings),
                'dimension': embeddings.shape[1]
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(full_metadata, f)
            
            # Store in memory for quick access
            self.indexes[pdf_id] = index
            self.metadata[pdf_id] = full_metadata
            
            logger.info(f"Embeddings saved for PDF {pdf_id}")
            
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            raise e
    
    def load_embeddings(self, pdf_id: str) -> Tuple[faiss.Index, Dict]:
        """Load embeddings and metadata from disk"""
        try:
            # Check if already in memory
            if pdf_id in self.indexes and pdf_id in self.metadata:
                return self.indexes[pdf_id], self.metadata[pdf_id]
            
            # Load from disk
            index_path = os.path.join(self.vector_db_path, f"{pdf_id}.index")
            metadata_path = os.path.join(self.vector_db_path, f"{pdf_id}.metadata")
            
            if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
                raise Exception(f"Embeddings not found for PDF {pdf_id}")
            
            # Load FAISS index
            index = faiss.read_index(index_path)
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Store in memory
            self.indexes[pdf_id] = index
            self.metadata[pdf_id] = metadata
            
            logger.info(f"Embeddings loaded for PDF {pdf_id}")
            return index, metadata
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise e
    
    async def process_pdf_embeddings(self, pdf_data: Dict[str, Any]) -> str:
        """Process PDF and create embeddings"""
        try:
            pdf_id = pdf_data['pdf_id']
            chunks = pdf_data['chunks']
            
            # Generate embeddings for chunks
            embeddings = await self.generate_embeddings(chunks)
            
            # Prepare metadata
            metadata = {
                'filename': pdf_data['filename'],
                'pages_count': pdf_data['pages_count'],
                'chunks_count': len(chunks),
                'detected_language': pdf_data['detected_language'],
                'content_hash': pdf_data['content_hash']
            }
            
            # Save embeddings
            self.save_embeddings(pdf_id, embeddings, chunks, metadata)
            
            return pdf_id
            
        except Exception as e:
            logger.error(f"PDF embedding processing failed: {e}")
            raise e
    
    async def search_similar_chunks(self, pdf_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks in the PDF"""
        try:
            # Load embeddings if not in memory
            index, metadata = self.load_embeddings(pdf_id)
            
            # Generate query embedding
            query_embedding = await self.generate_embeddings([query])
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search similar chunks
            scores, indices = index.search(query_embedding, top_k)
            
            # Prepare results
            results = []
            chunks = metadata['chunks']
            
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(chunks):  # Valid index
                    results.append({
                        'chunk_id': int(idx),
                        'content': chunks[idx],
                        'similarity_score': float(score),
                        'rank': i + 1
                    })
            
            logger.info(f"Found {len(results)} similar chunks for query in PDF {pdf_id}")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def list_available_pdfs(self) -> List[str]:
        """List all available PDF IDs"""
        try:
            pdf_ids = []
            for file in os.listdir(self.vector_db_path):
                if file.endswith('.index'):
                    pdf_id = file.replace('.index', '')
                    pdf_ids.append(pdf_id)
            return pdf_ids
        except Exception as e:
            logger.error(f"Failed to list PDFs: {e}")
            return []
    
    def delete_pdf_embeddings(self, pdf_id: str):
        """Delete embeddings for a PDF"""
        try:
            # Remove from memory
            if pdf_id in self.indexes:
                del self.indexes[pdf_id]
            if pdf_id in self.metadata:
                del self.metadata[pdf_id]
            
            # Remove from disk
            index_path = os.path.join(self.vector_db_path, f"{pdf_id}.index")
            metadata_path = os.path.join(self.vector_db_path, f"{pdf_id}.metadata")
            
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            logger.info(f"Embeddings deleted for PDF {pdf_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")

# Global instance
embedding_service = EmbeddingService()