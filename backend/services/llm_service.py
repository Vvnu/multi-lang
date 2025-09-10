import requests
import openai
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from backend.utils.config import config
import logging
import json
import asyncio

logger = logging.getLogger(__name__)
class LLMService:
    def __init__(self):
        # Initialize OpenAI
        if config.OPENAI_API_KEY:
            openai.api_key = config.OPENAI_API_KEY
        
        # Initialize Gemini
        if config.GEMINI_API_KEY:
            genai.configure(api_key=config.GEMINI_API_KEY)
            # Fixed: Use correct Gemini model name
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')  # Correct model name
        else:
            self.gemini_model = None
    
    def create_prompt_with_context(self, query: str, context_chunks: List[Dict[str, Any]], language: str = "English") -> str:
        """Create a well-structured prompt with context"""
        
        # Prepare context from chunks
        context_text = ""
        if context_chunks:
            context_text = "\n\n".join([
                f"Source {i+1}: {chunk['content']}"
                for i, chunk in enumerate(context_chunks[:5])  # Limit to top 5 chunks
            ])
        
        # Create the prompt
        prompt = f"""You are a helpful multilingual assistant. Answer the user's question based on the provided context.

Context from uploaded document:
{context_text if context_text else "No specific document context provided."}

User Question: {query}

Instructions:
1. ANALYZE THE CONTEXT: First, carefully examine the provided context to determine if it contains relevant information for answering the user's question.

2. INFORMATION SOURCING STRATEGY:
   - If the context contains sufficient relevant information: Use it as your primary source
   - If the context is insufficient or irrelevant: Search the internet for current and accurate information
   - If the question requires real-time data: Always perform a web search

3. RESPONSE REQUIREMENTS:
   - Provide accurate, well-researched answers
   - Be comprehensive yet concise
   - Structure your response clearly with proper formatting
   - Include specific examples when helpful

4. SOURCE ATTRIBUTION:
   - When using document context: Clearly indicate which parts of the document support your answer
   - When using web search results: Cite your sources appropriately
   - When combining sources: Distinguish between document context and external information

5. LANGUAGE AND TONE:
   - Respond in {language}
   - Use a professional but approachable tone
   - Adapt complexity to match the user's question level
   - Ensure cultural sensitivity in your response

6. QUALITY ASSURANCE:
   - Verify information accuracy before responding
   - If uncertain about any facts, clearly state your uncertainty
   - Provide disclaimers for medical, legal, or financial advice
   - Flag if the question requires expert consultation

RESPONSE FORMAT:
Begin your response directly with the answer. Structure it logically with clear paragraphs or sections as needed.

Now, please provide your response to the user's question."""

        return prompt

    
    async def query_sarvam_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        """Query Sarvam LLM API (primary)"""
        try:
            if not config.SARVAM_API_KEY:
                raise Exception("Sarvam API key not configured")
            
            # Fixed: Correct Sarvam API endpoint
            url = "https://api.sarvam.ai/chat/completions"
            headers = {
                "Authorization": f"Bearer {config.SARVAM_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "sarvam-m",  # Fixed: Correct model name
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful multilingual assistant specializing in document analysis and question answering."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()
            
            logger.info("Sarvam LLM response generated successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Sarvam LLM failed: {e}")
            raise e
    
    async def query_openai_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        """Query OpenAI GPT (fallback)"""
        try:
            if not config.OPENAI_API_KEY:
                raise Exception("OpenAI API key not configured")
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",  # or gpt-4 if you have access
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful multilingual assistant specializing in document analysis and question answering."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            
            logger.info("OpenAI LLM response generated successfully")
            return answer
            
        except Exception as e:
            logger.error(f"OpenAI LLM failed: {e}")
            raise e
    
    async def query_gemini_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        """Query Google Gemini (backup)"""
        try:
            if not self.gemini_model:
                raise Exception("Gemini API key not configured")
            
            # Configure generation settings
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.7,
            )
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            answer = response.text.strip()
            
            logger.info("Gemini LLM response generated successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Gemini LLM failed: {e}")
            raise e
    
    async def generate_response(self, query: str, context_chunks: List[Dict[str, Any]] = None, language: str = "English") -> str:
        """
        Main LLM query function with fallback mechanism
        Sarvam -> OpenAI -> Gemini
        """
        try:
            # Create prompt with context
            prompt = self.create_prompt_with_context(query, context_chunks or [], language)
            
            # Calculate max tokens (leave room for prompt)
            max_tokens = min(config.MAX_TOKENS - len(prompt.split()) * 2, 1000)
            if max_tokens < 100:
                max_tokens = 500  # Minimum reasonable response length
            
            # Try Sarvam first
            try:
                response = await self.query_sarvam_llm(prompt, max_tokens)
                logger.info("LLM response generated successfully with Sarvam")
                return response
            except Exception as e:
                logger.warning(f"Sarvam LLM failed, trying OpenAI: {e}")
            
            # Try OpenAI as fallback
            try:
                response = await self.query_openai_llm(prompt, max_tokens)
                logger.info("LLM response generated successfully with OpenAI")
                return response
            except Exception as e:
                logger.warning(f"OpenAI LLM failed, trying Gemini: {e}")
            
            # Try Gemini as backup
            try:
                response = await self.query_gemini_llm(prompt, max_tokens)
                logger.info("LLM response generated successfully with Gemini")
                return response
            except Exception as e:
                logger.error(f"Gemini LLM also failed: {e}")
                
                # Return a fallback response
                return self.generate_fallback_response(query, context_chunks, language)
                
        except Exception as e:
            logger.error(f"All LLM services failed: {e}")
            return self.generate_fallback_response(query, context_chunks, language)
    
    def generate_fallback_response(self, query: str, context_chunks: List[Dict[str, Any]], language: str) -> str:
        """Generate a fallback response when all LLM services fail"""
        
        fallback_responses = {
            "English": "I apologize, but I'm currently unable to process your question due to technical difficulties. Please try again in a few moments.",
            "Hindi": "मुझे खेद है, तकनीकी समस्याओं के कारण मैं वर्तमान में आपके प्रश्न को संसाधित नहीं कर सकता। कृपया कुछ देर बाद पुनः प्रयास करें।",
            "Tamil": "மன்னிக்கவும், தொழில்நுட்ப சிக்கல்களால் நான் தற்போது உங்கள் கேள்வியை செயலாக்க முடியவில்லை. சில நிமிடங்கள் கழித்து மீண்டும் முயற்சிக்கவும்.",
            "Spanish": "Me disculpo, pero actualmente no puedo procesar su pregunta debido a dificultades técnicas. Por favor, inténtelo de nuevo en unos momentos.",
            "French": "Je m'excuse, mais je ne peux pas traiter votre question en raison de difficultés techniques. Veuillez réessayer dans quelques instants."
        }
        
        return fallback_responses.get(language, fallback_responses["English"])
    
    async def summarize_document(self, text: str, language: str = "English") -> str:
        """Generate a summary of the document"""
        try:
            # Create summarization prompt
            prompt = f"""Please provide a comprehensive summary of the following document in {language}:

Document:
{text[:5000]}  # Limit text to avoid token limits

Instructions:
1. Provide a clear and concise summary
2. Highlight the main topics and key points
3. Organize the summary in a logical structure
4. Respond in {language}

Summary:"""
            
            response = await self.generate_response(prompt, language=language)
            return response
            
        except Exception as e:
            logger.error(f"Document summarization failed: {e}")
            return f"Unable to generate document summary in {language} at this time."
    
    async def answer_question_with_context(
        self, 
        question: str, 
        context_chunks: List[Dict[str, Any]], 
        language: str = "English",
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question with document context and return detailed response
        """
        try:
            # Generate the main response
            response = await self.generate_response(question, context_chunks, language)
            
            # Prepare source references if requested
            source_references = []
            if include_sources and context_chunks:
                for i, chunk in enumerate(context_chunks[:3]):  # Top 3 sources
                    source_references.append({
                        'source_number': i + 1,
                        'content_preview': chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'],
                        'similarity_score': chunk.get('similarity_score', 0.0),
                        'chunk_id': chunk.get('chunk_id', i)
                    })
            
            return {
                'answer': response,
                'sources': source_references,
                'context_used': len(context_chunks) > 0,
                'language': language
            }
            
        except Exception as e:
            logger.error(f"Question answering with context failed: {e}")
            return {
                'answer': self.generate_fallback_response(question, context_chunks, language),
                'sources': [],
                'context_used': False,
                'language': language,
                'error': str(e)
            }
    
    async def chat_without_context(self, message: str, language: str = "English") -> str:
        """Handle general chat without document context"""
        try:
            prompt = f"""You are a helpful multilingual assistant. Respond to the user's message naturally and conversationally.

User message: {message}

Please respond in {language} in a friendly and helpful manner.

Response:"""
            
            response = await self.generate_response(prompt, language=language)
            return response
            
        except Exception as e:
            logger.error(f"General chat failed: {e}")
            return self.generate_fallback_response(message, [], language)

# Global instance
llm_service = LLMService()