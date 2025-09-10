Multilingual Chat Assistant 🌍🤖
A powerful AI-powered chat assistant that can understand and respond in multiple languages, with document processing capabilities and web search integration.
✨ Features

Multi-language Support: Communicate in 50+ languages
Document Processing: Upload and analyze PDF
Web Search Integration: Real-time information retrieval
Context-Aware Responses: Maintains conversation history
Smart Source Attribution: Cites document sources and web results
Modern UI: Clean, responsive interface built with Streamlit
Real-time Translation: Instant language switching during conversations

🚀 Quick Start
Prerequisites

Python 3.8+
Streamlit
API keys for your chosen AI service (OpenAI, Gemini, etc.)

Installation

Clone the repository
git clone https://github.com/Vvnu/multi-lang.git
cd multi-lang

Set up Python environment
python -m venv venv
On Windows: venv\Scripts\activate
pip install -r requirements.txt

To run :
FrontEnd: python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload   #from the root folder like the multi-lang folder
Backend: streamlit run app.py --server.port 8501   # from the frontend folder
