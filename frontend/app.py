import streamlit as st
import requests
import json
from datetime import datetime
import time
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Multilingual Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API URL
BACKEND_URL = "http://localhost:8000"

# Supported languages
LANGUAGES = [
    "English", "Hindi", "Tamil", "Punjabi", "Bengali", "Telugu",
    "Marathi", "Gujarati", "Kannada", "Malayalam", "Urdu",
    "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Arabic"
]

def init_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_pdf_id' not in st.session_state:
        st.session_state.current_pdf_id = None
    if 'current_pdf_name' not in st.session_state:
        st.session_state.current_pdf_name = None
    if 'output_language' not in st.session_state:
        st.session_state.output_language = "English"

def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_pdf(uploaded_file):
    """Upload PDF to backend"""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        
        with st.spinner(f"Processing {uploaded_file.name}..."):
            response = requests.post(f"{BACKEND_URL}/upload_pdf", files=files)
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            st.error(f"Upload failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return None

def send_chat_message(message, pdf_id=None, output_language="English"):
    """Send chat message to backend"""
    try:
        payload = {
            "message": message,
            "pdf_id": pdf_id,
            "output_language": output_language
        }
        
        response = requests.post(f"{BACKEND_URL}/chat", json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Chat error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Communication error: {str(e)}")
        return None

def display_chat_message(message, is_user=True):
    """Display a chat message"""
    if is_user:
        with st.chat_message("user"):
            st.write(message)
    else:
        with st.chat_message("assistant"):
            st.write(message)

def display_sources(sources):
    """Display source references"""
    if sources:
        st.subheader("📚 Sources")
        for i, source in enumerate(sources):
            with st.expander(f"Source {i+1} (Similarity: {source.get('similarity_score', 0):.2f})"):
                st.write(source.get('content_preview', 'No preview available'))

def main():
    """Main Streamlit application"""
    init_session_state()
    
    # App header
    st.title("🤖 Multilingual Chat Assistant")
    st.markdown("*Chat in any language and get answers from your PDFs!*")
    
    # Check backend health
    if not check_backend_health():
        st.error("❌ Backend is not running! Please start the FastAPI backend first.")
        st.code("cd backend && python main.py")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Language selection
        output_language = st.selectbox(
            "Output Language",
            LANGUAGES,
            index=LANGUAGES.index(st.session_state.output_language)
        )
        st.session_state.output_language = output_language
        
        st.divider()
        
        # PDF Upload Section
        st.header("📄 PDF Upload")
        
        uploaded_file = st.file_uploader(
            "Upload a PDF file",
            type=['pdf'],
            help="Upload a PDF to ask questions about its content"
        )
        
        if uploaded_file is not None:
            if st.button("Process PDF"):
                result = upload_pdf(uploaded_file)
                if result:
                    st.session_state.current_pdf_id = result['pdf_id']
                    st.session_state.current_pdf_name = result['filename']
                    st.success(f"✅ PDF processed successfully!")
                    st.json({
                        "Filename": result['filename'],
                        "Pages": result['pages_count'],
                        "Chunks": result['chunks_count'],
                        "Language": result['language']
                    })
        
        # Current PDF info
        if st.session_state.current_pdf_id:
            st.success(f"📄 Current PDF: {st.session_state.current_pdf_name}")
            if st.button("Clear PDF"):
                st.session_state.current_pdf_id = None
                st.session_state.current_pdf_name = None
                st.rerun()
        
        st.divider()
        
        # Chat controls
        st.header("💬 Chat Controls")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        # App info
        st.divider()
        st.header("ℹ️ About")
        st.info(
            "This app supports multilingual chat with PDF document analysis. "
            "You can chat in any supported language and get responses in your preferred language."
        )
        
        # Backend status
        try:
            health_response = requests.get(f"{BACKEND_URL}/health")
            if health_response.status_code == 200:
                health_data = health_response.json()
                st.success("✅ Backend: Online")
                with st.expander("Service Status"):
                    for service, status in health_data['services'].items():
                        st.text(f"{service}: {status}")
        except:
            st.error("❌ Backend: Offline")
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message['content'], message['is_user'])
        
        # Show sources if available
        if not message['is_user'] and 'sources' in message:
            display_sources(message['sources'])
    
    # Chat input
    user_input = st.chat_input("Type your message in any language...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({
            'content': user_input,
            'is_user': True,
            'timestamp': datetime.now()
        })
        
        # Display user message
        display_chat_message(user_input, is_user=True)
        
        # Send message to backend and get response
        with st.spinner("Thinking..."):
            response = send_chat_message(
                user_input,
                st.session_state.current_pdf_id,
                st.session_state.output_language
            )
        
        if response:
            # Add assistant response to chat history
            assistant_message = {
                'content': response['response'],
                'is_user': False,
                'timestamp': datetime.now(),
                'detected_language': response.get('detected_language'),
                'sources': response.get('pdf_references')
            }
            
            st.session_state.messages.append(assistant_message)
            
            # Display assistant response
            display_chat_message(response['response'], is_user=False)
            
            # Show language info
            if response.get('detected_language') != st.session_state.output_language:
                st.info(f"🌐 Detected input language: {response.get('detected_language')}")
            
            # Show sources
            if response.get('pdf_references'):
                display_sources(response['pdf_references'])
            
            # Auto-rerun to update the chat
            st.rerun()
    
    # Example queries section
    if not st.session_state.messages:
        st.header("💡 Example Queries")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("General Chat")
            if st.button("Hello! How are you?"):
                st.session_state.messages.append({
                    'content': "Hello! How are you?",
                    'is_user': True,
                    'timestamp': datetime.now()
                })
                st.rerun()
        
        with col2:
            st.subheader("Multilingual")
            if st.button("नमस्ते, आप कैसे हैं?"):
                st.session_state.messages.append({
                    'content': "नमस्ते, आप कैसे हैं?",
                    'is_user': True,
                    'timestamp': datetime.now()
                })
                st.rerun()
        
        with col3:
            st.subheader("PDF Questions")
            if st.button("What is this document about?"):
                if st.session_state.current_pdf_id:
                    st.session_state.messages.append({
                        'content': "What is this document about?",
                        'is_user': True,
                        'timestamp': datetime.now()
                    })
                    st.rerun()
                else:
                    st.warning("Please upload a PDF first!")
    
    # Footer
    st.divider()
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Multilingual Chat Assistant with PDF Q&A | "
        f"Backend: {BACKEND_URL} | "
        f"Language: {st.session_state.output_language}"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()