import os


os.environ["GROQ_API_KEY"] = GROQ_API_KEY
import streamlit as st
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
import time
import re

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# ─── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stChatMessage { border-radius: 10px; margin: 5px 0; }
    .stat-box {
        background: #1e2130;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #2e3250;
    }
    .stat-number { font-size: 28px; font-weight: bold; color: #4CAF50; }
    .stat-label { font-size: 12px; color: #888; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────
defaults = {
    "messages": [],
    "pdf_text": "",
    "pdf_name": "",
    "pdf_pages": 0,
    "pdf_words": 0,
    "total_questions": 0,
    "model": "llama-3.1-8b-instant",
    "temperature": 0.7,
    "summary": ""
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─── Helper Functions ──────────────────────────────────────────
def get_llm():
    return ChatGroq(
        model=st.session_state.model,
        api_key=GROQ_API_KEY,
        temperature=st.session_state.temperature
    )

def count_words(text):
    return len(re.findall(r'\w+', text))

def generate_summary(text):
    llm = get_llm()
    response = llm.invoke(f"Summarize this document in 5 bullet points:\n\n{text[:6000]}")
    return response.content

def generate_quiz(text):
    llm = get_llm()
    response = llm.invoke(f"Generate 5 multiple choice questions from this document. Format each as Q: ... A) ... B) ... C) ... D) ... Answer: ...\n\n{text[:6000]}")
    return response.content

# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 PDF AI Assistant")
    st.divider()
    
    # Model selector
    st.subheader("⚙️ Settings")
    st.session_state.model = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        help="Larger models are smarter but slower"
    )
    st.session_state.temperature = st.slider(
        "Creativity", 0.0, 1.0, 0.7,
        help="Higher = more creative, Lower = more factual"
    )
    st.divider()

    # File uploader
    st.subheader("📂 Upload PDF")
    pdf = st.file_uploader("Choose a PDF", type="pdf", label_visibility="collapsed")
    
    if pdf and pdf.name != st.session_state.pdf_name:
        with st.spinner("Processing PDF..."):
            reader = PdfReader(pdf)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            st.session_state.pdf_text = text
            st.session_state.pdf_name = pdf.name
            st.session_state.pdf_pages = len(reader.pages)
            st.session_state.pdf_words = count_words(text)
            st.session_state.messages = []
            st.session_state.summary = ""
        st.success(f"✅ {pdf.name}")
    
    if st.session_state.pdf_text:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""<div class="stat-box">
                <div class="stat-number">{st.session_state.pdf_pages}</div>
                <div class="stat-label">Pages</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="stat-box">
                <div class="stat-number">{st.session_state.pdf_words:,}</div>
                <div class="stat-label">Words</div></div>""", unsafe_allow_html=True)
        
        st.divider()
        col3, col4 = st.columns(2)
        with col3:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col4:
            if st.button("📥 Export", use_container_width=True):
                chat_export = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
                st.download_button("Download", chat_export, "chat_history.txt", use_container_width=True)

# ─── Main Area ─────────────────────────────────────────────────
if not st.session_state.pdf_text:
    st.markdown("## 👋 Welcome to PDF AI Assistant")
    st.markdown("Upload a PDF in the sidebar to get started.")
    st.markdown("**Features:**")
    st.markdown("- 💬 Chat with your PDF\n- 📋 Auto-summarize\n- ❓ Quiz generator\n- 📥 Export chat\n- 🧠 Multiple AI models")
else:
    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📋 Summary", "❓ Quiz"])
    
    # ── Chat Tab ──
    with tab1:
        st.markdown(f"### Chatting with: `{st.session_state.pdf_name}`")
        
        # Quick question buttons
        st.markdown("**Quick questions:**")
        cols = st.columns(3)
        quick_questions = [
            "What is this document about?",
            "What are the key points?",
            "Give me the main conclusions."
        ]
        for i, q in enumerate(quick_questions):
            if cols[i].button(q, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                with st.spinner("Thinking..."):
                    llm = get_llm()
                    prompt = f"Document:\n{st.session_state.pdf_text[:8000]}\n\nQuestion: {q}"
                    answer = llm.invoke(prompt)
                    st.session_state.messages.append({"role": "assistant", "content": answer.content})
                    st.session_state.total_questions += 1
                st.rerun()
        
        st.divider()
        
        # Chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        question = st.chat_input("Ask anything about your PDF...")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    llm = get_llm()
                    history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]])
                    prompt = f"""You are a helpful assistant. Answer based on this document.

Document:
{st.session_state.pdf_text[:8000]}

History:
{history}

Question: {question}"""
                    answer = llm.invoke(prompt)
                    st.write(answer.content)
                    st.session_state.messages.append({"role": "assistant", "content": answer.content})
                    st.session_state.total_questions += 1

    # ── Summary Tab ──
    with tab2:
        st.markdown("### 📋 Document Summary")
        if not st.session_state.summary:
            if st.button("✨ Generate Summary", use_container_width=True):
                with st.spinner("Summarizing..."):
                    st.session_state.summary = generate_summary(st.session_state.pdf_text)
                st.rerun()
        else:
            st.markdown(st.session_state.summary)
            if st.button("🔄 Regenerate", use_container_width=True):
                with st.spinner("Regenerating..."):
                    st.session_state.summary = generate_summary(st.session_state.pdf_text)
                st.rerun()

    # ── Quiz Tab ──
    with tab3:
        st.markdown("### ❓ Quiz Generator")
        st.markdown("Test your understanding of the document!")
        if st.button("🎯 Generate Quiz", use_container_width=True):
            with st.spinner("Generating quiz..."):
                quiz = generate_quiz(st.session_state.pdf_text)
            st.markdown(quiz)