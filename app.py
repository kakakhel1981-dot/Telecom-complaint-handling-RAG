import os
import tempfile
import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

# ---------------------------------------------------------
# PAGE CONFIGURATION & PROFESSIONAL STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Telecom Complaint Assistant | RAG Portal",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Professional CSS (Deep Navy/Slate corporate aesthetic with clean interactive buttons)
st.markdown("""
    <style>
    /* Main Background & Font */
    .stApp {
        background-color: #0B132B;
        color: #E0FBFC;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1C2541;
        border-right: 1px solid #3A506B;
    }
    section[data-testid="stSidebar"] .stMarkdown h1, 
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #6FFFE9;
    }

    /* Card Containers */
    .css-1r6slb0, .stChatMessage {
        background-color: #1C2541;
        border-radius: 10px;
        border: 1px solid #3A506B;
        padding: 15px;
        margin-bottom: 10px;
    }
    
    /* Headers & Titles */
    h1, h2, h3 {
        color: #6FFFE9 !important;
    }

    /* Custom Buttons */
    .stButton>button {
        background-color: #00B4D8;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0077B6;
        color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(0, 180, 216, 0.3);
    }

    /* File Uploader Box */
    div[data-testid="stFileUploader"] {
        background-color: #1C2541;
        border: 2px dashed #3A506B;
        border-radius: 10px;
        padding: 15px;
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        background-color: #1C2541;
        color: #E0FBFC;
        border: 1px solid #3A506B;
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CACHED MODELS LOADING (Performance Optimization)
# ---------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedding_model()

# ---------------------------------------------------------
# HELPER FUNCTIONS: PDF Processing & FAISS Indexing
# ---------------------------------------------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        text += f"\n--- Page {page_num + 1} ---\n" + page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    return chunks

def create_faiss_index(chunks):
    if not chunks:
        return None, None
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, chunks

# ---------------------------------------------------------
# MAIN APP HEADER & SIDEBAR CONFIGURATION
# ---------------------------------------------------------
st.markdown("# 📡 Enterprise Telecom Complaint Handling Assistant")
st.markdown("##### *AI-Powered Dispute Resolution & Policy Reference System*")
st.markdown("---")

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/server.png", width=64)
    st.markdown("### ⚙️ System Configuration")
    
    # Secure API Key configuration (hidden from main interface)
    # Automatically checks Streamlit secrets (`st.secrets["GROQ_API_KEY"]`) or environment variables, with fallback password input in sidebar.
    default_groq_key = ""
    try:
        if "GROQ_API_KEY" in st.secrets:
            default_groq_key = st.secrets["GROQ_API_KEY"]
        elif os.environ.get("GROQ_API_KEY"):
            default_groq_key = os.environ.get("GROQ_API_KEY")
    except Exception:
        pass

    groq_api_key_input = st.text_input(
        "Groq API Key", 
        value=default_groq_key, 
        type="password", 
        help="Enter your Groq API key securely. It remains hidden."
    )
    
    # Model specification hidden or minimized in sidebar settings
    selected_model = st.selectbox(
        "LLM Model Engine",
        ["openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama3-8b-8192"],
        index=0,
        help="Backend Groq LLM model engine."
    )
    
    st.markdown("---")
    st.markdown("### 📂 Document Knowledge Base")
    uploaded_file = st.file_uploader("Upload Complaint Handling Policy / SOP (PDF)", type=["pdf"])
    
    st.markdown("---")
    st.markdown("### 💡 Quick Action Shortcuts")
    quick_action_preset = st.radio(
        "Select Inquiry Template:",
        [
            "None",
            "What is the SLA for billing dispute resolutions?",
            "How should agents handle dropped call complaints?",
            "What are the escalation tiers for network outages?",
            "What compensation applies for unauthorized roaming charges?"
        ]
    )
    
    st.markdown("---")
    if st.button("🗑️ Reset Session & Memory"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Telecom Complaint Handling Assistant. Please upload your telecom policy or SOP document in the sidebar to begin, or ask me general telecom service guidelines."}
    ]

# Process uploaded document if present and not yet indexed
if uploaded_file is not None:
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.spinner("Processing and indexing telecom document via FAISS & Sentence Transformers..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            extracted_text = extract_text_from_pdf(tmp_path)
            os.unlink(tmp_path)
            
            chunks = chunk_text(extracted_text)
            index, valid_chunks = create_faiss_index(chunks)
            
            st.session_state.faiss_index = index
            st.session_state.chunks = valid_chunks
            st.session_state.current_file = uploaded_file.name
            st.success(f"Successfully indexed **{uploaded_file.name}** ({len(valid_chunks)} knowledge segments created).")

# ---------------------------------------------------------
# CHAT INTERFACE & INTERACTIVE BUTTONS
# ---------------------------------------------------------
st.markdown("### 💬 Complaint Resolution & Policy Assistant")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle quick action templates
query_input = ""
if quick_action_preset != "None":
    query_input = quick_action_preset

user_prompt = st.chat_input("Type your customer complaint query or policy question here...")

if query_input and not user_prompt:
    prompt_to_process = query_input
elif user_prompt:
    prompt_to_process = user_prompt
else:
    prompt_to_process = None

if prompt_to_process:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt_to_process})
    with st.chat_message("user"):
        st.markdown(prompt_to_process)

    # Validate API Key
    api_key_to_use = groq_api_key_input or os.environ.get("GROQ_API_KEY", "")
    if not api_key_to_use:
        error_msg = "⚠️ **Groq API Key missing.** Please provide your API key in the sidebar configuration to execute queries."
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        with st.chat_message("assistant"):
            st.markdown(error_msg)
    else:
        with st.spinner("Analyzing complaint guidelines and generating response..."):
            # Retrieval Step
            context_text = ""
            if st.session_state.faiss_index is not None and len(st.session_state.chunks) > 0:
                query_embedding = embedder.encode([prompt_to_process]).astype('float32')
                k = min(3, len(st.session_state.chunks))
                distances, indices = st.session_state.faiss_index.search(query_embedding, k)
                retrieved_chunks = [st.session_state.chunks[idx] for idx in indices[0] if idx < len(st.session_state.chunks)]
                context_text = "\n\n".join(retrieved_chunks)
            
            # Construct Prompt for Groq
            system_prompt = (
                "You are an expert Telecom Customer Complaint Handling Assistant. "
                "Your job is to guide support agents and managers according to standard telecom dispute resolution procedures, "
                "SLAs, regulatory compliance, and the provided company document context.\n\n"
                f"Document Context:\n{context_text if context_text else 'No specific document uploaded yet. Answer based on standard telecom customer care best practices.'}"
            )
            
            try:
                client = Groq(api_key=api_key_to_use)
                chat_completion = client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_to_process}
                    ],
                    temperature=0.3,
                    max_completion_tokens=1024
                )
                response_content = chat_completion.choices[0].message.content
            except Exception as e:
                response_content = f"❌ Error communicating with Groq API: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": response_content})
        with st.chat_message("assistant"):
            st.markdown(response_content)

# Interactive helper buttons at the bottom of chat
st.markdown("---")
cols = st.columns(4)
if cols[0].button("📋 Summarize Policy"):
    if st.session_state.chunks:
        summary_prompt = "Provide a comprehensive executive summary of the uploaded telecom complaint handling policy document, highlighting key SLAs, compensation rules, and escalation steps."
        st.session_state.messages.append({"role": "user", "content": summary_prompt})
        st.rerun()
    else:
        st.warning("Please upload a PDF document first.")

if cols[1].button("⚠️ Check Escalation Rules"):
    esc_prompt = "What are the specific escalation criteria and tier transitions outlined in the complaint handling guidelines?"
    st.session_state.messages.append({"role": "user", "content": esc_prompt})
    st.rerun()

if cols[2].button("⏱️ Review SLA Timelines"):
    sla_prompt = "What are the mandated SLA response and resolution timelines for different priority levels of customer complaints?"
    st.session_state.messages.append({"role": "user", "content": sla_prompt})
    st.rerun()

if cols[3].button("💬 Draft Customer Response"):
    draft_prompt = "Provide a professional email template response for an agent dealing with an escalated billing dispute."
    st.session_state.messages.append({"role": "user", "content": draft_prompt})
    st.rerun()