
import streamlit as st

try:
    from rag_chatbot import answer_question
except Exception as exc:
    answer_question = None
    backend_import_error = str(exc)
else:
    backend_import_error = None

st.set_page_config(
    page_title="Nexus AI | RAG Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium UI Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #080b14;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background: #0d1220;
    border-right: 1px solid #202a40;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 25px 18px;
}

.brand {
    font-size: 27px;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

.brand-sub {
    color: #8491a8;
    font-size: 12px;
    margin-bottom: 35px;
}

.sidebar-label {
    color: #8290aa;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin: 25px 0 12px;
}

.side-card {
    background: #151d30;
    border: 1px solid #27334d;
    padding: 13px;
    border-radius: 12px;
    color: #dce7ff;
    font-size: 13px;
    margin-bottom: 10px;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1.8px;
    text-align: center;
    margin-top: 35px;
    background: linear-gradient(90deg, #ffffff, #8b9cff, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #8491a8;
    font-size: 14px;
    margin-bottom: 35px;
}

.status {
    display: inline-block;
    background: #102b28;
    color: #4ade80;
    border: 1px solid #1d5547;
    border-radius: 30px;
    padding: 7px 14px;
    font-size: 11px;
    font-weight: 600;
}

.feature-card {
    background: linear-gradient(145deg, #141d32, #101626);
    border: 1px solid #273653;
    border-radius: 17px;
    padding: 20px;
    min-height: 135px;
    transition: 0.3s;
}

.feature-card:hover {
    border-color: #647cff;
    transform: translateY(-3px);
}

.feature-icon {
    font-size: 25px;
    margin-bottom: 12px;
}

.feature-title {
    font-size: 14px;
    font-weight: 700;
    color: #e4eaff;
}

.feature-desc {
    color: #8d9bb5;
    font-size: 12px;
    margin-top: 7px;
}

.stChatInput textarea {
    background: #141c2d !important;
    color: white !important;
}

.stChatInput {
    border-color: #40547b !important;
}

[data-testid="stChatMessage"] {
    background: #111a2b;
    border: 1px solid #25334d;
    border-radius: 15px;
    padding: 14px;
    margin-bottom: 12px;
}

.stButton button {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 18px;
}

.stButton button:hover {
    border: 1px solid #93aaff;
    color: white;
}

.footer {
    text-align: center;
    color: #596780;
    font-size: 11px;
    margin-top: 45px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="brand">✦ NEXUS AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">INTELLIGENT KNOWLEDGE ASSISTANT</div>', unsafe_allow_html=True)

    st.markdown('<div class="status">● SYSTEM ONLINE</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">WORKSPACE</div>', unsafe_allow_html=True)

    if st.button("＋  New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown('<div class="sidebar-label">KNOWLEDGE BASE</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="side-card">
        📄 Document Intelligence<br>
        <small style="color:#8796b2">PDF knowledge retrieval</small>
    </div>
    <div class="side-card">
        🧠 Vector Database<br>
        <small style="color:#8796b2">Qdrant Local Storage</small>
    </div>
    <div class="side-card">
        ⚡ AI Engine<br>
        <small style="color:#8796b2">NVIDIA Nemotron</small>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">PROJECT</div>', unsafe_allow_html=True)
    st.caption("RAG Based AI Assistant")
    st.caption("Built with Python & Streamlit")

# Main Interface
st.markdown('<div class="main-title">Your Knowledge, Reimagined.</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask questions. Discover insights. Get intelligent answers from your documents.</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📚</div>
        <div class="feature-title">Document Intelligence</div>
        <div class="feature-desc">Retrieve information from your uploaded knowledge sources.</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔍</div>
        <div class="feature-title">Semantic Search</div>
        <div class="feature-desc">Find contextually relevant information with vector search.</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">✨</div>
        <div class="feature-title">AI Responses</div>
        <div class="feature-desc">Generate answers grounded in your document content.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 💬  AI Conversation")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask anything about your documents...")

if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if backend_import_error:
                response = f"Backend startup error: {backend_import_error}"
            else:
                try:
                    response = answer_question(prompt)
                except Exception as exc:
                    response = f"Backend error: {exc}"
            st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

st.markdown("""
<div class="footer">
    NEXUS AI • POWERED BY RETRIEVAL AUGMENTED GENERATION • BUILT WITH STREAMLIT
</div>
""", unsafe_allow_html=True)
