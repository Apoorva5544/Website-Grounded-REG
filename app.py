import streamlit as st
import time

# Page config to make it look premium
st.set_page_config(
    page_title="MyAdvice AI RAG Agent",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a cleaner, modern look
st.markdown("""
<style>
    /* Premium aesthetics for chat */
    .stTextInput>div>div>input {
        border-radius: 20px;
        border: 1px solid #e0e0e0;
        padding: 10px 15px;
    }
    .user-msg {
        background-color: #007AFF;
        color: white;
        padding: 10px 15px;
        border-radius: 18px 18px 0px 18px;
        margin: 10px 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    .bot-msg {
        background-color: #F1F1F1;
        color: black;
        padding: 10px 15px;
        border-radius: 18px 18px 18px 0px;
        margin: 10px 0;
        max-width: 80%;
        float: left;
        clear: both;
    }
    .source-metrics {
        font-size: 0.8em;
        color: #777;
        margin-top: 5px;
        float: left;
        clear: both;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Python Docs AI Assistant")
st.caption("A premium RAG Agent querying Python 3 documentation (v4 architecture)")

# Initialize caching for the heavy RAG Agent
@st.cache_resource(show_spinner="Initializing Model Weights from Disk... (instant startup)")
def load_agent():
    # Production deployment uses the totally separated pure-query app
    import query
    return query.query_engine

# Load agent
agent = load_agent()

# Chat history state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        if "meta" in msg and msg["meta"]["sources"]:
            sources = ", ".join(f'<a href="{s}" target="_blank">Doc</a>' for s in msg["meta"]["sources"])
            st.markdown(f'<div class="source-metrics">🔗 Sources: {sources}</div>', unsafe_allow_html=True)

st.markdown("<div style='clear:both'></div>", unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a question about Python..."):
    # Immediately show user question
    st.markdown(f'<div class="user-msg">{prompt}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        # We simulate the exact invoke structure defined in rag_agent.py
        result = agent.invoke({
            "question": prompt,
            "retrieved_docs": [],
            "reranked_docs": [],
            "reranker_scores": [],
            "answerable": False,
            "answer": "",
            "sources": []
        })
        
        answer = result.get("answer", "Error encountered")
        sources = result.get("sources", [])
        
        st.markdown(f'<div class="bot-msg">{answer}</div>', unsafe_allow_html=True)
        
        if sources:
            source_links = ", ".join(f'<a href="{s}" target="_blank">Doc</a>' for s in sources)
            st.markdown(f'<div class="source-metrics">🔗 Sources: {source_links}</div>', unsafe_allow_html=True)
            
        st.session_state.messages.append({"role": "assistant", "content": answer, "meta": {"sources": sources}})
        
    st.markdown("<div style='clear:both'></div>", unsafe_allow_html=True)
