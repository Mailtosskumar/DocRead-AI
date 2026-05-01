import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="DocMind AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #f8fafc; }
    .hero { background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%); border-radius: 16px; padding: 2.5rem 2rem; color: white; margin-bottom: 1.5rem; text-align: center; }
    .hero h1 { font-size: 2.4rem; font-weight: 700; margin: 0 0 0.4rem 0; }
    .hero p  { font-size: 1.05rem; opacity: 0.85; margin: 0; }
    .badge { display: inline-block; background: rgba(255,255,255,0.18); border-radius: 20px; padding: 4px 14px; font-size: 0.78rem; font-weight: 600; margin-bottom: 0.9rem; }
    .card { background: white; border-radius: 12px; padding: 1.4rem 1.6rem; box-shadow: 0 1px 4px rgba(0,0,0,0.07); margin-bottom: 1rem; }
    .answer-box { background: #eff6ff; border-left: 4px solid #2563eb; border-radius: 0 10px 10px 0; padding: 1.2rem 1.4rem; font-size: 1rem; line-height: 1.7; color: #1e293b; margin-top: 0.8rem; }
    .source-chip { display: inline-block; background: #e0e7ff; color: #3730a3; border-radius: 20px; padding: 3px 12px; font-size: 0.78rem; font-weight: 600; margin: 3px 4px 3px 0; }
    .source-text { background: #f1f5f9; border-radius: 8px; padding: 0.7rem 1rem; font-size: 0.82rem; color: #475569; margin-top: 0.4rem; border-left: 3px solid #94a3b8; font-style: italic; line-height: 1.5; }
    .tech-pill { display: inline-block; background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; border-radius: 20px; padding: 3px 11px; font-size: 0.75rem; font-weight: 600; margin: 2px; }
    .stat-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.9rem; text-align: center; }
    .stat-num { font-size: 1.5rem; font-weight: 700; color: #2563eb; }
    .stat-lbl { font-size: 0.75rem; color: #64748b; margin-top: 2px; }
    .stButton > button { background: linear-gradient(135deg, #2563eb, #1d4ed8); color: white; border: none; border-radius: 8px; padding: 0.55rem 1.4rem; font-weight: 600; font-size: 0.95rem; width: 100%; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

for k, v in {"retriever": None, "llm": None, "doc_stats": {}, "chat_history": [], "processed": False}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def answer_question(question, retriever, llm):
    """RAG pipeline: retrieve relevant chunks then generate answer."""
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert document analyst. Use ONLY the context below to answer the question accurately and concisely.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}"""),
        ("human", "{question}")
    ])
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content, docs

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DocMind AI")
    st.markdown("---")
    groq_key = st.text_input("🔑 Groq API Key", type="password", placeholder="gsk_...", help="Free at console.groq.com")
    st.markdown("---")
    st.markdown("### 📤 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file and groq_key:
        if st.button("⚡ Process Document"):
            with st.spinner("Reading & indexing your document..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    loader = PyPDFLoader(tmp_path)
                    pages = loader.load()
                    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
                    chunks = splitter.split_documents(pages)
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    vectordb = Chroma.from_documents(chunks, embeddings)

                    st.session_state.retriever = vectordb.as_retriever(search_kwargs={"k": 4})
                    st.session_state.llm = ChatGroq(groq_api_key=groq_key, model_name="llama-3.3-70b-versatile", temperature=0.2)
                    st.session_state.doc_stats = {"pages": len(pages), "chunks": len(chunks), "filename": uploaded_file.name}
                    st.session_state.processed = True
                    st.session_state.chat_history = []
                    os.unlink(tmp_path)
                    st.success("✅ Document ready!")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("### 🛠 Tech Stack")
    for tech in ["LangChain", "Groq (Llama 3)", "ChromaDB", "HuggingFace Embeddings", "Streamlit"]:
        st.markdown(f'<span class="tech-pill">{tech}</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### How it works")
    for n, step in enumerate(["PDF split into chunks","Chunks → vector embeddings","Stored in ChromaDB","Query matched semantically","Llama 3 generates answer"], 1):
        st.markdown(f"**{n}.** {step}")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="badge">⚡ POWERED BY RAG + LANGCHAIN + LLAMA 3</div>
    <h1>🧠 DocMind AI</h1>
    <p>Upload any PDF and have an intelligent conversation with your document.<br>Grounded answers, real sources — no hallucinations.</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.processed:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{st.session_state.doc_stats["pages"]}</div><div class="stat-lbl">Pages Indexed</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{st.session_state.doc_stats["chunks"]}</div><div class="stat-lbl">Text Chunks</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-box"><div class="stat-num">RAG</div><div class="stat-lbl">Retrieval Method</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align:center;color:#64748b;font-size:0.83rem;margin:0.4rem 0 1rem">📄 {st.session_state.doc_stats["filename"]}</div>', unsafe_allow_html=True)

if st.session_state.chat_history:
    st.markdown("### 💬 Conversation")
    for item in st.session_state.chat_history:
        st.markdown(f'<div class="card"><strong>🙋 {item["q"]}</strong><div class="answer-box">{item["a"]}</div></div>', unsafe_allow_html=True)
        if item.get("sources"):
            with st.expander("📎 View Sources"):
                for src in item["sources"]:
                    pg = src.metadata.get("page", "?")
                    st.markdown(f'<span class="source-chip">Page {int(pg)+1}</span>', unsafe_allow_html=True)
                    st.markdown(f'<div class="source-text">{src.page_content[:300]}...</div>', unsafe_allow_html=True)

if st.session_state.processed:
    st.markdown("### 🔍 Ask a Question")
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input("", placeholder="e.g. What are the key findings? Summarise section 3...", label_visibility="collapsed")
    with col2:
        ask = st.button("Ask →")

    st.markdown("**Try these:**")
    cols = st.columns(3)
    sample_qs = ["Summarise this document", "What are the key points?", "What conclusions are drawn?"]
    for i, sq in enumerate(sample_qs):
        with cols[i]:
            if st.button(sq, key=f"sq_{i}"):
                question = sq
                ask = True

    if ask and question:
        with st.spinner("🔍 Searching document & generating answer..."):
            try:
                answer, sources = answer_question(question, st.session_state.retriever, st.session_state.llm)
                st.session_state.chat_history.append({"q": question, "a": answer, "sources": sources})
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

else:
    st.markdown("""
    <div class="card" style="text-align:center;padding:3rem;">
        <div style="font-size:3rem;margin-bottom:1rem">📄</div>
        <h3 style="color:#1e293b;margin-bottom:0.5rem">No document loaded yet</h3>
        <p style="color:#64748b">Enter your Groq API key and upload a PDF in the sidebar to get started.</p>
        <div style="margin-top:1.2rem">
            <a href="https://console.groq.com" target="_blank" style="background:#2563eb;color:white;padding:0.5rem 1.2rem;border-radius:8px;text-decoration:none;font-weight:600;font-size:0.9rem">Get Free Groq API Key →</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### 💡 What can you do with DocMind AI?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card"><h4>📊 Research Reports</h4><p style="color:#64748b;font-size:0.9rem">Ask questions about any research paper, get grounded answers with page references.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h4>📋 Legal Documents</h4><p style="color:#64748b;font-size:0.9rem">Understand contracts, extract clauses, and get plain-language explanations.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card"><h4>📈 Annual Reports</h4><p style="color:#64748b;font-size:0.9rem">Analyse financial statements, identify key metrics and business highlights.</p></div>', unsafe_allow_html=True)
