# 🧠 DocMind AI — PDF Chat with RAG + LangChain + Llama 3

Upload any PDF and chat with it intelligently. Built with a full RAG pipeline:
**LangChain → HuggingFace Embeddings → ChromaDB → Groq (Llama 3)**

---

## 🚀 Deploy to Streamlit Cloud (Free, 15 minutes)

### Step 1 — Get a Free Groq API Key
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free, no credit card)
3. Go to **API Keys** → **Create API Key**
4. Copy the key (starts with `gsk_...`)

---

### Step 2 — Push this project to GitHub
1. Go to [github.com](https://github.com) → **New Repository**
2. Name it `docmind-ai` → **Create Repository**
3. Upload all files from this folder:
   - `app.py`
   - `requirements.txt`
   - `README.md`
4. Click **Commit changes**

---

### Step 3 — Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Select your `docmind-ai` repository
5. Set **Main file path** → `app.py`
6. Click **Deploy!**

> ⏱ First deploy takes ~3-5 minutes (installing packages). After that it's instant.

---

### Step 4 — Get your shareable link
After deploy, Streamlit gives you a public URL like:
```
https://your-username-docmind-ai-app-xxxx.streamlit.app
```
Share this link with anyone — including your interviewer! ✅

---

## 🛠 Tech Stack

| Layer | Tool | Why |
|---|---|---|
| LLM | Groq (Llama 3 8B) | Free, fast (500 tokens/sec) |
| Orchestration | LangChain | Industry standard |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Free, runs locally |
| Vector Store | ChromaDB | In-memory, no setup |
| UI + Hosting | Streamlit Cloud | Free public link |

---

## 💬 How to Use the App
1. Paste your **Groq API Key** in the sidebar
2. Upload any **PDF** (report, paper, contract, etc.)
3. Click **Process Document**
4. Ask any question in the chat box
5. Get grounded answers with **source page references**

---

## 🎤 Interview Talking Points

**"How does RAG work in this app?"**
> "When a PDF is uploaded, LangChain splits it into overlapping text chunks. HuggingFace converts each chunk into a vector embedding and stores them in ChromaDB. When a user asks a question, the query is also embedded and semantically matched against the stored vectors. The top 4 most relevant chunks are retrieved and passed as context to Llama 3, which generates a grounded answer — this prevents hallucinations since the model only uses the document content."

**"Why Groq over OpenAI?"**
> "Groq offers a completely free API with no credit card required, running Llama 3 at extremely high speeds — up to 500 tokens/second. For a demo this makes the app feel instant. In production I'd evaluate based on accuracy, cost, and compliance requirements."

**"What would you improve with more time?"**
> "Add persistent vector storage with Pinecone for multi-session support, user authentication, support for multiple documents simultaneously, and streaming responses for better UX."
