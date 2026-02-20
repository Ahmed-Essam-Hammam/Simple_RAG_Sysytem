# 📚 Simple RAG System

A simple Retrieval-Augmented Generation (RAG) application built with LangChain, ChromaDB, and Streamlit. Upload a `.txt` file, ask questions, and get answers grounded in your document's content — powered by the Cerebras API.

---

## 🗂 Project Structure

```
RAG_/
├── app.py              # Streamlit frontend
├── config.py           # Environment variables and constants
├── embeddings.py       # HuggingFace embedding model setup
├── llm.py              # LLM client (Cerebras via OpenAI-compatible API)
├── rag_pipeline.py     # Vector store creation and RAG chain
├── requirements.txt    # Python dependencies
├── .env                # Your local secrets (not committed)
└── .env.example        # Example env file
```

---

## ⚙️ How It Works

1. **Upload** a `.txt` file through the Streamlit interface.
2. The file is loaded and split into semantically meaningful chunks using LangChain's `SemanticChunker`.
3. Chunks are embedded using the `BAAI/bge-base-en-v1.5` HuggingFace model and stored in a local **ChromaDB** vector store.
4. When you ask a question, the top 4 most relevant chunks are retrieved and passed as context to the LLM.
5. The LLM (served via **Cerebras**) generates an answer strictly based on the retrieved context.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd RAG_
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

```env
CEREBRAS_API_KEY=your_cerebras_api_key_here
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## 🔧 Configuration

Key settings are managed in `config.py`:

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | HuggingFace embedding model |
| `LLM_MODEL` | `gpt-oss-120b` | Cerebras model name |
| `CHROMA_DIR` | `./chroma_db` | Local directory for ChromaDB persistence |

---

## 🧰 Tech Stack

- **[LangChain](https://www.langchain.com/)** — RAG pipeline orchestration
- **[ChromaDB](https://www.trychroma.com/)** — Local vector store
- **[HuggingFace](https://huggingface.co/)** — `BAAI/bge-base-en-v1.5` embeddings
- **[Cerebras](https://www.cerebras.ai/)** — LLM inference (OpenAI-compatible API)
- **[Streamlit](https://streamlit.io/)** — Web interface

---

## 📝 Notes

- Only `.txt` files are supported for upload at this time.
- The vector store is persisted locally in `./chroma_db` between sessions.
- The `SemanticChunker` splits documents based on semantic similarity rather than fixed character counts, which generally produces higher-quality chunks.
