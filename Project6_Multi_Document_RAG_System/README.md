# 🔎 Project 6: RAG Development for Unstructured Documents using Vector Search

## 🌐 Live Demo

🔗 https://genaiprojects-jduhezfy8qr9rvsflotf7r.streamlit.app

A focused Retrieval-Augmented Generation project centered on the **retrieval
pipeline itself** — ingesting unstructured documents, chunking them, embedding
them, and performing vector search over **Pinecone**. Unlike a full
conversational assistant, this app exposes the raw retrieved chunks (with
similarity scores) so you can directly inspect and tune retrieval quality.

## What this project focuses on

- **Ingestion** of unstructured documents: PDF, DOCX, TXT
- **Chunking** with configurable size/overlap
- **Embedding** with sentence-transformers
- **Vector search** in Pinecone — the core deliverable
- **Inspectable results**: every retrieved chunk shows its similarity score
  and source document
- A minimal, optional LLM pass (via Groq) to demonstrate an end-to-end
  answer — but the retrieval pipeline is the star of this project, not the
  chat experience

## Pipeline

```
Upload PDF/DOCX/TXT
        │
        ▼
Chunking (RecursiveCharacterTextSplitter)
        │
        ▼
Embedding (sentence-transformers/all-MiniLM-L6-v2)
        │
        ▼
Pinecone (vector upsert)
        │
        ▼
   ── User enters a search query ──
        │
        ▼
Vector similarity search (top-K, with scores)
        │
        ├──► Raw Retrieved Chunks tab (inspect scores/sources directly)
        │
        └──► Generated Answer tab (optional, via Groq)
```

## Tech stack

| Component        | Choice                                                  |
|-------------------|------------------------------------------------------------|
| Vector DB         | [Pinecone](https://www.pinecone.io/)                        |
| Embeddings        | sentence-transformers/all-MiniLM-L6-v2                       |
| LLM (optional)    | Llama 3.1 8B Instant via [Groq](https://console.groq.com)   |
| Loaders           | LangChain (`PyPDFLoader`, `UnstructuredWordDocumentLoader`, `TextLoader`) |
| UI                | Streamlit                                                     |

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a free Pinecone API key
Sign up at [pinecone.io](https://www.pinecone.io/) and create an API key.

### 3. (Optional) Get a free Groq API key
Sign up at [console.groq.com/keys](https://console.groq.com/keys) if you want
generated answers in addition to raw retrieved chunks.

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Ingest and search
- Paste your Pinecone API key (and optionally Groq API key) into the sidebar.
- Upload one or more PDF/DOCX/TXT files.
- Click **Ingest & index documents**.
- Enter a search query and a Top-K value, then click **Search**.
- Check both tabs:
  - **Generated Answer** — a short LLM-composed answer from the retrieved chunks
  - **Retrieved Chunks (raw)** — every chunk returned by vector search, with
    its similarity score and source document, for direct inspection

## Environment variables

| Variable              | Default                     | Purpose                                      |
|-------------------------|-------------------------------|--------------------------------------------------|
| `PINECONE_API_KEY`     | *(unset)*                    | Pre-fills the Pinecone API key field              |
| `PINECONE_INDEX_NAME`  | `unstructured-doc-search`    | Name of the Pinecone index used                   |
| `GROQ_API_KEY`         | *(unset)*                    | Pre-fills the optional Groq API key field         |
| `GROQ_MODEL`           | `llama-3.1-8b-instant`       | Which Groq-hosted model generates the answer      |

## Notes

- This project intentionally does **not** include conversational memory or
  multi-turn chat — that's the focus of Project 7 (the full Knowledge
  Assistant built with LangChain on top of this same retrieval foundation).
- Similarity scores are cosine similarity (higher = more relevant, since the
  Pinecone index is created with `metric="cosine"`).
- If a document format doesn't parse well, `unstructured` may need extra
  system dependencies (e.g. `libmagic`) depending on your OS — see the
  [Unstructured docs](https://docs.unstructured.io/) if you hit parsing errors.
