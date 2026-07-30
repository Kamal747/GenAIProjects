# 🧠 AI-Powered Knowledge Assistant

https://genaiprojects-9desxb94sfy3bxhkrzzgxx.streamlit.app

> Built with Retrieval-Augmented Generation (RAG), LangChain and Vector Search

A production-grade intelligent document Q&A system that lets you upload enterprise documents in any format and ask natural language questions — powered by Groq LLM, Pinecone Vector DB, and Sentence Transformers.

---

## 🚀 Features

- **Multi-format ingestion** — PDF, DOCX, TXT, RTF, XLSX, CSV, PNG, JPG
- **OCR support** — Extract text from scanned images using EasyOCR
- **Semantic search** — Vector similarity retrieval via Pinecone
- **RAG pipeline** — Retrieve → Augment → Generate with Groq (LLaMA 3.3 70B)
- **Department tagging** — Tag documents by HR, Finance, Legal, Operations
- **Version tracking** — Track document versions in metadata
- **Streaming responses** — Real-time token-by-token LLM output
- **Source citations** — Every answer shows grounding context with confidence scores
- **Clear chat** — Reset conversation history anytime

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| LLM | Groq API — LLaMA 3.3 70B Versatile |
| Embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Vector DB | Pinecone (Serverless, AWS us-east-1) |
| OCR | EasyOCR |
| Document Parsing | pypdf, docx2txt, striprtf, pandas |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ai-knowledge-assistant.git
cd ai-knowledge-assistant
```

### 2. Create virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run project7.py
```

---

## 🔑 API Keys Required

Get these before running:

| Key | Where to get |
|-----|-------------|
| **Groq API Key** | [console.groq.com](https://console.groq.com) — Free tier available |
| **Pinecone API Key** | [app.pinecone.io](https://app.pinecone.io) — Free Serverless tier |

Enter both in the sidebar when the app launches.

---

## 📁 Project Structure

```
ai-knowledge-assistant/
├── project7.py          # Main Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── .gitignore           # Git ignore rules
```

---

## 🧠 How It Works

```
User Uploads Documents
        ↓
Text Extraction (PDF/DOCX/TXT/RTF/XLSX/CSV/Image OCR)
        ↓
Chunking (600 words, 200 overlap)
        ↓
Embedding (all-MiniLM-L6-v2 → 384-dim vectors)
        ↓
Store in Pinecone with Metadata
        ↓
User Asks a Question
        ↓
Embed Query → Vector Search → Top-K Chunks Retrieved
        ↓
Groq LLM (LLaMA 3.3 70B) → Grounded Answer
        ↓
Streamed Response + Source Citations
```

---

## 🎛️ Sidebar Controls

| Control | Description |
|---------|-------------|
| Temperature | Lower = more factual (recommended: 0.1) |
| Top-p | Nucleus sampling diversity |
| Top-k | Number of chunks retrieved (recommended: 6) |
| Similarity Threshold | Minimum retrieval confidence (recommended: 0.25) |
| Department Tag | Metadata label for uploaded documents |
| Document Version | Version tracking for document lifecycle |

---

## 💡 Use Cases

- **HR** — Query leave policies, appraisal guidelines, onboarding documents
- **Finance** — Retrieve budget reports, expense policies, audit records
- **Legal** — Search contracts, NDAs, compliance documents
- **Operations** — Access SOPs, manuals, process documents

---

## 🙋 Author

**Kamaleshwar**
Final Year B.E. Computer Science Engineering
St. Joseph's College of Engineering and Technology, Thanjavur
Generative AI Engineer (in progress) 🚀

---

## 📄 License

MIT License — Free to use and modify.
