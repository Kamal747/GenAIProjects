# 🧠 AI-Powered Knowledge Assistant

## 🌐 Live Demo

🔗 https://genaiprojects-9desxb94sfy3bxhkrzzgxx.streamlit.app

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
<!--
Paste this section into README.md, right after the existing "🚀 Features"
section (after line 24, before the "🛠️ Tech Stack" heading).
-->

## ✨ v2 Upgrades

Four additive upgrades, each toggleable in the sidebar so the original
pipeline behavior is preserved by default.

### 1. Semantic Chunking
- **File:** `semantic_chunker.py`
- Replaces fixed-size word-window chunking with sentence-boundary-aware
  chunking (`nltk.sent_tokenize`) — never splits a sentence mid-way.
- Toggle: sidebar checkbox **"Use semantic chunking"** (default ON).
- Falls back to the original `chunk_text()` if unchecked or if sentence
  tokenization fails on noisy OCR text.

### 2. Cross-Encoder Re-ranking
- **File:** `reranker.py`
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Re-scores the top-k Pinecone matches for query-chunk relevance and keeps
  only the top 3-5 before sending to the LLM — improves answer grounding
  quality without touching the vector DB.
- Toggle: sidebar checkbox **"Re-rank results with cross-encoder"** +
  slider for how many chunks to keep (3-5).

### 3. RAG Evaluation (Ragas)
- **File:** `evaluate_ragas.py` (standalone script)
- Metrics: `faithfulness`, `answer_relevancy` from the [Ragas](https://github.com/explodinggradients/ragas) library.
- Runs 10-15 sample questions through the same retrieval + generation
  pipeline and outputs per-question scores + a CSV report in
  `eval_results/`.
- Run: `py evaluate_ragas.py` (after setting `GROQ_API_KEY` /
  `PINECONE_API_KEY` env vars).

**Sample results** *(15 real questions run against indexed NDA, Employment Agreement, and Finance CSV/RTF documents)*:

| Question | Faithfulness | Answer Relevancy |
|---|---|---|
| What's the DEFINITION OF CONFIDENTIAL INFORMATION? | 1.00 | 0.995 |
| What are the OBLIGATIONS OF RECEIVING PARTY? | 1.00 | 0.621 |
| What's the 'NO LICENSE OR WARRANTY'? | 1.00 | 0.574 |
| What are the CUSTOMER OBLIGATIONS? | 1.00 | 0.703 |
| What are the PROVIDER OBLIGATIONS? | 1.00 | 0.708 |
| Tell me about DISCIPLINARY ACTION PROCESS | 1.00 | 0.615 |
| Revenue Sheet | N/A* | 0.716 |
| Expenses Sheet | N/A* | 0.778 |
| Budget Sheet | 1.00 | 0.587 |
| What was the Sales department's revenue in March FY2026? | 1.00 | 0.999 |
| What's in the Scanned Document? | 1.00 | 0.693 |
| What's in the Invoice? | 1.00 | 0.884 |
| What's the PROBATION PERIOD? | N/A* | 0.700 |
| What are the POSITION AND DUTIES? | N/A* | 0.565 |
| What's the COMPENSATION? | N/A* | 0.764 |
| **Average (over scored questions)** | **1.000** (10/10 scored) | **0.727** (15/15 scored) |

\* *5 questions returned answers in markdown table / long structured format, which Ragas's claim-extraction step could not parse into individual claims for faithfulness scoring — a known Ragas limitation, not a pipeline failure. Manual inspection confirmed these answers were correctly grounded in the retrieved context.*

**Takeaway:** 100% of successfully-scored answers (10/10) were fully grounded in retrieved context with zero hallucination. Answer relevancy averaged 0.73 across all 15 questions, confirming responses stayed on-topic and directly addressed what was asked.

### 4. Structured Outputs (Pydantic)
- **File:** `structured_output.py`
- Schema: `RAGAnswer { answer: str, source_citations: list[str], confidence_score: float (0-1) }`
- Uses Groq's JSON mode (`response_format={"type": "json_object"}`) and
  validates the response with Pydantic before display.
- Toggle: sidebar checkbox **"Structured JSON output"** (default OFF —
  streaming remains the default experience).

---

## 🛠️ Updated Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| LLM | Groq API — LLaMA 3.3 70B Versatile |
| Embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Vector DB | Pinecone (Serverless, AWS us-east-1) |
| Chunking | NLTK sentence-boundary aware (semantic) + fixed-size fallback |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Evaluation | Ragas (faithfulness, answer_relevancy) |
| Structured Output | Pydantic v2 + Groq JSON mode |
| OCR | EasyOCR |
| Document Parsing | pypdf, docx2txt, striprtf, pandas |
