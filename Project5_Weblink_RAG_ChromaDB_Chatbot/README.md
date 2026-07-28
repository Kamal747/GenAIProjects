# 🌐 Weblink RAG Chatbot (Groq + Streamlit + LangChain + Chroma Cloud)

A Retrieval-Augmented Generation chatbot that extracts content from multiple
websites, embeds it, stores it in **Chroma Cloud** (hosted vector database),
and answers user questions with conversational (session) memory — powered by
**Groq's** hosted **Llama 3.1** model.

## Architecture

```
1. Data Sources (multiple URLs)
        │
        ▼
2. Web Scraping/Loading  ──  WebBaseLoader (LangChain)
        │
        ▼
3. Text Chunking  ──  RecursiveCharacterTextSplitter (chunk 1000 / overlap 200)
        │
        ▼
4. Embedding Generation  ──  HuggingFace sentence-transformers/all-MiniLM-L6-v2
        │
        ▼
5. Vector Store  ──  Chroma Cloud (hosted)
        │
        ▼
   ── User asks a question ──
        │
        ▼
6. Retrieval (similarity search over Chroma Cloud)
        │
        ▼
7. Memory (chat history from st.session_state)
        │
        ▼
8. Prompt Construction (system instructions + context + history)
        │
        ▼
9. LLM  ──  Groq (Llama 3.1 8B Instant)
        │
        ▼
10. Response  ──  shown to user, saved back to memory
```

## Key Features

- Multi-website knowledge ingestion — paste any number of URLs
- Conversational memory (follow-up questions work naturally)
- Fast responses via Groq's LLM
- Accurate answers grounded in your ingested sites (RAG)
- Vector data stored in the cloud (Chroma Cloud) — no local DB files, works
  the same whether you run the app on your laptop or deploy it elsewhere
- Simple Streamlit UI

## Tech stack

| Component        | Choice                                                       |
|-------------------|---------------------------------------------------------------|
| LLM               | Llama 3.1 8B Instant via [Groq API](https://console.groq.com) |
| Embeddings        | sentence-transformers/all-MiniLM-L6-v2                        |
| Vector DB         | [Chroma Cloud](https://www.trychroma.com/) (hosted)            |
| Loader            | LangChain `WebBaseLoader`                                     |
| Framework         | LangChain                                                      |
| UI                | Streamlit                                                       |

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a free Groq API key
Sign up at [console.groq.com/keys](https://console.groq.com/keys).

### 3. Get free Chroma Cloud credentials
Sign up at [trychroma.com](https://www.trychroma.com/) and create a database.
You'll need three values from your Chroma Cloud dashboard:
- **API Key**
- **Tenant ID**
- **Database name**

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Paste credentials and ingest websites
- Paste your Groq API key into the **Groq API Key** field.
- Paste your Chroma Cloud API key, tenant ID, and database name into the
  **Chroma Cloud** section.
- Paste one or more website URLs (one per line) into the **Website URLs** box.
- Click **Ingest websites** — this scrapes, chunks, embeds, and stores the
  content in your Chroma Cloud database.
- Start asking questions in the chat box below.

## Example usage

```
https://www.aimoretechnologies.com/courses
https://www.aimoretechnologies.com/placements
https://www.aimoretechnologies.com/locations
```
Then ask:
- "What courses does AIMore offer?"
- "Where are AIMore's branches located?"
- "What are the placement highlights?" (follow-up, uses chat memory)

## Environment variables

All fields can also be pre-filled via environment variables instead of typing
them into the sidebar each time:

| Variable                 | Default                   | Purpose                                  |
|---------------------------|----------------------------|--------------------------------------------|
| `GROQ_API_KEY`            | *(unset)*                 | Pre-fills the Groq API key field            |
| `GROQ_MODEL`              | `llama-3.1-8b-instant`    | Which Groq-hosted model to use              |
| `CHROMA_API_KEY`          | *(unset)*                 | Pre-fills the Chroma Cloud API key field    |
| `CHROMA_TENANT`           | *(unset)*                 | Pre-fills the Chroma Cloud tenant ID field  |
| `CHROMA_DATABASE`         | *(unset)*                 | Pre-fills the Chroma Cloud database field   |
| `CHROMA_COLLECTION_NAME`  | `weblink-rag-chatbot`     | Name of the collection created in Chroma Cloud |

## Notes

- Since data lives in Chroma Cloud (not a local folder), the same ingested
  knowledge base is available no matter where you run the Streamlit app from.
- Some websites block automated scraping (WebBaseLoader will return an error
  or empty content) — try a different URL or check the site's robots.txt.
- Re-ingesting adds to the existing Chroma Cloud collection; use the Chroma
  Cloud dashboard to delete/reset the collection if you want to start fresh.