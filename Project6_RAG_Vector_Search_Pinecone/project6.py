"""
Project 6: RAG Development for Unstructured Documents using Vector Search
--------------------------------------------------------------------------
Focus: building and proving out a solid RETRIEVAL pipeline over unstructured
documents (PDF, DOCX, TXT) using Pinecone vector search. Unlike a full
conversational assistant, this project emphasizes the ingestion -> chunking
-> embedding -> vector search mechanics, and exposes the raw retrieved
chunks (with similarity scores) so retrieval quality can be inspected and
tuned directly. A minimal LLM pass is included just to demonstrate an
end-to-end answer, but the core deliverable is the search pipeline itself.

Pipeline:
    1. Upload unstructured documents (PDF / DOCX / TXT)
    2. Chunk with configurable size/overlap
    3. Embed with sentence-transformers
    4. Upsert into Pinecone (vector search)
    5. Query -> retrieve top-k chunks with similarity scores (inspectable)
    6. (Optional) Generate a short answer from the retrieved chunks via Groq

Tech stack:
    Vector DB   : Pinecone
    Embeddings  : sentence-transformers/all-MiniLM-L6-v2
    LLM         : Groq (Llama 3.1) — minimal, for the optional answer step
    Framework   : LangChain (loaders/splitters only — no chains/memory)
    UI          : Streamlit

Run:
    pip install -r requirements.txt
    streamlit run app.py
    # Paste your Pinecone API key and (optionally) Groq API key into the sidebar.
"""

import os
import tempfile

import streamlit as st
from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
st.set_page_config(page_title="RAG Vector Search (Pinecone)", page_icon="🔎", layout="wide")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "unstructured-doc-search")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def get_pinecone_client(api_key: str):
    return Pinecone(api_key=api_key)


def ensure_index(pc: Pinecone, index_name: str, dim: int):
    existing = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)


def load_document(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif suffix in (".docx", ".doc"):
        loader = UnstructuredWordDocumentLoader(tmp_path)
    else:  # .txt
        loader = TextLoader(tmp_path, encoding="utf-8")

    docs = loader.load()
    for d in docs:
        d.metadata["source"] = uploaded_file.name
    return docs


def generate_answer_from_chunks(query: str, chunks: list, groq_api_key: str) -> str:
    """Minimal LLM pass: summarize the retrieved chunks into a short answer."""
    if not groq_api_key:
        return "*(No Groq API key provided — showing raw retrieved chunks only.)*"

    from langchain_groq import ChatGroq

    context = "\n\n".join(f"[{i+1}] {c.page_content}" for i, c in enumerate(chunks))
    prompt = (
        "Answer the question below using ONLY the numbered context passages. "
        "If the answer isn't in the context, say so plainly.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    try:
        llm = ChatGroq(model=GROQ_MODEL, groq_api_key=groq_api_key, temperature=0.1)
        return llm.invoke(prompt).content.strip()
    except Exception as e:
        return f"⚠️ LLM call failed: {e}"


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "pinecone_api_key" not in st.session_state:
    st.session_state.pinecone_api_key = os.environ.get("PINECONE_API_KEY", "")
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = os.environ.get("GROQ_API_KEY", "")
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []


# --------------------------------------------------------------------------
# Sidebar: setup + ingestion
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")
    st.session_state.pinecone_api_key = st.text_input(
        "Pinecone API Key",
        value=st.session_state.pinecone_api_key,
        type="password",
        placeholder="pcsk_...",
    )
    st.session_state.groq_api_key = st.text_input(
        "Groq API Key (optional — for generated answers)",
        value=st.session_state.groq_api_key,
        type="password",
        placeholder="gsk_... (leave blank to see raw retrieved chunks only)",
    )
    st.caption(f"Pinecone index: `{PINECONE_INDEX_NAME}`")

    st.divider()
    st.header("📄 Unstructured Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF / DOCX / TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    chunk_size = st.slider("Chunk size", 200, 2000, 800, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 100, step=50)

    if st.button("🔄 Ingest & index documents", use_container_width=True):
        if not st.session_state.pinecone_api_key:
            st.error("Please provide a Pinecone API key.")
        elif not uploaded_files:
            st.error("Please upload at least one document.")
        else:
            os.environ["PINECONE_API_KEY"] = st.session_state.pinecone_api_key
            with st.spinner("Loading, chunking, and embedding documents..."):
                try:
                    all_docs = []
                    for f in uploaded_files:
                        all_docs.extend(load_document(f))

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap
                    )
                    chunks = splitter.split_documents(all_docs)

                    embeddings = get_embeddings()
                    pc = get_pinecone_client(st.session_state.pinecone_api_key)
                    ensure_index(pc, PINECONE_INDEX_NAME, EMBED_DIM)

                    vectorstore = PineconeVectorStore.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        index_name=PINECONE_INDEX_NAME,
                        pinecone_api_key=st.session_state.pinecone_api_key,
                    )
                    st.session_state.vectorstore = vectorstore
                    st.session_state.ingested_files = [f.name for f in uploaded_files]
                    st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} document(s).")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    if st.session_state.ingested_files:
        with st.expander("📚 Indexed documents"):
            for name in st.session_state.ingested_files:
                st.markdown(f"- {name}")


# --------------------------------------------------------------------------
# Main: query + retrieval inspection
# --------------------------------------------------------------------------
st.title("🔎 RAG Development for Unstructured Documents using Vector Search")
st.caption(
    "Upload unstructured documents, index them in Pinecone, and inspect vector "
    "search retrieval quality directly — with similarity scores and an optional generated answer."
)

if "search_history" not in st.session_state:
    st.session_state.search_history = []  # list of {"query":..., "top_k":..., "results": [(doc, score), ...]}

top_k = st.slider("Top-K (number of chunks to retrieve per search)", min_value=1, max_value=20, value=5)

if st.session_state.search_history:
    if st.button("🗑️ Clear search history"):
        st.session_state.search_history = []
        st.rerun()

# Render every past search as its own block, most recent last (chat-like)
for entry in st.session_state.search_history:
    st.markdown(f"**🔍 {entry['query']}**")
    tab_answer, tab_chunks = st.tabs(["💬 Generated Answer", "📊 Retrieved Chunks (raw)"])

    with tab_answer:
        st.markdown(entry["answer"])

    with tab_chunks:
        st.caption("Raw vector search results — inspect chunk content, source, and similarity score directly.")
        for i, (doc, score) in enumerate(entry["results"], 1):
            with st.expander(f"#{i} — score: {score:.4f} — source: {doc.metadata.get('source', 'unknown')}"):
                st.markdown(doc.page_content)
                st.json(doc.metadata)
    st.divider()

# Chat-style input that auto-clears after each submission
new_query = st.chat_input("Ask something about your uploaded documents...")

if new_query:
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Please ingest documents in the sidebar first.")
    else:
        with st.spinner("Running vector search..."):
            results = st.session_state.vectorstore.similarity_search_with_score(new_query, k=top_k)

        chunks_only = [doc for doc, score in results]
        with st.spinner("Generating answer from retrieved context..."):
            answer = generate_answer_from_chunks(new_query, chunks_only, st.session_state.groq_api_key)

        st.session_state.search_history.append(
            {"query": new_query, "results": results, "answer": answer}
        )
        st.rerun()