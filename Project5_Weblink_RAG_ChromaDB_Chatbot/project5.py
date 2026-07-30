"""
Weblink RAG Chatbot (Groq + Streamlit + LangChain + Chroma Cloud)
--------------------------------------------------------------
A Retrieval-Augmented Generation chatbot that ingests content from
multiple websites, embeds it, stores it in a **Chroma Cloud** vector
database (hosted, not local), and answers user questions with
conversational (session) memory, powered by Groq's hosted Llama 3.1 model.

Architecture (matches the "Multi-Source Conversational RAG Chatbot" diagram):
    1. Data Sources        -> multiple website URLs
    2. Web Scraping/Loading -> WebBaseLoader (LangChain)
    3. Text Chunking       -> RecursiveCharacterTextSplitter (chunk 1000 / overlap 200)
    4. Embedding Generation -> HuggingFace sentence-transformers/all-MiniLM-L6-v2
    5. Vector Store        -> Chroma Cloud (hosted)
    6. Retrieval           -> similarity search over Chroma Cloud
    7. Memory              -> Streamlit session state chat history
    8. Prompt Construction -> system instructions + retrieved context + chat history
    9. LLM                 -> Groq (Llama 3.1 8B Instant)
    10. Response           -> shown to user, saved back to memory

Run:
    pip install -r requirements.txt
    streamlit run app.py
    # Paste your Groq API key AND Chroma Cloud credentials into the sidebar.
    # Get a free Groq key: https://console.groq.com/keys
    # Get free Chroma Cloud credentials: https://www.trychroma.com/
"""

import os

import chromadb
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
st.set_page_config(page_title="Weblink RAG Chatbot", page_icon="🌐", layout="wide")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "weblink-rag-chatbot")


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def get_llm(api_key: str):
    return ChatGroq(model=GROQ_MODEL, groq_api_key=api_key, temperature=0.2)


def get_chroma_cloud_client(api_key: str, tenant: str, database: str):
    """Connect to a Chroma Cloud instance (hosted, not local)."""
    return chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database,
    )


def load_and_split_urls(urls: list[str], chunk_size: int, chunk_overlap: int):
    """Scrape the given URLs and split them into chunks ready for embedding."""
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        loaded = loader.load()
        for d in loaded:
            d.metadata["source"] = url
        docs.extend(loaded)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, text)
if "chain" not in st.session_state:
    st.session_state.chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = os.environ.get("GROQ_API_KEY", "")
if "chroma_api_key" not in st.session_state:
    st.session_state.chroma_api_key = os.environ.get("CHROMA_API_KEY", "")
if "chroma_tenant" not in st.session_state:
    st.session_state.chroma_tenant = os.environ.get("CHROMA_TENANT", "")
if "chroma_database" not in st.session_state:
    st.session_state.chroma_database = os.environ.get("CHROMA_DATABASE", "")
if "ingested_urls" not in st.session_state:
    st.session_state.ingested_urls = []

# --------------------------------------------------------------------------
# Sidebar: setup + ingestion
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")
    st.session_state.groq_api_key = st.text_input(
        "Groq API Key",
        value=st.session_state.groq_api_key,
        type="password",
        placeholder="gsk_...",
        help="Get a free key at https://console.groq.com/keys",
    )
    st.caption(f"Model: `{GROQ_MODEL}` (set env var `GROQ_MODEL` to change)")

    st.divider()
    st.header("☁️ Chroma Cloud")
    st.session_state.chroma_api_key = st.text_input(
        "Chroma Cloud API Key",
        value=st.session_state.chroma_api_key,
        type="password",
        placeholder="ck-...",
        help="Get free credentials at https://www.trychroma.com/",
    )
    st.session_state.chroma_tenant = st.text_input(
        "Tenant ID",
        value=st.session_state.chroma_tenant,
        placeholder="your-tenant-id",
    )
    st.session_state.chroma_database = st.text_input(
        "Database name",
        value=st.session_state.chroma_database,
        placeholder="your-database-name",
    )
    st.caption(f"Collection: `{CHROMA_COLLECTION_NAME}` (set env var `CHROMA_COLLECTION_NAME` to change)")

    st.divider()
    st.header("🌐 Data Sources")
    urls_text = st.text_area(
        "Website URLs (one per line)",
        placeholder="https://example.com/courses\nhttps://example.com/placements\nhttps://example.com/locations",
        height=140,
    )

    chunk_size = st.slider("Chunk size", 300, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 200, step=50)

    if st.button("🔄 Ingest websites", use_container_width=True):
        urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
        if not urls:
            st.error("Please enter at least one URL.")
        elif not (st.session_state.chroma_api_key and st.session_state.chroma_tenant and st.session_state.chroma_database):
            st.error("Please fill in all Chroma Cloud fields (API key, tenant, database).")
        else:
            with st.spinner(f"Scraping and indexing {len(urls)} website(s)..."):
                try:
                    chunks = load_and_split_urls(urls, chunk_size, chunk_overlap)
                    embeddings = get_embeddings()

                    chroma_client = get_chroma_cloud_client(
                        st.session_state.chroma_api_key,
                        st.session_state.chroma_tenant,
                        st.session_state.chroma_database,
                    )

                    vectorstore = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        client=chroma_client,
                        collection_name=CHROMA_COLLECTION_NAME,
                    )
                    st.session_state.vectorstore = vectorstore
                    st.session_state.ingested_urls = urls

                    memory = ConversationBufferMemory(
                        memory_key="chat_history", return_messages=True, output_key="answer"
                    )
                    st.session_state.chain = ConversationalRetrievalChain.from_llm(
                        llm=get_llm(st.session_state.groq_api_key),
                        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
                        memory=memory,
                        return_source_documents=True,
                    )
                    st.success(f"Indexed {len(chunks)} chunks from {len(urls)} website(s) into Chroma Cloud.")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    if st.session_state.ingested_urls:
        with st.expander("📄 Ingested sources"):
            for u in st.session_state.ingested_urls:
                st.markdown(f"- {u}")

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# --------------------------------------------------------------------------
# Main chat UI
# --------------------------------------------------------------------------
st.title("🌐 Weblink RAG Chatbot")
st.caption("Ask questions grounded in the content of the websites you've ingested.")

for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(text)

query = st.chat_input("Ask a question about the ingested websites...")

if query:
    st.session_state.chat_history.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)

    if not st.session_state.groq_api_key:
        answer = "⚠️ Please paste your Groq API key into the sidebar first."
        sources = []
    elif not (st.session_state.chroma_api_key and st.session_state.chroma_tenant and st.session_state.chroma_database):
        answer = "⚠️ Please fill in your Chroma Cloud credentials in the sidebar first."
        sources = []
    elif st.session_state.chain is None:
        answer = "⚠️ Please enter website URLs and click **Ingest websites** in the sidebar first."
        sources = []
    else:
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.chain({"question": query})
                answer = result["answer"]
                sources = result.get("source_documents", [])
            except Exception as e:
                answer = f"⚠️ Something went wrong: {e}"
                sources = []

    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            with st.expander("🔗 Sources"):
                seen = set()
                for doc in sources:
                    src = doc.metadata.get("source", "unknown")
                    if src not in seen:
                        seen.add(src)
                        st.markdown(f"- {src}")

    st.session_state.chat_history.append(("assistant", answer))