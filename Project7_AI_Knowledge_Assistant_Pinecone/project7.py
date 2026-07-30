import streamlit as st
import pandas as pd
import io
import os
import time
from datetime import datetime

# --- Parsing & Imaging Libraries ---
from pypdf import PdfReader
import docx2txt
from striprtf.striprtf import rtf_to_text
import easyocr
import numpy as np
from PIL import Image

# --- AI & Vector DB Libraries ---
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from groq import Groq

# --- 1. CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="AI Knowledge Assistant", page_icon="🧠", layout="wide")

# Persistent Session State Initializations
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- FIX 1: EasyOCR → cache_resource (session state pannaa reload aagum) ---
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_pinecone_index(api_key, index_name):
    pc = Pinecone(api_key=api_key)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        time.sleep(2)
    return pc.Index(index_name)

@st.cache_resource
def get_groq_client(api_key):
    return Groq(api_key=api_key)

ocr_reader = load_ocr_reader()
embedding_model = load_embedding_model()

# --- SIDEBAR CONTROL UNIT ---
with st.sidebar:
    st.header("🔑 API Credentials")
    groq_api_key = st.text_input("Groq API Key:", type="password", help="Enter your gsk_... key")
    pinecone_api_key = st.text_input("Pinecone API Key:", type="password", help="Enter your Pinecone API key")
    pinecone_index_name = st.text_input("Pinecone Index Name:", value="enterprise-rag")

    st.write("---")
    st.header("⚙️ Generation Hyperparameters")
    temperature = st.slider("Temperature (Creativity Control):", min_value=0.0, max_value=1.0, value=0.1, step=0.05,
                            help="Keep low (0.0 - 0.2) for strict factual grounding and policy compliance.")
    top_p = st.slider("Top-p (Nucleus Sampling):", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
    top_k = st.slider("Top-k (Context Chunk Count):", min_value=1, max_value=15, value=6, step=1,
                      help="Increase this value if you need broader paragraph contexts to be retrieved.")
    similarity_threshold = st.slider("Retrieval Confidence Threshold:", min_value=0.0, max_value=1.0, value=0.25, step=0.05,
                                     help="Lower this threshold slightly if some spreadsheet cells or short sentences are missing.")

    st.write("---")
    st.header("📁 Document Ingestion Engine")
    doc_dept = st.selectbox("Department Tag:", ["HR", "Finance", "Legal", "Operations", "General"])
    doc_version = st.text_input("Document Version:", value="1.0", placeholder="e.g., 1.2 or 2026.Q1")

    uploaded_files = st.file_uploader(
        "Upload Enterprise Documents:",
        type=["pdf", "docx", "rtf", "txt", "xlsx", "csv", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

# --- 2. MULTI-FORMAT DOCUMENT INGESTION LOGIC ---
def extract_text_from_file(file_obj, file_name):
    ext = file_name.split('.')[-1].lower()
    text = ""

    if ext == "txt":
        # FIX 2: TXT — encoding fallback chain (utf-8 → latin-1 → ignore)
        raw_bytes = file_obj.read()
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                text = raw_bytes.decode(encoding)
                break
            except (UnicodeDecodeError, Exception):
                continue
        if not text:
            text = raw_bytes.decode("utf-8", errors="ignore")

    elif ext == "pdf":
        reader = PdfReader(file_obj)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    elif ext == "docx":
        text = docx2txt.process(file_obj)

    elif ext == "rtf":
        raw_rtf = file_obj.read().decode("utf-8", errors="ignore")
        text = rtf_to_text(raw_rtf)

    elif ext == "csv":
        df = pd.read_csv(file_obj)
        text = df.to_string(index=False)

    elif ext in ["xlsx", "xls"]:
        # FIX 3: Excel — each row as separate readable sentence (not one giant line)
        excel_sheets_dict = pd.read_excel(file_obj, sheet_name=None)
        combined_excel_text = []
        for sheet_name, sheet_df in excel_sheets_dict.items():
            if sheet_df.empty:
                continue
            combined_excel_text.append(f"=== SHEET: {sheet_name} ===")
            headers = sheet_df.columns.tolist()
            for _, row in sheet_df.iterrows():
                # Each row → "Column: Value | Column: Value" format
                row_parts = []
                for col in headers:
                    val = row[col]
                    if pd.notna(val) and str(val).strip():
                        row_parts.append(f"{col}: {val}")
                if row_parts:
                    combined_excel_text.append(" | ".join(row_parts))
            combined_excel_text.append("")
        text = "\n".join(combined_excel_text)

    elif ext in ["png", "jpg", "jpeg"]:
        # FIX 4: Image — direct numpy array to EasyOCR (no BytesIO pointer issue)
        pil_image = Image.open(file_obj).convert('RGB')
        img_array = np.array(pil_image)
        ocr_result = ocr_reader.readtext(img_array, detail=0)
        text = "\n".join(ocr_result)
        if not text.strip():
            st.sidebar.warning(f"⚠️ No text detected in image: {file_name}")

    return text.strip()


def chunk_text(text, chunk_size=600, overlap=200):
    words = text.split()
    chunks = []
    if not words:
        return chunks
    if len(words) <= chunk_size:
        return [text]
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
    return chunks


# --- 3. PINECONE VECTOR STORAGE ---
if uploaded_files and st.sidebar.button("⚡ Process & Index Documents"):
    if not pinecone_api_key:
        st.sidebar.error("Pinecone API Key input missing!")
    else:
        try:
            index = get_pinecone_index(pinecone_api_key, pinecone_index_name)
            total_chunks_indexed = 0

            for f in uploaded_files:
                # File size check (10MB limit)
                if f.size > 10 * 1024 * 1024:
                    st.sidebar.warning(f"⚠️ {f.name} exceeds 10MB limit, skipping.")
                    continue

                with st.spinner(f"Extracting text from {f.name}..."):
                    raw_extracted_text = extract_text_from_file(f, f.name)

                if not raw_extracted_text:
                    st.sidebar.warning(f"⚠️ Could not extract text from {f.name}")
                    continue

                chunks = chunk_text(raw_extracted_text)
                upsert_batch = []
                upload_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                for idx, chunk_content in enumerate(chunks):
                    vector_embedding = embedding_model.encode(chunk_content).tolist()
                    unique_id = f"{f.name}_v{doc_version}_c{idx}"
                    metadata = {
                        "source_file": f.name,
                        "document_type": f.name.split('.')[-1].lower(),
                        "department": doc_dept,
                        "version_id": doc_version,
                        "updated_at": upload_timestamp,
                        "text_content": chunk_content
                    }
                    upsert_batch.append((unique_id, vector_embedding, metadata))

                if upsert_batch:
                    index.upsert(vectors=upsert_batch)
                    total_chunks_indexed += len(upsert_batch)
                    st.sidebar.success(f"✅ {f.name} → {len(upsert_batch)} chunks indexed")

            st.sidebar.success(f"🎉 Total: {total_chunks_indexed} chunks indexed to Pinecone!")

        except Exception as err:
            st.sidebar.error(f"Vector DB Error: {err}")


# --- MAIN DASHBOARD PANEL ---
st.title("🧠 AI-Powered Knowledge Assistant")
st.caption("Chat with your documents using RAG, LangChain & Vector Search")

# FIX 5: Clear Chat Button
col1, col2 = st.columns([8, 1])
with col2:
    if st.button("🗑️ Clear", help="Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

if not groq_api_key or not pinecone_api_key:
    st.info("💡 Fill your Groq and Pinecone API keys in the left sidebar to get started.")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 Verified Grounding Context Records"):
                for src in message["sources"]:
                    st.write(f"📄 **File:** {src['file']} (v{src['version']}) | **Dept:** {src['dept']} | **Score:** {src['score']:.3f}")
                    st.caption(f"✍️ *Text:* {src['text']}")


# --- 4. QUERY ENGINE ---
if user_query := st.chat_input("Ask any enterprise policy, operational protocol or system metrics query..."):

    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    if not groq_api_key or not pinecone_api_key:
        with st.chat_message("assistant"):
            st.error("Missing API credentials in the sidebar.")
        st.stop()

    try:
        query_vector = embedding_model.encode(user_query).tolist()

        # FIX 1: Cached Pinecone index (no re-init every query)
        index = get_pinecone_index(pinecone_api_key, pinecone_index_name)

        retrieved_response = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        matches = retrieved_response.get("matches", [])
        valid_chunks = []
        source_citations = []

        for m in matches:
            score = m.get("score", 0.0)
            if score >= similarity_threshold:
                meta = m.get("metadata", {})
                chunk_text_content = meta.get("text_content", "")
                valid_chunks.append(chunk_text_content)
                source_citations.append({
                    "file": meta.get("source_file", "Unknown"),
                    "version": meta.get("version_id", "1.0"),
                    "dept": meta.get("department", "General"),
                    "score": score,
                    "text": chunk_text_content
                })

        if not valid_chunks:
            ai_response = "I cannot find any verified information regarding this request in the indexed documents."
            with st.chat_message("assistant"):
                st.markdown(ai_response)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response, "sources": []})
        else:
            context_str = "\n\n---\n\n".join(valid_chunks)

            system_prompt = (
                "You are an expert Enterprise Knowledge Assistant. Answer queries with strict factual grounding.\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. Answer SOLELY based on the CONTEXT below. Give full details without summarizing away important data.\n"
                "2. If context does not contain the answer, say: 'The provided documents do not contain this information.'\n"
                "3. Do NOT use external knowledge or fabricate information.\n"
                "4. Keep language objective, professional, and traceable to context.\n\n"
                f"=== CONTEXT ===\n{context_str}"
            )

            messages_payload = [{"role": "system", "content": system_prompt}]

            # FIX: Extended chat history context (last 6 messages)
            for prev in st.session_state.chat_history[-7:-1]:
                messages_payload.append({"role": prev["role"], "content": prev["content"]})

            messages_payload.append({"role": "user", "content": f"Query: {user_query}"})

            # FIX 1: Cached Groq client
            groq_client = get_groq_client(groq_api_key)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_stream_response = ""

                completion = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages_payload,
                    temperature=temperature,  # FIX: sidebar slider value use pannrom
                    top_p=top_p,
                    frequency_penalty=1.2,
                    stream=True
                )

                for chunk in completion:
                    content_token = chunk.choices[0].delta.content
                    if content_token:
                        full_stream_response += content_token
                        message_placeholder.markdown(full_stream_response + "▌")

                message_placeholder.markdown(full_stream_response)

                with st.expander("🔍 Verified Grounding Context Records"):
                    for src in source_citations:
                        st.write(f"📄 **File:** {src['file']} (v{src['version']}) | **Dept:** {src['dept']} | **Score:** {src['score']:.3f}")
                        st.caption(f"✍️ *Text:* {src['text']}")

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_stream_response,
                "sources": source_citations
            })

    except Exception as runtime_error:
        with st.chat_message("assistant"):
            st.error(f"Runtime Error: {runtime_error}")