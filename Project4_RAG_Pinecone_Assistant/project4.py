# pip install streamlit pinecone sentence-transformers groq pypdf

import streamlit as st
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from groq import Groq
from pypdf import PdfReader

# ---------------- UI ----------------
st.set_page_config(page_title="Doc-Query", page_icon="⚡")
st.title("Query Documents with Pinecone")

# ---------------- Sidebar ----------------
st.sidebar.header("🔐 API Config")

groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password")

init = st.sidebar.button("Initialize")

# ---------------- Initialize ----------------
if init:
    if not groq_api_key or not pinecone_api_key:
        st.sidebar.error("Enter all keys")
        st.stop()

    os.environ["GROQ_API_KEY"] = groq_api_key

    pc = Pinecone(api_key=pinecone_api_key)

    index_name = "docmind"

    if index_name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    st.session_state.index = pc.Index(index_name)

    st.session_state.embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    st.session_state.client = Groq()

    st.sidebar.success("✅ Initialized")

# ---------------- Stop if not initialized ----------------
if "index" not in st.session_state:
    st.warning("Initialize from sidebar")
    st.stop()

index = st.session_state.index
embedder = st.session_state.embedder
client = st.session_state.client

# ---------------- Chat History ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- Upload PDF ----------------
uploaded = st.file_uploader("📄 Upload PDF", type="pdf")

if uploaded:

    reader = PdfReader(uploaded)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    chunks = [
        text[i:i + 500]
        for i in range(0, len(text), 500)
    ]

    vectors = embedder.encode(chunks).tolist()

    filename = uploaded.name

    to_upsert = []

    for i, chunk in enumerate(chunks):
        to_upsert.append(
            (
                f"{filename}-{i}",
                vectors[i],
                {
                    "text": chunk,
                    "source": filename
                }
            )
        )

    index.upsert(vectors=to_upsert)

    st.success(f"✅ Indexed {filename}")

# ---------------- Chat ----------------
if prompt := st.chat_input("Ask your document..."):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    query_vec = embedder.encode([prompt])[0].tolist()

    results = index.query(
        vector=query_vec,
        top_k=3,
        include_metadata=True
    )

    matches = results.get("matches", [])

    context = "\n\n".join(
        m["metadata"]["text"]
        for m in matches
    )

    # Collect source document names
    sources = sorted(
        set(
            m["metadata"].get("source", "Unknown")
            for m in matches
        )
    )

    # If nothing relevant was retrieved
    if not context.strip():

        answer = "Not in documents."

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    else:

        with st.chat_message("assistant"):

            placeholder = st.empty()

            full_text = ""

            stream = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": """
You are a document question-answering assistant.

Answer ONLY using the supplied context.

If the answer is not available in the context, reply exactly:

Not in documents.

Do not use outside knowledge.
Do not make assumptions.
"""
                    },
                    {
                        "role": "user",
                        "content": f"""
Context:
{context}

Question:
{prompt}
"""
                    }
                ],
                stream=True
            )

            for chunk in stream:

                token = chunk.choices[0].delta.content or ""

                full_text += token

                placeholder.markdown(full_text + "▌")

            # Append source names
            full_text += "\n\n---\n**Source Document(s):**\n"
            full_text += "\n".join(f"- {s}" for s in sources)

            placeholder.markdown(full_text)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_text
	    }
            )