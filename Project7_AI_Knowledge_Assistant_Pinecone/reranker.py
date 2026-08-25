"""
reranker.py
-----------
ADDITIVE MODULE — Feature 2: Cross-encoder re-ranking of Pinecone results

project7.py currently retrieves top_k chunks from Pinecone purely by cosine
similarity (bi-encoder score) and sends ALL of them (above
similarity_threshold) straight to the LLM. Bi-encoder similarity is fast
but coarse. A cross-encoder scores the (query, chunk) pair jointly, which
is slower but far more accurate for relevance — so we use it as a second
pass on the already-small top_k set, then keep only the best 3-5.

No new dependency needed: CrossEncoder ships inside the sentence-transformers
package that project7.py already depends on.

File path (new file, add next to project7.py):
    Project7_AI_Knowledge_Assistant_Pinecone/reranker.py
"""
import streamlit as st
from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@st.cache_resource
def load_reranker():
    return CrossEncoder(RERANKER_MODEL_NAME)


def rerank_citations(user_query, source_citations, top_n=4):
    """
    Re-ranks the source_citations list (same dict structure project7.py
    already builds: {"file", "version", "dept", "score", "text"}) using a
    cross-encoder, and returns only the top_n most relevant ones.

    Adds a new "rerank_score" key to each dict (does not remove the
    original Pinecone "score" key, so existing display code that reads
    src['score'] keeps working unchanged).

    If source_citations is empty, returns it unchanged (no-op) so callers
    don't need extra guard logic.
    """
    if not source_citations:
        return source_citations

    reranker = load_reranker()
    pairs = [[user_query, c["text"]] for c in source_citations]
    scores = reranker.predict(pairs)

    for citation, score in zip(source_citations, scores):
        citation["rerank_score"] = float(score)

    reranked = sorted(source_citations, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_n]
