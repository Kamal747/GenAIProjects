"""
evaluate_ragas.py
------------------
ADDITIVE MODULE — Feature 3: RAG evaluation with Ragas
(faithfulness + answer_relevancy)

This is a STANDALONE script — it does NOT import or modify project7.py.
It re-implements the same retrieval + generation calls (same embedding
model, same Pinecone index, same Groq model) so it can run outside
Streamlit, feed 10-15 sample questions through the pipeline, and score
each answer.

Run this AFTER you've already indexed documents into Pinecone via the
Streamlit app (project7.py), using the same index name.

Install:
    pip install ragas datasets langchain-groq langchain-huggingface

Set environment variables before running (Windows CMD shown, since you use
py + CMD):
    set GROQ_API_KEY=gsk_your_key
    set PINECONE_API_KEY=your_pinecone_key
    set PINECONE_INDEX_NAME=enterprise-rag

Run:
    py evaluate_ragas.py

File path (new file, add next to project7.py):
    Project7_AI_Knowledge_Assistant_Pinecone/evaluate_ragas.py
"""
import os
import time
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIG ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "enterprise-rag")
LLM_MODEL = "openai/gpt-oss-120b"
TOP_K = 6
SIMILARITY_THRESHOLD = 0.25

# --- SAMPLE QUESTIONS ---
# IMPORTANT: replace these 10-15 with real questions about the documents
# YOU actually indexed (HR policy, finance docs, etc.) — generic placeholder
# questions below won't score meaningfully against your real index.
SAMPLE_QUESTIONS = [
    "What's the DEFINITION OF CONFIDENTIAL INFORMATION?",
    "What are the OBLIGATIONS OF RECEIVING PARTY.",
    "What's the 'NO LICENSE OR WARRANTY'.",
    "What are the CUSTOMER OBLIGATIONS.",
    "What are the PROVIDER OBLIGATIONS.",
    "Tell me about DISCIPLINARY ACTION PROCESS.",
    "Budget Sheet",
    "What was the Sales department's revenue in March FY2026?",
    "What's in the Scanned Document?",
    "What's in the Invoice?"
    # add 2-5 more specific to your indexed documents to reach 12-15
]


def retrieve_context(index, embedding_model, query, top_k=TOP_K, threshold=SIMILARITY_THRESHOLD):
    vector = embedding_model.encode(query).tolist()
    result = index.query(vector=vector, top_k=top_k, include_metadata=True)
    contexts = []
    for m in result.get("matches", []):
        if m.get("score", 0.0) >= threshold:
            contexts.append(m.get("metadata", {}).get("text_content", ""))
    return contexts


def generate_answer(groq_client, query, contexts):
    context_str = "\n\n---\n\n".join(contexts) if contexts else "No relevant context found."
    system_prompt = (
        "You are an expert Enterprise Knowledge Assistant. Answer queries with strict factual grounding.\n"
        "Answer SOLELY based on the CONTEXT below. If context does not contain the answer, "
        "say: 'The provided documents do not contain this information.'\n\n"
        f"=== CONTEXT ===\n{context_str}"
    )
    completion = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"},
        ],
        temperature=0.1,
    )
    return completion.choices[0].message.content


def main():
    if not GROQ_API_KEY or not PINECONE_API_KEY:
        raise SystemExit(
            "Missing credentials. Set GROQ_API_KEY and PINECONE_API_KEY environment variables first."
        )

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    groq_client = Groq(api_key=GROQ_API_KEY)

    records = []
    for question in SAMPLE_QUESTIONS:
        contexts = retrieve_context(index, embedding_model, question)
        answer = generate_answer(groq_client, question, contexts)
        records.append(
            {
                "question": question,
                "contexts": contexts if contexts else ["No relevant context found."],
                "answer": answer,
            }
        )
        time.sleep(2)  # gentle on Groq free-tier rate limits

    dataset = Dataset.from_list(records)

    # Ragas needs a judge LLM + judge embeddings for these metrics.
    # We reuse Groq (via LangChain wrapper) so no extra API key is needed.
    ragas_llm = LangchainLLMWrapper(ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL))
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    result = evaluate(
        dataset,
        # strictness=1 forces a SINGLE completion per judgement call instead of
        # ragas's default self-consistency sampling (n=3). Groq's API rejects
        # any n > 1 ("'n' : number must be at most 1"), which is what was
        # causing most evaluations to fail above.
        metrics=[Faithfulness(llm=ragas_llm), AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1)],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        # Faithfulness makes several sequential LLM calls per question (claim
        # extraction, then NLI verification per claim). Under the default
        # RunConfig (short timeout, high concurrency) most of these were
        # timing out against Groq -> almost all faithfulness scores came back
        # as NaN. A longer timeout + lower concurrency (fewer parallel calls
        # hitting Groq's rate limit at once) fixes that.
        run_config=RunConfig(timeout=180, max_workers=2, max_retries=3, max_wait=60),
    )

    df = result.to_pandas()

    if "question" not in df.columns and "user_input" in df.columns:
        # Newer ragas versions name the column "user_input" instead of "question"
        df = df.rename(columns={"user_input": "question"})

    available_cols = [c for c in ["question", "faithfulness", "answer_relevancy"] if c in df.columns]
    print(df[available_cols].to_string(index=False))

    os.makedirs("eval_results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("eval_results", f"ragas_eval_{timestamp}.csv")
    df.to_csv(out_path, index=False)

    print(f"\nSaved detailed results to {out_path}")
    print(f"Average faithfulness:     {df['faithfulness'].mean():.3f}")
    print(f"Average answer_relevancy: {df['answer_relevancy'].mean():.3f}")


if __name__ == "__main__":
    main()
