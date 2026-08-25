# Setup Notes — Project7 v2 Upgrade Package

This folder is a drop-in replacement for your existing
`Project7_AI_Knowledge_Assistant_Pinecone/` folder in the GenAIProjects repo.

## What changed

- `project7.py` — your ORIGINAL file, with the 4 upgrades wired in as
  **opt-in sidebar toggles**. Every original code path (Pinecone retrieval,
  EasyOCR, Groq streaming, fixed-size chunking) is still present and still
  runs by default when a toggle is off — nothing was deleted.
- `semantic_chunker.py`, `reranker.py`, `structured_output.py` — new
  modules imported by `project7.py`.
- `evaluate_ragas.py` — new standalone script, not imported by
  `project7.py`, run it separately.
- `requirements.txt` — original content kept, new packages appended at
  the bottom.
- `README.md` — original content kept, new "v2 Upgrades" section +
  updated tech stack table appended at the bottom.
- `eval_results/` — empty folder where `evaluate_ragas.py` will write its
  CSV reports.

## How to deploy this into your repo

1. Copy every file in this zip into your existing
   `Project7_AI_Knowledge_Assistant_Pinecone/` folder, overwriting
   `project7.py`, `requirements.txt`, and `README.md`.
2. Install the new dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app as usual:
   ```
   streamlit run project7.py
   ```
   You'll see a new "🧪 v2 Upgrades" section in the sidebar with 3
   checkboxes + a slider — semantic chunking and re-ranking are ON by
   default, structured output is OFF by default (streaming stays the
   default UX).
4. Before your first commit, re-index your documents (re-upload them
   through the app) so the new semantic chunker is applied — chunks
   already stored in Pinecone from before this upgrade keep their old
   fixed-size boundaries until re-indexed.
5. For the Ragas evaluation:
   - Open `evaluate_ragas.py`, replace the 10 placeholder
     `SAMPLE_QUESTIONS` with real questions about the documents you
     actually indexed (add 2-5 more to reach 12-15).
   - Set env vars: `GROQ_API_KEY`, `PINECONE_API_KEY`,
     `PINECONE_INDEX_NAME`.
   - Run: `py evaluate_ragas.py`
   - Paste the printed averages / CSV rows into the "Sample results"
     table in `README.md` under "3. RAG Evaluation (Ragas)".
6. Commit and push to `Kamal747/GenAIProjects`.

## For your resume / LinkedIn

Everything here is real, runnable code against your actual stack
(Pinecone, Groq LLaMA 3.3 70B, Sentence Transformers, EasyOCR) — once you
run the eval script and fill in real numbers, the README section is
accurate to cite directly, e.g.:

> "Upgraded a production RAG pipeline with semantic chunking, cross-encoder
> re-ranking (ms-marco-MiniLM-L-6-v2), Ragas-based faithfulness/relevancy
> evaluation, and Pydantic-validated structured LLM outputs via Groq JSON
> mode."
