"""
semantic_chunker.py
--------------------
ADDITIVE MODULE — Feature 1: Semantic / sentence-boundary-aware chunking

Existing project7.py has this fixed-size chunker (unchanged, still works):

    def chunk_text(text, chunk_size=600, overlap=200):
        words = text.split()
        ...
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunks.append(" ".join(chunk_words))
            ...

Problem: it can slice a sentence in half mid-word-window, which hurts
embedding quality and retrieval precision.

This module is a DROP-IN REPLACEMENT with the same signature
(text, chunk_size, overlap) -> list[str], but it never cuts a sentence.
It groups whole sentences into chunks until the soft word-count target
is hit, then carries the last few sentences forward as overlap.

Install:
    pip install nltk

File path (new file, add next to project7.py):
    Project7_AI_Knowledge_Assistant_Pinecone/semantic_chunker.py
"""
import nltk

# Download tokenizer data once (safe to call every run — it's a no-op if cached)
for _resource in ("punkt_tab", "punkt"):
    try:
        nltk.data.find(f"tokenizers/{_resource}")
    except LookupError:
        try:
            nltk.download(_resource, quiet=True)
        except Exception:
            pass

from nltk.tokenize import sent_tokenize


def semantic_chunk_text(text, chunk_size=600, overlap=100):
    """
    Sentence-boundary aware chunking.

    - chunk_size: soft target word count per chunk (same meaning as the
      original chunk_text's chunk_size)
    - overlap: number of trailing WORDS carried into the next chunk for
      context continuity (kept smaller than the original 200 by default,
      since whole sentences are already coherent — tune as needed)

    Falls back gracefully: if NLTK's sentence tokenizer fails for any
    reason (e.g. noisy OCR text with no punctuation), the whole text is
    treated as one "sentence" so the function never throws and never
    returns an empty result for non-empty input.
    """
    text = text.strip()
    if not text:
        return []

    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = [text]

    if not sentences:
        return []

    chunks = []
    current_sentences = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        # Rare case: a single "sentence" (e.g. OCR blob with no punctuation)
        # already exceeds chunk_size. Flush what we have, emit it standalone.
        if sentence_word_count > chunk_size:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_word_count = 0
            chunks.append(sentence)
            continue

        if current_word_count + sentence_word_count > chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences))

            # Sentence-level overlap: carry trailing sentences worth ~overlap words
            overlap_sentences = []
            overlap_word_count = 0
            for s in reversed(current_sentences):
                w = len(s.split())
                if overlap_word_count + w > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_word_count += w

            current_sentences = overlap_sentences
            current_word_count = overlap_word_count

        current_sentences.append(sentence)
        current_word_count += sentence_word_count

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks
