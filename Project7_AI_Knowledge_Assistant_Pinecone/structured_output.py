"""
structured_output.py
---------------------
ADDITIVE MODULE — Feature 4: Pydantic-validated structured JSON output

project7.py currently streams a free-text answer token-by-token from Groq.
This module adds an OPT-IN structured mode: instead of streaming, it makes
one Groq call with JSON mode enabled, and validates the response against a
strict Pydantic schema (answer, source_citations, confidence_score).

Use this alongside the existing streaming path — don't replace it. Suggested
integration: a sidebar checkbox "Enable structured JSON output" that, when
checked, calls get_structured_answer() instead of the streaming completion
block (see INTEGRATION_GUIDE.md).

Install:
    pip install pydantic>=2.0

File path (new file, add next to project7.py):
    Project7_AI_Knowledge_Assistant_Pinecone/structured_output.py
"""
import json
from typing import List
from pydantic import BaseModel, Field, ValidationError


class RAGAnswer(BaseModel):
    answer: str = Field(..., description="The factual answer, grounded strictly in the provided context")
    source_citations: List[str] = Field(
        default_factory=list, description="Source file names actually used to build the answer"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Model's self-assessed confidence in the answer, 0.0 to 1.0"
    )


STRUCTURED_SYSTEM_SUFFIX = """

You MUST respond with ONLY a valid JSON object — no markdown fences, no
commentary before or after — matching EXACTLY this schema:
{
  "answer": "<your grounded answer as a plain string>",
  "source_citations": ["<file_name_1>", "<file_name_2>"],
  "confidence_score": <float between 0.0 and 1.0>
}
"""


def get_structured_answer(groq_client, system_prompt, user_query, model="openai/gpt-oss-120b", temperature=0.1):
    """
    Calls Groq with JSON mode (response_format={"type": "json_object"}) using
    the SAME system_prompt project7.py already builds (grounding context +
    instructions), with the schema appended, then validates the result with
    Pydantic.

    Returns: a validated RAGAnswer instance on success.
    Raises: ValueError (with the raw text attached) on JSON/validation
    failure, so the caller can catch it and fall back to the existing
    streaming path — this never silently corrupts chat_history.
    """
    completion = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt + STRUCTURED_SYSTEM_SUFFIX},
            {"role": "user", "content": f"Query: {user_query}"},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )

    raw_text = completion.choices[0].message.content

    try:
        parsed = json.loads(raw_text)
        return RAGAnswer(**parsed)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Structured output validation failed: {e}\nRaw response: {raw_text}")
