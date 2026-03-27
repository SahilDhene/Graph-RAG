"""Answer generation via Groq (llama-3.3-70b-versatile)."""
from __future__ import annotations

from groq import Groq, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import config
from retrieval.searcher import RetrievedContext

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise EnvironmentError("GROQ_API_KEY not set in environment.")
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


_SYSTEM = """You are a precise, helpful assistant answering questions from a knowledge graph.
You are given ENTITY CONTEXT and DOCUMENT CHUNKS retrieved from the user's documents.

Formatting rules (strictly follow):
- Use **markdown** for all responses.
- Use `##` headings to separate major sections when the answer has multiple parts.
- Use bullet lists (`-`) for enumerating items, people, facts, or steps.
- Use numbered lists for sequences or ranked items.
- Use **bold** for key terms, names, and important values.
- Keep paragraphs short (2-3 sentences max).
- If citing a source, write it as: *(source: filename)*

Content rules:
- Answer ONLY from the provided context.
- If the context is insufficient, state clearly: "The documents do not contain enough information to answer this."
- Never invent facts not present in the context.
"""


def _build_user_message(query: str, context: RetrievedContext) -> str:
    parts: list[str] = [f"QUESTION: {query}\n"]

    if context.entity_summaries:
        parts.append("ENTITY CONTEXT:")
        parts.extend(f"  • {s}" for s in context.entity_summaries[:15])
        parts.append("")

    if context.chunks:
        parts.append("DOCUMENT CHUNKS:")
        for i, chunk in enumerate(context.chunks[:config.TOP_K_CHUNKS], 1):
            src = chunk.get("source", "unknown")
            text = chunk.get("text", "")[:800]  # cap per chunk
            parts.append(f"[{i}] (source: {src})\n{text}")
            parts.append("")

    return "\n".join(parts)


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(4),
)
def generate(query: str, context: RetrievedContext) -> str:
    """Generate a grounded answer from retrieved context."""
    user_msg = _build_user_message(query, context)
    response = _get_client().chat.completions.create(
        model=config.GROQ_ANSWER_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()
