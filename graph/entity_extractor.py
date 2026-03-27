"""Entity and relationship extraction via Groq (llama-3.1-8b-instant).

Returns structured JSON; uses tenacity for rate-limit retries.
"""
from __future__ import annotations

import json
import re
import time

from groq import Groq, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import config

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise EnvironmentError("GROQ_API_KEY not set in environment.")
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


_SYSTEM_PROMPT = """You are an expert knowledge graph builder.
Given a text chunk, extract:
1. ENTITIES: important named concepts (people, organizations, locations, technologies, events, concepts).
2. RELATIONSHIPS: directed relations between entities found in the text.

Respond ONLY with valid JSON in this exact schema — no prose:
{
  "entities": [
    {"name": "string", "type": "PERSON|ORG|LOCATION|CONCEPT|EVENT|TECH|OTHER",
     "description": "one sentence"}
  ],
  "relationships": [
    {"source": "entity name", "relation": "verb phrase", "target": "entity name",
     "description": "one sentence"}
  ]
}

Rules:
- Keep entity names normalized (title-case, no duplicates).
- Extract at most 15 entities and 20 relationships.
- Only include relationships where BOTH source and target appear in your entities list.
- If nothing meaningful exists, return {"entities": [], "relationships": []}.
"""


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
)
def _call_groq(text: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=config.GROQ_EXTRACT_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"TEXT:\n{text[:3000]}"},  # cap per call
        ],
        temperature=0.0,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _safe_parse(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Attempt to extract JSON block from mixed output
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"entities": [], "relationships": []}


def extract(chunk_text: str) -> dict:
    """Return {'entities': [...], 'relationships': [...]} for a chunk."""
    raw = _call_groq(chunk_text)
    result = _safe_parse(raw)

    # Validate and sanitize
    entities = [
        e for e in result.get("entities", [])
        if isinstance(e.get("name"), str) and e["name"].strip()
    ]
    entity_names = {e["name"] for e in entities}

    relationships = [
        r for r in result.get("relationships", [])
        if (isinstance(r.get("source"), str) and
            isinstance(r.get("target"), str) and
            r["source"] in entity_names and
            r["target"] in entity_names)
    ]

    return {"entities": entities[:15], "relationships": relationships[:20]}
