"""Local embeddings using fastembed (ONNX, no GPU needed).

Model: BAAI/bge-small-en-v1.5  — 384 dims, ~130 MB download on first run.
No API key required. Runs fully on CPU.
"""
from __future__ import annotations

from functools import lru_cache

from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    return TextEmbedding(model_name=MODEL_NAME)


def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vecs = list(_model().embed(texts))
    return [v.tolist() for v in vecs]


def embed_query(query: str) -> list[float]:
    vecs = list(_model().query_embed(query))
    return vecs[0].tolist()
