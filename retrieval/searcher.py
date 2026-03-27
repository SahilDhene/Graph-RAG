"""Hybrid retrieval: vector similarity + graph neighbourhood expansion.

Two modes
---------
local_search  — best for specific, entity-centric questions.
global_search — best for broad, thematic questions (uses entity-cluster summaries).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from arango.database import StandardDatabase

import config


@dataclass
class RetrievedContext:
    chunks: list[dict]        # list of chunk docs
    entity_summaries: list[str]  # entity neighbourhood snippets


# ---------------------------------------------------------------------------
# Cosine fallback (when ArangoDB vector index unavailable)
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _vector_search_chunks(
    db: StandardDatabase,
    query_vec: list[float],
    top_k: int,
) -> list[dict]:
    """Try native ArangoDB vector search; fall back to in-memory cosine."""
    try:
        aql = """
        FOR doc IN @@col
          LET score = APPROX_NEAR_COSINE(doc.embedding, @qvec)
          FILTER score != null
          SORT score DESC
          LIMIT @k
          RETURN MERGE(doc, {_score: score})
        """
        cursor = db.aql.execute(aql, bind_vars={
            "@col": config.COL_CHUNKS,
            "qvec": query_vec,
            "k": top_k,
        })
        results = list(cursor)
        if results:
            return results
    except Exception:
        pass

    # Fallback: scan all chunks and rank by cosine
    cursor = db.aql.execute(
        "FOR doc IN @@col FILTER LENGTH(doc.embedding) > 0 RETURN doc",
        bind_vars={"@col": config.COL_CHUNKS},
    )
    docs = list(cursor)
    scored = sorted(docs, key=lambda d: _cosine(query_vec, d["embedding"]), reverse=True)
    return scored[:top_k]


def _vector_search_entities(
    db: StandardDatabase,
    query_vec: list[float],
    top_k: int,
) -> list[dict]:
    try:
        aql = """
        FOR doc IN @@col
          LET score = APPROX_NEAR_COSINE(doc.embedding, @qvec)
          FILTER score != null
          SORT score DESC
          LIMIT @k
          RETURN doc
        """
        cursor = db.aql.execute(aql, bind_vars={
            "@col": config.COL_ENTITIES,
            "qvec": query_vec,
            "k": top_k,
        })
        results = list(cursor)
        if results:
            return results
    except Exception:
        pass

    cursor = db.aql.execute(
        "FOR doc IN @@col FILTER LENGTH(doc.embedding) > 0 RETURN doc",
        bind_vars={"@col": config.COL_ENTITIES},
    )
    docs = list(cursor)
    scored = sorted(docs, key=lambda d: _cosine(query_vec, d["embedding"]), reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Graph neighbourhood expansion
# ---------------------------------------------------------------------------

def _get_chunks_for_entities(
    db: StandardDatabase,
    entity_ids: list[str],
    limit: int = 10,
) -> list[dict]:
    """Retrieve chunks that mention any of the given entities."""
    if not entity_ids:
        return []
    aql = """
    FOR eid IN @entity_ids
      FOR chunk, edge IN 1..1 INBOUND eid @@mentions
        LIMIT @limit
        RETURN chunk
    """
    cursor = db.aql.execute(aql, bind_vars={
        "entity_ids": entity_ids,
        "@mentions": config.EDGE_MENTIONS,
        "limit": limit,
    })
    seen, result = set(), []
    for c in cursor:
        uid = c.get("uid", c.get("_id"))
        if uid not in seen:
            seen.add(uid)
            result.append(c)
    return result


def _expand_entity_neighbourhood(
    db: StandardDatabase,
    entity_ids: list[str],
    depth: int = 2,
) -> list[dict]:
    """Traverse the entity graph to find related entities."""
    if not entity_ids:
        return []
    aql = """
    FOR start_id IN @start_ids
      FOR v IN 1..@depth ANY start_id @@relations
        LIMIT 30
        RETURN v
    """
    cursor = db.aql.execute(aql, bind_vars={
        "start_ids": entity_ids,
        "depth": depth,
        "@relations": config.EDGE_RELATIONS,
    })
    seen, result = set(), []
    for v in cursor:
        if v.get("_id") not in seen:
            seen.add(v["_id"])
            result.append(v)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def local_search(
    db: StandardDatabase,
    query_vec: list[float],
    top_k: int = config.TOP_K_CHUNKS,
) -> RetrievedContext:
    """
    1. Find top-K similar chunks via vector search.
    2. Find top-K similar entities.
    3. Expand entity neighbourhood (graph hops).
    4. Collect additional chunks from expanded entities.
    5. Merge and deduplicate.
    """
    # Step 1: vector-similar chunks
    seed_chunks = _vector_search_chunks(db, query_vec, top_k)
    seen_uids: set[str] = {c.get("uid", c.get("_id")) for c in seed_chunks}

    # Step 2: vector-similar entities
    seed_entities = _vector_search_entities(db, query_vec, config.TOP_K_ENTITIES)
    seed_entity_ids = [e["_id"] for e in seed_entities]

    # Step 3: expand graph neighbourhood
    expanded_entities = _expand_entity_neighbourhood(
        db, seed_entity_ids, config.GRAPH_HOP_DEPTH
    )
    all_entity_ids = list({e["_id"] for e in seed_entities + expanded_entities})

    # Step 4: chunks from expanded entities
    extra_chunks = _get_chunks_for_entities(db, all_entity_ids, limit=top_k * 2)
    for c in extra_chunks:
        uid = c.get("uid", c.get("_id"))
        if uid not in seen_uids:
            seed_chunks.append(c)
            seen_uids.add(uid)

    # Step 5: entity neighbourhood summaries
    entity_summaries = [
        f"{e['name']} ({e.get('type', '?')}): {e.get('description', '')}"
        for e in (seed_entities + expanded_entities)
        if e.get("name")
    ]

    return RetrievedContext(
        chunks=seed_chunks[: top_k * 2],
        entity_summaries=list(dict.fromkeys(entity_summaries))[:20],
    )


def global_search(
    db: StandardDatabase,
    query_vec: list[float],
    top_k: int = config.TOP_K_CHUNKS,
) -> RetrievedContext:
    """
    Community-aware search: groups entities by type, returns a broad set of
    chunks and cross-cluster entity summaries.
    """
    # Get top entities matching the query
    top_entities = _vector_search_entities(db, query_vec, top_k=30)

    # Group by type (our simplified "communities")
    communities: dict[str, list[dict]] = {}
    for ent in top_entities:
        t = ent.get("type", "OTHER")
        communities.setdefault(t, []).append(ent)

    # Pick representative entities from each community
    rep_ids: list[str] = []
    summaries: list[str] = []
    for etype, members in communities.items():
        rep = members[:3]
        rep_ids.extend(m["_id"] for m in rep)
        desc = "; ".join(f"{m['name']}: {m.get('description','')}" for m in rep)
        summaries.append(f"[{etype}] {desc}")

    # Chunks from representative entities
    chunks = _get_chunks_for_entities(db, rep_ids, limit=top_k * 3)

    # Also include vector-similar chunks
    vec_chunks = _vector_search_chunks(db, query_vec, top_k)
    seen = {c.get("uid", c.get("_id")) for c in chunks}
    for c in vec_chunks:
        uid = c.get("uid", c.get("_id"))
        if uid not in seen:
            chunks.append(c)
            seen.add(uid)

    return RetrievedContext(
        chunks=chunks[:top_k * 2],
        entity_summaries=summaries[:15],
    )


def search(
    db: StandardDatabase,
    query_vec: list[float],
    mode: str = "local",
    top_k: int = config.TOP_K_CHUNKS,
) -> RetrievedContext:
    if mode == "global":
        return global_search(db, query_vec, top_k)
    return local_search(db, query_vec, top_k)
