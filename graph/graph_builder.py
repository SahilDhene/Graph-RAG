"""Build the knowledge graph in ArangoDB from chunks + extracted entities."""
from __future__ import annotations

import time
from typing import Callable

from arango.database import StandardDatabase
from tqdm import tqdm

import config
from graph.arango_client import (
    insert_edge,
    upsert_chunk,
    upsert_document,
    upsert_entity,
)
from graph.entity_extractor import extract
from ingestion.chunker import Chunk


def build_graph(
    db: StandardDatabase,
    chunks: list[Chunk],
    source_path: str,
    embed_fn: Callable[[list[str]], list[list[float]]],
    groq_delay: float = 2.1,  # seconds between extraction calls (~28 RPM)
) -> None:
    """
    1. Upsert document vertex.
    2. For each chunk: embed, upsert, link to document.
    3. Extract entities/relations, upsert, link to chunks.
    """
    # --- Document vertex ---
    doc_id = upsert_document(db, {
        "source": source_path,
        "chunk_count": len(chunks),
    })

    # --- Embed all chunks in one batch call ---
    texts = [c.text for c in chunks]
    embeddings = embed_fn(texts)

    # --- Process each chunk ---
    for chunk, embedding in tqdm(zip(chunks, embeddings),
                                 total=len(chunks), desc="Building graph", unit="chunk"):
        chunk_doc = {
            "uid": chunk.uid,
            "text": chunk.text,
            "source": chunk.source,
            "page_idx": chunk.page_idx,
            "chunk_idx": chunk.chunk_idx,
            "embedding": embedding,
        }
        chunk_id = upsert_chunk(db, chunk_doc)

        # chunk -> document
        insert_edge(db, config.EDGE_BELONGS, chunk_id, doc_id)

        # --- Entity extraction ---
        extracted = extract(chunk.text)
        time.sleep(groq_delay)  # respect Groq free-tier rate limit

        entity_ids: dict[str, str] = {}  # name -> _id
        for ent in extracted["entities"]:
            ent_doc = {
                "name": ent["name"],
                "type": ent["type"],
                "description": ent.get("description", ""),
                "embedding": [],  # filled in embed_entities() pass
            }
            ent_id = upsert_entity(db, ent_doc)
            entity_ids[ent["name"]] = ent_id

            # chunk mentions entity
            insert_edge(db, config.EDGE_MENTIONS, chunk_id, ent_id)

        # --- Relationships ---
        for rel in extracted["relationships"]:
            src_id = entity_ids.get(rel["source"])
            tgt_id = entity_ids.get(rel["target"])
            if src_id and tgt_id:
                insert_edge(db, config.EDGE_RELATIONS, src_id, tgt_id, {
                    "relation": rel.get("relation", ""),
                    "description": rel.get("description", ""),
                })


def embed_entities(
    db: StandardDatabase,
    embed_fn: Callable[[list[str]], list[list[float]]],
    batch_size: int = 100,
) -> None:
    """Fill in embedding vectors for all entities that don't have one yet."""
    aql = """
    FOR e IN @@col
      FILTER LENGTH(e.embedding) == 0
      RETURN {_id: e._id, name: e.name, description: e.description}
    """
    cursor = db.aql.execute(aql, bind_vars={"@col": config.COL_ENTITIES})
    rows = list(cursor)
    if not rows:
        return

    for i in tqdm(range(0, len(rows), batch_size), desc="Embedding entities"):
        batch = rows[i : i + batch_size]
        texts = [f"{r['name']}: {r['description']}" for r in batch]
        vecs = embed_fn(texts)
        for row, vec in zip(batch, vecs):
            key = row["_id"].split("/")[-1]
            db.aql.execute(
                "UPDATE @key WITH {embedding: @vec} IN @@col",
                bind_vars={"key": key, "vec": vec,
                           "@col": config.COL_ENTITIES},
            )
