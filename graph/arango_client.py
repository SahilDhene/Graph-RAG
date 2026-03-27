"""ArangoDB connection, schema creation, and parameterized query helpers."""
from __future__ import annotations

from arango import ArangoClient as _ArangoClient
from arango.database import StandardDatabase

import config


def get_db() -> StandardDatabase:
    client = _ArangoClient(hosts=config.ARANGO_URL)
    sys_db = client.db("_system", username=config.ARANGO_USER,
                       password=config.ARANGO_PASSWORD, verify=True)

    if not sys_db.has_database(config.ARANGO_DB):
        sys_db.create_database(config.ARANGO_DB)

    db = client.db(config.ARANGO_DB, username=config.ARANGO_USER,
                   password=config.ARANGO_PASSWORD, verify=True)
    return db


def setup_schema(db: StandardDatabase) -> None:
    """Create collections, edge collections, graph, and indexes."""
    # --- Vertex collections ---
    for name in (config.COL_DOCUMENTS, config.COL_CHUNKS, config.COL_ENTITIES):
        if not db.has_collection(name):
            db.create_collection(name)

    # --- Edge collections ---
    for name in (config.EDGE_RELATIONS, config.EDGE_MENTIONS, config.EDGE_BELONGS):
        if not db.has_collection(name):
            db.create_collection(name, edge=True)

    # --- Named graph ---
    if not db.has_graph(config.GRAPH_NAME):
        db.create_graph(
            config.GRAPH_NAME,
            edge_definitions=[
                {
                    "edge_collection": config.EDGE_RELATIONS,
                    "from_vertex_collections": [config.COL_ENTITIES],
                    "to_vertex_collections": [config.COL_ENTITIES],
                },
                {
                    "edge_collection": config.EDGE_MENTIONS,
                    "from_vertex_collections": [config.COL_CHUNKS],
                    "to_vertex_collections": [config.COL_ENTITIES],
                },
                {
                    "edge_collection": config.EDGE_BELONGS,
                    "from_vertex_collections": [config.COL_CHUNKS],
                    "to_vertex_collections": [config.COL_DOCUMENTS],
                },
            ],
        )

    # --- Persistent index on source field ---
    chunks = db.collection(config.COL_CHUNKS)
    _ensure_index(chunks, {"type": "persistent", "fields": ["source"], "unique": False})

    entities = db.collection(config.COL_ENTITIES)
    _ensure_index(entities, {"type": "persistent", "fields": ["name"], "unique": False})

    # --- Vector indexes (ArangoDB 3.12+) ---
    _ensure_vector_index(chunks, "embedding", config.EMBEDDING_DIM, "cosine")
    _ensure_vector_index(entities, "embedding", config.EMBEDDING_DIM, "cosine")


def _ensure_index(col, index_def: dict) -> None:
    try:
        col.add_index(index_def)
    except Exception:
        pass  # index already exists


def _ensure_vector_index(col, field: str, dim: int, metric: str) -> None:
    try:
        col.add_index({
            "type": "vector",
            "fields": [field],
            "params": {"metric": metric, "dimension": dim, "nLists": 10},
        })
    except Exception:
        pass  # not supported or already exists — fallback handled in searcher


# ---------------------------------------------------------------------------
# Parameterized AQL helpers (never use string interpolation for user data)
# ---------------------------------------------------------------------------

def upsert_document(db: StandardDatabase, doc: dict) -> str:
    """Insert or update a document vertex, return _id."""
    aql = """
    UPSERT {source: @source}
    INSERT @doc
    UPDATE @doc
    IN @@col
    RETURN NEW._id
    """
    cursor = db.aql.execute(aql, bind_vars={
        "source": doc["source"],
        "doc": doc,
        "@col": config.COL_DOCUMENTS,
    })
    return list(cursor)[0]


def upsert_chunk(db: StandardDatabase, chunk: dict) -> str:
    aql = """
    UPSERT {uid: @uid}
    INSERT @chunk
    UPDATE @chunk
    IN @@col
    RETURN NEW._id
    """
    cursor = db.aql.execute(aql, bind_vars={
        "uid": chunk["uid"],
        "chunk": chunk,
        "@col": config.COL_CHUNKS,
    })
    return list(cursor)[0]


def upsert_entity(db: StandardDatabase, entity: dict) -> str:
    """Upsert entity by (name, type) pair."""
    aql = """
    UPSERT {name: @name, type: @type}
    INSERT @entity
    UPDATE {description: @desc}
    IN @@col
    RETURN NEW._id
    """
    cursor = db.aql.execute(aql, bind_vars={
        "name": entity["name"],
        "type": entity["type"],
        "desc": entity.get("description", ""),
        "entity": entity,
        "@col": config.COL_ENTITIES,
    })
    return list(cursor)[0]


def insert_edge(db: StandardDatabase, collection: str,
                from_id: str, to_id: str, attrs: dict | None = None) -> None:
    doc = {"_from": from_id, "_to": to_id, **(attrs or {})}
    aql = """
    UPSERT {_from: @from, _to: @to}
    INSERT @doc
    UPDATE {}
    IN @@col
    """
    db.aql.execute(aql, bind_vars={
        "from": from_id,
        "to": to_id,
        "doc": doc,
        "@col": collection,
    })
