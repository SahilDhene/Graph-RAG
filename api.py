"""FastAPI server for GraphRAG.

Endpoints
---------
POST /api/ingest          — upload PDF or DOCX, triggers ingestion
GET  /api/query           — SSE streaming query (EventSource)
GET  /api/documents       — list ingested documents
GET  /api/stats           — database statistics
DELETE /api/documents/{id} — remove a document and its chunks/entities
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
from embeddings.embedder import embed, embed_query
from generation.answer_gen import _build_user_message, _get_client
from graph.arango_client import get_db, setup_schema
from graph.graph_builder import build_graph, embed_entities
from ingestion.chunker import chunk_document
from ingestion.docx_parser import parse_docx
from ingestion.pdf_parser import parse_pdf
from retrieval.searcher import search

app = FastAPI(title="GraphRAG", version="1.0.0")

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_initialized_db():
    try:
        db = get_db()
        setup_schema(db)
        return db
    except Exception as e:
        msg = str(e)
        if "bad username/password" in msg or "token is expired" in msg:
            raise HTTPException(status_code=503,
                detail="ArangoDB auth failed — check ARANGO_USER/ARANGO_PASSWORD in .env")
        if "Connection refused" in msg or "Failed to establish" in msg:
            raise HTTPException(status_code=503,
                detail="ArangoDB not reachable — is Docker running? (docker compose up -d)")
        raise HTTPException(status_code=503, detail=f"Database error: {msg}")


def _parse_uploaded(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path)
    if ext in (".docx", ".doc"):
        return parse_docx(path)
    raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    index = STATIC_DIR / "index.html"
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.post("/api/ingest")
async def ingest_file(file: UploadFile = File(...)):
    """Accept a PDF or DOCX upload, ingest it into the graph."""
    allowed = {".pdf", ".docx", ".doc"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400,
                            detail=f"Unsupported file type '{ext}'. Allowed: {allowed}")

    # Save upload to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        db = _get_initialized_db()
        doc = _parse_uploaded(tmp_path)

        # Use the original filename as the source identifier
        doc.source = file.filename
        chunks = chunk_document(doc.source, doc.pages)
        if not chunks:
            raise HTTPException(status_code=422,
                                detail="No text could be extracted from the file.")

        build_graph(db, chunks, doc.source, embed_fn=embed)
        embed_entities(db, embed_fn=embed)

        return JSONResponse({
            "status": "ok",
            "filename": file.filename,
            "chunks": len(chunks),
            "pages": len(doc.pages),
        })
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/api/query")
async def query_stream(q: str, mode: str = "local"):
    """Server-Sent Events stream for a query."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if mode not in ("local", "global"):
        raise HTTPException(status_code=400, detail="mode must be 'local' or 'global'.")

    async def event_stream():
        try:
            db = _get_initialized_db()

            # Embed query
            yield _sse("status", {"message": "Embedding query..."})
            query_vec = await asyncio.to_thread(embed_query, q)

            # Retrieve context
            yield _sse("status", {"message": f"Searching graph [{mode}]..."})
            context = await asyncio.to_thread(search, db, query_vec, mode)

            # Send entity summaries
            yield _sse("entities", {"summaries": context.entity_summaries})
            yield _sse("status", {"message": "Generating answer..."})

            # Stream answer from Groq
            user_msg = _build_user_message(q, context)
            client = _get_client()
            stream = client.chat.completions.create(
                model=config.GROQ_ANSWER_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=1024,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield _sse("token", {"text": delta})

            # Send source citations
            sources = list({c.get("source", "") for c in context.chunks if c.get("source")})
            yield _sse("done", {"sources": sources})

        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/documents")
async def list_documents():
    db = _get_initialized_db()
    cursor = db.aql.execute(
        "FOR d IN @@col RETURN {id: d._id, source: d.source, chunks: d.chunk_count}",
        bind_vars={"@col": config.COL_DOCUMENTS},
    )
    return JSONResponse({"documents": list(cursor)})


@app.get("/api/stats")
async def stats():
    db = _get_initialized_db()

    def _count(col: str) -> int:
        try:
            return db.collection(col).count()
        except Exception:
            return 0

    return JSONResponse({
        "documents": _count(config.COL_DOCUMENTS),
        "chunks": _count(config.COL_CHUNKS),
        "entities": _count(config.COL_ENTITIES),
        "relationships": _count(config.EDGE_RELATIONS),
    })


@app.delete("/api/documents/{doc_id:path}")
async def delete_document(doc_id: str):
    """Remove a document and cascade-delete its chunks and entity mentions."""
    db = _get_initialized_db()

    # 1. Collect chunk _ids that belong to this document
    cursor = db.aql.execute(
        "FOR e IN @@col FILTER e._to == @doc_id RETURN e._from",
        bind_vars={"@col": config.EDGE_BELONGS, "doc_id": doc_id},
    )
    chunk_ids = list(cursor)

    if chunk_ids:
        # 2. Delete mention edges for those chunks
        db.aql.execute(
            "FOR e IN @@col FILTER e._from IN @ids REMOVE e IN @@col",
            bind_vars={"@col": config.EDGE_MENTIONS, "ids": chunk_ids},
        )
        # 3. Delete belongs_to edges
        db.aql.execute(
            "FOR e IN @@col FILTER e._from IN @ids REMOVE e IN @@col",
            bind_vars={"@col": config.EDGE_BELONGS, "ids": chunk_ids},
        )
        # 4. Delete chunk documents (use keys only)
        keys = [cid.split("/")[-1] for cid in chunk_ids]
        db.aql.execute(
            "FOR k IN @keys REMOVE k IN @@col",
            bind_vars={"keys": keys, "@col": config.COL_CHUNKS},
        )

    # 5. Delete document vertex
    try:
        db.collection(config.COL_DOCUMENTS).delete(doc_id.split("/")[-1])
    except Exception:
        pass

    return JSONResponse({"status": "deleted", "id": doc_id})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a precise, helpful assistant answering questions from a knowledge graph.
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
- If the context is insufficient, state: "The documents do not contain enough information to answer this."
- Never invent facts not present in the context."""


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
