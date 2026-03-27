"""Ingestion pipeline: parse -> chunk -> embed -> build graph."""
from __future__ import annotations

from pathlib import Path

from arango.database import StandardDatabase
from tqdm import tqdm

from embeddings.embedder import embed
from graph.arango_client import setup_schema
from graph.graph_builder import build_graph, embed_entities
from ingestion.chunker import chunk_document
from ingestion.docx_parser import parse_docx
from ingestion.pdf_parser import parse_pdf


def _parse(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path)
    if ext in (".docx", ".doc"):
        return parse_docx(path)
    raise ValueError(f"Unsupported file type: {ext}")


def ingest_file(db: StandardDatabase, file_path: str | Path) -> int:
    """Parse, chunk, embed, and build graph for one file. Returns chunk count."""
    path = Path(file_path)
    doc = _parse(path)
    chunks = chunk_document(doc.source, doc.pages)
    if not chunks:
        print(f"[WARN] No chunks extracted from {path.name}")
        return 0

    build_graph(db, chunks, doc.source, embed_fn=embed)
    return len(chunks)


def ingest_directory(db: StandardDatabase, directory: str | Path,
                     glob: str = "**/*") -> None:
    """Ingest all supported files under a directory."""
    setup_schema(db)
    directory = Path(directory)
    files = [
        f for f in directory.glob(glob)
        if f.suffix.lower() in (".pdf", ".docx", ".doc") and f.is_file()
    ]
    if not files:
        print(f"No supported files found in {directory}")
        return

    total_chunks = 0
    for f in tqdm(files, desc="Ingesting files", unit="file"):
        try:
            n = ingest_file(db, f)
            total_chunks += n
            print(f"  ✓ {f.name} → {n} chunks")
        except Exception as exc:
            print(f"  ✗ {f.name} — {exc}")

    print(f"\nEmbedding entities ...")
    embed_entities(db, embed_fn=embed)
    print(f"Done. Total chunks indexed: {total_chunks}")
