"""GraphRAG CLI

Usage
-----
  # Ingest a single file
  python main.py ingest path/to/document.pdf

  # Ingest all files in a directory
  python main.py ingest path/to/docs/

  # Query (local mode — entity-centric, default)
  python main.py query "Who are the key stakeholders?"

  # Query (global mode — thematic / broad questions)
  python main.py query "Summarize the main themes" --mode global
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from graph.arango_client import get_db, setup_schema


def cmd_ingest(args: argparse.Namespace) -> None:
    from pipeline import ingest_directory, ingest_file

    db = get_db()
    setup_schema(db)

    target = Path(args.path)
    if target.is_dir():
        ingest_directory(db, target)
    elif target.is_file():
        setup_schema(db)
        n = ingest_file(db, target)
        print(f"Ingested {n} chunks from {target.name}")

        from embeddings.embedder import embed
        from graph.graph_builder import embed_entities
        embed_entities(db, embed_fn=embed)
    else:
        print(f"Error: {target} does not exist.", file=sys.stderr)
        sys.exit(1)


def cmd_query(args: argparse.Namespace) -> None:
    from embeddings.embedder import embed_query
    from generation.answer_gen import generate
    from retrieval.searcher import search

    db = get_db()
    query = args.query
    mode = args.mode

    print(f"\nSearching [{mode}] for: {query!r}\n")
    query_vec = embed_query(query)
    context = search(db, query_vec, mode=mode)

    print(f"Retrieved {len(context.chunks)} chunks, "
          f"{len(context.entity_summaries)} entity summaries.\n")

    answer = generate(query, context)
    print("=" * 60)
    print(answer)
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="graphrag",
        description="GraphRAG: graph-augmented retrieval over PDFs and DOCX files.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Index a file or directory.")
    p_ingest.add_argument("path", help="Path to a .pdf/.docx file or directory.")
    p_ingest.set_defaults(func=cmd_ingest)

    # query
    p_query = sub.add_parser("query", help="Ask a question.")
    p_query.add_argument("query", help="Natural-language question.")
    p_query.add_argument(
        "--mode", choices=["local", "global"], default="local",
        help="local = entity-centric (default); global = thematic/broad.",
    )
    p_query.set_defaults(func=cmd_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
