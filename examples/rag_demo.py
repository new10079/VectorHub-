#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VectorHub Phase 3 demo: an end-to-end RAG (Retrieval-Augmented Generation) pipeline.

Pipeline
--------
1. Load a source document (.txt, optionally .pdf)
2. Split it into overlapping chunks (with source/chunk_id metadata)
3. Embed each chunk with sentence-transformers
4. Index the embeddings in a VectorCollection (reusing the existing
   add() / search() / save() / load() API from Phase 1 & 2 -- no core
   index changes)
5. Embed the user's query and retrieve the top-k most relevant chunks
6. Generate an answer -- either a dependency-free "mock" summary, or (with
   an OPENAI_API_KEY) a real call to the OpenAI API

Usage
-----
    python examples/rag_demo.py --data examples/data/sample.txt \\
        --query "What is VectorHub?" --top-k 3 --llm mock

    # Persist the index and reuse it on the next run:
    python examples/rag_demo.py --data examples/data/sample.txt \\
        --query "What is VectorHub?" --top-k 3 --llm mock \\
        --persist-path examples/data/vectorhub_index.bin
"""

import argparse
import os
import sys

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Auto-load API keys (OPENAI_API_KEY / DEEPSEEK_API_KEY / ...) from a .env
# file at the repo root, if python-dotenv is installed. Purely optional --
# env vars set some other way (shell, CI secrets) still work without it.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

from vectorhub import VectorCollection
from vectorhub.rag import (
    load_document,
    chunk_text,
    get_default_embedder,
    index_chunks,
    retrieve,
    generate_answer,
)


def parse_args():
    parser = argparse.ArgumentParser(description="VectorHub RAG demo (Phase 3)")
    parser.add_argument("--data", required=True,
                         help="Path to the source document (.txt, or .pdf with pypdf installed)")
    parser.add_argument("--query", required=True, help="Question to ask")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--chunk-size", type=int, default=256, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=32, help="Overlap between chunks in characters")
    parser.add_argument("--llm", choices=["mock", "openai", "deepseek"], default="mock",
                         help="Answer generation backend (default: mock, no external API needed)")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                         help="sentence-transformers model name")
    parser.add_argument("--persist-path", default=None,
                         help="If set, save the built index here; reload it if it already exists")
    return parser.parse_args()


def main():
    args = parse_args()

    print("VectorHub RAG Demo (Phase 3)")
    print("=" * 60)

    # ── embeddings (needed for both indexing and query, regardless of --llm) ──
    try:
        embed = get_default_embedder(args.embedding_model)
    except (ImportError, RuntimeError) as e:
        print(f"\n[ERROR] Could not initialize the embedding model:\n  {e}")
        sys.exit(1)

    meta_path = (args.persist_path + ".meta.json") if args.persist_path else None
    can_load = args.persist_path and os.path.exists(args.persist_path) and os.path.exists(meta_path)

    if can_load:
        # ── reuse a previously persisted index (Phase 2 save/load) ──
        print(f"[OK] Found existing index at '{args.persist_path}', loading instead of re-indexing")
        db = VectorCollection.load(args.persist_path)
    else:
        # ── 1. load ──
        text = load_document(args.data)
        source_name = os.path.basename(args.data)

        # ── 2. chunk ──
        chunks = chunk_text(text, source=source_name,
                             chunk_size=args.chunk_size, overlap=args.overlap)
        if not chunks:
            print("[ERROR] No chunks were produced from the input document; aborting.")
            sys.exit(1)
        print(f"[OK] Loaded '{args.data}' -> {len(chunks)} chunks "
              f"(chunk_size={args.chunk_size}, overlap={args.overlap})")

        # ── 3 & 4. embed + index ──
        sample_dim = len(embed([chunks[0]["text"]])[0])
        db = VectorCollection(dim=sample_dim, metric="cosine")
        index_chunks(db, chunks, embed)
        print(f"[OK] Indexed {len(chunks)} chunks into VectorCollection(dim={sample_dim}, metric=cosine)")

        if args.persist_path:
            db.save(args.persist_path)
            print(f"[OK] Persisted index to '{args.persist_path}' (+ .meta.json sidecar)")

    # ── 5. retrieve ──
    results = retrieve(db, args.query, embed, k=args.top_k)

    print(f"\n[OK] Top-{args.top_k} retrieved chunks for query: {args.query!r}")
    for r in results:
        text_preview = r.get("text", "")
        if len(text_preview) > 120:
            text_preview = text_preview[:120] + "..."
        print(f"  id={r['id']:<4} distance={r['distance']:.4f} "
              f"source={r.get('source')} chunk_id={r.get('chunk_id')}")
        print(f"    text: {text_preview}")

    # ── 6. generate ──
    print("\n" + "=" * 60)
    print(f"Generating answer (mode={args.llm})...")
    try:
        answer = generate_answer(args.query, results, mode=args.llm)
    except (RuntimeError, ImportError) as e:
        print(f"\n[ERROR] Answer generation failed:\n  {e}")
        sys.exit(1)

    print("\nAnswer:\n" + answer)


if __name__ == "__main__":
    main()
