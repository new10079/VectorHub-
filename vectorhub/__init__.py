"""VectorHub: Lightweight vector database with HNSW indexing."""

from .collection import VectorCollection

__version__ = "0.1.0"
__all__ = ["VectorCollection"]

# Phase 3 (RAG helpers) lives in vectorhub.rag and is intentionally NOT
# imported here by default -- it's a thin add-on over VectorCollection, not
# part of the core API, and keeps `import vectorhub` free of any embedding
# backend dependency. Use `from vectorhub.rag import ...` or
# `from vectorhub import rag`.
