"""Phase 3 tests: RAG workflow (chunking, indexing, retrieval, mock generation).

These tests do NOT depend on:
- sentence-transformers actually being installed / downloading a model
- a real OpenAI API key being present

Embeddings are produced by a small deterministic, hash-based fake embedder
defined in this file, so the tests stay fast and fully offline while still
exercising the real VectorCollection + vectorhub.rag code paths.
"""

import math
import os

import pytest

from vectorhub import VectorCollection
from vectorhub.rag import (
    build_prompt,
    chunk_text,
    generate_answer,
    index_chunks,
    mock_generate_answer,
    openai_generate_answer,
    retrieve,
)

FAKE_DIM = 16


def fake_embed(texts):
    """Deterministic, dependency-free stand-in for a real embedding model.

    Hashes words into a fixed-size bag-of-words vector and L2-normalizes it.
    Good enough to make semantically-similar text land closer together for
    retrieval tests, without requiring network access or a real model.
    """
    vectors = []
    for text in texts:
        vec = [0.0] * FAKE_DIM
        for word in text.lower().split():
            idx = sum(ord(c) for c in word) % FAKE_DIM
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        vectors.append([x / norm for x in vec])
    return vectors


SAMPLE_TEXT = (
    "VectorHub is a lightweight hybrid vector database engine. "
    "It combines a Python client layer with a C++ HNSW index core. "
    "The project targets RAG applications, recommendation systems, "
    "and semantic search use cases. "
    "Bananas are a good source of potassium and are yellow when ripe."
)


# ─── chunk_text ──────────────────────────────────────────────────────────

class TestChunkText:
    def test_produces_nonempty_chunks(self):
        chunks = chunk_text(SAMPLE_TEXT, source="sample.txt", chunk_size=64, overlap=8)
        assert len(chunks) > 0
        for c in chunks:
            assert c["text"].strip() != ""

    def test_chunk_metadata_fields(self):
        chunks = chunk_text(SAMPLE_TEXT, source="sample.txt", chunk_size=64, overlap=8)
        for i, c in enumerate(chunks):
            assert c["chunk_id"] == i
            assert c["source"] == "sample.txt"
            assert isinstance(c["text"], str)

    def test_no_empty_chunks_from_whitespace_only_text(self):
        chunks = chunk_text("   \n\n   \t  ", source="empty.txt")
        assert chunks == []

    def test_empty_string_returns_no_chunks(self):
        assert chunk_text("", source="x.txt") == []

    def test_short_text_single_chunk(self):
        chunks = chunk_text("hello world", source="x.txt", chunk_size=256, overlap=32)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "hello world"

    def test_overlap_shares_content_between_chunks(self):
        text = "abcdefghijklmnopqrstuvwxyz" * 4  # 104 chars
        chunks = chunk_text(text, source="x.txt", chunk_size=40, overlap=10)
        assert len(chunks) >= 2
        # tail of chunk i should reappear at head of chunk i+1
        for i in range(len(chunks) - 1):
            tail = chunks[i]["text"][-10:]
            head = chunks[i + 1]["text"][:10]
            assert tail == head

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError):
            chunk_text("hello", chunk_size=0)

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            chunk_text("hello", chunk_size=10, overlap=10)
        with pytest.raises(ValueError):
            chunk_text("hello", chunk_size=10, overlap=-1)


# ─── build_prompt / mock_generate_answer ────────────────────────────────

class TestPromptAndMockGeneration:
    def test_build_prompt_includes_query_and_context(self):
        chunks = [{"source": "a.txt", "chunk_id": 0, "text": "VectorHub is great."}]
        prompt = build_prompt("What is VectorHub?", chunks)
        assert "What is VectorHub?" in prompt
        assert "VectorHub is great." in prompt
        assert "a.txt" in prompt

    def test_mock_generate_answer_with_chunks(self):
        chunks = [
            {"source": "a.txt", "chunk_id": 0, "text": "VectorHub is a vector database."},
            {"source": "a.txt", "chunk_id": 1, "text": "It uses HNSW indexing."},
        ]
        answer = mock_generate_answer("What is VectorHub?", chunks)
        assert "VectorHub is a vector database." in answer
        assert "HNSW indexing" in answer
        assert "a.txt" in answer

    def test_mock_generate_answer_no_chunks(self):
        answer = mock_generate_answer("anything", [])
        assert "No relevant context" in answer

    def test_generate_answer_dispatches_to_mock_by_default(self):
        chunks = [{"source": "a.txt", "chunk_id": 0, "text": "hello"}]
        answer = generate_answer("q", chunks)  # mode defaults to "mock"
        assert "mock" in answer.lower()

    def test_generate_answer_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown llm mode"):
            generate_answer("q", [], mode="not-a-real-backend")


# ─── openai backend (no real API key needed) ────────────────────────────

class TestOpenAIBackendNoKey:
    def test_openai_without_key_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="API key"):
            openai_generate_answer("What is VectorHub?", [])

    def test_generate_answer_openai_mode_without_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="API key"):
            generate_answer("q", [], mode="openai")


# ─── end-to-end RAG workflow (mock embedder + mock LLM) ─────────────────

class TestRAGWorkflow:
    def _build_index(self):
        chunks = chunk_text(SAMPLE_TEXT, source="sample.txt", chunk_size=64, overlap=8)
        db = VectorCollection(dim=FAKE_DIM, metric="cosine")
        index_chunks(db, chunks, fake_embed)
        return db, chunks

    def test_index_chunks_populates_collection(self):
        db, chunks = self._build_index()
        assert len(db) == len(chunks)

    def test_retrieve_returns_text_and_source(self):
        db, _ = self._build_index()
        results = retrieve(db, "What is VectorHub?", fake_embed, k=2)
        assert len(results) > 0
        for r in results:
            assert "text" in r and r["text"]
            assert "source" in r and r["source"] == "sample.txt"
            assert "id" in r
            assert "distance" in r

    def test_full_workflow_mock_llm(self):
        """The whole load -> chunk -> index -> retrieve -> generate pipeline,
        end to end, using the mock LLM backend (no external API)."""
        db, _ = self._build_index()
        query = "What is VectorHub?"
        results = retrieve(db, query, fake_embed, k=3)
        assert len(results) > 0

        answer = generate_answer(query, results, mode="mock")
        assert isinstance(answer, str)
        assert len(answer) > 0
        # answer should reference retrieved content, not be empty boilerplate
        assert "mock" in answer.lower()

    def test_persistence_roundtrip_preserves_retrieval(self, tmp_path):
        db, _ = self._build_index()
        path = str(tmp_path / "rag_index.bin")
        db.save(path)

        db2 = VectorCollection.load(path)
        results_before = retrieve(db, "semantic search", fake_embed, k=2)
        results_after = retrieve(db2, "semantic search", fake_embed, k=2)

        assert [r["id"] for r in results_before] == [r["id"] for r in results_after]
        assert [r["text"] for r in results_before] == [r["text"] for r in results_after]

    def test_search_with_filter_by_source(self):
        db, _ = self._build_index()
        extra_chunks = chunk_text(
            "This is a completely different document about cats.",
            source="other.txt", chunk_size=64, overlap=8,
        )
        index_chunks(db, extra_chunks, fake_embed, id_offset=1000)

        results = retrieve(db, "VectorHub", fake_embed, k=10, filter={"source": "other.txt"})
        assert all(r["source"] == "other.txt" for r in results)
