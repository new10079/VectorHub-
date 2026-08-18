"""Phase 2 tests: persistence, metadata, delete/update, HNSW recall."""

import os
import random
import tempfile

import numpy as np
import pytest

from vectorhub import VectorCollection


# ─── fixtures ────────────────────────────────────────────────────────────────

def make_collection(n=20, dim=8, metric="l2", seed=0):
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    ids = list(range(n))
    col = VectorCollection(dim=dim, metric=metric, M=8, ef_construction=50, ef_search=20)
    col.add(vecs, ids)
    return col, vecs, ids


# ─── HNSW parameters ────────────────────────────────────────────────────────

class TestHNSWParams:
    def test_default_params(self):
        col = VectorCollection(dim=4)
        assert col.M == 16
        assert col.ef_construction == 200
        assert col.ef_search == 50

    def test_custom_params(self):
        col = VectorCollection(dim=4, M=8, ef_construction=100, ef_search=30)
        assert col.M == 8
        assert col.ef_construction == 100
        assert col.ef_search == 30

    def test_cosine_metric(self):
        col = VectorCollection(dim=4, metric="cosine")
        col.add([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], [1, 2])
        results = col.search([1.0, 0.0, 0.0, 0.0], k=2)
        assert results[0][0] == 1  # identical vector is closest
        assert results[0][1] < 1e-5

    def test_set_ef_search(self):
        col = VectorCollection(dim=4)
        col.set_ef_search(100)
        assert col.ef_search == 100

    def test_repr(self):
        col = VectorCollection(dim=4)
        assert "VectorCollection" in repr(col)
        assert "dim=4" in repr(col)


# ─── save / load ────────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_load_search_consistent(self):
        col, vecs, ids = make_collection(n=30, dim=8)
        query = vecs[5].tolist()
        before = col.search(query, k=5)

        with tempfile.NamedTemporaryFile(suffix=".vhb", delete=False) as f:
            path = f.name
        try:
            col.save(path)
            col2 = VectorCollection.load(path)
            after = col2.search(query, k=5)
            assert [r[0] for r in before] == [r[0] for r in after], \
                f"ID order changed after save/load: {before} vs {after}"
        finally:
            os.unlink(path)
            meta_path = path + ".meta.json"
            if os.path.exists(meta_path):
                os.unlink(meta_path)

    def test_load_continues_add(self):
        col, vecs, _ = make_collection(n=10, dim=4)
        with tempfile.NamedTemporaryFile(suffix=".vhb", delete=False) as f:
            path = f.name
        try:
            col.save(path)
            col2 = VectorCollection.load(path)
            # Should be able to add more vectors after load
            col2.add([[9.9, 8.8, 7.7, 6.6]], [999])
            results = col2.search([9.9, 8.8, 7.7, 6.6], k=1)
            assert results[0][0] == 999
        finally:
            os.unlink(path)
            if os.path.exists(path + ".meta.json"):
                os.unlink(path + ".meta.json")

    def test_save_load_metadata(self):
        col = VectorCollection(dim=3)
        col.add(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [10, 20],
            metadatas=[{"tag": "a"}, {"tag": "b"}],
        )
        with tempfile.NamedTemporaryFile(suffix=".vhb", delete=False) as f:
            path = f.name
        try:
            col.save(path)
            col2 = VectorCollection.load(path)
            res = col2.search([1.0, 0.0, 0.0], k=2, include_metadata=True)
            meta_map = {r[0]: r[2] for r in res}
            assert meta_map[10] == {"tag": "a"}
            assert meta_map[20] == {"tag": "b"}
        finally:
            os.unlink(path)
            if os.path.exists(path + ".meta.json"):
                os.unlink(path + ".meta.json")

    def test_save_load_preserves_deleted(self):
        col, _, _ = make_collection(n=5, dim=4)
        col.delete(2)
        with tempfile.NamedTemporaryFile(suffix=".vhb", delete=False) as f:
            path = f.name
        try:
            col.save(path)
            col2 = VectorCollection.load(path)
            results = col2.search([0.0] * 4, k=5)
            returned_ids = [r[0] for r in results]
            assert 2 not in returned_ids
        finally:
            os.unlink(path)
            if os.path.exists(path + ".meta.json"):
                os.unlink(path + ".meta.json")

    def test_load_missing_meta_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".vhb", delete=False) as f:
            path = f.name
        try:
            with pytest.raises(FileNotFoundError, match="Metadata sidecar not found"):
                VectorCollection.load(path)
        finally:
            os.unlink(path)


# ─── metadata ────────────────────────────────────────────────────────────────

class TestMetadata:
    def test_add_and_return_metadata(self):
        col = VectorCollection(dim=3)
        col.add(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [1, 2, 3],
            metadatas=[{"cat": "A"}, {"cat": "B"}, {"cat": "A"}],
        )
        res = col.search([1.0, 0.0, 0.0], k=3, include_metadata=True)
        assert len(res[0]) == 3  # (id, dist, meta)
        meta_map = {r[0]: r[2] for r in res}
        assert meta_map[1] == {"cat": "A"}
        assert meta_map[2] == {"cat": "B"}

    def test_search_without_metadata_backward_compat(self):
        col = VectorCollection(dim=3)
        col.add([[1.0, 0.0, 0.0]], [1], metadatas=[{"x": 1}])
        res = col.search([1.0, 0.0, 0.0], k=1)
        assert len(res[0]) == 2  # (id, dist) only

    def test_filter_equality(self):
        col = VectorCollection(dim=3)
        col.add(
            [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]],
            [1, 2, 3],
            metadatas=[{"cat": "book"}, {"cat": "movie"}, {"cat": "book"}],
        )
        res = col.search([1.0, 0.0, 0.0], k=10, filter={"cat": "book"})
        returned = {r[0] for r in res}
        assert 1 in returned
        assert 3 in returned
        assert 2 not in returned

    def test_filter_no_match_returns_empty(self):
        col = VectorCollection(dim=3)
        col.add([[1.0, 0.0, 0.0]], [1], metadatas=[{"cat": "book"}])
        res = col.search([1.0, 0.0, 0.0], k=5, filter={"cat": "music"})
        assert res == []

    def test_metadata_none_entry(self):
        col = VectorCollection(dim=3)
        col.add(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [1, 2],
            metadatas=[{"x": 1}, None],
        )
        res = col.search([0.0, 1.0, 0.0], k=2, include_metadata=True)
        meta_map = {r[0]: r[2] for r in res}
        assert meta_map[2] == {}

    def test_no_metadatas_arg(self):
        col = VectorCollection(dim=3)
        col.add([[1.0, 0.0, 0.0]], [1])
        res = col.search([1.0, 0.0, 0.0], k=1, include_metadata=True)
        assert res[0][2] == {}

    def test_metadatas_length_mismatch(self):
        col = VectorCollection(dim=3)
        with pytest.raises(ValueError, match="metadatas must have the same length"):
            col.add([[1.0, 0.0, 0.0]], [1], metadatas=[{"a": 1}, {"b": 2}])


# ─── delete ──────────────────────────────────────────────────────────────────

class TestDelete:
    def test_delete_not_in_search(self):
        col = VectorCollection(dim=3)
        col.add(
            [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]],
            [1, 2, 3],
        )
        col.delete(1)
        res = col.search([1.0, 0.0, 0.0], k=3)
        ids = [r[0] for r in res]
        assert 1 not in ids

    def test_delete_nonexistent_raises(self):
        col = VectorCollection(dim=3)
        with pytest.raises(KeyError):
            col.delete(999)

    def test_delete_already_deleted_raises(self):
        col = VectorCollection(dim=3)
        col.add([[1.0, 0.0, 0.0]], [1])
        col.delete(1)
        with pytest.raises(KeyError):
            col.delete(1)

    def test_delete_clears_metadata(self):
        col = VectorCollection(dim=3)
        col.add([[1.0, 0.0, 0.0]], [1], metadatas=[{"x": 1}])
        col.delete(1)
        assert 1 not in col._metadata

    def test_len_decreases_after_delete(self):
        col = VectorCollection(dim=3)
        col.add([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [1, 2])
        assert len(col) == 2
        col.delete(1)
        assert len(col) == 1


# ─── update ──────────────────────────────────────────────────────────────────

class TestUpdate:
    def test_update_changes_search_result(self):
        col = VectorCollection(dim=3)
        # Use more vectors so HNSW graph has enough context to reconnect properly
        col.add(
            [
                [1.0, 0.0, 0.0],  # id 1 – will be moved
                [0.0, 1.0, 0.0],  # id 2
                [0.5, 0.5, 0.0],  # id 3
                [0.8, 0.2, 0.0],  # id 4
                [0.2, 0.8, 0.0],  # id 5
            ],
            [1, 2, 3, 4, 5],
        )
        # Brute-force search: before update [0,1,0] nearest is id 2
        bf_before = col.brute_force_search([0.0, 1.0, 0.0], k=1)
        assert bf_before[0][0] == 2

        # Move id 1 to [0.01, 0.99, 0.0] — nearly identical to id 2
        col.update(1, [0.01, 0.99, 0.0])

        # Brute-force confirms top-2 near [0,1,0] are now ids 1 and 2
        bf_after = col.brute_force_search([0.0, 1.0, 0.0], k=2)
        bf_ids = {r[0] for r in bf_after}
        assert 1 in bf_ids, f"Updated id 1 should be in brute-force top-2: {bf_after}"

        # HNSW search should also find id 1 in top-2
        hnsw_res = col.search([0.0, 1.0, 0.0], k=2)
        hnsw_ids = {r[0] for r in hnsw_res}
        assert 1 in hnsw_ids, f"Updated id 1 should be in HNSW top-2: {hnsw_res}"

    def test_update_with_metadata(self):
        col = VectorCollection(dim=3)
        col.add([[1.0, 0.0, 0.0]], [1], metadatas=[{"v": "old"}])
        col.update(1, [1.0, 0.0, 0.0], metadata={"v": "new"})
        res = col.search([1.0, 0.0, 0.0], k=1, include_metadata=True)
        assert res[0][2] == {"v": "new"}

    def test_update_nonexistent_raises(self):
        col = VectorCollection(dim=3)
        with pytest.raises(KeyError):
            col.update(999, [1.0, 0.0, 0.0])

    def test_update_wrong_dimension_raises(self):
        col = VectorCollection(dim=3)
        col.add([[1.0, 0.0, 0.0]], [1])
        with pytest.raises(ValueError):
            col.update(1, [1.0, 0.0])

    def test_update_clears_metadata_when_none(self):
        col = VectorCollection(dim=3)
        col.add([[1.0, 0.0, 0.0]], [1], metadatas=[{"x": 1}])
        col.update(1, [1.0, 0.0, 0.0])  # no metadata arg → cleared
        res = col.search([1.0, 0.0, 0.0], k=1, include_metadata=True)
        assert res[0][2] == {}


# ─── brute-force search ───────────────────────────────────────────────────────

class TestBruteForceSearch:
    def test_brute_force_exact(self):
        col, vecs, ids = make_collection(n=20, dim=8)
        query = vecs[0].tolist()
        bf = col.brute_force_search(query, k=5)
        assert bf[0][0] == 0  # closest is itself
        assert bf[0][1] < 1e-5

    def test_brute_force_after_delete(self):
        col = VectorCollection(dim=3)
        col.add([[1.0, 0.0, 0.0], [0.9, 0.0, 0.0]], [1, 2])
        col.delete(1)
        res = col.brute_force_search([1.0, 0.0, 0.0], k=5)
        ids = [r[0] for r in res]
        assert 1 not in ids


# ─── HNSW recall@k ───────────────────────────────────────────────────────────

class TestHNSWRecall:
    """Verify that HNSW approximate search achieves reasonable recall."""

    def _recall(self, hnsw_ids, bf_ids):
        """Fraction of brute-force top-k that appear in HNSW top-k."""
        bf_set = set(bf_ids)
        hits = sum(1 for id_ in hnsw_ids if id_ in bf_set)
        return hits / len(bf_ids)

    def test_recall_l2(self):
        rng = np.random.default_rng(42)
        n, dim, k = 500, 32, 10
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        col = VectorCollection(dim=dim, metric="l2", M=16, ef_construction=100, ef_search=50)
        col.add(vecs, list(range(n)))

        recalls = []
        for q_vec in rng.standard_normal((20, dim)).astype(np.float32):
            query = q_vec.tolist()
            hnsw_ids = [r[0] for r in col.search(query, k=k)]
            bf_ids   = [r[0] for r in col.brute_force_search(query, k=k)]
            recalls.append(self._recall(hnsw_ids, bf_ids))

        mean_recall = sum(recalls) / len(recalls)
        assert mean_recall >= 0.7, f"HNSW recall@{k} too low: {mean_recall:.3f}"

    def test_recall_cosine(self):
        rng = np.random.default_rng(7)
        n, dim, k = 300, 16, 10
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / (norms + 1e-8)

        col = VectorCollection(dim=dim, metric="cosine", M=16, ef_construction=100, ef_search=50)
        col.add(vecs, list(range(n)))

        recalls = []
        for q_vec in rng.standard_normal((20, dim)).astype(np.float32):
            q_norm = np.linalg.norm(q_vec)
            query = (q_vec / (q_norm + 1e-8)).tolist()
            hnsw_ids = [r[0] for r in col.search(query, k=k)]
            bf_ids   = [r[0] for r in col.brute_force_search(query, k=k)]
            recalls.append(self._recall(hnsw_ids, bf_ids))

        mean_recall = sum(recalls) / len(recalls)
        assert mean_recall >= 0.7, f"HNSW cosine recall@{k} too low: {mean_recall:.3f}"


# ─── edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_collection_search(self):
        col = VectorCollection(dim=3)
        res = col.search([1.0, 0.0, 0.0], k=5)
        assert res == []

    def test_single_vector_search(self):
        col = VectorCollection(dim=3)
        col.add([[1.0, 0.0, 0.0]], [42])
        res = col.search([1.0, 0.0, 0.0], k=10)
        assert len(res) == 1
        assert res[0][0] == 42

    def test_len_reflects_live_count(self):
        col = VectorCollection(dim=3)
        assert len(col) == 0
        col.add([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [1, 2])
        assert len(col) == 2
        col.delete(1)
        assert len(col) == 1

    def test_duplicate_id_in_batch_raises(self):
        col = VectorCollection(dim=3)
        with pytest.raises(RuntimeError, match="Duplicate id"):
            col.add([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [5, 5])
