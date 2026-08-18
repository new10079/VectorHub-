"""Tests for VectorCollection."""

import pytest
import numpy as np
from vectorhub import VectorCollection


class TestVectorCollection:
    """Test suite for VectorCollection."""

    def test_create_collection(self):
        """Test creating a VectorCollection."""
        db = VectorCollection(dim=10)
        assert db.dim == 10
        assert db.metric == "l2"

    def test_add_and_search(self):
        """Test adding vectors and searching for nearest neighbors."""
        db = VectorCollection(dim=3)
        vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        ids = [101, 102, 103]

        db.add(vectors=vectors, ids=ids)

        # Search for exact match
        query = [1.0, 0.0, 0.0]
        results = db.search(query, k=1)

        assert len(results) == 1
        assert results[0][0] == 101  # Should return id 101
        assert results[0][1] < 1e-6  # Distance should be near 0

    def test_search_returns_correct_order(self):
        """Test that search results are sorted by distance (ascending)."""
        db = VectorCollection(dim=2)
        vectors = [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ]
        ids = [1, 2, 3]

        db.add(vectors=vectors, ids=ids)

        query = [0.0, 0.0]
        results = db.search(query, k=3)

        # Should be sorted by distance: [1, 2, 3]
        assert results[0][0] == 1
        assert results[1][0] == 2
        assert results[2][0] == 3

        # Check distances are increasing
        assert results[0][1] <= results[1][1] <= results[2][1]

    def test_k_larger_than_data(self):
        """Test that k larger than data size returns all data."""
        db = VectorCollection(dim=2)
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        ids = [1, 2]

        db.add(vectors=vectors, ids=ids)

        results = db.search([0.0, 0.0], k=10)
        assert len(results) == 2

    def test_dimension_mismatch_on_add(self):
        """Test that dimension mismatch raises error."""
        db = VectorCollection(dim=3)

        with pytest.raises(ValueError, match="Vector .* has dimension"):
            db.add(vectors=[[1.0, 2.0]], ids=[1])

    def test_dimension_mismatch_on_search(self):
        """Test that query dimension mismatch raises error."""
        db = VectorCollection(dim=3)
        db.add(vectors=[[1.0, 2.0, 3.0]], ids=[1])

        with pytest.raises(ValueError, match="Query has dimension"):
            db.search([1.0, 2.0])

    def test_ids_vectors_length_mismatch(self):
        """Test that mismatched ids and vectors lengths raise error."""
        db = VectorCollection(dim=2)

        with pytest.raises(ValueError, match="same length"):
            db.add(vectors=[[1.0, 2.0]], ids=[1, 2])

    def test_duplicate_ids(self):
        """Test that duplicate ids raise error."""
        db = VectorCollection(dim=2)

        db.add(vectors=[[1.0, 0.0]], ids=[1])

        with pytest.raises(RuntimeError, match="Duplicate id"):
            db.add(vectors=[[0.0, 1.0]], ids=[1])

    def test_invalid_k(self):
        """Test that invalid k values raise error."""
        db = VectorCollection(dim=2)
        db.add(vectors=[[1.0, 0.0]], ids=[1])

        with pytest.raises(ValueError, match="k must be positive"):
            db.search([1.0, 0.0], k=0)

        with pytest.raises(ValueError, match="k must be positive"):
            db.search([1.0, 0.0], k=-1)

    def test_numpy_array_support(self):
        """Test that numpy arrays are supported."""
        db = VectorCollection(dim=3)

        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        ids = np.array([101, 102, 103])

        db.add(vectors=vectors, ids=ids)

        query = np.array([1.0, 0.0, 0.0])
        results = db.search(query, k=1)

        assert len(results) == 1
        assert results[0][0] == 101

    def test_distance_calculation(self):
        """Test L2 distance calculation."""
        db = VectorCollection(dim=2)
        vectors = [
            [0.0, 0.0],
            [3.0, 4.0],  # Distance = 5.0 from origin
        ]
        ids = [1, 2]

        db.add(vectors=vectors, ids=ids)

        query = [0.0, 0.0]
        results = db.search(query, k=2)

        # Check distances
        assert abs(results[0][1] - 0.0) < 1e-6  # First vector at origin
        assert abs(results[1][1] - 5.0) < 1e-6  # Second vector at distance 5

    def test_invalid_metric(self):
        """Test that invalid metric raises error."""
        with pytest.raises(ValueError, match="Only 'l2' and 'cosine' metrics"):
            VectorCollection(dim=3, metric="euclidean")

    def test_invalid_dimension(self):
        """Test that invalid dimension raises error."""
        with pytest.raises(ValueError, match="dim must be positive"):
            VectorCollection(dim=0)

        with pytest.raises(ValueError, match="dim must be positive"):
            VectorCollection(dim=-1)

    def test_batch_insert(self):
        """Test batch insertion."""
        db = VectorCollection(dim=2)

        vectors = [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
        ids = [1, 2, 3]

        db.add(vectors=vectors, ids=ids)

        results = db.search([1.0, 0.0], k=3)
        assert len(results) == 3

    def test_multiple_adds(self):
        """Test multiple add calls."""
        db = VectorCollection(dim=2)

        db.add(vectors=[[1.0, 0.0]], ids=[1])
        db.add(vectors=[[0.0, 1.0]], ids=[2])
        db.add(vectors=[[1.0, 1.0]], ids=[3])

        results = db.search([1.0, 0.0], k=3)
        assert len(results) == 3
