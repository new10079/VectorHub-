"""VectorCollection: Python wrapper for HNSW vector search."""

import json
import os
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from . import _vectorhub
    HAS_CPP = True
except ImportError:
    HAS_CPP = False


# ─── Pure-Python fallback ─────────────────────────────────────────────────────

class _SimplePythonHNSW:
    """Brute-force fallback used when the C++ extension is not compiled."""

    def __init__(self, dim: int, metric: str = "l2",
                 M: int = 16, ef_construction: int = 200, ef_search: int = 50):
        if dim <= 0:
            raise ValueError("Dimension must be positive")
        if metric not in ("l2", "cosine"):
            raise ValueError("Only 'l2' and 'cosine' metrics are supported")
        self.dim = dim
        self.metric = metric
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._vectors: List[List[float]] = []
        self._ids: List[int] = []
        self._id_to_idx: Dict[int, int] = {}
        self._deleted: set = set()

    # ── insert ────────────────────────────────────────────────────────────────

    def insert(self, id: int, vector: List[float]):
        if len(vector) != self.dim:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dim}, got {len(vector)}"
            )
        if id in self._id_to_idx and id not in self._deleted:
            raise RuntimeError(f"Duplicate id: {id}")
        if id in self._deleted:
            self._deleted.discard(id)
            idx = self._id_to_idx[id]
            self._vectors[idx] = vector
            return
        idx = len(self._ids)
        self._ids.append(id)
        self._vectors.append(vector)
        self._id_to_idx[id] = idx

    def insert_batch(self, ids: List[int], vectors: List[List[float]]):
        if len(ids) != len(vectors):
            raise ValueError("Number of ids and vectors must match")
        for id_, vec in zip(ids, vectors):
            self.insert(id_, vec)

    # ── search ────────────────────────────────────────────────────────────────

    def _dist(self, a: List[float], b: List[float]) -> float:
        if self.metric == "l2":
            return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
        # cosine
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na < 1e-10 or nb < 1e-10:
            return 1.0
        return 1.0 - dot / (na * nb)

    def search(self, query: List[float], k: int) -> List[Tuple[int, float]]:
        if len(query) != self.dim:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dim}, got {len(query)}"
            )
        if k <= 0:
            raise ValueError("k must be positive")

        results = [
            (self._ids[i], self._dist(query, self._vectors[i]))
            for i in range(len(self._ids))
            if self._ids[i] not in self._deleted
        ]
        results.sort(key=lambda x: x[1])
        return results[:k]

    def brute_force_search(self, query: List[float], k: int) -> List[Tuple[int, float]]:
        return self.search(query, k)

    # ── remove ────────────────────────────────────────────────────────────────

    def remove(self, id: int):
        if id not in self._id_to_idx:
            raise KeyError(f"ID not found: {id}")
        if id in self._deleted:
            raise KeyError(f"ID already deleted: {id}")
        self._deleted.add(id)

    @property
    def size(self) -> int:
        return len(self._ids) - len(self._deleted)


# ─── VectorCollection ─────────────────────────────────────────────────────────

class VectorCollection:
    """
    Lightweight vector collection using HNSW index.

    Phase 2 features
    ----------------
    - True HNSW graph index (M, ef_construction, ef_search parameters)
    - Cosine distance support in addition to L2
    - Persistence: save() / load()
    - Metadata: store and filter by arbitrary key-value pairs
    - delete() and update() operations
    - brute_force_search() for exact recall evaluation
    """

    def __init__(
        self,
        dim: int,
        metric: str = "l2",
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ):
        """
        Parameters
        ----------
        dim : int
            Vector dimensionality (must be > 0).
        metric : str
            Distance metric – 'l2' (Euclidean) or 'cosine'. Default 'l2'.
        M : int
            Max neighbours per node per layer in the HNSW graph. Default 16.
            Higher M → better recall at higher memory cost.
        ef_construction : int
            Candidate-pool size during index build. Default 200.
            Higher → better graph quality, slower inserts.
        ef_search : int
            Candidate-pool size during search. Default 50.
            Higher → better recall, slower queries. Can be changed at query
            time via set_ef_search().
        """
        if dim <= 0:
            raise ValueError("dim must be positive")
        if metric not in ("l2", "cosine"):
            raise ValueError("Only 'l2' and 'cosine' metrics are supported")

        self.dim = dim
        self.metric = metric
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        if HAS_CPP:
            self._index = _vectorhub.HNSW(dim, metric, M, ef_construction, ef_search)
        else:
            self._index = _SimplePythonHNSW(dim, metric, M, ef_construction, ef_search)

        # Metadata store: id -> dict
        self._metadata: Dict[int, Dict[str, Any]] = {}

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_list_of_lists(vectors) -> List[List[float]]:
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()
        return [list(v) for v in vectors]

    @staticmethod
    def _to_int_list(ids) -> List[int]:
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        return [int(i) for i in ids]

    @staticmethod
    def _to_float_list(query) -> List[float]:
        if isinstance(query, np.ndarray):
            return query.tolist()
        return list(query)

    def set_ef_search(self, ef: int):
        """Adjust ef_search at query time without rebuilding the index."""
        if ef < 1:
            raise ValueError("ef_search must be >= 1")
        self.ef_search = ef
        self._index.set_ef_search(ef)

    # ── add ───────────────────────────────────────────────────────────────────

    def add(
        self,
        vectors: Union[List[List[float]], np.ndarray],
        ids: Union[List[int], np.ndarray],
        metadatas: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> None:
        """Add vectors to the collection.

        Parameters
        ----------
        vectors : array-like, shape (n, dim)
        ids : list[int], length n
        metadatas : list[dict] or None, length n.
            Each element is a free-form dict attached to the corresponding
            vector. Pass None to omit metadata for all vectors.

        Raises
        ------
        ValueError
            If lengths mismatch, any vector has wrong dimension, or
            metadatas length doesn't match vectors.
        RuntimeError
            If any ID is already present (and not deleted).
        """
        vectors = self._to_list_of_lists(vectors)
        ids = self._to_int_list(ids)

        if len(vectors) != len(ids):
            raise ValueError("vectors and ids must have the same length")
        if metadatas is not None and len(metadatas) != len(vectors):
            raise ValueError("metadatas must have the same length as vectors")

        for i, vec in enumerate(vectors):
            if len(vec) != self.dim:
                raise ValueError(
                    f"Vector {i} has dimension {len(vec)}, expected {self.dim}"
                )

        self._index.insert_batch(ids, vectors)

        if metadatas is not None:
            for id_, meta in zip(ids, metadatas):
                if meta is not None:
                    self._metadata[id_] = dict(meta)
                else:
                    self._metadata.pop(id_, None)

    # ── search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: Union[List[float], np.ndarray],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = False,
    ) -> List:
        """Search for k nearest neighbours.

        Parameters
        ----------
        query : array-like, shape (dim,)
        k : int
            Number of results to return.
        filter : dict or None
            Simple equality filter on metadata, e.g. ``{"category": "book"}``.
            Only vectors whose metadata contains **all** key-value pairs in
            filter are returned.  Applied **after** HNSW search with an
            internally inflated k to account for filtered-out candidates.
        include_metadata : bool
            If True, each result is ``(id, distance, metadata_dict)``.
            If False (default), each result is ``(id, distance)``.

        Returns
        -------
        list of (id, distance) or (id, distance, metadata) tuples.
        """
        query = self._to_float_list(query)
        if len(query) != self.dim:
            raise ValueError(f"Query has dimension {len(query)}, expected {self.dim}")
        if k <= 0:
            raise ValueError("k must be positive")

        if filter is None:
            raw = self._index.search(query, k)
        else:
            # Over-fetch to account for filtered-out results, then trim
            fetch_k = max(k * 10, k + 50)
            raw = self._index.search(query, fetch_k)
            raw = [
                (id_, dist)
                for id_, dist in raw
                if self._matches_filter(id_, filter)
            ]
            raw = raw[:k]

        if not include_metadata:
            return list(raw)

        return [
            (id_, dist, self._metadata.get(id_, {}))
            for id_, dist in raw
        ]

    def brute_force_search(
        self,
        query: Union[List[float], np.ndarray],
        k: int = 10,
    ) -> List[Tuple[int, float]]:
        """Exact brute-force search (for recall benchmarking)."""
        query = self._to_float_list(query)
        if len(query) != self.dim:
            raise ValueError(f"Query has dimension {len(query)}, expected {self.dim}")
        if k <= 0:
            raise ValueError("k must be positive")
        return list(self._index.brute_force_search(query, k))

    # ── delete / update ───────────────────────────────────────────────────────

    def delete(self, id: int) -> None:
        """Remove a vector from the collection (lazy delete).

        Raises
        ------
        KeyError
            If ``id`` does not exist or has already been deleted.
        """
        id = int(id)
        try:
            self._index.remove(id)
        except IndexError as e:
            # The compiled C++ backend raises std::out_of_range, which pybind11
            # auto-translates to IndexError. The pure-Python fallback already
            # raises KeyError directly. Normalize to KeyError here so behavior
            # (and the documented contract above) is consistent across backends.
            raise KeyError(str(e)) from e
        self._metadata.pop(id, None)

    def update(
        self,
        id: int,
        vector: Union[List[float], np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update a vector (and optionally its metadata).

        Implemented as delete + re-insert so the HNSW graph is updated.

        Parameters
        ----------
        id : int
        vector : array-like, shape (dim,)
        metadata : dict or None
            New metadata.  Pass None to clear existing metadata.

        Raises
        ------
        KeyError
            If ``id`` does not exist.
        ValueError
            If vector has wrong dimension.
        """
        id = int(id)
        vector = self._to_float_list(vector)
        if len(vector) != self.dim:
            raise ValueError(
                f"Vector has dimension {len(vector)}, expected {self.dim}"
            )
        self.delete(id)
        self._index.insert(id, vector)
        if metadata is not None:
            self._metadata[id] = dict(metadata)
        # else metadata was cleared by delete()

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the collection (index + metadata) to disk.

        Creates two files:
        - ``<path>``          – binary HNSW index
        - ``<path>.meta.json``– JSON metadata store

        Parameters
        ----------
        path : str
            Destination path for the index file.
        """
        self._index.save(path)
        meta_path = path + ".meta.json"
        payload = {
            "version": 2,
            "dim": self.dim,
            "metric": self.metric,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "metadata": {str(k): v for k, v in self._metadata.items()},
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "VectorCollection":
        """Load a collection previously saved with :meth:`save`.

        Parameters
        ----------
        path : str
            Path to the index file (the ``.meta.json`` sidecar is loaded
            automatically).

        Returns
        -------
        VectorCollection
        """
        meta_path = path + ".meta.json"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata sidecar not found: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        col = cls(
            dim=payload["dim"],
            metric=payload["metric"],
            M=payload.get("M", 16),
            ef_construction=payload.get("ef_construction", 200),
            ef_search=payload.get("ef_search", 50),
        )
        col._index.load(path)
        col._metadata = {int(k): v for k, v in payload.get("metadata", {}).items()}
        return col

    # ── internal ──────────────────────────────────────────────────────────────

    def _matches_filter(self, id_: int, filter_: Dict[str, Any]) -> bool:
        meta = self._metadata.get(id_, {})
        return all(meta.get(k) == v for k, v in filter_.items())

    def __len__(self) -> int:
        return self._index.size

    def __repr__(self) -> str:
        return (
            f"VectorCollection(dim={self.dim}, metric={self.metric!r}, "
            f"M={self.M}, ef_construction={self.ef_construction}, "
            f"ef_search={self.ef_search}, size={len(self)})"
        )
