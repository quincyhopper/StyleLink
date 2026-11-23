from __future__ import annotations

from pathlib import Path
from typing import Sequence

import faiss
import numpy as np


class VectorIndex:
    """Class for managing document or centroid FAISS indexes."""

    def __init__(self) -> None:
        """Initialise empty FAISS index."""
        self.dimension = 1024
        base_index = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap2(base_index)

    @property
    def ntotal(self) -> int:
        """Total number of vectors currently indexed."""
        return int(self.index.ntotal)

    @property
    def ids(self) -> set[int]:
        """The set of all vector IDs currently indexed."""
        if self.ntotal == 0:
            return set()
        return set(faiss.vector_to_array(self.index.id_map).tolist())

    def __len__(self):
        return self.ntotal

    def add(self, x: np.ndarray, ids: Sequence[int] | np.ndarray) -> None:
        """Add vectors to index, assigning them the provided ids.

        Args:
            x: 2D float32 array of shape (N, 1024).
            ids: Sequence or 1D array of N integer document ids.

        Raises:
            ValueError: if wrong shapes or shapes do not align.
        """
        x = np.ascontiguousarray(x, dtype=np.float32)
        ids = np.ascontiguousarray(ids, np.int64)  # Int64 for larger ID capacity

        if x.ndim != 2:
            raise ValueError(f"x must have 2 dims, got {x.ndim}")

        if ids.ndim != 1:
            raise ValueError(f"ids must be 1D, got {ids.ndim}")

        if x.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of dimensionality {self.dimension}, got {x.shape[1]}")

        if ids.shape != (x.shape[0],):
            raise ValueError(f"Number of vectors ({x.shape[0]}) must equal number of IDs ({ids.shape})")

        self.index.add_with_ids(x, ids)

    def remove(self, ids: Sequence[int] | np.ndarray) -> int:
        """Remove vectors associated with the given document IDs.

        Args:
            ids: Sequence or 1D array of document IDs to remove.

        Returns:
            Number of vectors removed.

        Raises:
            ValueError: if ids is not 1D after casting to array.
        """
        ids = np.ascontiguousarray(ids, dtype=np.int64)

        if ids.ndim != 1:
            raise ValueError(f"Expected 1D array, got ndim {ids.ndim}")

        selector = faiss.IDSelectorArray(ids)
        removed = self.index.remove_ids(selector)
        return int(removed)

    def reset(self) -> None:
        """Clear all indexed vectors and ID mappings, returning the index to an empty state."""
        self.index.reset()

    def reconstruct(self, doc_id: Sequence[int] | int | np.ndarray) -> np.ndarray:
        """Retrieve the raw vector associated with a specific document ID or sequence of IDs.

        Args:
            doc_id: 1D NumPy array of IDs to retrieve.

        Returns:
            2D NumPy array of shape (n, d) corresponding to given ID(s).

        Raises:
            ValueError: if doc_id is incorrect shape.
            KeyError: if reconstructing the vectors fails.
        """
        # Check int64 before casting to integers, to avoid potential rounding errors if floats are passed
        if isinstance(doc_id, np.ndarray):
            if doc_id.dtype != np.int64:
                raise ValueError(f"doc_id passed as NumPy arrays must have dtype np.int64, got {doc_id.dtype}")

        ids = np.atleast_1d(np.asarray(doc_id, dtype=np.int64))

        if ids.ndim != 1:
            raise ValueError(f"doc_id must be 1D, got {ids.ndim}")

        if ids.size == 0:
            raise KeyError("doc_id cannot be empty")

        try:
            return self.index.reconstruct_batch(ids)
        except RuntimeError as err:
            raise KeyError("Failed to reconstruct one or more IDs") from err

    def centroid(
        self,
        doc_ids: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        """Calculate the mean centroid vector for a given sequence of document IDs.

        Retrieves vectors stored for the given document IDs, computes their mean,
        and optionally unit-normalizes the resulting vector (defaults to True if metric
        is 'cosine' or 'ip').

        Args:
            doc_ids: Sequence or 1D array of integer document IDs.

        Returns:
            2D float32 NumPy array of shape (1, dimension).

        Raises:
            ValueError: If norm is zero or non-finite.
        """
        vecs = self.reconstruct(doc_ids)
        centroid_vec = np.mean(vecs, axis=0, keepdims=True).astype(np.float32)

        # Normalise the centroids
        norm = float(np.linalg.norm(centroid_vec))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Norm is zero or non-finite.")
        centroid_vec = centroid_vec / norm

        return centroid_vec

    def search(
        self,
        queries: np.ndarray,
        k: int = 5,
        allowed_ids: Sequence[int] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for the k nearest neighbors for each query vector.

        Args:
            queries: NumPy array of shape (N, dimension) or (dimension,).
            k: Number of nearest neighbors to retrieve per query.
            allowed_ids: Optional sequence of document IDs to constrain the search.
                If provided, only vectors with matching IDs will be evaluated.

        Returns:
            Tuple of (distances, ids):
                - distances: shape (N, k) float32 distances/scores
                - ids: shape (N, k) int64 document IDs (-1 if fewer than k vectors exist)

        Raises:
            RuntimeError: if index is empty.
            ValueError: if query vectors are wrong shape.
        """
        if self.ntotal == 0:
            raise RuntimeError("Cannot search an empty index. Add vectors before searching.")

        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        if queries.shape[1] != self.dimension:
            raise ValueError(f"Query vectors must have dimension {self.dimension}, got {queries.shape[1]}")

        n_queries = queries.shape[0]

        # Query a subset of the index
        if allowed_ids is not None:
            ids_arr = np.atleast_1d(np.asarray(allowed_ids, dtype=np.int64))

            if ids_arr.size == 0:
                # FAISS returns -1 automatically, but we can return early before building the selector
                return np.full((n_queries, 0), np.nan, dtype=np.float32), np.full((n_queries, 0), -1, dtype=np.int64)

            selector = faiss.IDSelectorBatch(ids_arr)
            params = faiss.SearchParameters()
            params.sel = selector
            k_clamped = min(k, self.ntotal)
            distances, indices = self.index.search(queries, k_clamped, params=params)
            return distances, indices

        k_clamped = min(k, self.ntotal)
        distances, indices = self.index.search(queries, k_clamped)
        return distances, indices

    def search_by_id(
        self,
        query_doc_id: Sequence[int] | int,
        k: int = 5,
        exclude_self: bool = True,
        allowed_ids: Sequence[int] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for the k nearest neighbors using existing document ID(s).

        Retrieves the precomputed vector(s) for query_doc_id directly from the index and finds
        nearest neighbors.

        Args:
            query_doc_id: Integer document ID (or Sequence of integers) to use as query.
            k: Maximum number of nearest neighbors to retrieve per query.
            exclude_self: If True (default), excludes each query document ID from its results.
            allowed_ids: Optional sequence of document IDs to constrain the search.

        Returns:
            Tuple of (distances, ids)
                - distances: 2D float32 of shape (N, k)
                - ids: 2D int64 of shape (N, k)

        Raises:
            ValueError: if query ids is > 1 ndim or k is negative.
        """
        query_ids = np.asarray(query_doc_id, dtype=np.int64)

        if query_ids.ndim == 0:
            # Integer
            query_ids = query_ids.reshape(1)
        elif query_ids.ndim != 1:
            raise ValueError(f"query_doc_id must be 1D, got {query_ids.ndim}")

        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")

        n_queries = query_ids.size

        # Init output arrays
        out_dist = np.full((n_queries, k), np.nan, dtype=np.float32)
        out_ids = np.full((n_queries, k), -1, dtype=np.int64)

        if self.ntotal == 0 or n_queries == 0 or k == 0:
            return out_dist, out_ids

        query_ids = np.ascontiguousarray(query_ids, dtype=np.int64)
        query_vecs = self.reconstruct(query_ids)  # Shape (n_queries, d)

        # Fetch 1 extra result per query if we need to remove the query itself
        fetch_k = min(k + int(exclude_self), self.ntotal)

        distances, indices = self.search(
            queries=query_vecs,
            k=fetch_k,
            allowed_ids=allowed_ids,
        )

        # Mask out invalid results (indices are -1 if fewer than k neighbours are available)
        mask = indices != -1

        if exclude_self:
            # Add own index to the mask
            mask &= indices != query_ids.reshape(-1, 1)

        rank = np.cumsum(mask, axis=1) - 1  # Destination column of valid results
        rows, cols = np.nonzero(mask & (rank < k))  # Get coordinates to fill k results per query

        out_dist[rows, rank[rows, cols]] = distances[rows, cols]
        out_ids[rows, rank[rows, cols]] = indices[rows, cols]

        return out_dist, out_ids

    def save(self, file_path: Path | str) -> None:
        """Save vector index to a file on disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, file_path: Path | str) -> VectorIndex:
        """Load a vector index from disk.

        Args:
            file_path: path to .faiss file.

        Returns:
            Instance of vector index.

        Raises:
            FileNotFoundError: if no index is found at specified file_path.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Index file not found: {path}")

        loaded_index = faiss.read_index(str(path))
        if not isinstance(loaded_index, faiss.IndexIDMap2):
            # Ensure index has ID mapping enabled
            loaded_index = faiss.IndexIDMap2(loaded_index)

        instance = cls.__new__(cls)
        instance.dimension = int(loaded_index.d)
        instance.index = loaded_index
        return instance
