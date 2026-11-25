from pathlib import Path

import numpy as np
import pytest
from conftest import make_unit_vectors

from stylelink.core.index import VectorIndex


def test_initial_state(empty_index: VectorIndex) -> None:
    """A new VectorIndex should start empty with 1024 dimensions."""
    assert empty_index.dimension == 1024
    assert empty_index.ntotal == 0
    assert len(empty_index) == 0
    assert empty_index.ids == set()


def test_add_vectors_updates_count_and_ids(empty_index: VectorIndex) -> None:
    """Adding valid vector should update ntotal, len() and the ids set."""
    vecs = make_unit_vectors(3, 1024)
    ids = [101, 102, 103]

    empty_index.add(vecs, ids)

    assert empty_index.ntotal == 3
    assert len(empty_index) == 3
    assert empty_index.ids == {101, 102, 103}


@pytest.mark.parametrize(
    ("bad_x", "bad_ids", "error_match"),
    [
        # x is 1D instesad of 2D
        (np.zeros(1024, dtype=np.float32), [1], "x must have 2 dims"),
        # ids is 2D instead of 1D
        (np.zeros((2, 1024), dtype=np.float32), np.array([[1, 2]]), "ids must be 1D"),
        # Wrong vector dimensionality
        (np.zeros((2, 512), dtype=np.float32), [1, 2], "Expected vectors of dimensionality 1024"),
        # Mismatched number of vectors (2) vs. IDs (3)
        (np.zeros((2, 1024), dtype=np.float32), [1, 2, 3], "Number of vectors .* must equal number of IDs"),
    ],
)
def test_add_invalid_shapes_raises_value_error(
    empty_index: VectorIndex, bad_x: np.ndarray, bad_ids: list[int] | np.ndarray, error_match: str
) -> None:
    """Adding vectors or IDs with invalid shapes should raise a clear ValueError."""
    with pytest.raises(ValueError, match=error_match):
        empty_index.add(bad_x, bad_ids)

    # Ensure the failed add didn't change the number of vectors at all
    assert empty_index.ntotal == 0


def test_remove_existing_id(populated_index: VectorIndex) -> None:
    """Removing existing IDs returns the count of removed vectors, decreases ntotal and removes only those ids from index.ids while keeping remaining vectors."""
    index, _, _ = populated_index
    ids_to_remove = np.array([10, 20])
    assert index.remove(ids_to_remove) == 2
    assert index.ntotal == 3
    assert index.ids == {30, 40, 50}


def test_remove_non_existing_ids(populated_index: VectorIndex) -> None:
    """Attempting to remove non-existent IDs should return 0 and leave ntotal unchanged."""
    index, _, _ = populated_index
    ids_to_remove = np.array([78, 84])
    assert index.remove(ids_to_remove) == 0
    assert index.ntotal == 5
    assert index.ids == {10, 20, 30, 40, 50}


def test_remove_invalid_id_shape(populated_index: VectorIndex) -> None:
    """Attempting to remove an 2D ids raises ValueError."""
    index, _, _ = populated_index
    with pytest.raises(ValueError, match="Expected 1D array, got ndim .*"):
        index.remove(np.array([[10, 20]]))


def test_reset(populated_index: VectorIndex) -> None:
    """`index.reset()` clears ntotal to 0 and index.ids back to set()."""
    index, _, _ = populated_index
    index.reset()
    assert index.ntotal == 0
    assert index.ids == set()


def test_reconstruct(empty_index: VectorIndex) -> None:
    """reconstruct() should return the same vector that was added."""
    vecs = make_unit_vectors(5)
    ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    empty_index.add(vecs, ids)

    np.testing.assert_allclose(vecs[0].reshape(1, -1), empty_index.reconstruct(1))

    with pytest.raises(KeyError):
        empty_index.reconstruct(999)

    with pytest.raises(KeyError):
        empty_index.reconstruct([])


def test_centroid(populated_index: tuple[VectorIndex, np.ndarray, np.ndarray]) -> None:
    """Centroids should be of shape (1, 1024) and an L2 norm of 1.0."""
    index, _, _ = populated_index
    ids = np.array([10, 20], dtype=np.int64)
    centroid = index.centroid(ids)
    assert centroid.shape == (1, 1024)
    assert np.isclose(np.linalg.norm(centroid), 1.0)


def test_search(populated_index: tuple[VectorIndex, np.ndarray, np.ndarray], empty_index: VectorIndex) -> None:
    """search() should return the closest matching IDs and support allowed_ids filtering."""
    index, vecs, _ = populated_index

    # Searching with vecs[0] (ID 10) should return ID 10 as the top match with score ~1.0
    distances, result_ids = index.search(vecs[0], k=3)
    assert distances.shape == (1, 3)
    assert result_ids.shape == (1, 3)
    assert result_ids[0, 0] == 10
    assert np.isclose(distances[0, 0], 1.0)

    # Restricting search with allowed_ids should only return IDs from that subset
    _, filtered_ids = index.search(vecs[0], k=2, allowed_ids=[20, 40])
    assert set(filtered_ids[0]).issubset({20, 40})

    # Searching an empty index should raise RuntimeError
    with pytest.raises(RuntimeError, match="Cannot search an empty index"):
        empty_index.search(vecs[0], k=1)


def test_search_by_id(populated_index: tuple[VectorIndex, np.ndarray, np.ndarray]) -> None:
    """search_by_id() should exclude the query ID by default and include it when exclude_self=False."""
    index, _, _ = populated_index

    # Default exclude_self=True: ID 10 should NOT appear in its own results
    distances, result_ids = index.search_by_id(10, k=3)
    assert distances.shape == (1, 3)
    assert result_ids.shape == (1, 3)
    assert 10 not in result_ids[0]

    # exclude_self=False: ID 10 should be the top result (rank 0) with score ~1.0
    distances_self, result_ids_self = index.search_by_id(10, k=3, exclude_self=False)
    assert result_ids_self[0, 0] == 10
    assert np.isclose(distances_self[0, 0], 1.0)


def test_save_and_load(populated_index: tuple[VectorIndex, np.ndarray, np.ndarray], tmp_path: Path) -> None:
    """Saving an index to disk and loading it back should preserve all IDs and vectors."""
    index, vecs, _ = populated_index
    file_path = tmp_path / "docs.faiss"

    index.save(file_path)
    loaded = VectorIndex.load(file_path)

    assert loaded.ntotal == index.ntotal
    assert loaded.ids == index.ids
    np.testing.assert_allclose(loaded.reconstruct(10), vecs[0].reshape(1, -1))

    with pytest.raises(FileNotFoundError):
        VectorIndex.load(tmp_path / "missing.faiss")
