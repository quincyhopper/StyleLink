from pathlib import Path

import pytest

from stylelink.core.project import Project
from stylelink.core.search import CentroidSearchResult, DocumentSearchResult, run_search


@pytest.fixture
def synced_project(tmp_path: Path, mock_embedder) -> Project:
    """Create a synced project with 3 authors and 4 documents."""
    project = Project.create(tmp_path / "search_case")
    project.database.docs_table.add(
        [
            {"author": "Alice", "text": "Alice doc one", "source": "reddit"},
            {"author": "Alice", "text": "Alice doc two", "source": "twitter"},
            {"author": "Bob", "text": "Bob doc one", "source": "reddit"},
            {"author": "Charlie", "text": "Charlie doc one", "source": "blog"},
        ]
    )
    project.sync_embeddings()
    yield project
    project.database.close()


def test_run_search_guards(tmp_path: Path, synced_project: Project, mock_embedder) -> None:
    """run_search() should reject unsynced projects and empty query or target ID lists."""
    unsynced = Project.create(tmp_path / "unsynced_case")
    unsynced.database.docs_table.add([{"author": "Alice", "text": "Unindexed doc"}])

    with pytest.raises(RuntimeError, match="Project embeddings and centroids are not synced with the database."):
        run_search(unsynced, "Document-to-Document", query_ids=[1], target_ids=[1], k=1, exclude_self=True)
    unsynced.database.close()

    with pytest.raises(ValueError, match="query_ids cannot be empty."):
        run_search(synced_project, "Document-to-Document", query_ids=[], target_ids=[1, 2], k=1, exclude_self=True)

    with pytest.raises(ValueError, match="target_ids cannot be empty."):
        run_search(synced_project, "Document-to-Document", query_ids=[1], target_ids=[], k=1, exclude_self=True)


def test_doc_to_doc_search(synced_project: Project) -> None:
    """Document-to-Document search should return DocumentSearchResult objects, unpack metadata, and respect exclude_self."""
    results, labels = run_search(
        project=synced_project,
        mode="Document-to-Document",
        query_ids=[1],
        target_ids=[1, 2, 3, 4],
        k=2,
        exclude_self=True,
    )

    assert labels == ["ID #1 (Alice)"]
    assert len(results) == 1
    matches = results[0]
    assert len(matches) == 2

    # All matches should be DocumentSearchResult with sequential ranks 1..k, excluding query doc ID 1
    assert all(isinstance(m, DocumentSearchResult) for m in matches)
    assert [m.rank for m in matches] == [1, 2]
    assert all(m.id != 1 for m in matches)
    assert all(m.llr is not None for m in matches)
    assert all("source" in m.metadata for m in matches)


def test_doc_to_author_exclude_self_filters_matching_author(synced_project: Project) -> None:
    """In Document-to-Author mode, exclude_self=True must filter out the centroid belonging to the query document's author."""
    # Query with Document ID 1 (author: "Alice") against all 3 author centroids (IDs 1, 2, 3)
    results_excluded, _ = run_search(
        project=synced_project,
        mode="Document-to-Author",
        query_ids=[1],
        target_ids=[1, 2, 3],
        k=2,
        exclude_self=True,
    )

    matches = results_excluded[0]
    assert len(matches) == 2
    assert all(isinstance(m, CentroidSearchResult) for m in matches)
    assert [m.rank for m in matches] == [1, 2]
    # Alice's own centroid must be excluded even though her ID in centroids (1) is a centroid ID
    assert all(m.author != "Alice" for m in matches)

    # With exclude_self=False, Alice's centroid should be allowed in the results
    results_included, _ = run_search(
        project=synced_project,
        mode="Document-to-Author",
        query_ids=[1],
        target_ids=[1, 2, 3],
        k=3,
        exclude_self=False,
    )
    assert "Alice" in {m.author for m in results_included[0]}


def test_author_to_author_search(synced_project: Project) -> None:
    """Author-to-Author search should query the centroid index and return CentroidSearchResult objects with doc_count."""
    results, labels = run_search(
        project=synced_project,
        mode="Author-to-Author",
        query_ids=[1, 2],  # Centroid 1 (Alice), Centroid 2 (Bob)
        target_ids=[1, 2, 3],
        k=2,
        exclude_self=True,
    )

    assert labels == ["ID #1 (Alice)", "ID #2 (Bob)"]
    assert len(results) == 2

    # Centroid 1 (Alice) results should exclude Alice and only contain Bob and Charlie
    alice_matches = results[0]
    assert [m.rank for m in alice_matches] == [1, 2]
    assert {m.author for m in alice_matches} == {"Bob", "Charlie"}

    # Centroid 2 (Bob) results should include Alice with doc_count == 2
    bob_matches = results[1]
    alice_match = next(m for m in bob_matches if m.author == "Alice")
    assert alice_match.doc_count == 2
