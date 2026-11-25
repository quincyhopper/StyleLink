from pathlib import Path

from stylelink.core.project import Project


def test_create_and_open_project(tmp_path: Path, mock_embedder) -> None:
    """Creating a project initializes the SQLite DB and FAISS files on disk, which Project.open() can reload."""
    project_dir = tmp_path / "test_case"
    project = Project.create(project_dir)

    assert (project_dir / Project.DB_FILENAME).is_file()
    assert (project_dir / Project.DOC_INDEX_FILENAME).is_file()
    assert (project_dir / Project.CENTROID_INDEX_FILENAME).is_file()
    assert project.is_synced is True  # 0 docs == 0 vectors

    project.database.close()

    reopened = Project.open(project_dir)
    assert reopened.doc_index.ntotal == 0
    assert reopened.centroid_index.ntotal == 0
    reopened.database.close()


def test_sync_embeddings_and_incremental_sync(tmp_path: Path, mock_embedder, monkeypatch) -> None:
    """sync_embeddings() should embed unindexed documents, build author centroids, and only embed newly added documents on subsequent syncs."""
    project = Project.create(tmp_path / "case")
    project.database.docs_table.add(
        [
            {"author": "Alice", "text": "Alice text 1"},
            {"author": "Alice", "text": "Alice text 2"},
            {"author": "Bob", "text": "Bob text 1"},
        ]
    )

    # Before syncing: 3 documents in DB, 0 in FAISS
    assert project.is_synced is False

    project.sync_embeddings()

    # After syncing: 3 docs indexed, 2 distinct author centroids built
    assert project.is_synced is True
    assert project.doc_index.ids == {1, 2, 3}
    assert project.centroid_index.ntotal == 2

    centroids = project.database.conn.execute(
        "SELECT author, doc_count FROM centroids ORDER BY author"
    ).fetchall()
    assert [(r["author"], r["doc_count"]) for r in centroids] == [("Alice", 2), ("Bob", 1)]

    # Add a new document for an existing author (Bob) and track what gets passed to embed_texts
    project.database.docs_table.add([{"author": "Bob", "text": "Bob text 2"}])
    assert project.is_synced is False

    embedded_Batches: list[list[str]] = []
    orig_embed = project.embedder.embed_texts

    def spy_embed_texts(texts, **kwargs):
        embedded_Batches.append(list(texts))
        return orig_embed(texts=texts, **kwargs)

    monkeypatch.setattr(project.embedder, "embed_texts", spy_embed_texts)

    project.sync_embeddings()

    # Only the 1 new document should have been sent to the embedder, not all 4
    assert embedded_Batches == [["Bob text 2"]]
    assert project.is_synced is True
    assert project.doc_index.ids == {1, 2, 3, 4}

    # Bob's centroid doc_count should now be updated to 2
    bob_count = project.database.conn.execute(
        "SELECT doc_count FROM centroids WHERE author = 'Bob'"
    ).fetchone()["doc_count"]
    assert bob_count == 2
    project.database.close()


def test_delete_documents_updates_indexes_and_centroids(tmp_path: Path, mock_embedder) -> None:
    """Deleting documents removes them from SQLite and FAISS, and updates or removes affected author centroids."""
    project = Project.create(tmp_path / "case")
    project.database.docs_table.add(
        [
            {"author": "Alice", "text": "Alice text 1"},
            {"author": "Alice", "text": "Alice text 2"},
            {"author": "Bob", "text": "Bob only text"},
        ]
    )
    project.sync_embeddings()

    # Delete one of Alice's documents (ID 1) and Bob's only document (ID 3)
    project.delete_documents([1, 3])

    assert project.is_synced is True
    assert project.database.docs_table.count() == 1
    assert project.doc_index.ids == {2}

    # Only Alice should remain in centroids, with doc_count == 1
    assert project.centroid_index.ntotal == 1
    remaining_centroids = project.database.conn.execute(
        "SELECT author, doc_count FROM centroids"
    ).fetchall()
    assert len(remaining_centroids) == 1
    assert remaining_centroids[0]["author"] == "Alice"
    assert remaining_centroids[0]["doc_count"] == 1
    project.database.close()
