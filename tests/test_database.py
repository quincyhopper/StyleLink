import json

import pytest

from stylelink.core.database import Database


def test_add_documents_extracts_metadata(memory_db: Database) -> None:
    """Extra fields beyond 'author' and 'text' should be serialized into the JSON metadata column."""
    records = [
        {"author": "Alice", "text": "Hello world", "subreddit": "python", "year": 2025, "": "ignored_empty_key"},
        {"author": "Bob", "text": "Only core fields"},
    ]

    memory_db.docs_table.add(records)

    assert memory_db.docs_table.count() == 2

    rows = memory_db.conn.execute("SELECT id, author, text, metadata FROM documents ORDER BY id").fetchall()

    assert rows[0]["author"] == "Alice"
    assert rows[0]["text"] == "Hello world"
    assert json.loads(rows[0]["metadata"]) == {"subreddit": "python", "year": 2025}

    assert rows[1]["author"] == "Bob"
    assert json.loads(rows[1]["metadata"]) == {}


def test_add_missing_required_fields_raises_key_error(memory_db: Database) -> None:
    """Adding a record without 'author' or 'text' should raise a KeyError and rollback."""
    with pytest.raises(KeyError):
        memory_db.docs_table.add([{"author": "Alice"}])  # Missing 'text'

    assert memory_db.docs_table.count() == 0


def test_delete_documents_handles_empty_and_targeted_ids(memory_db: Database) -> None:
    """delete() should safely no-op on empty input and remove only the specified document IDs."""
    memory_db.docs_table.add(
        [
            {"author": "Alice", "text": "Doc 1"},
            {"author": "Bob", "text": "Doc 2"},
            {"author": "Charlie", "text": "Doc 3"},
        ]
    )

    # Empty list should be a safe no-op
    memory_db.docs_table.delete([])
    assert memory_db.docs_table.count() == 3

    # Delete first and third documents
    memory_db.docs_table.delete([1, 3])
    remaining = memory_db.docs_table.get_all()
    assert len(remaining) == 1
    assert remaining[0]["id"] == 2
    assert remaining[0]["text"] == "Doc 2"


def test_centroids_clear_resets_autoincrement(memory_db: Database) -> None:
    """centroids_table.clear() should remove all centroids and reset the AUTOINCREMENT ID sequence to 1."""
    with memory_db.conn as conn:
        conn.execute("INSERT INTO centroids (author, doc_count) VALUES ('Alice', 3)")
        conn.execute("INSERT INTO centroids (author, doc_count) VALUES ('Bob', 5)")

    memory_db.centroids_table.clear()

    count = memory_db.conn.execute("SELECT COUNT(*) FROM centroids").fetchone()[0]
    assert count == 0

    # Because sqlite_sequence was cleared, the next inserted centroid must get ID 1 again
    with memory_db.conn as conn:
        cursor = conn.execute("INSERT INTO centroids (author, doc_count) VALUES ('Charlie', 2)")
        assert cursor.lastrowid == 1
