from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

INIT_PRAGMAS = [
    "PRAGMA foreign_keys = ON;",
    "PRAGMA busy_timeout = 5000;",
]

FILE_PRAGMAS = [
    "PRAGMA journal_mode = WAL;",
    "PRAGMA synchronous = NORMAL;",
]

CREATE_DOCUMENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    author TEXT NOT NULL,
    metadata JSON NOT NULL DEFAULT '{}'
);
"""

# Combinations of 'author' and 'metadata' must be unique
CREATE_CENTROIDS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS centroids (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    author TEXT NOT NULL,
    doc_count INTEGER NOT NULL DEFAULT 0,
    metadata JSON NOT NULL DEFAULT '{}',
    UNIQUE(author, metadata)
);
"""


class Database:
    """Database manager for a Project's data."""

    def __init__(self, path: Path | str = ":memory:"):
        """Initialise a database.

        Args:
            path: path to database.
        """
        # Support a :memory: database (for tests)
        self.path = Path(path) if path != ":memory:" else ":memory:"
        self._is_memory = self.path == ":memory:"

        if not self._is_memory:
            assert isinstance(self.path, Path)

        # Establish database connection
        # Check same thread is False to allow queries from QThread
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # Init the database (schemas, etc)
        self._init_database()

    @property
    def conn(self) -> sqlite3.Connection:
        """Expose underlying SQLite connection."""
        return self._conn

    def _init_database(self) -> None:
        """Configure schemas."""
        cursor = self._conn.cursor()
        for pragma in INIT_PRAGMAS:
            cursor.execute(pragma)

        if not self._is_memory:
            for pragma in FILE_PRAGMAS:
                cursor.execute(pragma)

        cursor.execute(CREATE_DOCUMENTS_TABLE_SQL)
        cursor.execute(CREATE_CENTROIDS_TABLE_SQL)
        self._conn.commit()

    def close(self) -> None:
        """Close database connection."""
        self._conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # --- Documents table methods ---
    def add_documents(self, records: Iterable[dict[str, Any]]) -> None:
        """Insert documents into the documents table."""

        def _rows():
            for row in records:
                text = row["text"]
                author = row["author"]
                metadata = {k: v for k, v in row.items() if k and k not in ("text", "author")}
                yield (text, author, json.dumps(metadata))

        with self._conn:
            self._conn.executemany("INSERT INTO documents (text, author, metadata) VALUES (?, ?, ?)", _rows())

    def count_documents(self) -> int:
        """Return the total number of documents in database."""
        row = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return row

    def get_documents(self) -> list[sqlite3.Row]:
        """Return id and text for all documents."""
        # TODO: this may benefit from streaming?
        return self._conn.execute("SELECT id, text FROM documents").fetchall()
