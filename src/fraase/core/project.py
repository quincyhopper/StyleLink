from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from fraase.core.database import Database
from fraase.core.index import VectorIndex
from fraase.embedding.star import STAR


class Project:
    """Manage a project directory.

    Directory layout:
        case_dir/
        ├── database.db      # SQLite document and centroid store
        ├── centroids.faiss  # FAISS centroid index
        └── docs.faiss       # FAISS document index
    """

    DB_FILENAME = "database.db"
    DOC_INDEX_FILENAME = "docs.faiss"
    CENTROID_INDEX_FILENAME = "centroids.faiss"

    def __init__(self, project_dir: Path, database: Database, doc_index: VectorIndex, centroid_index: VectorIndex):
        self.project_dir = project_dir
        self.database = database
        self.doc_index = doc_index
        self.centroid_index = centroid_index
        self.embedder = STAR()

    @classmethod
    def create(cls, project_dir: Path) -> Project:
        """Create a new project.

        Args:
            project_dir: the name of the directory in which to store the project.

        Returns:
            Instance of a new Project.
        """
        # Create directory for project
        project_dir.mkdir(parents=True, exist_ok=True)

        # Create/open a database
        database = Database(project_dir / cls.DB_FILENAME)

        # Create FAISS indexes
        doc_index = VectorIndex()
        doc_index.save(project_dir / cls.DOC_INDEX_FILENAME)
        centroid_index = VectorIndex()
        centroid_index.save(project_dir / cls.CENTROID_INDEX_FILENAME)

        return cls(
            project_dir=project_dir,
            database=database,
            doc_index=doc_index,
            centroid_index=centroid_index,
        )

    @classmethod
    def open(cls, project_dir: Path) -> Project:
        """Open an existing project.

        Returns:
            Project instance from specified directory.
        """
        database = Database(project_dir / cls.DB_FILENAME)
        doc_index = VectorIndex.load(project_dir / cls.DOC_INDEX_FILENAME)
        centroid_index = VectorIndex.load(project_dir / cls.CENTROID_INDEX_FILENAME)

        return cls(
            project_dir=project_dir,
            database=database,
            doc_index=doc_index,
            centroid_index=centroid_index,
        )

    @property
    def is_synced(self) -> bool:
        """True if number of documents in database equals vectors in document index."""
        return self.database.docs_table.count() == self.doc_index.ntotal

    def sync_embeddings(self, progress_callback: Callable[[int, int], None] | None = None) -> None:
        """Embed any documents that are NOT already in the document index. If a dataset has just been imported, this will simply embed the entire dataset."""
        existing_ids = self.doc_index.ids
        missing = [
            row for row in self.database.docs_table.get_all() if row["id"] not in existing_ids
        ]  # TODO: more efficient way to do this?
        if not missing:
            return

        ids = [row["id"] for row in missing]
        texts = [row["text"] for row in missing]

        # Embed and add to indexes
        embeddings = self.embedder.embed_texts(texts=texts, progress_callback=progress_callback)
        self.embedder.release()
        self.doc_index.add(embeddings, ids)
        self.doc_index.save(self.project_dir / self.DOC_INDEX_FILENAME)

        # TODO: build centroids

    def build_author_centroids(self, authors: Sequence[str] | None = None) -> None:
        """Build centroids for the given authors and populate the SQL and FAISS databases.

        Args:
            authors: sequence of authors for whom to construct centroids.
        """
        # Find target IDs - the authors we need to build
        if authors is None:
            self.centroid_index.reset()
            self.database.centroids_table.clear()
            target_ids = self.doc_index.ids
        else:
            target_ids = list(authors)

        if not target_ids:
            return
