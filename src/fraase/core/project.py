from __future__ import annotations

from pathlib import Path

from fraase.core.database import Database
from fraase.core.index import VectorIndex


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
