from __future__ import annotations

from pathlib import Path

from fraase.core.database import Database
from fraase.utils.file import read_file


class Project:
    """Manage a project directory."""

    def __init__(self, project_dir: Path, database: Database | None = None):
        self.project_dir = project_dir
        self.db_path = project_dir / "database.db"
        self.database = database or Database(self.db_path)

    @classmethod
    def create(cls, project_dir: Path, source_file: Path) -> Project:
        """Create a new project.

        Args:
            project_dir: the name of the directory in which to store the project.
            source_file: path to the data to be imported.

        Returns:
            Instance of a new Project.
        """
        project_dir.mkdir(parents=True, exist_ok=True)

        project = cls(project_dir=project_dir)

        # Read the provided source file
        file_stream = read_file(source_file)
        project.database.add_documents(file_stream)

        return project

    @classmethod
    def open(cls, project_dir: Path) -> Project:
        """Open an existing project.

        Returns:
            Project instance from specified directory.
        """
        # Will need to initialise all services here
        return cls(project_dir=project_dir)
