from __future__ import annotations

from pathlib import Path


class Project:
    """Manage a project directory."""

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir

    @classmethod
    def create(cls, project_dir: Path, name: str) -> Project:
        """Create a new project.

        Returns:
            Instance of a new Project.
        """
        project_dir.mkdir(parents=True, exist_ok=True)

        # Init services here

        return cls(project_dir=project_dir)

    @classmethod
    def open(cls, project_dir: Path) -> Project:
        """Open an existing project.

        Returns:
            Project instance from specified directory.
        """
        # Will need to initialise all services here
        return cls(project_dir=project_dir)
