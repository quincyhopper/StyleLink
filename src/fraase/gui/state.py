from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Signal

from fraase.core.project import Project
from fraase.embedding.star import STAR


class AppState(QObject):
    """Single source of truth for the desktop session, essentially just holding the active project and embedder, and broadcasting events."""

    # Setup some signals that the app state can broadcast
    # It will be useful to broadcast when a project has been opened/closed
    project_changed = Signal(object)  # `object` is just any Qt object

    def __init__(self, settings: QSettings | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.settings = settings or QSettings("Quincy Hopper", "FRAASE")
        self.active_project: Project | None = None
        self._embedder: STAR | None = None

    @property
    def embedder(self) -> STAR:
        """The embedding model."""
        if self._embedder is None:
            self._embedder = STAR()
        return self._embedder

    def open_project(self, project_dir: Path) -> Project:
        """Open a new project."""
        # Close existing case if open
        if self.active_project:
            self.close_project()

        self.active_project = Project.open(project_dir)
        self.project_changed.emit(self.active_project)

    def close_project(self) -> None:
        """Close the active project."""
        if self.active_project:
            # Close the case
            pass
