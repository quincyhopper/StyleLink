from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QStackedWidget, QWidget

from fraase.core.project import Project
from fraase.gui.state import AppState


class HomePage(QStackedWidget):
    """Landing page."""

    def __init__(self, state: AppState) -> None:
        super().__init__()
        self.state = state

        # The home page will contain 2 views:
        # - If there is NO active project: empty page
        # - If there is an active project: Explore and Search button

        self.empty_view = self.init_empty_view()
        self.project_view = self.init_project_view()

        self.addWidget(self.empty_view)
        self.addWidget(self.project_view)

        self.state.project_changed.connect(self.on_project_changed)
        self.on_project_changed(self.state.active_project)

    def init_empty_view(self) -> QWidget:
        """Initialise the empty view."""  # ruff:ignore[docstring-missing-returns]
        label = QLabel("Please create a new project or open an existing one.")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        return label

    def init_project_view(self) -> QWidget:
        """Initialise the project view."""  # ruff:ignore[docstring-missing-returns]
        page = QWidget()
        layout = QHBoxLayout(page)

        explore_btn = QPushButton("Explore")
        explore_btn.clicked.connect(self.on_explore_btn_clicked)

        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self.on_search_btn_clicked)

        layout.addWidget(QPushButton("Explore"))
        layout.addWidget(QPushButton("Search"))

        return page

    @Slot(Project)
    def on_project_changed(self, project: Project | None = None) -> None:
        """Set current widget in page stack according to active project status."""
        self.setCurrentWidget(self.empty_view if project is None else self.project_view)

    def on_explore_btn_clicked(self) -> None:
        pass

    def on_search_btn_clicked(self) -> None:
        pass
