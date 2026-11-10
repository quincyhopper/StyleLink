from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import QLabel, QProgressDialog, QPushButton, QStackedWidget, QVBoxLayout, QWidget

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
        layout = QVBoxLayout(page)

        explore_btn = QPushButton("Explore")
        explore_btn.clicked.connect(self.on_explore_btn_clicked)

        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self.on_search_btn_clicked)

        embed_btn = QPushButton("Create Styles")
        embed_btn.clicked.connect(self.on_embed_btn_clicked)

        # Shows whether database and FAISS are synced or not
        self.sync_label = QLabel()

        layout.addWidget(explore_btn)
        layout.addWidget(search_btn)
        layout.addWidget(embed_btn)

        return page

    @Slot(Project)
    def on_project_changed(self, project: Project | None = None) -> None:
        """Set current widget in page stack according to active project status."""
        self.setCurrentWidget(self.empty_view if project is None else self.project_view)
        self.update_sync_status()

    def on_explore_btn_clicked(self) -> None:
        pass

    def on_search_btn_clicked(self) -> None:
        pass

    def on_embed_btn_clicked(self) -> None:
        """Syncs embeddings."""
        if self.state.active_project is None:
            return

        self.progress = QProgressDialog(labelText="Embedding documents...", cancelButtonText=None, minimum=0, maximum=0)
        self.progress.setWindowTitle("Creating Styles")
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)  # Block input to
        self.progress.setMinimumDuration(0)  # Show immediately when clicked
        self.progress.show()

        self.worker = EmbedWorker(self.state.active_project, self)
        self.worker.progress.connect(self.on_embed_progress)
        self.worker.finished.connect(self.on_embed_finished)
        self.worker.start()

    def on_embed_progress(self, processed: int, total: int) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(processed)

    def on_embed_finished(self) -> None:
        self.progress.close()
        self.update_sync_status()

    def update_sync_status(self) -> None:
        """Updates sync label with sync status."""
        if self.state.active_project is None:
            return

        synced = self.state.active_project.is_synced
        self.sync_label.setText("Index: In sync" if synced else "Index: Out of sync")


class EmbedWorker(QThread):
    """Background thread for embedding documents."""

    progress = Signal(int, int)

    def __init__(self, project: Project, parent=None) -> None:
        super().__init__(parent)
        self.project = project

    def run(self) -> None:
        self.project.sync_embeddings(progress_callback=self.progress.emit)
