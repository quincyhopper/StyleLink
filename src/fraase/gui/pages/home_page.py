from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import QGroupBox, QLabel, QProgressDialog, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from fraase.core.project import Project
from fraase.gui.state import AppState


class HomePage(QStackedWidget):
    """Landing page."""

    # Quick document-to-document search
    quick_search_requested = Signal()

    def __init__(self, state: AppState) -> None:
        super().__init__()
        self.state = state

        # The home page will contain 2 views:
        # - If there is NO active project: empty page
        # - If there is a project, some buttons

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

        # Sync section
        sync_box = QGroupBox("Project status")
        sync_layout = QVBoxLayout(sync_box)
        self.sync_label = QLabel()
        self.embed_btn = QPushButton("Create Styles")
        self.embed_btn.clicked.connect(self.on_embed_btn_clicked)
        sync_layout.addWidget(self.sync_label)
        sync_layout.addWidget(self.embed_btn)

        # Quick search option
        search_box = QGroupBox("Quick search")
        search_layout = QVBoxLayout(search_box)
        search_info = QLabel(
            "Go to the Search page to perform a custom search, or press below "
            "to compare all document styles against each other."
        )
        quick_search_btn = QPushButton("Quick search")
        quick_search_btn.clicked.connect(self.quick_search_requested.emit)
        search_layout.addWidget(search_info)
        search_layout.addWidget(quick_search_btn)

        layout.addWidget(sync_box)
        layout.addWidget(search_box)
        layout.addStretch()

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

        if self.state.active_project.is_synced:
            self.sync_label.setText("All documents and authors have Styles. Perform a search when ready!")
            self.embed_btn.setVisible(False)
        else:
            self.embed_btn.setVisible(True)
            self.sync_label.setText(
                "Some documents or authors dont't have Styles. Create them with the Create Styles button."
            )


class EmbedWorker(QThread):
    """Background thread for embedding documents."""

    progress = Signal(int, int)

    def __init__(self, project: Project, parent=None) -> None:
        super().__init__(parent)
        self.project = project

    def run(self) -> None:
        self.project.sync_embeddings(progress_callback=self.progress.emit)
