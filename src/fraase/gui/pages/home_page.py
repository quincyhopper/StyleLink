from __future__ import annotations

import time

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from fraase.core.project import Project
from fraase.gui.state import AppState


def format_eta(seconds: float) -> str:
    """Convert seconds ETA to human-readable string."""
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours}h {mins:02d}m"
    if mins > 0:
        return f"{mins}m {secs:02d}s"
    return f"{secs}s"


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

        self.progress_frame = QFrame()
        self.progress_frame.setFrameShape(QFrame.Shape.StyledPanel)
        prog_layout = QHBoxLayout(self.progress_frame)
        prog_layout.setContentsMargins(10, 6, 10, 6)

        self.progress_label = QLabel("Processing...")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumWidth(260)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)

        prog_layout.addWidget(self.progress_label)
        prog_layout.addWidget(self.progress_bar, stretch=1)
        prog_layout.addWidget(self.cancel_btn)
        self.progress_frame.setVisible(False)

        sync_layout.addWidget(self.sync_label)
        sync_layout.addWidget(self.embed_btn)
        sync_layout.addWidget(self.progress_frame)

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

    @Slot()
    def _on_cancel_clicked(self) -> None:
        self.cancel_btn.setEnabled(False)
        self.progress_label.setText("Cancelling...")
        self.worker.requestInterruption()

    def on_embed_btn_clicked(self) -> None:
        """Syncs embeddings."""
        if self.state.active_project is None:
            return

        self.embed_btn.setEnabled(False)
        self._embed_start = time.monotonic()
        self.progress_frame.setVisible(True)
        self.progress_label.setText("Loading Style creator...")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.worker = EmbedWorker(self.state.active_project, self)
        self.worker.progress.connect(self.on_embed_progress)
        self.worker.finished.connect(self.on_embed_finished)
        self.worker.start()

    def on_embed_progress(self, processed: int, total: int) -> None:
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(processed)

        if processed == 0:
            self._embed_start = time.monotonic()
            self.progress_label.setText(f"Embedding documents: 0 / {total:,}")
            return

        elapsed = time.monotonic() - self._embed_start
        eta_str = format_eta((elapsed / processed) * (total - processed))
        self.progress_label.setText(f"Embedding documents: {processed:,} / {total:,} (~{eta_str} remaining)")

    def on_embed_finished(self) -> None:
        self.progress_frame.setVisible(False)
        self.embed_btn.setEnabled(True)
        self.update_sync_status()

    def update_sync_status(self) -> None:
        """Updates sync label with sync status."""
        if self.state.active_project is None:
            return

        if self.state.active_project.is_synced:
            self.sync_label.setText("All documents and authors have Styles. Perform a search when ready!")
            self.sync_label.setStyleSheet("")  # Return to default colours
            self.embed_btn.setVisible(False)
        else:
            self.embed_btn.setVisible(True)

            docs_total = self.state.active_project.database.docs_table.count()
            indexed_docs = self.state.active_project.doc_index.ntotal

            self.sync_label.setText(
                f"{indexed_docs}/{docs_total} documents are missing Styles. Create them with the button below:"
            )
            self.sync_label.setStyleSheet("color: #d32f2f;")


class EmbedWorker(QThread):
    """Background thread for embedding documents."""

    progress = Signal(int, int)

    def __init__(self, project: Project, parent=None) -> None:
        super().__init__(parent)
        self.project = project

    def run(self) -> None:
        try:
            self.project.sync_embeddings(
                progress_callback=self.progress.emit, cancel_check=self.isInterruptionRequested
            )
        except InterruptedError:
            pass
