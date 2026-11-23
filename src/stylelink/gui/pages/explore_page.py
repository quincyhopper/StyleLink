from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMessageBox, QPushButton, QVBoxLayout, QWidget

from stylelink.gui.state import AppState
from stylelink.gui.widgets.filter_table import SqlFilterTable


class ExplorePage(QWidget):
    """Class for exploring the database."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.table = SqlFilterTable(parent=self)

        delete_btn = QPushButton()
        delete_btn.clicked.connect(self._on_delete_clicked)
        delete_btn.setIcon(QIcon.fromTheme(QIcon.ThemeIcon.EditDelete))
        delete_btn.setToolTip("Delete documents")

        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addWidget(delete_btn, alignment=Qt.AlignmentFlag.AlignRight)

        # Set the database only if there is an active project
        self.state.project_changed.connect(lambda p: self.table.set_database(p.database if p else None))

    @Slot()
    def _on_delete_clicked(self) -> None:
        project = self.state.active_project
        if project is None:
            return

        ids = self.table.get_effective_ids()
        if not ids:
            return

        reply = QMessageBox.question(
            self, "Delete documents", f"Are you sure you want to delete {len(ids)} document(s)?"
        )

        if reply == QMessageBox.StandardButton.Yes:
            project.delete_documents(ids)
            self.state.project_changed.emit(project)
