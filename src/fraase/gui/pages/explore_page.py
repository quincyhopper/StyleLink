from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from fraase.gui.state import AppState
from fraase.gui.widgets.filter_table import SqlFilterTable


class ExplorePage(QWidget):
    """Class for exploring the database."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.table = SqlFilterTable(parent=self, show_selection_scope=False)

        layout = QVBoxLayout(self)
        layout.addWidget(self.table)

        # Set the database only if there is an active project
        self.state.project_changed.connect(lambda p: self.table.set_database(p.database if p else None))
