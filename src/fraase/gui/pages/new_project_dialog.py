from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fraase.core.project import Project
from fraase.gui.state import AppState
from fraase.utils.file import preview_file


class NewProjectDialog(QDialog):
    """Dialog for importing a new project."""

    def __init__(self, state: AppState) -> None:
        super().__init__()
        self.setWindowTitle("New Project")
        self.state = state
        self.build_ui()

    def build_ui(self) -> None:
        """Build the apperance of this dialog."""
        # Set the main layout for the dialog
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 28, 40, 28)
        layout.setSpacing(18)
        self.setMinimumSize(600, 400)

        # Form layout will hold Project Name and Import fields
        form = QFormLayout()

        # Project name
        self.project_name = QLineEdit()
        self.project_name.setPlaceholderText("Choose a name for the new project")
        form.addRow("New project name:", self.project_name)

        # Importing
        file_row = QHBoxLayout()
        self.file_name = QLineEdit()
        self.file_name.setReadOnly(True)
        self.file_name.setPlaceholderText("Select a file to import")
        file_row.addWidget(self.file_name)

        # Button for importing a file
        choose_file_btn = QPushButton("Choose file")
        choose_file_btn.clicked.connect(self.on_choose_file_btn)
        file_row.addWidget(choose_file_btn)
        form.addRow("Import:", file_row)  # Add everything to form

        # Add the form to the main layout
        layout.addLayout(form)

        # Add a separation line in betwen the form and the file preview
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)

        # File preview
        self.preview_widget = ImportPreviewWidget()
        self.preview_widget.hide()
        layout.addWidget(self.preview_widget, stretch=1)

        # Last two "Cancel" and "Create Project" buttons
        final_btn_layout = QHBoxLayout()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.on_cancel_btn_clicked)

        create_btn = QPushButton("Create Project")
        create_btn.clicked.connect(self.on_create_btn_clicked)
        create_btn.setProperty("primary", True)

        final_btn_layout.addWidget(cancel_btn)
        final_btn_layout.addWidget(create_btn)
        layout.addLayout(final_btn_layout)

    @Slot()
    def on_choose_file_btn(self) -> None:
        """Go the file selector and import the file."""
        file_str, _ = QFileDialog.getOpenFileName(
            parent=self, caption="Select file to import", filter="Tabular Files (*.xlsx *.csv *.tsv)"
        )

        if file_str:
            path = Path(file_str)

            # Assemble preview
            preview = preview_file(path)
            self.preview_widget.set_preview(preview)
            self.preview_widget.show()

            # Update forms with selected file
            self.file_name.setText(str(path))

    @Slot()
    def on_cancel_btn_clicked(self) -> None:
        """Close the dialog."""
        self.close()

    @Slot()
    def on_create_btn_clicked(self) -> None:
        """Create new project given the inputs."""
        # TODO: probably need to make sure that this can't be pressed unless a file has been chosen
        new_project = Project(project_name=self.project_name)
        self.state.open_project(new_project.dir)


class ImportPreviewWidget(QWidget):
    """Class for parsing a file preview into a tabular display widget."""

    def __init__(self, preview: tuple[list[dict[str, Any]], list[str]] | None = None):
        super().__init__()

        self.preview = preview
        self.build_ui()

        if self.preview is not None:
            self.set_preview(preview)

    def build_ui(self):
        """Build preview table."""
        layout = QVBoxLayout(self)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)

        # Fix the column width
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 150)
        self.table.setColumnWidth(1, 200)

        # Preview needs to be able to expand once it is revealed
        self.table.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        layout.addWidget(self.table)

    def set_preview(self, preview: tuple[list[dict[str, Any]], list[str]]) -> None:
        """Populate table."""
        rows, cols = preview

        self.table.setColumnCount(len(cols))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels(cols)

        # For truncating text
        metrics = QFontMetrics(self.table.font())
        mode = Qt.TextElideMode.ElideRight

        for row_idx, row in enumerate(rows):
            for col_idx, col in enumerate(cols):
                value = str(row.get(col, ""))

                # Truncate long text
                item = QTableWidgetItem(metrics.elidedText(value, mode, 180))
                item.setToolTip(value)  # This preserves the original text
                self.table.setItem(row_idx, col_idx, item)

        self.preview = preview
