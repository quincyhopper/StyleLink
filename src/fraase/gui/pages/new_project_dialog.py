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
from fraase.utils.file import preview_file, read_file


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
        self.setMinimumSize(600, 400)

        # Form layout will hold Project Name and Import fields
        form = QFormLayout()
        layout.addLayout(form)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)  # use full width

        # Project name
        self.project_name = QLineEdit()
        self.project_name.setPlaceholderText("Choose a name for the new project")
        self.project_name.textChanged.connect(self.validate_form)
        form.addRow("New project name:", self.project_name)

        # Importing
        file_row = QHBoxLayout()
        self.file_name = QLineEdit()
        self.file_name.setReadOnly(True)
        self.file_name.setPlaceholderText("Select a .csv, .tsv, or .xlsx dataset file to import...")
        file_row.addWidget(self.file_name)

        # Button for importing a file
        choose_file_btn = QPushButton("Choose file")
        choose_file_btn.clicked.connect(self.on_choose_file_btn)
        file_row.addWidget(choose_file_btn)
        form.addRow("Import:", file_row)  # Add everything to form

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

        self.create_btn = QPushButton("Create Project")
        self.create_btn.clicked.connect(self.on_create_btn_clicked)
        self.create_btn.setProperty("primary", True)
        self.create_btn.setEnabled(False)

        final_btn_layout.addWidget(cancel_btn)
        final_btn_layout.addWidget(self.create_btn)
        layout.addLayout(final_btn_layout)

    def validate_form(self) -> None:
        """Enable Create Project button only when project name and file are valid."""
        has_name = bool(self.project_name.text().strip())
        has_file = bool(self.file_name.text().strip())

        self.create_btn.setEnabled(has_name and has_file)

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
            self.validate_form()

    @Slot()
    def on_cancel_btn_clicked(self) -> None:
        """Close the dialog."""
        self.close()

    @Slot()
    def on_create_btn_clicked(self) -> None:
        """Create new project given the inputs."""
        # The project directory will just have the project name
        project_dir = Path(self.project_name.text())

        # Create the project
        source_file = Path(self.file_name.text())
        project = Project.create(project_dir=project_dir)

        # Populate database
        project.database.add_documents(read_file(source_file))

        # Set the active project to the new project
        self.state.open_project(project_dir=project_dir)

        # Close the dialog box successfully
        self.accept()


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
