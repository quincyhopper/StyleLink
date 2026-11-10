from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

OPERATORS = {
    "=": ("{col} = ?", lambda v: v),
    "!=": ("{col} != ?", lambda v: v),
    "CONTAINS": ("{col} LIKE ?", lambda v: f"%{v}%"),
    ">": ("{col} > ?", lambda v: v),
    "<": ("{col} < ?", lambda v: v),
}

CORE_FIELDS = ["id", "author", "text"]


def compile_filter(rules: list[tuple[str, str, str]]) -> tuple[str, list[Any]]:
    """Convert UI rules [(field, op, value)] into a WHERE clause.

    Args:
        rules: list of tuples, such as [("author", "equals", "John"), ("corpus", "equals", "Reddit"), ("text", "contains", "apples")]

    Returns:
        where: the SQL query.
        params: the parameters of the SQL query.
    """
    clauses = []
    params = []

    for field, op, val in rules:
        val = val.strip()

        if not val:
            # Ignore empty values
            continue

        if field in CORE_FIELDS:
            col = field
        else:
            # Allows us to query arbitrary JSON metadata fields
            col = f"json_extract(metadata, '$.\"{field}\"')"

        template, transform = OPERATORS[op]
        clauses.append(template.format(col=col))
        params.append(transform(val))

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


class FilterRuleRow(QWidget):
    """Class for a filter row/entry."""

    removed = Signal(QWidget)
    changed = Signal()

    def __init__(self, fields: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        self.field_combo = QComboBox()  # Dropdown box to select field of SQL table
        self.field_combo.addItems(fields)

        self.op_combo = QComboBox()  # Dropdown box to select operators
        self.op_combo.addItems(list(OPERATORS.keys()))

        self.val_edit = QLineEdit("Filter value...")  # Where the user enters the value

        del_btn = QPushButton()
        del_btn.setFixedWidth(28)
        del_btn.clicked.connect(lambda: self.removed.emit(self))  # Emit the signal for if this row is removed

        self.field_combo.currentTextChanged.connect(lambda _: self.changed.emit())
        self.op_combo.currentTextChanged.connect(lambda _: self.changed.emit())
        self.val_edit.textChanged.connect(lambda _: self.changed.emit())

        layout.addWidget(self.field_combo, stretch=2)
        layout.addWidget(self.op_combo, stretch=1)
        layout.addWidget(self.val_edit, stretch=3)
        layout.addWidget(del_btn)

    def rule(self) -> tuple[str, str, str]:
        """Return the row as it currently is."""
        return (self.field_combo.currentText(), self.op_combo.currentText(), self.val_edit.text())


class SqlFilterTable(QWidget):
    """Class for holding the SQL table widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.db_path = Path
        self.meta_keys: list[str] = []  # Will scan the db to find these
        self._rule_rows: list[FilterRuleRow]  # List of filter rows above the SQL viewer

        self.build_ui()

    def build_ui(self):
        """Build the UI."""
