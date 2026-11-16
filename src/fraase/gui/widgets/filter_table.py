from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from fraase.core.database import Database

OPERATORS = {
    "=": ("{col} = ?", lambda v: v),
    "!=": ("{col} != ?", lambda v: v),
    "CONTAINS": ("{col} LIKE ?", lambda v: f"%{v}%"),
    ">": ("{col} > ?", lambda v: v),
    "<": ("{col} < ?", lambda v: v),
}

# Fields that are guaranteed for the two tables
TABLE_FIELDS = {
    "documents": ["id", "author", "text"],
    "centroids": ["id", "author", "doc_count"],
}


def compile_filter(rules: list[tuple[str, str, str]], core_fields: list[str]) -> tuple[str, list[Any]]:
    """Convert UI rules [(field, op, value)] into a WHERE clause.

    Args:
        rules: list of tuples, such as [("author", "equals", "John"), ("corpus", "equals", "Reddit"), ("text", "contains", "apples")]
        core_fields: list of core table columns.

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

        if field in core_fields:
            col = field
        else:
            # Allows us to query arbitrary JSON metadata fields
            col = f"json_extract(metadata, '$.\"{field}\"')"

        template, transform = OPERATORS[op]
        clauses.append(template.format(col=col))
        params.append(transform(val))

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


class SqlTableModel(QAbstractTableModel):
    """Class for viewing the SQL table."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.rows: list[dict[str, Any]] = []
        self.columns: list[str] = []
        self.core_fields: set[str] = set()

    def set_data(self, rows: list[dict[str, Any]], core_fields: list[str], meta_keys: list[str]) -> None:
        """Fill in the data in the SQL viewer."""
        self.beginResetModel()
        self.rows = rows
        self.core_fields = set(core_fields)
        self.columns = [f for f in core_fields if f != "text"] + meta_keys
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of rows."""
        return len(self.rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of columns."""
        return len(self.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        row = self.rows[index.row()]
        col = self.columns[index.column()]
        val = row.get(col) if col in self.core_fields else row["parsed_metadata"].get(col)
        return "" if val is None else str(val)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.columns[section]
        return None


class FilterRuleRow(QWidget):
    """A single visual filter row: [Field combo] [Operator combo] [Value input] [Remove btn]."""

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

        self.val_edit = QLineEdit()  # Where the user enters the value
        self.val_edit.setPlaceholderText("Filter value...")

        del_btn = QPushButton("✕")
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
    """Class for holding the filter functionality, SQL table viewer, and text inspector."""

    selection_changed = Signal()

    def __init__(
        self,
        database: Database | None = None,
        table: str = "documents",
        show_selection_scope: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        """Initialise the SQL table viewer.

        Args:
            database: instance of a database.
            table: "documents" or "centroids".
            show_selection_scope: if True, display the "All filtered rows" vs. "Highlighted rows only" radio.
            parent: parent widget.

        Raises:
            ValueError: if table is not "documents" or "centroids".
        """
        super().__init__(parent)
        if table not in TABLE_FIELDS:
            raise ValueError(f"Unsupported table: {table}")
        self.table_name = table
        self.core_fields = TABLE_FIELDS[table]
        self.db: Database | None = None
        self.meta_keys: list[str] = []  # Will scan the db to find these
        self._rule_rows: list[FilterRuleRow] = []  # List of filter rows above the SQL viewer

        self.show_selection_scope = show_selection_scope

        self.build_ui()
        if database is not None:
            self.set_database(database)

    def build_ui(self):
        """Build the UI for the SQL table viewer."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Filter builder box
        self.filter_container = QVBoxLayout()
        add_btn = QPushButton("+ Add Filter")
        add_btn.clicked.connect(self._add_rule_row)

        # Add filter builder to layout
        layout.addLayout(self.filter_container)
        layout.addWidget(add_btn)

        # Status bar
        bar = QHBoxLayout()
        self.count_lbl = QLabel(f"0 {'authors' if self.table_name == 'centroids' else self.table_name}")
        bar.addWidget(self.count_lbl)
        bar.addStretch()

        # Option button to select all filtered rows or only the ones the user highlights
        self.radio_all = QRadioButton("All filtered rows (0)")
        self.radio_all.setChecked(True)  # Filtered rows are default
        self.radio_selected = QRadioButton("Highlighted rows only (0)")
        self.radio_all.setVisible(self.show_selection_scope)  # Hide if we're on the explore page
        self.radio_selected.setVisible(self.show_selection_scope)

        # Group the options into buttons
        self.scope_group = QButtonGroup(self)
        self.scope_group.addButton(self.radio_all)
        self.scope_group.addButton(self.radio_selected)
        self.scope_group.buttonClicked.connect(lambda _: self.selection_changed.emit())
        bar.addWidget(self.radio_all)
        bar.addWidget(self.radio_selected)
        layout.addLayout(bar)

        # Spliter: Table on the left, inspector on the right
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.model = SqlTableModel(self)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.selectionModel().selectionChanged.connect(self._on_row_selected)
        splitter.addWidget(self.table)

        inspector = QGroupBox("Record Inspector")
        insp_layout = QVBoxLayout(inspector)
        self.insp_text = QTextEdit()
        self.insp_text.setReadOnly(True)
        self.insp_meta = QTextEdit()
        self.insp_meta.setReadOnly(True)
        self.insp_meta.setMaximumHeight(140)
        insp_layout.addWidget(QLabel("Full Text:"))
        insp_layout.addWidget(self.insp_text, 2)
        insp_layout.addWidget(QLabel("Metadata:"))
        insp_layout.addWidget(self.insp_meta, 1)
        splitter.addWidget(inspector)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

    def set_database(self, db: Database | None) -> None:
        """Link to the database."""
        self.db = db
        self.meta_keys = self._discover_meta_keys()
        self._clear_rules()
        self.refresh()

    def set_table(self, table: str) -> None:
        """Switch the active SQLite table (documents, centroids)."""
        if table == self.table_name:
            return
        self.table_name = table
        self.core_fields = TABLE_FIELDS[table]
        self.meta_keys = self._discover_meta_keys()
        self.radio_all.setChecked(True)  # Reset to "All filtered rows" when we switch to Author-to-Author search mode
        self._clear_rules()
        self.refresh()

    def _discover_meta_keys(self) -> list[str]:
        """Fetch the unique metadata from the table.

        Returns:
            list of metadata keys.
        """
        if not self.db:
            return []

        cursor = self.db.conn.execute(
            f"SELECT DISTINCT json_each.key FROM {self.table_name}, json_each({self.table_name}.metadata) "
            f"WHERE {self.table_name}.metadata IS NOT NULL AND {self.table_name}.metadata != '{{}}' LIMIT 30"
        )
        return [str(row[0]) for row in cursor if row[0]]

    def _add_rule_row(self) -> None:
        """Add a given filter to the rules for filtering."""
        fields = self.core_fields + self.meta_keys
        row = FilterRuleRow(fields, self)
        row.removed.connect(self._remove_rule_row)
        row.changed.connect(self.refresh)
        self._rule_rows.append(row)
        self.filter_container.addWidget(row)  # Add the row to the view of rules

    def _remove_rule_row(self, row: QWidget):
        if row in self._rule_rows:
            self._rule_rows.remove(row)
            row.deleteLater()
            self.refresh()

    def _clear_rules(self):
        for r in self._rule_rows:
            r.deleteLater()
        self._rule_rows.clear()

    def _current_query(self) -> tuple[str, list[Any]]:
        rules = [r.rule() for r in self._rule_rows]
        return compile_filter(rules, self.core_fields)

    def refresh(self):
        """Query SQLite and update table in memory."""
        if not self.db:
            self.model.set_data([], self.core_fields, [])
            return

        where_clause, params = self._current_query()
        cols = ", ".join([*self.core_fields, "metadata"])
        sql = f"SELECT {cols} FROM {self.table_name} {where_clause} LIMIT 1000"

        rows = []
        for r in self.db.conn.execute(sql, params):
            d = dict(r)
            d["parsed_metadata"] = json.loads(d.get("metadata") or "{}")
            rows.append(d)

        self.model.set_data(rows, self.core_fields, self.meta_keys)
        self.count_lbl.setText(f"{len(rows):,} {'authors' if self.table_name == 'centroids' else self.table_name}")
        self.radio_all.setText(f"All filtered rows ({len(rows)})")
        self._on_row_selected()
        self.selection_changed.emit()

    def _on_row_selected(self):
        selected_indexes = self.table.selectionModel().selectedRows()
        self.radio_selected.setText(f"Highlighted rows only ({len(selected_indexes)})")
        if not selected_indexes:
            self.insp_text.clear()
            self.insp_meta.clear()
            return

        row_data = self.model.rows[selected_indexes[0].row()]
        self.insp_text.setText(str(row_data.get("text", "")))
        self.insp_meta.setText(json.dumps(row_data.get("parsed_metadata", {}), indent=2))
        self.selection_changed.emit()

    def get_effective_ids(self) -> list[int]:
        """Returns the list of integer IDs to pass to FAISS VectorIndex."""
        if not self.db:
            return []
        if self.radio_all.isChecked():
            where_clause, params = self._current_query()
            cursor = self.db.conn.execute(f"SELECT id FROM {self.table_name} {where_clause}", params)
            return [int(r[0]) for r in cursor]
        return [int(self.model.rows[idx.row()]["id"]) for idx in self.table.selectionModel().selectedRows()]
