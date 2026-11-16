from __future__ import annotations

import csv
import json
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from fraase.embedding.calibrator import LLRCalibrator, llr_to_verbal
from fraase.gui.state import AppState
from fraase.gui.widgets.filter_table import SqlFilterTable

# Maps the search mode to the query and target tables
MODES_MAP = {
    "Document-to-Document": ("documents", "documents"),
    "Document-to-Author": ("documents", "centroids"),
    "Author-to-Author": ("centroids", "centroids"),
}


@dataclass(frozen=True)
class BaseResult:
    """Base result class for document and centroid searches."""

    rank: int
    score: float
    llr: float | None
    author: str

    HEADERS: ClassVar[list[str]]

    @abstractmethod
    def to_table_row(self) -> list[str]:
        """Translate a result to a list of strings."""
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Translate a result to a dictionary."""
        pass

    def to_inspector_text(self) -> str:
        """Translate result into string format to be printed in inspector box."""
        pass


@dataclass(frozen=True)
class DocumentSearchResult(BaseResult):
    """Dataclass representing a single document search result."""

    rank: int
    score: float
    llr: float | None  # If calibrated
    id: int
    author: str
    text: str
    metadata: dict

    HEADERS = ["Rank", "ID", "Author", "Score", "LLR", "Verbal"]

    def to_table_row(self) -> list[str]:
        llr_str = f"{self.llr:.3f}" if self.llr is not None else "N/A"
        verbal = llr_to_verbal(self.llr)
        # preview = self.text.replace("\n", " ")[:85]  # No need to display the full text
        return [str(self.rank), str(self.id), self.author, f"{self.score:.4f}", llr_str, verbal]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "llr": self.llr,
            "id": self.id,
            "author": self.author,
            "metadata": self.metadata,
        }

    def to_inspector_text(self) -> str:
        return (
            f"Author: {self.author}\n\n"
            f"--- Full Text ---\n{self.text}\n\n"
            f"--- Metadata ---\n{json.dumps(self.metadata, indent=2)}"
        )


@dataclass(frozen=True)
class CentroidSearchResult(BaseResult):
    """Dataclass representing a single centroid search result."""

    rank: int
    score: float
    llr: float | None
    id: int
    author: str
    doc_count: int
    sample_texts: list[str]  # A few lines of each of this author's texts

    HEADERS = ["Rank", "ID", "Author", "Doc count", "Score", "LLR", "Verbal"]

    def to_table_row(self) -> list[str]:
        llr_str = f"{self.llr:.3f}" if self.llr is not None else "N/A"
        verbal = llr_to_verbal(self.llr)
        return [str(self.rank), str(self.id), self.author, str(self.doc_count), f"{self.score:.4f}", llr_str, verbal]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "llr": self.llr,
            "id": self.id,
            "author": self.author,
            "doc_count": self.doc_count,
        }

    def to_inspector_text(self) -> str:
        snippets = "\n\n---\n\n".join("\n".join(t.splitlines()[:3])[:300] for t in self.sample_texts)
        return f"Author: {self.author} ({self.doc_count} documents)\n\n--- Sample Texts ---\n{snippets}"


class SearchPage(QWidget):
    """Page for performing vector searches.

    Workflow:
        - Step 1: Define queries.
        - Step 2: Optionally restrict the search space.
        - Step 3: Observe result and optionally export

    Search modes:
        - Document to Document
        - Document to Centroid
        - Centroid to Centroid
    """

    def __init__(self, state: AppState, parent: QWidget) -> None:
        super().__init__(parent)
        self.state = state
        self.calibrator = LLRCalibrator()
        self._build_ui()

        # Update the filter tables on initialisation or if project is changed
        self.state.project_changed.connect(self._on_project_changed)
        self._on_project_changed(self.state.active_project)

    def _build_ui(self) -> None:
        """Build the layout of the search page."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        # This row will hold everything to set parameters for the search (mode, k, exclude self, etc)
        param_row = QHBoxLayout()
        param_row.setSpacing(10)

        # Title of the mode selector
        mode_label = QLabel("Search mode: ")
        mode_label.setStyleSheet("font-weight: bold;")
        param_row.addWidget(mode_label)

        # This is where the user actually selects the search mdoe
        self.mode_combo = QComboBox()
        self.mode_combo.setMinimumWidth(260)
        for mode in ("Document-to-Document", "Document-to-Author", "Author-to-Author"):
            self.mode_combo.addItem(mode)

        # Make any changes related to changing the search mode
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        param_row.addWidget(self.mode_combo)

        # K parameter selector
        param_row.addSpacing(8)
        k_label = QLabel("Number of matches to find: ")
        k_label.setStyleSheet("font-weight: bold;")
        param_row.addWidget(k_label)
        self.k_spinbox = QSpinBox()
        self.k_spinbox.setValue(3)
        self.k_spinbox.setToolTip("How many related documents/authors do you want to find?")
        param_row.addWidget(self.k_spinbox)

        self.exclude_checkbox = QCheckBox("Exclude query author")
        self.exclude_checkbox.setChecked(False)
        self.exclude_checkbox.setToolTip("If this is checked, the query author won't appear in the results.")
        param_row.addWidget(self.exclude_checkbox)

        param_row.addStretch(1)

        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self._on_search_btn_clicked)
        param_row.addWidget(search_btn)

        layout.addLayout(param_row)

        # Now we need the query definition and optional search restriction. This will be two tabs that the user follows in linear order:
        # they define a query and then move on to search restriction. This stage will be optional, so they can just click
        # "Search" from this point and they will be directed to the results page.

        # This will hold the two tabs
        self.steps_tab = QTabWidget()

        self.query_table = SqlFilterTable(table="documents", parent=self)
        self.target_table = SqlFilterTable(table="documents", parent=self)

        self.steps_tab.addTab(self.query_table, "Search Items")
        self.steps_tab.addTab(self.target_table, "Comparison Items")
        self.steps_tab.addTab(self._build_results_tab(), "Results")

        layout.addWidget(self.steps_tab)

    def _build_results_tab(self) -> None:
        """Build results tab with an empty placeholder and the results widget."""
        # If no search has been run, we just display some text advising the user to do a search. Otherwise, we display
        # the results.
        self.results_stack = QStackedWidget()

        self.empty_results_label = QLabel("Please run a search to view results.")
        self.empty_results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.results_widget = SearchResultsWidget(self.results_stack)

        self.results_stack.addWidget(self.empty_results_label)
        self.results_stack.addWidget(self.results_widget)

        return self.results_stack

    @Slot(object)
    def _on_project_changed(self, project: object) -> None:
        """Attach the active project's database to the filter tables."""
        db = project.database if project is not None else None
        self.query_table.set_database(db)
        self.target_table.set_database(db)
        self.results_stack.setCurrentWidget(self.empty_results_label)

    @Slot(str)
    def _on_mode_changed(self, mode: str) -> None:
        """Update tables to select centroids or documents."""
        query_table, target_table = MODES_MAP[mode]
        self.query_table.set_table(query_table)
        self.target_table.set_table(target_table)

    @Slot()
    def _on_search_btn_clicked(self) -> None:
        try:
            self.run_search()
        except (ValueError, KeyError, RuntimeError) as err:
            QMessageBox.warning(self, "Cannot run search", str(err))
            return
        self.steps_tab.setCurrentWidget(self.results_stack)
        self.results_stack.setCurrentWidget(self.results_widget)

    def run_search(self) -> None:
        """Peform a vector search given the currently selected parameters."""
        # Retrieve all the parameters of the search
        query_ids = self.query_table.get_effective_ids()
        target_ids = self.target_table.get_effective_ids()
        k: int = self.k_spinbox.value()
        exclude_self: bool = self.exclude_checkbox.isChecked()

        # A few checks before we begin the search
        if not self.state.active_project.is_synced:
            raise RuntimeError(
                "Not all documents/authors have Styles. Please run Create Styles on the Home Page first."
            )
        if not query_ids:
            raise ValueError("No search items selected. Please select at least one in the Search Items tab.")
        if not target_ids:
            raise ValueError("No comparison items selected. Please select at least one in the Comparison Items tab.")

        # Determine which FAISS indexes we need to use
        query_table, target_table = MODES_MAP[self.mode_combo.currentText()]

        # Doc2Doc search
        if query_table == "documents" and target_table == "documents":
            distances, result_ids = self.state.active_project.doc_index.search_by_id(
                query_doc_id=query_ids, k=k, exclude_self=exclude_self, allowed_ids=target_ids
            )
        # Doc2Author search
        elif query_table == "documents" and target_table == "centroids":
            # Extract queries from the document index
            query_vecs = self.state.active_project.doc_index.reconstruct(query_ids)

            # because .search() does not handle exclude_self, we can just do it manually here
            distances, result_ids = self.state.active_project.centroid_index.search(
                queries=query_vecs, k=k + int(exclude_self), allowed_ids=target_ids
            )
        # Author2Author search
        else:
            distances, result_ids = self.state.active_project.centroid_index.search_by_id(
                query_doc_id=query_ids, k=k, exclude_self=exclude_self, allowed_ids=target_ids
            )

        # Compute llrs
        llrs = self.calibrator.compute_llr(distances)

        conn = self.state.active_project.database.conn
        final_results: list[list[BaseResult]] = []

        # Get list of query author names
        query_authors = [
            conn.execute(f"SELECT author FROM {query_table} WHERE id = ?", (q_id,)).fetchone()["author"]
            for q_id in query_ids
        ]

        # Convert results into BaseResult objects to pass to self.results_widget
        for q_idx in range(len(result_ids)):
            matches: list[BaseResult] = []
            for res_id, score, llr in zip(result_ids[q_idx], distances[q_idx], llrs[q_idx]):
                if res_id == -1:
                    # FAISS returns -1 if it couldn't find K results, so just ignore it
                    continue

                # Get the database info for this match
                row = conn.execute(f"SELECT * FROM {target_table} WHERE id = ?", (int(res_id),)).fetchone()
                meta = json.loads(row["metadata"]) or "{}"

                if exclude_self and query_table == "documents" and target_table == "centroids":
                    if row["author"] == query_authors[q_idx]:
                        continue

                # Current rank updates dynamically based on the number of matches we have accumulated
                rank = len(matches) + 1

                if target_table == "documents":
                    matches.append(
                        DocumentSearchResult(
                            rank=rank,
                            score=float(score),
                            llr=float(llr),
                            author=row["author"],
                            id=int(res_id),
                            text=row["text"],
                            metadata=meta,
                        )
                    )
                else:
                    sample_texts = [
                        r["text"]
                        for r in conn.execute("SELECT text FROM documents WHERE author = ? LIMIT 5", (row["author"],))
                    ]
                    matches.append(
                        CentroidSearchResult(
                            rank=rank,
                            score=float(score),
                            llr=float(llr),
                            author=row["author"],
                            id=res_id,
                            doc_count=row["doc_count"],
                            sample_texts=sample_texts,
                        )
                    )

                if len(matches) == k:
                    # No need to build more results if we already have K results
                    break

            final_results.append(matches)

        # Get visual representation of query authors
        query_labels = [f"ID #{id} ({author})" for id, author in zip(query_ids, query_authors)]

        self.results_widget.populate_results(query_labels, final_results)


class SearchResultsWidget(QWidget):
    """Widget for displaying the search results.

    Each query text is displayed in a clickable list on the left.
    For each query text, the k results are shown in the central results box
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._results: list[list[BaseResult]] = []  # List for every query, k sublists
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        def _pane(title: str, widget: QWidget) -> QGroupBox:
            """Helper function for grouping a pane with a title on top."""
            box = QGroupBox(title)
            QVBoxLayout(box).addWidget(widget)
            return box

        # Left query list
        self.query_list = QListWidget()
        self.query_list.currentRowChanged.connect(self._on_query_selected)
        splitter.addWidget(_pane("Searched items", self.query_list))

        # Middle top K matches table
        self.matches_table = QTableWidget()
        self.matches_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)  # Allow user to select rows
        self.matches_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)  # Immutable table
        self.matches_table.horizontalHeader().setStretchLastSection(True)
        self.matches_table.currentCellChanged.connect(self._on_match_selected)
        splitter.addWidget(_pane("Closest matches", self.matches_table))

        # Right match inspector
        self.inspector = QTextEdit()
        self.inspector.setReadOnly(True)
        splitter.addWidget(_pane("Match details", self.inspector))

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 2)
        layout.addWidget(splitter)

        # Button for exporting to csv
        top_bar = QHBoxLayout()
        top_bar.addStretch()
        export_btn = QPushButton("Export to CSV...")
        export_btn.clicked.connect(self._on_export_btn_clicked)
        top_bar.addWidget(export_btn)
        layout.addLayout(top_bar)

    def populate_results(self, query_labels: list[str], results: list[list[BaseResult]]) -> None:
        """Fill the results widget with the results."""
        self._results = results
        self.query_list.clear()
        self.query_list.addItems(query_labels)
        if query_labels:
            self.query_list.setCurrentRow(0)

    @Slot(int)
    def _on_query_selected(self, query_idx: int) -> None:

        # Clear middle and right
        self.matches_table.clear()
        self.inspector.clear()
        if query_idx < 0 or query_idx >= len(self._results):
            return

        matches = self._results[query_idx]  # Access sublist full of k results
        if not matches:
            return

        headers = matches[0].HEADERS
        self.matches_table.setColumnCount(len(headers))
        self.matches_table.setHorizontalHeaderLabels(headers)
        self.matches_table.setRowCount(len(matches))

        # Populate table
        for row_idx, match in enumerate(matches):
            for col_idx, val in enumerate(match.to_table_row()):
                self.matches_table.setItem(row_idx, col_idx, QTableWidgetItem(val))

        self.matches_table.selectRow(0)

    @Slot(int)
    def _on_match_selected(self, row_idx: int, *_) -> None:
        query_idx = self.query_list.currentRow()
        if 0 <= query_idx < len(self._results) and 0 <= row_idx < len(self._results[query_idx]):
            self.inspector.setText(self._results[query_idx][row_idx].to_inspector_text())

    @Slot()
    def _on_export_btn_clicked(self) -> None:
        if not any(self._results):
            return

        file_str, _ = QFileDialog.getSaveFileName(self, "Export Search Results", "results.csv", "CSV Files (*csv)")

        if not file_str:
            return

        rows = []
        for q_idx, matches in enumerate(self._results):
            query_label = self.query_list.item(q_idx).text()
            for match in matches:
                d = match.to_dict()
                meta = d.pop("metadata", {})
                rows.append({"query": query_label, **d, **meta})

        fieldnames = list(dict.fromkeys(k for r in rows for k in r))

        with open(file_str, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
