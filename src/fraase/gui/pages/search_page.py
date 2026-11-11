from __future__ import annotations

import json
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from fraase.core.project import Project
from fraase.embedding.calibrator import LLRCalibrator
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

    HEADERS = ["Rank", "Score", "LLR", "ID", "Author", "Text Preview"]

    def to_table_row(self) -> list[str]:
        llr_str = f"{self.llr:.3f}" if self.llr is not None else "N/A"
        preview = self.text.replace("\n", " ")[:85]  # No need to display the full text
        return [str(self.rank), f"{self.score:.4f}", llr_str, str(self.id), self.author, preview]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "llr": self.llr,
            "id": self.id,
            "author": self.author,
            "text": self.text,
            "metadata": self.metadata,
        }

    def to_inspector_text(self) -> str:
        return (
            f"Rank #{self.rank} | Sim: {self.score:.4f}\n"
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
    metadata: dict

    HEADERS = ["Rank", "Score", "LLR", "ID", "Author", "Doc Count"]

    def to_table_row(self) -> list[str]:
        llr_str = f"{self.llr:.3f}" if self.llr is not None else "N/A"
        return [str(self.rank), f"{self.score:.4f}", llr_str, str(self.id), self.author]

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
        # TODO: this could be a concatenation of say a few lines of each text that makes up the centroid??
        pass


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
        k_label = QLabel("Top k: ")
        k_label.setStyleSheet("font-weight: bold;")
        param_row.addWidget(k_label)
        self.k_spinbox = QSpinBox()
        self.k_spinbox.setValue(3)
        self.k_spinbox.setToolTip("How many related documents/authors do you want to find?")
        param_row.addWidget(self.k_spinbox)

        self.exclude_checkbox = QCheckBox("Exclude self")
        self.exclude_checkbox.setChecked(False)
        self.exclude_checkbox.setToolTip(
            "Don't find matches to the author you're searching for. This is useful when doing Document-to-Document searches, as a document will always have 100\% similarity to itself."
        )
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

        self.steps_tab.addTab(self.query_table, "Define queries")
        self.steps_tab.addTab(self.target_table, "Define targets")
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

    @Slot(object, object)
    def _on_search_btn_clicked(self, queries: list[int], allowed_ids: list[int]) -> None:
        # TODO: display any errors with input (i.e. no query, index empty, etc)
        pass

    @Slot()
    def _on_query_row_selected(self, row_idx: int) -> None:
        """Update the results view to show the selected query and its matches."""
        pass

    def run_search(
        self, mode: str, queries: list[int], k: int, exclude_self: bool, allowed_ids: list[int] | None = None
    ) -> None:
        # TODO: Dispatch a search worker to perform the vector search. Worker will just call VectorIndex methods
        pass


class SearchResultsWidget(QWidget):
    """Widget for displaying the search results.

    Each query text is displayed in a clickable list on the left.
    For each query text, the k results are shown in the central results box
    """

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.query_list = QListWidget(self)

    def populate_results(self, results: BaseResult) -> None:
        """Fill the results widget with the results."""
        pass
