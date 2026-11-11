from __future__ import annotations

import json
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fraase.embedding.calibrator import LLRCalibrator
from fraase.gui.state import AppState
from fraase.gui.widgets.filter_table import SqlFilterTable


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

        self.build_ui()

    def build_ui(self) -> None:
        """Build the layout of the search page."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        # Persistent search mode bar
        self.mode_bar = QFrame()
        self.mode_bar.setFrameShape(QFrame.Shape.StyledPanel)
        mode_bar_vlayout = QVBoxLayout(self.mode_bar)
        mode_bar_vlayout.setContentsMargins(12, 10, 12, 10)
        mode_bar_vlayout.setSpacing(8)

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
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
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

        layout.addLayout(param_row)

    @Slot()
    def _on_mode_changed(self) -> None:
        """Update tables to select centroids or documents."""
        pass

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
