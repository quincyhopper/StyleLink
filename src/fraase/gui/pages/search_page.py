from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QWidget

from fraase.embedding.calibrator import LLRCalibrator
from fraase.gui.state import AppState
from fraase.gui.widgets.filter_table import SqlFilterTable


@dataclass
class DocumentSearchResult:
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
            f"Rank #{self.rank} | Sim: {self.similarity:.4f}\n"
            f"Author: {self.user_id}\n\n"
            f"--- Full Text ---\n{self.text}\n\n"
            f"--- Metadata ---\n{json.dumps(self.metadata, indent=2)}"
        )


@dataclass
class CentroidSearchResult:
    rank: int
    score: float
    llr: float | None
    id: int
    author: str
    text: str
    metadata: dict

    HEADERS = ["Rank", "Score", "LLR", "ID", "Author"]

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
        pass

    @Slot(list[int], list[int])
    def on_search_btn_clicked(self, queries: list[int], allowed_ids: list[int]) -> None:
        # TODO: display any errors with input (i.e. no query, index empty, etc)
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

    def __init__(self):
        pass
