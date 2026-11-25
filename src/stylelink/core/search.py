from __future__ import annotations

import json
import sqlite3
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from stylelink.core.project import Project
from stylelink.embedding.calibrator import LLRCalibrator, llr_to_verbal

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
            "id": self.id,
            "author": self.author,
            "score": self.score,
            "llr": self.llr,
            "verbal": llr_to_verbal(self.llr),
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

    HEADERS = ["Rank", "ID", "Author", "Doc count", "Score", "LLR", "Verbal"]

    def to_table_row(self) -> list[str]:
        llr_str = f"{self.llr:.3f}" if self.llr is not None else "N/A"
        verbal = llr_to_verbal(self.llr)
        return [str(self.rank), str(self.id), self.author, str(self.doc_count), f"{self.score:.4f}", llr_str, verbal]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "id": self.id,
            "author": self.author,
            "score": self.score,
            "llr": self.llr,
            "verbal": llr_to_verbal(self.llr),
            "doc_count": self.doc_count,
        }

    def to_inspector_text(self, sample_texts: list[str] = ()) -> str:
        snippets = "...\n\n---\n\n".join("\n".join(t.splitlines()[:3])[:300] for t in sample_texts)
        return f"Author: {self.author} ({self.doc_count} documents)\n\n--- Sample Texts ---\n{snippets}"


def run_search(
    project: Project,
    mode: str,
    query_ids: list[int],
    target_ids: list[int],
    k: int,
    exclude_self: bool,
    calibrator_name: str = "General",
) -> tuple[list[list[BaseResult]], list[str]]:
    """Peform a vector search given the currently selected parameters.

    Args:
        project: active Project instance.
        mode: Document-to-Document, Document-to-Author, or Author-to-Author.
        query_ids: IDs of all query documents/authors.
        target_ids: IDs of the docs/authors to search against.
        k: number of matches to find.
        exclude_self: if True, query author is excluded from results.
        calibrator_name: General, Forum posts or Chat logs.

    Returns:
        results: list of results, where each element is the k results for each query.
        query_labels: formatted strings to display in the UI.

    Raises:
        RuntimeError: if project is not synced.
        ValueError: if queyry/target ids are empty
    """
    if not project.is_synced:
        raise RuntimeError("Project embeddings and centroids are not synced with the database.")
    if not query_ids:
        raise ValueError("query_ids cannot be empty.")
    if not target_ids:
        raise ValueError("target_ids cannot be empty.")

    # Determine which FAISS indexes we need to use
    query_table, target_table = MODES_MAP[mode]

    # Doc2Doc search
    if query_table == "documents" and target_table == "documents":
        distances, result_ids = project.doc_index.search_by_id(
            query_doc_id=query_ids, k=k, exclude_self=exclude_self, allowed_ids=target_ids
        )
    # Doc2Author search
    elif query_table == "documents" and target_table == "centroids":
        # Extract queries from the document index
        query_vecs = project.doc_index.reconstruct(query_ids)

        # because .search() does not handle exclude_self, we can just do it manually here
        distances, result_ids = project.centroid_index.search(
            queries=query_vecs, k=k + int(exclude_self), allowed_ids=target_ids
        )
    # Author2Author search
    else:
        distances, result_ids = project.centroid_index.search_by_id(
            query_doc_id=query_ids, k=k, exclude_self=exclude_self, allowed_ids=target_ids
        )

    final_results, query_labels = to_results(
        project=project,
        distances=distances,
        result_ids=result_ids,
        query_ids=query_ids,
        k=k,
        exclude_self=exclude_self,
        query_table=query_table,
        target_table=target_table,
        calibrator_name=calibrator_name,
    )

    return final_results, query_labels


def to_results(
    project: Project,
    distances: np.ndarray,
    result_ids: np.ndarray,
    query_ids: list[int],
    k: int,
    exclude_self: bool,
    query_table: str,
    target_table: str,
    calibrator_name: str = "General",
) -> tuple[list[list[BaseResult]], list[str]]:
    """Convert the result of a vector search into BaseResult objects.

    Args:
        project: active Project instance.
        distances: distances from FAISS.
        result_ids: target IDs corresponding to the distances.
        query_ids: IDs of all query documents/authors.
        k: number of matches to find.
        exclude_self: if True, query author is excluded from results.
        query_table: name of the query table.
        target_table: name of the target table.
        calibrator_name: name of the LLR calibration model.

    Returns:
        results: list of lists of BaseResult objects.
        query_labels: formatted strings to display the query author.
    """
    conn = project.database.conn

    def _fetch_rows_by_id(conn: sqlite3.Connection, table: str, cols: str, ids: list[int]) -> dict[int, Any]:
        rows = {}
        max_params = 32766  # Max parameters sqlite3 >=3.32.0
        for i in range(0, len(ids), max_params):
            chunk = ids[i : i + max_params]
            placeholders = ",".join("?" * len(chunk))
            for r in conn.execute(f"SELECT {cols} FROM {table} WHERE id IN ({placeholders})", chunk):
                # r is sqlite3.Row object allowing us to access column names via indexing
                rows[r["id"]] = r

        return rows

    # Get the IDs and columns for the query authors
    query_author_map = _fetch_rows_by_id(conn, query_table, "id, author", query_ids)  # id: sqlite3.Row (id, author)
    query_authors = [query_author_map[q_id]["author"] for q_id in query_ids]

    # Get all unique result IDs - we will need to query the database to fill the results table for these documents/authors
    needed_ids = [int(i) for i in np.unique(result_ids) if i != -1]

    # Build a map of all the columns for each of the needed result ids
    row_cache = _fetch_rows_by_id(conn, target_table, "*", needed_ids)

    # Compute LLRs
    calibrator = LLRCalibrator(name=calibrator_name)
    llrs = calibrator.compute_llr(distances)

    # Build the results objects
    final_results: list[list[BaseResult]] = []
    for q_idx in range(len(result_ids)):
        matches: list[BaseResult] = []

        for res_id, score, llr in zip(result_ids[q_idx], distances[q_idx], llrs[q_idx]):
            if res_id == -1:
                continue

            res_id = int(res_id)
            row = row_cache[res_id]
            meta = json.loads(row["metadata"]) or "{}"

            if exclude_self and query_table == "documents" and target_table == "centroids":
                if row["author"] == query_authors[q_idx]:
                    continue

            rank = len(matches) + 1

            if target_table == "documents":
                matches.append(
                    DocumentSearchResult(
                        rank=rank,
                        score=float(score),
                        llr=float(llr),
                        author=row["author"],
                        id=res_id,
                        text=row["text"],
                        metadata=meta,
                    )
                )
            else:
                matches.append(
                    CentroidSearchResult(
                        rank=rank,
                        score=float(score),
                        llr=float(llr),
                        author=row["author"],
                        id=res_id,
                        doc_count=row["doc_count"],
                    )
                )

            if len(matches) == k:
                break

        final_results.append(matches)

    query_labels = [f"ID #{id} ({author})" for id, author in zip(query_ids, query_authors)]

    return final_results, query_labels
