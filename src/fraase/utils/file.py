"""Useful classes and functions for importing, parsing and previewing files."""

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO


def get_reader_for_file(file: TextIO, path: Path):
    """Return an appropriate file reader (csv, xlsx, etc).

    Args:
        file (TextIO): the return of open() on the desired file.
        path (Path): the path to the file.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return csv.DictReader(file)
    if suffix == ".tsv":
        return csv.DictReader(file, delimiter="\t")


def preview_file(path: Path, size: int = 5) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract the first 5 rows of a tabular dataset.

    Useful for displaying a preview before a user decides to import a file.

    Args:
        path: filepath to file to be previewed.
        size: number of rows to extract

    Returns:
        rows: rows
        cols: columns
    """
    with open(path, mode="r") as file:
        reader = get_reader_for_file(file, path)

        # Extract `size` rows
        rows: list[dict[str, Any]] = [row for _, row in zip(range(size), reader)]
        cols: list[str] = reader.fieldnames

        return rows, cols


def read_file(path: Path) -> Iterator[dict[str, Any]]:
    """Stream all rows from a tabular dataset.

    Yields:
        file reader.
    """
    with open(path, mode="r", encoding="utf-8") as file:
        yield from get_reader_for_file(file, path)
