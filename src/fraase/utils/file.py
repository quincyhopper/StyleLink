import csv
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any, TextIO

import openpyxl


def _read_xlsx(path: Path) -> tuple[tuple[list[str]], Iterator[dict[str, Any]]]:
    """Read an .xlsx file.

    Returns:
        cols (list): list of column names.
        iter_rows: Iterator of dictionaries (key is colname, value is row value).
    """
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = wb.active.iter_rows(values_only=True)
    cols = [str(c) if c is not None else "" for c in next(rows, ())]

    def _iter() -> Iterator[dict[str, Any]]:
        try:
            for row in rows:
                yield dict(zip(cols, ("" if v is None else v for v in row)))
        finally:
            wb.close()

    return cols, _iter()


def get_csv_reader(file: TextIO, path: Path):
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
    if path.suffix.lower() == ".xlsx":
        cols, rows = _read_xlsx(path)
        return list(islice(rows, size)), cols

    with open(path, mode="r") as file:
        reader = get_csv_reader(file, path)

        # Extract `size` rows
        rows: list[dict[str, Any]] = [row for _, row in zip(range(size), reader)]
        cols: list[str] = reader.fieldnames

        return rows, cols


def read_file(path: Path) -> Iterator[dict[str, Any]]:
    """Stream all rows from a tabular dataset.

    Yields:
        file reader.
    """
    if path.suffix.lower() == ".xlsx":
        _, rows = _read_xlsx(path)
        yield from rows
        return

    with open(path, mode="r", encoding="utf-8") as file:
        yield from get_csv_reader(file, path)
