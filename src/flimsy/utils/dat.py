"""
dat.py
------
Utilities for reading sequentially-named .dat (TSV) files produced by
LabJack acquisitions into a single Polars DataFrame.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable

import polars as pl

from flimsy.utils.ioer import find_files_matching_pattern


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_float(value: str) -> bool:
    """Return True if *value* can be parsed as a float."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def _natural_sort_key(path: Path) -> list[int | str]:
    """
    Key function that sorts path names naturally (e.g. data_2 < data_10).
    Splits the stem into alternating non-digit / digit chunks and converts
    digit chunks to ints so numeric ordering is preserved.
    """
    parts: list[int | str] = []
    for chunk in re.split(r"(\d+)", path.stem):
        parts.append(int(chunk) if chunk.isdigit() else chunk)
    return parts


def _find_data_start(filepath: Path, sep: str, encoding: str) -> tuple[int, bool]:
    """
    Scan a single file to find how many rows of metadata precede the
    tabular data, and whether a header row is present.

    Only the first file in the folder needs to be scanned — all files
    produced in the same acquisition pass share the same metadata structure.

    Returns (skip_rows, has_header).
    """
    with filepath.open(encoding=encoding) as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if stripped:
                first_field = stripped.split(sep)[0]
                if first_field == "Time":
                    return i, True
                if _is_float(first_field):
                    return i, False
    raise ValueError(f"Could not locate the start of tabular data in {filepath.name}.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def consolidate_dat_files(
    directory: str | Path,
    *,
    pattern: str = "data_*.dat",
    sep: str = "\t",
    encoding: str = "utf8",
    output_path: str | Path | None = None,
    sort_key: Callable[[Path], object] | None = None,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Consolidate all .dat files matching *pattern* inside *directory* into a
    single DataFrame.

    Parameters
    ----------
    directory:
        Folder that contains the .dat files.
    pattern:
        Glob pattern used to discover files (default ``"data_*.dat"``).
    sep:
        Field separator used in the tabular section (default ``"\\t"``).
    encoding:
        File encoding (default ``"utf8"``). Must be a Polars-compatible
        encoding name; use ``"utf8"`` rather than ``"utf-8"``.
    output_path:
        If given, the consolidated DataFrame is written to this path as a
        TSV file.  Defaults to ``None`` (no file written).
    sort_key:
        Custom sort key callable.  Defaults to natural (human) sort order so
        that ``data_2`` comes before ``data_10``.
    verbose:
        If True, print per-phase timing to stdout.

    Returns
    -------
    pl.DataFrame
        Concatenated tabular data from all matching files, in natural sorted
        order with a ``source_file`` column added for provenance.

    Raises
    ------
    FileNotFoundError
        If *directory* does not exist or no files match *pattern*.
    ValueError
        If the tabular-data start cannot be located in the first file.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    t0 = time.perf_counter()
    files = sorted(find_files_matching_pattern(directory, recursive=True, pattern=pattern)) #sorted(directory.glob(pattern), key=sort_key or _natural_sort_key)
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {directory}"
        )
    if verbose:
        print(f"[timing] discovery & sort : {time.perf_counter() - t0:.3f}s  ({len(files)} files)")

    # --- Detect metadata row count from first file -----------------------
    t1 = time.perf_counter()
    skip_rows, has_header = _find_data_start(files[0], sep=sep, encoding=encoding)
    if verbose:
        print(f"[timing] metadata scan    : {time.perf_counter() - t1:.3f}s  (skip_rows={skip_rows}, has_header={has_header})")

    # --- Read and concat all files natively in parallel ------------------
    # scan_csv accepts a list of paths and reads them in parallel via
    # Polars' lazy engine.  include_file_paths adds the source filename
    # column at no extra cost.  collect() triggers execution.
    t2 = time.perf_counter()
    result = (
        pl.scan_csv(
            files,
            separator=sep,
            skip_rows=skip_rows,
            has_header=has_header,
            include_file_paths="source_file",
            encoding=encoding,
        )
        .collect()
    )
    if verbose:
        print(f"[timing] read & concat    : {time.perf_counter() - t2:.3f}s  ({len(result):,} rows total)")

    if output_path is not None:
        t3 = time.perf_counter()
        result.write_csv(output_path, separator=sep)
        if verbose:
            print(f"[timing] write output     : {time.perf_counter() - t3:.3f}s")

    if verbose:
        print(f"[timing] total            : {time.perf_counter() - t0:.3f}s")

    return result
