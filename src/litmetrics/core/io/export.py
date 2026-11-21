"""Writes a dataset to a file of one of the formats the app can read, for the user to download.

Each format is one ``ExportFormat`` in ``EXPORT_FORMATS``, named like its reader in
``formats.FORMATS`` so the user sees the same name in Load and in Export. A file written here
reads back with the app's own loader with the same columns, values and types: the dataset
already has the dtypes of its schema (``schema.apply_schema``), and each format keeps them or
the loader guesses them again from the text (CSV).

Before downloading, ``export_warnings`` tells the user what the file will not hold as in the
original: values the app could not read, values changed by the loader (binary data replaced by
its size...) and a file name some systems refuse. ``slow_export`` estimates how long a slow
format (CSV) takes to write the dataset, and whether the browser may give up waiting for it.
Known limits of this release, to be fixed later; for now the user is warned. For the values
that will be empty, ``unparsed_report`` lists each one with its row and its text in the
original file, so the user can find and fix them there.

The DataFrame is only read, never changed: it is the shared, cached dataset.
"""

import io
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePath

import pandas as pd
import pyarrow as pa
import pyarrow.feather as pa_feather
import pyarrow.parquet as pq
from pandas.api import types as ptypes

from litmetrics.core.io import csv_loader, feather_loader, parquet_loader
from litmetrics.core.io.base import LoadedDataset

# Characters Windows, macOS or Linux refuse in a file name: a table of a database can have
# any name, and it becomes part of the file name.
_UNSAFE_FILE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True)
class ExportFormat:
    """One format a dataset can be written to.

    Attributes:
        name: Name shown to the user, the same as its reader ("CSV").
        extension: File extension, lowercase and with the dot (".csv").
        mime: MIME type of the file, for the browser.
        write: Returns the bytes of the file holding the DataFrame, without its index.
        slow: Whether it writes value by value as text, so a large dataset takes a while
            (``slow_export`` estimates how long).
    """

    name: str
    extension: str
    mime: str
    write: Callable[[pd.DataFrame], bytes]
    slow: bool = False


def _write_csv(df: pd.DataFrame) -> bytes:
    """CSV in UTF-8 with a header row, comma separator and decimal point."""
    return df.to_csv(index=False).encode("utf-8")


def _arrow_table(df: pd.DataFrame) -> pa.Table:
    """``df`` as an Arrow table without its index (a RangeIndex would come back as a column)."""
    return pa.Table.from_pandas(df, preserve_index=False)


def _write_parquet(df: pd.DataFrame) -> bytes:
    """Parquet with the default compression of pyarrow (Snappy)."""
    buffer = io.BytesIO()
    pq.write_table(_arrow_table(df), buffer)
    return buffer.getvalue()


def _write_feather(df: pd.DataFrame) -> bytes:
    """Feather v2 (Arrow IPC) with the default compression of pyarrow (LZ4)."""
    buffer = io.BytesIO()
    pa_feather.write_feather(_arrow_table(df), buffer)
    return buffer.getvalue()


# Formats a dataset can be exported to, in the order offered to the user.
EXPORT_FORMATS: tuple[ExportFormat, ...] = (
    ExportFormat(csv_loader.FORMAT_NAME, ".csv", "text/csv", _write_csv, slow=True),
    ExportFormat(
        parquet_loader.FORMAT_NAME, ".parquet", "application/vnd.apache.parquet", _write_parquet
    ),
    ExportFormat(
        feather_loader.FORMAT_NAME, ".feather", "application/vnd.apache.arrow.file", _write_feather
    ),
)


def export_format(name: str) -> ExportFormat:
    """Return the export format called ``name`` (as in ``EXPORT_FORMATS``).

    Raises:
        ValueError: No export format has that name (a bug, not a user error).
    """
    for fmt in EXPORT_FORMATS:
        if fmt.name == name:
            return fmt
    known = ", ".join(fmt.name for fmt in EXPORT_FORMATS)
    raise ValueError(f"No export format named {name!r}. Known: {known}.")


def export(df: pd.DataFrame, name: str) -> bytes:
    """Return the bytes of a file of format ``name`` holding ``df``, without its index.

    Raises:
        ValueError: No export format has that name.
    """
    return export_format(name).write(df)


def export_file_name(source_name: str, part: str | None, name: str) -> str:
    """File name of the export: the source's name with the extension of format ``name``.

    ``part`` is the dataset of a file with several (a table of a database), added to the name
    so two tables of one file do not download as the same file: ``shop.db`` and ``orders``
    give ``shop_orders.csv``.

    Raises:
        ValueError: No export format has that name.
    """
    stem = PurePath(source_name).stem or "dataset"
    if part:
        stem = f"{stem}_{part}"
    return stem + export_format(name).extension


# Columns of ``unparsed_report``, in order: row number (1 = first row of data, the header not
# counted), column, value as written in the file, and what the column expected there.
REPORT_COLUMNS = ("row", "column", "value", "expected")


def unparsed_report(dataset: LoadedDataset) -> pd.DataFrame:
    """Every value the app could not read, which an export writes empty, one per line.

    Built from ``dataset.unparsed_values``: only the lost values are read, not the rows.

    Returns:
        A DataFrame with ``REPORT_COLUMNS``, sorted by row and then by the order of the
        columns in the dataset. ``row`` starts at 1 with the first row of data. Without
        lines when every value was read.
    """
    order = {name: i for i, name in enumerate(dataset.df.columns)}
    parts = [
        pd.DataFrame(
            {
                "row": values.index.to_numpy(dtype="int64") + 1,
                "column": name,
                "value": values.astype("str").to_numpy(),
                "expected": _expected(dataset.df[name]),
                "_order": order[name],
            }
        )
        for name, values in dataset.unparsed_values.items()
        if len(values) and name in order
    ]
    if not parts:
        return pd.DataFrame(
            {
                "row": pd.Series(dtype="int64"),
                **{c: pd.Series(dtype="str") for c in REPORT_COLUMNS[1:]},
            }
        )
    report = pd.concat(parts, ignore_index=True).sort_values(["row", "_order"], kind="stable")
    return report.drop(columns="_order").reset_index(drop=True)


def affected_rows(report: pd.DataFrame) -> int:
    """Rows with at least one value of ``report`` (``unparsed_report``) written empty."""
    return int(report["row"].nunique())


def report_file_name(file_name: str) -> str:
    """Name of the report of the export ``file_name``: ``shop_orders.csv`` gives
    ``shop_orders_problems.csv``. Always CSV, to open it in any spreadsheet."""
    return f"{PurePath(file_name).stem}_problems.csv"


def write_report(report: pd.DataFrame) -> bytes:
    """Bytes of ``report`` (``unparsed_report``) as a CSV file in UTF-8."""
    return _write_csv(report)


def _expected(column: pd.Series) -> str:
    """What a typed column holds, as the report says it: "date" or "number"."""
    return "date" if ptypes.is_datetime64_any_dtype(column) else "number"


def export_warnings(dataset: LoadedDataset, file_name: str) -> tuple[str, ...]:
    """What the user should know before downloading ``dataset``, in any format.

    Only looks at what the loader recorded: no row is read, so it is cheap on every rerun. How
    long the file takes to write is ``slow_export``.

    Args:
        dataset: The dataset to export, as loaded (with ``unparsed`` and ``changed``).
        file_name: Name the file will be downloaded with (``export_file_name``).

    Returns:
        Messages for the user, most serious first: values written empty, values changed by
        the loader, an unsafe file name. Empty when there is nothing to say.
    """
    messages: list[str] = []
    for column, count in dataset.unparsed.items():
        if count and column in dataset.df.columns:
            kind = "dates" if ptypes.is_datetime64_any_dtype(dataset.df[column]) else "numbers"
            values = "value" if count == 1 else "values"
            messages.append(
                f"'{column}': {count:,} {values} that could not be read as {kind} will be empty."
            )
    by_note: dict[str, list[str]] = {}
    for column, note in dataset.changed.items():
        by_note.setdefault(note, []).append(f"'{column}'")
    messages.extend(
        f"{', '.join(columns)}: {note}, not as in the original file."
        for note, columns in by_note.items()
    )
    if _UNSAFE_FILE_CHARS.search(file_name):
        messages.append(
            f"'{file_name}' has characters some systems refuse in file names: "
            "your browser may change them."
        )
    return tuple(messages)


# Values written per second as CSV by ``_write_csv``, by kind of column. Measured with pandas 3
# on 1 M rows (scratch benchmark, Nov 2026): floats 0.9-1.4 M/s depending on their digits (the
# 3.1 M-row housing x 150 took 19.8 s), dates 2.4 M/s, integers 5.9 M/s, text 7.7 M/s,
# booleans 8.4 M/s. A slower or busier server takes longer: the estimate is shown as "about".
_CSV_FLOATS_PER_SECOND = 1.2e6
_CSV_DATES_PER_SECOND = 2.4e6
_CSV_OTHER_PER_SECOND = 6e6


def csv_seconds(df: pd.DataFrame) -> float:
    """Estimated seconds ``_write_csv`` takes to write ``df``, from its size and dtypes.

    Reads no value: cheap on every rerun. Floats dominate: they are written ~5 times slower
    than integers or text.
    """
    seconds = 0.0
    for dtype in df.dtypes:
        if ptypes.is_float_dtype(dtype):
            rate = _CSV_FLOATS_PER_SECOND
        elif ptypes.is_datetime64_any_dtype(dtype):
            rate = _CSV_DATES_PER_SECOND
        else:
            rate = _CSV_OTHER_PER_SECOND
        seconds += len(df) / rate
    return seconds


@dataclass(frozen=True)
class SlowExport:
    """Warning that writing the file takes long, from ``slow_export``.

    Attributes:
        message: Text for the user: how long, what the download button does meanwhile, and
            the faster formats.
        may_fail: The estimate is close to or past the time the browser waits for the file:
            the download may fail, not just be slow.
    """

    message: str
    may_fail: bool


def slow_export(
    dataset: LoadedDataset, name: str, slow_seconds: float, timeout_seconds: float
) -> SlowExport | None:
    """How long writing ``dataset`` as format ``name`` takes, when it is long enough to say.

    Args:
        dataset: The dataset to export.
        name: Name of the export format.
        slow_seconds: Estimated seconds from which the user is warned
            (``ExportConfig.slow_seconds``).
        timeout_seconds: Seconds the browser waits for the file before failing the download
            (``ExportConfig.download_timeout_seconds``). From half of it, the download may fail.

    Returns:
        ``None`` for a fast format or an estimate under ``slow_seconds``.

    Raises:
        ValueError: No export format has that name.
    """
    fmt = export_format(name)
    if not fmt.slow:
        return None
    seconds = csv_seconds(dataset.df)
    if seconds < slow_seconds:
        return None
    columns = "column" if dataset.n_cols == 1 else "columns"
    size = f"{dataset.n_rows:,} rows and {dataset.n_cols:,} {columns}"
    faster = " or ".join(f.name for f in EXPORT_FORMATS if not f.slow)
    took = f"Writing {size} as {fmt.name} takes about {duration_text(seconds)}"
    if seconds >= timeout_seconds / 2:
        chance = "will probably fail" if seconds >= timeout_seconds else "may fail"
        message = (
            f"{took}. Your browser stops waiting for the file after "
            f"{duration_text(timeout_seconds)}, so the download {chance}. "
            f"Use {faster} instead: they take a few seconds."
        )
        return SlowExport(message, may_fail=True)
    message = (
        f"{took}. After you click Download, the button spins until the file is ready: keep "
        f"this dialog open and do not click again. {faster} take a few seconds."
    )
    return SlowExport(message, may_fail=False)


def duration_text(seconds: float) -> str:
    """``seconds`` rounded for an estimate, shown after "about": "25 s", "3 min"."""
    if seconds < 60:
        # Up to the next 5 s: an estimate below the real time makes the user click again.
        return f"{max(5, math.ceil(seconds / 5) * 5)} s"
    return f"{round(seconds / 60)} min"
