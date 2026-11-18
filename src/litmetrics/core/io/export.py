"""Writes a dataset to a file of one of the formats the app can read, for the user to download.

Each format is one ``ExportFormat`` in ``EXPORT_FORMATS``, named like its reader in
``formats.FORMATS`` so the user sees the same name in Load and in Export. A file written here
reads back with the app's own loader with the same columns, values and types: the dataset
already has the dtypes of its schema (``schema.apply_schema``), and each format keeps them or
the loader guesses them again from the text (CSV).

Before downloading, ``export_warnings`` tells the user what the file will not hold as in the
original: values the app could not read, values changed by the loader (binary data replaced by
its size...), a file name some systems refuse, and a slow conversion. Known limits of this
release, to be fixed later; for now the user is warned.

The DataFrame is only read, never changed: it is the shared, cached dataset.
"""

import io
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
            (``export_warnings`` warns from ``ExportConfig.slow_rows``).
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


def export_warnings(
    dataset: LoadedDataset, file_name: str, name: str, slow_rows: int
) -> tuple[str, ...]:
    """What the user should know before downloading ``dataset`` as format ``name``.

    Only looks at what the loader recorded and at the size: no row is read, so it is cheap
    on every rerun.

    Args:
        dataset: The dataset to export, as loaded (with ``unparsed`` and ``changed``).
        file_name: Name the file will be downloaded with (``export_file_name``).
        name: Name of the export format.
        slow_rows: Rows from which a slow format gets a warning (``ExportConfig.slow_rows``).

    Returns:
        Messages for the user, most serious first: values written empty, values changed by
        the loader, an unsafe file name, a slow conversion. Empty when there is nothing to say.

    Raises:
        ValueError: No export format has that name.
    """
    fmt = export_format(name)
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
    if fmt.slow and dataset.n_rows >= slow_rows:
        faster = " or ".join(f.name for f in EXPORT_FORMATS if not f.slow)
        messages.append(
            f"Writing {dataset.n_rows:,} rows as {fmt.name} takes a while: {faster} are much "
            "faster."
        )
    return tuple(messages)
