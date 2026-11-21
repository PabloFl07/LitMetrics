"""Read one sheet of a spreadsheet workbook (Excel or OpenDocument) into a LoadedDataset.

A workbook holds several sheets, so reading one needs a choice the bytes cannot make: which
sheet. ``describe_sheets`` lists them for the user, ``ExcelOptions.sheet`` carries the choice
and ``ExcelLoader`` reads it, like the tables of a SQLite database (``sqlite_loader.py``).

Workbooks are read with calamine (``python-calamine``, pandas' ``"calamine"`` engine), a
reader written in Rust that opens .xlsx, .xlsm, .xlsb, .xls and .ods alike, from the bytes
of the upload. The first row of a sheet holds the column names.

Cells keep the type Excel stored (numbers, dates, booleans). Two kinds of columns are turned
into text, as a CSV column would be, and listed in ``details``:

- times of day (``"09:30:00"``): pandas keeps them as ``datetime.time`` objects, which the
  schema would take for dates and fail to convert, losing every value (Parquet and Feather
  do the same, see ``arrow_table.py``);
- columns whose cells have different types (numbers and a few ``"unknown"``), so the schema
  can find they are numeric and count the values that are not.
"""

import datetime as dt
import io
from dataclasses import dataclass, replace

import pandas as pd
import python_calamine as calamine

from litmetrics.core.io.base import MIXED_AS_TEXT, TIME_AS_TEXT, LoadedDataset, ReadError, Source

# Name of the format, shown to the user.
FORMAT_NAME = "Excel"

# What one of the datasets of a workbook is called, shown to the user.
PART_LABEL = "Sheet"

# Largest workbook (bytes) whose sheets are listed with their size. Finding the size of a sheet
# parses it whole: about 0.7 s for 1.5 M cells, which fit in roughly 8 MB of .xlsx. Above this,
# listing the sheets would take seconds at every upload, so they are listed without it.
_SIZES_MAX_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class ExcelOptions:
    """How a workbook is read.

    Attributes:
        sheet: Sheet to read. ``None`` reads the first one listed by ``sheet_names``.
    """

    sheet: str | None = None


def detect_options(data: bytes) -> ExcelOptions:
    """Return the default options: the first sheet. Nothing about a workbook needs guessing.

    Never raises: problems with the file are reported by ``describe_sheets`` and the loader.
    """
    return ExcelOptions()


def with_sheet(options: ExcelOptions, sheet: str) -> ExcelOptions:
    """Return ``options`` reading ``sheet`` instead."""
    return replace(options, sheet=sheet)


def sheet_names(source: Source) -> tuple[str, ...]:
    """Return the sheets of the workbook in ``source`` that hold cells.

    In workbook order, visible sheets first and hidden ones after them. Chart and macro
    sheets are left out.

    Raises:
        ReadError: The file is empty, is not a workbook, is damaged, is protected with a
            password, or has no sheets.
    """
    with _open(source) as workbook:
        return tuple(name for name, _ in _sheets(workbook.book, source.name))


def describe_sheets(source: Source) -> dict[str, str]:
    """Return each sheet of ``sheet_names``, in that order, with a short description.

    The description is text for the user: its rows and columns (``"1,200 rows · 8 columns"``,
    the first row being the column names), ``"empty"``, and ``"hidden"`` for hidden sheets.
    Finding the size parses every sheet once, so it is left out of workbooks larger than
    ``_SIZES_MAX_BYTES``: their sheets are described only when hidden (``""`` otherwise).

    Raises:
        ReadError: The same cases as ``sheet_names``.
    """
    sized = len(source.data) <= _SIZES_MAX_BYTES
    with _open(source) as workbook:
        return {
            name: _describe(workbook.book, name, hidden, sized)
            for name, hidden in _sheets(workbook.book, source.name)
        }


class ExcelLoader:
    """``Loader`` for workbooks: reads one sheet."""

    def load(self, source: Source, options: ExcelOptions) -> LoadedDataset:
        """Read the sheet ``options.sheet`` of the workbook in ``source``.

        Raises:
            ReadError: The file is empty, is not a workbook, is damaged or is protected with
                a password; it has no sheets; the sheet does not exist; or the sheet has no
                data rows.
        """
        name = source.name
        # Opened once: the list of sheets and the sheet come from the same workbook.
        with _open(source) as workbook:
            sheets = [sheet for sheet, _ in _sheets(workbook.book, name)]
            sheet = options.sheet if options.sheet is not None else sheets[0]
            if sheet not in sheets:
                raise ReadError(f"'{name}' has no sheet called '{sheet}'.")
            try:
                df = pd.read_excel(workbook, sheet_name=sheet)
            except (calamine.CalamineError, ValueError) as e:
                raise ReadError(f"Sheet '{sheet}' of '{name}' cannot be read: {e}") from e
        if df.empty:
            raise ReadError(f"Sheet '{sheet}' of '{name}' has no data rows.")
        df.columns = [str(column) for column in df.columns]
        df, times, mixed = _to_text(df)
        details = {PART_LABEL.lower(): sheet}
        if times:
            details["time columns"] = ", ".join(times) + " (read as text)"
        if mixed:
            details["mixed columns"] = ", ".join(mixed) + " (read as text)"
        changed = dict.fromkeys(times, TIME_AS_TEXT) | dict.fromkeys(mixed, MIXED_AS_TEXT)
        return LoadedDataset(df, name, FORMAT_NAME, details, changed=changed)


def _open(source: Source) -> pd.ExcelFile:
    """Open the bytes of ``source`` as a workbook. Use it as a context manager, to close it.

    ``.book`` is calamine's workbook (the sheets and their metadata), and ``pd.read_excel``
    reads a sheet from the same object without opening the file again.

    Raises:
        ReadError: The file is empty, is not a workbook, is damaged or has a password.
    """
    name = source.name
    if not source.data:
        raise ReadError(f"'{name}' is empty.")
    try:
        return pd.ExcelFile(io.BytesIO(source.data), engine="calamine")
    except calamine.PasswordError as e:
        raise ReadError(f"'{name}' is protected with a password.") from e
    except calamine.CalamineError as e:
        raise ReadError(f"'{name}' is not a spreadsheet or is damaged: {e}") from e


def _sheets(workbook: calamine.CalamineWorkbook, name: str) -> list[tuple[str, bool]]:
    """Sheets with cells as ``(name, hidden)``, visible first, each group in workbook order.

    Raises:
        ReadError: The workbook has no sheets with cells.
    """
    sheets = [
        (meta.name, meta.visible != calamine.SheetVisibleEnum.Visible)
        for meta in workbook.sheets_metadata
        if meta.typ == calamine.SheetTypeEnum.WorkSheet
    ]
    if not sheets:
        raise ReadError(f"'{name}' has no sheets.")
    # sorted() is stable: the workbook order stays within each group.
    return sorted(sheets, key=lambda sheet: sheet[1])


def _describe(workbook: calamine.CalamineWorkbook, sheet: str, hidden: bool, sized: bool) -> str:
    """Rows (without the header) and columns of ``sheet``, as text for the user.

    Without ``sized``, only whether it is hidden: the size would mean parsing the sheet.
    """
    if not sized:
        return "hidden" if hidden else ""
    try:
        cells = workbook.get_sheet_by_name(sheet)
    except calamine.CalamineError:
        return "cannot be read"
    rows, columns = max(cells.height - 1, 0), cells.width
    if rows == 0:
        size = "empty"
    else:
        size = f"{rows:,} row{'' if rows == 1 else 's'} · {columns} column"
        size += "" if columns == 1 else "s"
    return f"hidden · {size}" if hidden else size


def _to_text(df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Turn the columns of times of day, and those whose cells have several types, into text.

    Missing cells stay missing.

    Returns:
        The new DataFrame, the names of the time columns and of the mixed columns.
    """
    times: list[str] = []
    mixed: list[str] = []
    for column in df.columns:
        # Only object columns can hold these: a column of one plain type gets its own dtype.
        if df[column].dtype != object:
            continue
        kinds = set(map(type, df[column].dropna()))
        if kinds == {dt.time}:
            times.append(column)
        elif len(kinds) > 1:
            mixed.append(column)
    if not times and not mixed:
        return df, (), ()
    text = {c: df[c].astype("str").where(df[c].notna()) for c in (*times, *mixed)}
    return df.assign(**text), tuple(times), tuple(mixed)
