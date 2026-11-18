"""Unit tests of ``core/io/excel_loader.py``: listing and reading the sheets of a workbook.

The workbooks are built by ``fixtures/xlsx_builder.py``; ``samples.WORKBOOK`` has a typed
sheet, a hidden one, an empty one and a small one. How a sheet becomes typed columns
(``prepare_dataset``) is in ``tests/integration/test_file_to_profile.py``.
"""

import io
import re
import zipfile
from datetime import time

import pandas as pd
import pytest
from fixtures.samples import WORKBOOK
from fixtures.xlsx_builder import xlsx

from litmetrics.core.io.base import TIME_AS_TEXT, ReadError, Source
from litmetrics.core.io.excel_loader import (
    MIXED_AS_TEXT,
    ExcelLoader,
    ExcelOptions,
    describe_sheets,
    sheet_names,
)


def _source(data: bytes, name: str = "book.xlsx") -> Source:
    return Source(data, name, "id")


def _load(data: bytes, options: ExcelOptions | None = None):
    return ExcelLoader().load(_source(data), options or ExcelOptions())


# Listing sheets


def test_sheet_names_lists_visible_sheets_first_in_workbook_order():
    """Visible sheets come first in workbook order, hidden ones last.

    Case: sheets are sorted by name, or a hidden sheet is offered first and opened by default.
    Affects: core/io/excel_loader.py (sheet_names)
    How: lists the sheets of ``WORKBOOK`` (Archive is hidden and second in the file).
    """
    assert sheet_names(_source(WORKBOOK)) == ("Sales", "Blank", "Notes", "Archive")


def test_describe_sheets_gives_rows_without_header_and_columns():
    """Each sheet is described by its data rows (header excluded) and columns.

    Case: the header row is counted as data, or an empty sheet raises instead of "empty".
    Affects: core/io/excel_loader.py (describe_sheets), shown in the Load page and sidebar
    How: describes ``WORKBOOK`` and compares every description.
    """
    assert describe_sheets(_source(WORKBOOK)) == {
        "Sales": "3 rows · 5 columns",
        "Blank": "empty",
        "Notes": "2 rows · 2 columns",
        "Archive": "hidden · 1 row · 1 column",
    }


def test_describe_sheets_of_a_large_workbook_leaves_the_size_out(monkeypatch):
    """Above the size limit, sheets are not measured: only "hidden" is said.

    Case: measuring parses every sheet, freezing the page for a large workbook (R-L rules).
    Affects: core/io/excel_loader.py (describe_sheets, _SIZES_MAX_BYTES)
    How: lowers the limit just below the size of ``WORKBOOK`` and compares the descriptions.
    """
    monkeypatch.setattr("litmetrics.core.io.excel_loader._SIZES_MAX_BYTES", len(WORKBOOK) - 1)
    assert describe_sheets(_source(WORKBOOK)) == {
        "Sales": "",
        "Blank": "",
        "Notes": "",
        "Archive": "hidden",
    }


# Reading a sheet


def test_load_reads_the_first_sheet_by_default():
    """Without a sheet chosen, the first visible sheet is read and named in ``details``.

    Case: the default options read nothing, or read a hidden sheet.
    Affects: core/io/excel_loader.py (ExcelLoader.load)
    How: loads ``WORKBOOK`` with default options and checks format, details, columns, rows.
    """
    dataset = _load(WORKBOOK)
    assert dataset.format == "Excel"
    assert dataset.details == {"sheet": "Sales"}
    assert list(dataset.df.columns) == ["id", "city", "when", "paid", "amount"]
    assert dataset.n_rows == 3


def test_load_keeps_the_types_stored_in_the_sheet():
    """Ints, dates, booleans and empty cells keep the type Excel stored.

    Case: dates come back as Excel serial numbers (45293) or booleans as 0 / 1.
    Affects: core/io/excel_loader.py
    How: loads the Sales sheet and checks dtype and value of each typed column.
    """
    df = _load(WORKBOOK).df
    assert pd.api.types.is_integer_dtype(df["id"])
    assert pd.api.types.is_datetime64_any_dtype(df["when"])
    assert df["when"].iloc[0] == pd.Timestamp("2024-01-02")
    assert pd.api.types.is_bool_dtype(df["paid"])
    assert pd.isna(df["amount"].iloc[1])


@pytest.mark.parametrize(
    ("sheet", "column", "values"),
    [("Notes", "y", [2, 4]), ("Archive", "code", ["a"])],
    ids=["visible", "hidden"],
)
def test_load_reads_the_chosen_sheet(sheet, column, values):
    """The sheet named in the options is read, hidden or not.

    Case: the chosen sheet is ignored and the first one read again.
    Affects: core/io/excel_loader.py
    How: loads one visible and one hidden sheet and checks a column of each.
    """
    dataset = _load(WORKBOOK, ExcelOptions(sheet))
    assert dataset.details == {"sheet": sheet}
    assert dataset.df[column].tolist() == values


def test_load_turns_mixed_columns_into_text():
    """A column mixing numbers and text becomes text, and the conversion is reported.

    Case: pandas keeps an object column of floats and strings, which breaks numeric stats; or
    the change is not recorded in ``changed``, so an export does not warn about it.
    Affects: core/io/excel_loader.py
    How: loads a column with numbers, a word and an empty cell and checks values, details and
    ``changed``.
    """
    dataset = _load(xlsx({"S": [["price"], [1.5], ["unknown"], [None], [3]]}))
    assert dataset.details["mixed columns"] == "price (read as text)"
    assert dataset.changed == {"price": MIXED_AS_TEXT}
    assert dataset.df["price"].iloc[1] == "unknown"
    assert pd.isna(dataset.df["price"].iloc[2])


def test_load_turns_time_columns_into_text():
    """Times of day become ``HH:MM:SS`` text, and the conversion is reported.

    Case: times become datetimes dated 1899-12-30 or today, inventing a date; or the change is
    not recorded in ``changed``, so an export does not warn about it.
    Affects: core/io/excel_loader.py
    How: loads a column of times with an empty cell and checks values, details and ``changed``.
    """
    dataset = _load(xlsx({"S": [["start"], [time(9, 30)], [None], [time(17, 5, 10)]]}))
    assert dataset.details["time columns"] == "start (read as text)"
    assert dataset.changed == {"start": TIME_AS_TEXT}
    assert dataset.df["start"].iloc[0] == "09:30:00"
    assert pd.isna(dataset.df["start"].iloc[1])


def test_load_column_names_are_strings():
    """A header cell holding a number (a year) becomes a string name.

    Case: integer column names break code that treats names as text.
    Affects: core/io/excel_loader.py
    How: loads a sheet whose first header cell is 2024 and checks the names.
    """
    assert list(_load(xlsx({"S": [[2024, "name"], [1, "a"]]})).df.columns) == ["2024", "name"]


def test_load_opens_the_workbook_once(monkeypatch):
    """Reading a sheet opens the workbook once, not once per step.

    Case: listing sheets and reading one each parse the whole file again (slow for big files).
    Affects: core/io/excel_loader.py, python_calamine
    How: counts the calls to calamine's two entry points while reading one sheet.
    """
    import python_calamine

    opened = []

    def counting(real):
        def open_counted(*args, **kwargs):
            opened.append(1)
            return real(*args, **kwargs)

        return open_counted

    # pandas opens it with load_workbook; calamine's own entry point is from_filelike.
    monkeypatch.setattr(python_calamine, "load_workbook", counting(python_calamine.load_workbook))
    monkeypatch.setattr(
        python_calamine.CalamineWorkbook,
        "from_filelike",
        counting(python_calamine.CalamineWorkbook.from_filelike),
    )
    _load(WORKBOOK, ExcelOptions("Notes"))
    assert len(opened) == 1


# Errors


@pytest.mark.parametrize(
    ("data", "sheet", "message"),
    [
        (WORKBOOK, "Blank", "Sheet 'Blank' of 'book.xlsx' has no data rows."),
        (xlsx({"S": [["a", "b"]]}), None, "Sheet 'S' of 'book.xlsx' has no data rows."),
        (WORKBOOK, "Missing", "'book.xlsx' has no sheet called 'Missing'."),
    ],
    ids=["empty-sheet", "header-only", "unknown-sheet"],
)
def test_load_sheet_without_data_is_a_read_error(data, sheet, message):
    """A sheet with nothing to profile gives a message naming the sheet and the file.

    Case: an empty DataFrame reaches the Profile, or calamine's error reaches the user.
    Affects: core/io/excel_loader.py, core/io/base.py (ReadError)
    How: loads an empty sheet, a header-only sheet and a missing sheet.
    """
    with pytest.raises(ReadError, match=re.escape(message)):
        _load(data, ExcelOptions(sheet))


def test_empty_file_is_a_read_error():
    """An empty upload is reported as empty before trying to open it.

    Case: calamine raises an obscure zip error for a 0-byte file.
    Affects: core/io/excel_loader.py (sheet_names)
    How: lists the sheets of an empty file.
    """
    with pytest.raises(ReadError, match=re.escape("'book.xlsx' is empty.")):
        sheet_names(_source(b""))


def _zip_without_workbook() -> bytes:
    """A zip, as an .xlsx is, without the parts of a workbook."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("hello.txt", "not a workbook")
    return buffer.getvalue()


@pytest.mark.parametrize(
    "data",
    [b"id,name\n1,Ana\n", _zip_without_workbook(), WORKBOOK[: len(WORKBOOK) // 2]],
    ids=["csv-renamed", "zip-not-workbook", "truncated"],
)
def test_file_that_is_not_a_workbook_is_a_read_error(data):
    """A renamed, foreign or truncated file says it is not a spreadsheet or is damaged.

    Case: one of these files raises an exception that is not ``ReadError`` and crashes the page.
    Affects: core/io/excel_loader.py (describe_sheets, ExcelLoader.load)
    How: describes and loads each bad file; both must raise ``ReadError`` naming the file.
    """
    with pytest.raises(ReadError, match=re.escape("'book.xlsx'")):
        describe_sheets(_source(data))
    with pytest.raises(ReadError, match=re.escape("'book.xlsx'")):
        _load(data)
