import io
import re
import zipfile
from datetime import date, time

import pandas as pd
import pytest
from xlsx_builder import xlsx

from litmetrics.core.io.base import ReadError, Source
from litmetrics.core.io.excel_loader import (
    ExcelLoader,
    ExcelOptions,
    describe_sheets,
    detect_options,
    sheet_names,
    with_sheet,
)
from litmetrics.core.schema import SemanticType, prepare_dataset

BOOK = xlsx(
    {
        "Sales": [
            ["id", "city", "when", "paid", "amount"],
            [1, "Lugo", date(2024, 1, 2), True, 10.5],
            [2, "Vigo", date(2024, 3, 4), False, None],
            [3, "Ourense", date(2024, 5, 6), True, 7],
        ],
        "Archive": [["code"], ["a"]],
        "Blank": [],
        "Notes": [["x", "y"], [1, 2], [3, 4]],
    },
    hidden=["Archive"],
)


def _source(data: bytes, name: str = "book.xlsx") -> Source:
    return Source(data, name, "id")


def _load(data: bytes, options: ExcelOptions | None = None):
    return ExcelLoader().load(_source(data), options or ExcelOptions())


def test_sheet_names_lists_visible_sheets_first_in_workbook_order():
    assert sheet_names(_source(BOOK)) == ("Sales", "Blank", "Notes", "Archive")


def test_describe_sheets_gives_rows_without_header_and_columns():
    assert describe_sheets(_source(BOOK)) == {
        "Sales": "3 rows · 5 columns",
        "Blank": "empty",
        "Notes": "2 rows · 2 columns",
        "Archive": "hidden · 1 row · 1 column",
    }


def test_detect_options_reads_the_first_sheet():
    assert detect_options(BOOK) == ExcelOptions()


def test_with_sheet_chooses_the_sheet():
    assert with_sheet(ExcelOptions(), "Notes") == ExcelOptions(sheet="Notes")


def test_load_reads_the_first_sheet_by_default():
    dataset = _load(BOOK)
    assert dataset.format == "Excel"
    assert dataset.details == {"sheet": "Sales"}
    assert list(dataset.df.columns) == ["id", "city", "when", "paid", "amount"]
    assert dataset.n_rows == 3


def test_load_keeps_the_types_stored_in_the_sheet():
    df = _load(BOOK).df
    assert pd.api.types.is_integer_dtype(df["id"])
    assert pd.api.types.is_datetime64_any_dtype(df["when"])
    assert df["when"].iloc[0] == pd.Timestamp("2024-01-02")
    assert pd.api.types.is_bool_dtype(df["paid"])
    assert pd.isna(df["amount"].iloc[1])


def test_load_reads_the_chosen_sheet():
    dataset = _load(BOOK, ExcelOptions("Notes"))
    assert dataset.details == {"sheet": "Notes"}
    assert dataset.df["y"].tolist() == [2, 4]


def test_load_reads_a_hidden_sheet():
    assert _load(BOOK, ExcelOptions("Archive")).df["code"].tolist() == ["a"]


def test_load_turns_mixed_columns_into_text():
    data = xlsx({"S": [["price"], [1.5], ["unknown"], [None], [3]]})
    dataset = _load(data)
    assert dataset.details["mixed columns"] == "price (read as text)"
    assert dataset.df["price"].iloc[1] == "unknown"
    assert pd.isna(dataset.df["price"].iloc[2])


def test_mixed_column_is_typed_numeric_with_unparsed_values():
    # One text value in 30: under the 5% the schema allows in a numeric column.
    values = [i + 0.5 for i in range(29)] + ["unknown"]
    data = xlsx({"S": [["price"], *([v] for v in values)]})
    dataset, schema = prepare_dataset(_load(data))
    assert schema.type_of("price") is SemanticType.NUMERIC
    assert dataset.unparsed == {"price": 1}


def test_load_turns_time_columns_into_text():
    data = xlsx({"S": [["start"], [time(9, 30)], [None], [time(17, 5, 10)]]})
    dataset = _load(data)
    assert dataset.details["time columns"] == "start (read as text)"
    assert dataset.df["start"].iloc[0] == "09:30:00"
    assert pd.isna(dataset.df["start"].iloc[1])


def test_time_column_keeps_its_values_once_typed():
    # Read as dates, every time would be lost (or dated today).
    rows = [[time(8 + i % 9, 30)] for i in range(30)]
    dataset, schema = prepare_dataset(_load(xlsx({"S": [["start"], *rows]})))
    assert schema.type_of("start") is not SemanticType.DATETIME
    assert dataset.df["start"].notna().all()
    assert dataset.unparsed == {}


def test_load_column_names_are_strings():
    data = xlsx({"S": [[2024, "name"], [1, "a"]]})
    assert list(_load(data).df.columns) == ["2024", "name"]


def test_load_empty_sheet_is_a_read_error():
    with pytest.raises(
        ReadError, match=re.escape("Sheet 'Blank' of 'book.xlsx' has no data rows.")
    ):
        _load(BOOK, ExcelOptions("Blank"))


def test_load_header_only_sheet_is_a_read_error():
    with pytest.raises(ReadError, match=re.escape("has no data rows")):
        _load(xlsx({"S": [["a", "b"]]}))


def test_load_unknown_sheet_is_a_read_error():
    with pytest.raises(ReadError, match=re.escape("'book.xlsx' has no sheet called 'Missing'.")):
        _load(BOOK, ExcelOptions("Missing"))


def test_empty_file_is_a_read_error():
    with pytest.raises(ReadError, match=re.escape("'book.xlsx' is empty.")):
        sheet_names(_source(b""))


def test_another_format_renamed_is_a_read_error():
    with pytest.raises(
        ReadError, match=re.escape("'book.xlsx' is not a spreadsheet or is damaged")
    ):
        _load(b"id,name\n1,Ana\n")


def test_damaged_workbook_is_a_read_error():
    # A zip, as an .xlsx is, without the parts of a workbook.
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("hello.txt", "not a workbook")
    with pytest.raises(
        ReadError, match=re.escape("'book.xlsx' is not a spreadsheet or is damaged")
    ):
        describe_sheets(_source(buffer.getvalue()))


def test_truncated_workbook_is_a_read_error():
    with pytest.raises(ReadError, match=re.escape("'book.xlsx'")):
        _load(BOOK[: len(BOOK) // 2])


def test_describe_sheets_of_a_large_workbook_leaves_the_size_out(monkeypatch):
    # Finding the size parses every sheet: above the limit, only "hidden" is said.
    monkeypatch.setattr("litmetrics.core.io.excel_loader._SIZES_MAX_BYTES", len(BOOK) - 1)
    assert describe_sheets(_source(BOOK)) == {
        "Sales": "",
        "Blank": "",
        "Notes": "",
        "Archive": "hidden",
    }


def test_load_opens_the_workbook_once(monkeypatch):
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
    _load(BOOK, ExcelOptions("Notes"))
    assert len(opened) == 1
