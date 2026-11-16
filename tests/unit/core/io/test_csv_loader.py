"""Unit tests of ``core/io/csv_loader.py``: reading a CSV with options already chosen.

Options are given explicitly here; how they are detected is in ``test_sniffing.py``, and the
two together in ``tests/integration/test_file_to_profile.py``.
"""

import pandas as pd
import pytest

from litmetrics.core.io.base import ReadError, Source
from litmetrics.core.io.csv_loader import CsvLoader
from litmetrics.core.io.sniffing import CsvOptions


def _load(data: bytes, options: CsvOptions | None = None):
    return CsvLoader().load(Source(data, "f.csv", "id"), options or CsvOptions())


def test_reads_types():
    """Numbers, booleans and text are read with their own dtype, and rows are counted.

    Case: every column is read as text (``dtype=str``), so nothing numeric can be profiled.
    Affects: core/io/csv_loader.py (CsvLoader.load), core/io/base.py (LoadedDataset)
    How: reads a 2-row CSV and checks column order, dtypes and ``n_rows``.
    """
    dataset = _load(b"n,text,flag,when\n1,a,True,2024-01-02\n2,b,False,2024-01-03\n")
    df = dataset.df
    assert list(df.columns) == ["n", "text", "flag", "when"]
    assert pd.api.types.is_integer_dtype(df["n"])
    assert pd.api.types.is_bool_dtype(df["flag"])
    assert dataset.n_rows == 2


def test_names_columns_without_header():
    """A file without a header gets ``column_1``, ``column_2``... as names.

    Case: the first data row is used as the header and lost, or names are integers.
    Affects: core/io/csv_loader.py
    How: reads 2 rows with ``has_header=False`` and checks the generated names.
    """
    df = _load(b"1,x\n2,y\n", CsvOptions(has_header=False)).df
    assert list(df.columns) == ["column_1", "column_2"]


def test_reads_cp1252_and_semicolon():
    """A Windows-encoded file with ``;`` keeps its accented letters.

    Case: the encoding option is ignored and ``café`` becomes ``cafÃ©`` or fails to decode.
    Affects: core/io/csv_loader.py
    How: reads cp1252 bytes with matching options and checks the text value.
    """
    data = "a;b\n1;café\n".encode("cp1252")
    df = _load(data, CsvOptions(separator=";", encoding="cp1252")).df
    assert df.loc[0, "b"] == "café"


def test_reads_utf8_bom():
    """The UTF-8 byte order mark is not part of the first column name.

    Case: the first column is called ``\\ufeffa`` and the user cannot find column ``a``.
    Affects: core/io/csv_loader.py
    How: reads a file starting with a BOM and checks the column names.
    """
    df = _load("﻿a,b\n1,2\n".encode(), CsvOptions(encoding="utf-8-sig")).df
    assert list(df.columns) == ["a", "b"]


def test_empty_values_are_missing():
    """Empty fields are missing values, in numeric and text columns alike.

    Case: empty text fields are read as ``""`` and not counted as missing by the Profile.
    Affects: core/io/csv_loader.py
    How: reads one empty field per column and checks the missing count per column.
    """
    df = _load(b"a,b\n1,\n,x\n").df
    assert df.isna().sum().to_dict() == {"a": 1, "b": 1}


def test_ragged_rows_are_a_read_error():
    """A row with more fields than the header is a ``ReadError`` that says "malformed".

    Case: extra fields are silently dropped or shift values into the wrong columns.
    Affects: core/io/csv_loader.py, core/io/base.py (ReadError)
    How: reads a file whose second row has 3 fields under a 2-column header.
    """
    with pytest.raises(ReadError, match="malformed"):
        _load(b"a,b\n1,2\n3,4,5\n")


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"  \n", "'f.csv' is empty."),
        (b"a,b\n", "'f.csv' has no data rows."),
        (b"a,b\n\x00\x01", "'f.csv' is not a text file. It may be an Excel file renamed to .csv."),
    ],
    ids=["blank", "header-only", "binary"],
)
def test_read_errors(data, message):
    """Files that hold no table give a message the user can act on, shown as is.

    Case: pandas' own error (``EmptyDataError``, ``UnicodeDecodeError``) reaches the user.
    Affects: core/io/csv_loader.py, core/io/base.py (ReadError)
    How: reads each bad file and compares the full message.
    """
    with pytest.raises(ReadError) as error:
        _load(data)
    assert str(error.value) == message


def test_says_how_the_file_was_read():
    """``details`` names separator, encoding, decimal mark and header in words.

    Case: the Load page shows ``\\t`` or ``None`` instead of something a user can read.
    Affects: core/io/csv_loader.py (details), shown by ui/pages/load.py
    How: reads a tab-separated file without header and compares format and details.
    """
    dataset = _load(b"1\tx\n2\ty\n", CsvOptions(separator="\t", has_header=False))
    assert dataset.format == "CSV"
    assert dataset.details == {
        "separator": "Tab",
        "encoding": "utf-8",
        "decimal mark": "Dot (.)",
        "column names": "none (columns numbered)",
    }
