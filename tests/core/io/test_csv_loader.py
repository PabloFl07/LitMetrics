import pandas as pd
import pytest

from litmetrics.core.io.base import ReadError, Source
from litmetrics.core.io.csv_loader import CsvLoader
from litmetrics.core.io.sniffing import CsvOptions, detect_options


def _load(data: bytes, options: CsvOptions | None = None):
    return CsvLoader().load(Source(data, "f.csv", "id"), options or CsvOptions())


def test_reads_types():
    dataset = _load(b"n,text,flag,when\n1,a,True,2024-01-02\n2,b,False,2024-01-03\n")
    df = dataset.df
    assert list(df.columns) == ["n", "text", "flag", "when"]
    assert pd.api.types.is_integer_dtype(df["n"])
    assert pd.api.types.is_bool_dtype(df["flag"])
    assert dataset.n_rows == 2


def test_names_columns_without_header():
    df = _load(b"1,x\n2,y\n", CsvOptions(has_header=False)).df
    assert list(df.columns) == ["column_1", "column_2"]


def test_reads_cp1252_and_semicolon():
    data = "a;b\n1;café\n".encode("cp1252")
    df = _load(data, CsvOptions(separator=";", encoding="cp1252")).df
    assert df.loc[0, "b"] == "café"


def test_reads_utf8_bom():
    df = _load("﻿a,b\n1,2\n".encode(), CsvOptions(encoding="utf-8-sig")).df
    assert list(df.columns) == ["a", "b"]


def test_empty_values_are_missing():
    df = _load(b"a,b\n1,\n,x\n").df
    assert df.isna().sum().to_dict() == {"a": 1, "b": 1}


def test_ragged_rows_fall_back_and_report_malformed():
    with pytest.raises(ReadError, match="malformed"):
        _load(b"a,b\n1,2\n3,4,5\n")


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"  \n", "'f.csv' is empty."),
        (b"a,b\n", "'f.csv' has no data rows."),
        (b"a,b\n\x00\x01", "'f.csv' is not a text file. It may be an Excel file renamed to .csv."),
    ],
)
def test_read_errors(data, message):
    with pytest.raises(ReadError) as error:
        _load(data)
    assert str(error.value) == message


def test_reads_comma_decimals_as_numbers():
    # Without the decimal mark, "0,5" stays text and the column looked like an identifier.
    data = b"precio;ciudad\n" + b"".join(b"%d,5;Lugo\n" % i for i in range(30))
    options = detect_options(data)
    dataset = CsvLoader().load(Source(data, "eu.csv", "1"), options)
    assert dataset.df["precio"].dtype == "float64"
    assert dataset.df["precio"].iloc[1] == pytest.approx(1.5)
    assert dataset.details["decimal mark"] == "Comma (,)"


def test_says_how_the_file_was_read():
    dataset = _load(b"1\tx\n2\ty\n", CsvOptions(separator="\t", has_header=False))
    assert dataset.format == "CSV"
    assert dataset.details == {
        "separator": "Tab",
        "encoding": "utf-8",
        "decimal mark": "Dot (.)",
        "column names": "none (columns numbered)",
    }
