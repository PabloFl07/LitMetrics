"""Unit tests of ``core/io/arrow_table.py``: Arrow tables (from Parquet and Feather) to DataFrames.

Arrow can hold types pandas or the Profile cannot use as they come (times, decimals,
durations, binary, nested values). These tests check each one becomes a usable column and that
the conversion is reported in ``details``, which the Load page shows to the user.
"""

import datetime as dt
import decimal
import json

import pandas as pd
import pyarrow as pa
import pytest

from litmetrics.core.io.arrow_table import to_frame


def test_plain_types_keep_their_dtype():
    """Types pandas handles natively arrive with their own dtype and no details.

    Case: a conversion meant for exotic types also touches ints, bools or categories (e.g.
    every column through JSON), turning numbers into text.
    Affects: core/io/arrow_table.py (to_frame)
    How: converts one column of each plain Arrow type and checks each dtype and empty details.
    """
    table = pa.table(
        {
            "i": pa.array([1, 2], pa.int64()),
            "f": [1.5, None],
            "s": ["a", None],
            "b": [True, False],
            "cat": pa.array(["x", "y"]).dictionary_encode(),
            "ts": pa.array([dt.datetime(2020, 1, 1)] * 2, pa.timestamp("us", tz="UTC")),
        }
    )
    df, details = to_frame(table)
    assert pd.api.types.is_integer_dtype(df["i"])
    assert pd.api.types.is_float_dtype(df["f"])
    assert pd.api.types.is_string_dtype(df["s"])
    assert pd.api.types.is_bool_dtype(df["b"])
    assert isinstance(df["cat"].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_datetime64_any_dtype(df["ts"])
    assert details == {}


def test_dates_become_datetimes():
    """A date32 column becomes datetime64, so the Dates view can profile it.

    Case: dates arrive as Python ``date`` objects (dtype object) and are profiled as text.
    Affects: core/io/arrow_table.py (to_frame)
    How: converts a date32 column and checks its dtype.
    """
    df, details = to_frame(pa.table({"d": pa.array([dt.date(2020, 1, 2)], pa.date32())}))
    assert pd.api.types.is_datetime64_any_dtype(df["d"])
    assert details == {}


def test_times_become_text():
    """Times of day become ``HH:MM:SS`` text, and the conversion is reported.

    Case: times become datetimes dated today, inventing a date that is not in the file.
    Affects: core/io/arrow_table.py (to_frame)
    How: converts a time64 column with a missing value and checks text values and details.
    """
    times = pa.array([dt.time(13, 45), dt.time(1, 2, 3, 500), None], pa.time64("us"))
    df, details = to_frame(pa.table({"t": times}))
    assert list(df["t"].fillna("-")) == ["13:45:00", "01:02:03.000500", "-"]
    assert details == {"time columns": "t (read as text)"}


def test_decimals_become_floats():
    """Decimal columns become float64 with the same value, and the conversion is reported.

    Case: decimals stay ``Decimal`` objects (dtype object), so no statistics are computed.
    Affects: core/io/arrow_table.py (to_frame)
    How: converts a decimal128 column and checks dtype, value and details.
    """
    prices = pa.array([decimal.Decimal("1.25"), None], pa.decimal128(5, 2))
    df, details = to_frame(pa.table({"price": prices}))
    assert df["price"].dtype == "float64"
    assert df.loc[0, "price"] == 1.25
    assert details == {"decimal columns": "price (read as floating-point numbers)"}


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_durations_become_seconds(unit):
    """A duration in any unit becomes a number of seconds.

    Case: the unit is ignored, so 90 s written in milliseconds reads as 90,000 s.
    Affects: core/io/arrow_table.py (to_frame)
    How: writes 90 seconds in each Arrow unit and checks the value, the missing one and details.
    """
    per_second = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}[unit]
    durations = pa.array([90 * per_second, None], pa.duration(unit))
    df, details = to_frame(pa.table({"wait": durations}))
    assert df.loc[0, "wait"] == 90.0
    assert df["wait"].isna().sum() == 1
    assert details == {"duration columns": "wait (read as seconds)"}


def test_binary_becomes_its_size():
    """Binary values are shown as their size, never as raw bytes.

    Case: raw bytes reach the preview table (unreadable, and heavy to send to the browser).
    Affects: core/io/arrow_table.py (to_frame)
    How: converts a binary column with a missing value and checks the size labels and details.
    """
    df, details = to_frame(pa.table({"img": pa.array([b"\x00" * 1024, None, b"ab"])}))
    assert list(df["img"].fillna("-")) == ["<1,024 bytes>", "-", "<2 bytes>"]
    assert details == {"binary columns": "img (shown as their size)"}


def test_nested_values_become_json():
    """Lists and structs become JSON text that the Profile can count.

    Case: lists stay Python lists, which are unhashable: counting distinct values raises.
    Affects: core/io/arrow_table.py (to_frame)
    How: converts a list and a struct column, checks the JSON (with non-ASCII text kept) and
    that ``value_counts`` works on the result.
    """
    table = pa.table(
        {
            "tags": pa.array([["a", "ñ"], None]),
            "point": pa.array([{"x": 1, "y": 2}, {"x": 3, "y": None}]),
        }
    )
    df, details = to_frame(table)
    assert df.loc[0, "tags"] == '["a", "ñ"]'
    assert pd.isna(df.loc[1, "tags"])
    assert json.loads(df.loc[1, "point"]) == {"x": 3, "y": None}
    assert details == {"nested columns": "tags, point (shown as JSON text)"}
    assert df["tags"].value_counts().size == 1


def test_dictionary_of_a_converted_type_is_converted():
    """A dictionary-encoded column of a converted type is converted too.

    Case: the dictionary check runs first and keeps binary values as categories of raw bytes.
    Affects: core/io/arrow_table.py (to_frame)
    How: converts a dictionary-encoded binary column and checks the size labels.
    """
    sizes = pa.array([b"ab", b"ab"]).dictionary_encode()
    df, details = to_frame(pa.table({"blob": sizes}))
    assert list(df["blob"]) == ["<2 bytes>", "<2 bytes>"]
    assert "binary columns" in details


def test_pandas_index_becomes_columns():
    """A named index saved by pandas comes back as a column, with a plain row index.

    Case: the saved index is lost, or kept as the index and left out of the profile.
    Affects: core/io/arrow_table.py (to_frame)
    How: writes a frame with a named index to Arrow and checks columns and index type.
    """
    frame = pd.DataFrame({"x": [1, 2]}, index=pd.Index(["r1", "r2"], name="row"))
    df, _ = to_frame(pa.Table.from_pandas(frame))
    assert list(df.columns) == ["row", "x"]
    assert isinstance(df.index, pd.RangeIndex)


def test_range_index_is_not_a_column():
    """The default 0..n index pandas writes is not added as a column.

    Case: every Parquet file written by pandas gets a spurious ``__index_level_0__`` column.
    Affects: core/io/arrow_table.py (to_frame)
    How: writes a frame with a RangeIndex to Arrow and checks the only column is ``x``.
    """
    df, _ = to_frame(pa.Table.from_pandas(pd.DataFrame({"x": [1, 2]})))
    assert list(df.columns) == ["x"]


def test_column_names_are_unique_strings():
    """Repeated column names are made unique without clashing with existing ones.

    Case: renaming the second ``a`` to ``a.1`` clashes with a column already called ``a.1``.
    Affects: core/io/arrow_table.py (to_frame)
    How: converts a table with columns ``a, a, a.1`` and checks the new names.
    """
    table = pa.Table.from_arrays([pa.array([1]), pa.array([2]), pa.array([3])], ["a", "a", "a.1"])
    df, _ = to_frame(table)
    assert list(df.columns) == ["a", "a.2", "a.1"]


def test_number_column_names_become_strings():
    """Column names that are numbers become strings.

    Case: integer names break code that formats or escapes names as text (``text.md``).
    Affects: core/io/arrow_table.py (to_frame)
    How: converts a frame whose columns are 0 and 1 and checks the names are ``"0"``, ``"1"``.
    """
    df, _ = to_frame(pa.Table.from_pandas(pd.DataFrame({0: [1], 1: [2]})))
    assert list(df.columns) == ["0", "1"]
