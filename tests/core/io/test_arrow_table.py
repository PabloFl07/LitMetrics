import datetime as dt
import decimal
import json

import pandas as pd
import pyarrow as pa
import pytest

from litmetrics.core.io.arrow_table import to_frame


def test_plain_types_keep_their_dtype():
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
    df, details = to_frame(pa.table({"d": pa.array([dt.date(2020, 1, 2)], pa.date32())}))
    assert pd.api.types.is_datetime64_any_dtype(df["d"])
    assert details == {}


def test_times_become_text():
    times = pa.array([dt.time(13, 45), dt.time(1, 2, 3, 500), None], pa.time64("us"))
    df, details = to_frame(pa.table({"t": times}))
    assert list(df["t"].fillna("-")) == ["13:45:00", "01:02:03.000500", "-"]
    assert details == {"time columns": "t (read as text)"}


def test_decimals_become_floats():
    prices = pa.array([decimal.Decimal("1.25"), None], pa.decimal128(5, 2))
    df, details = to_frame(pa.table({"price": prices}))
    assert df["price"].dtype == "float64"
    assert df.loc[0, "price"] == 1.25
    assert details == {"decimal columns": "price (read as floating-point numbers)"}


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_durations_become_seconds(unit):
    per_second = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}[unit]
    durations = pa.array([90 * per_second, None], pa.duration(unit))
    df, details = to_frame(pa.table({"wait": durations}))
    assert df.loc[0, "wait"] == 90.0
    assert df["wait"].isna().sum() == 1
    assert details == {"duration columns": "wait (read as seconds)"}


def test_binary_becomes_its_size():
    df, details = to_frame(pa.table({"img": pa.array([b"\x00" * 1024, None, b"ab"])}))
    assert list(df["img"].fillna("-")) == ["<1,024 bytes>", "-", "<2 bytes>"]
    assert details == {"binary columns": "img (shown as their size)"}


def test_nested_values_become_json():
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
    # Hashable now: the profile can count them.
    assert df["tags"].value_counts().size == 1


def test_dictionary_of_a_converted_type_is_converted():
    sizes = pa.array([b"ab", b"ab"]).dictionary_encode()
    df, details = to_frame(pa.table({"blob": sizes}))
    assert list(df["blob"]) == ["<2 bytes>", "<2 bytes>"]
    assert "binary columns" in details


def test_pandas_index_becomes_columns():
    frame = pd.DataFrame({"x": [1, 2]}, index=pd.Index(["r1", "r2"], name="row"))
    df, _ = to_frame(pa.Table.from_pandas(frame))
    assert list(df.columns) == ["row", "x"]
    assert isinstance(df.index, pd.RangeIndex)


def test_range_index_is_not_a_column():
    df, _ = to_frame(pa.Table.from_pandas(pd.DataFrame({"x": [1, 2]})))
    assert list(df.columns) == ["x"]


def test_column_names_are_unique_strings():
    table = pa.Table.from_arrays([pa.array([1]), pa.array([2]), pa.array([3])], ["a", "a", "a.1"])
    df, _ = to_frame(table)
    assert list(df.columns) == ["a", "a.2", "a.1"]


def test_number_column_names_become_strings():
    df, _ = to_frame(pa.Table.from_pandas(pd.DataFrame({0: [1], 1: [2]})))
    assert list(df.columns) == ["0", "1"]
