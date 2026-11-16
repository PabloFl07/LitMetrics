import pandas as pd
import pytest

from litmetrics.core import schema as sch
from litmetrics.core.io.base import LoadedDataset
from litmetrics.core.schema import SemanticType as T

N = sch.IDENTIFIER_MIN_ROWS


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.5, 2.5, 3.5], T.NUMERIC),
        ([True, False, True], T.BOOLEAN),
        (["yes", "no", "yes"], T.BOOLEAN),
        (["2024-01-01", "2024-02-01", "2024-03-01"], T.DATETIME),
        (["a", "b", "a", "b"], T.CATEGORICAL),
        ([f"code-{i}" for i in range(N)], T.IDENTIFIER),
        ([f"free text {i}" for i in range(N)], T.TEXT),
    ],
)
def test_infer_semantic_type(values, expected):
    assert sch.infer_semantic_type(pd.Series(values)) is expected


def test_integer_id_by_name():
    assert sch.infer_semantic_type(pd.Series([3, 1, 2], name="user_id")) is T.IDENTIFIER
    assert sch.infer_semantic_type(pd.Series([3, 1, 2], name="paid")) is T.NUMERIC


@pytest.mark.parametrize(
    "times",
    [
        ["09:30", "10:15", "17:45"],
        ["09:30:00", "10:15:30", "17:45:00"],
        ["09:30:00.250", "10:15:30.5", "17:45:00.0"],
        ["9:30 AM", "10:15 pm", "5:45 p.m."],
    ],
)
def test_times_of_day_are_not_dates(times):
    # pandas would read them as today at that time.
    assert sch.infer_semantic_type(pd.Series(times * 10)) is not T.DATETIME


def test_dates_with_a_time_are_still_dates():
    series = pd.Series(["2024-01-02 09:30:00", "2024-01-03 10:15:00", "2024-01-04 17:45:00"])
    assert sch.infer_semantic_type(series) is T.DATETIME


def test_dates_with_mixed_offsets_are_dates():
    series = pd.Series(["2024-01-02T10:00:00+02:00", "2024-01-03T10:00:00+01:00"])
    assert sch.infer_semantic_type(series) is T.DATETIME


def test_apply_schema_converts_dates_and_categories():
    df = pd.DataFrame({"d": ["2024-01-01", "2024-01-02", "2024-01-03"], "c": ["a", "b", "a"]})
    schema = sch.Schema(
        (
            sch.ColumnSchema("d", "str", T.DATETIME),
            sch.ColumnSchema("c", "str", T.CATEGORICAL),
        )
    )
    typed = sch.apply_schema(df, schema)
    assert pd.api.types.is_datetime64_any_dtype(typed.df["d"])
    assert isinstance(typed.df["c"].dtype, pd.CategoricalDtype)
    assert typed.schema.columns[0].pandas_dtype.startswith("datetime64")
    assert typed.schema.columns[1].pandas_dtype == "category"
    assert typed.unparsed == {"d": 0}
    # The argument is not changed.
    assert df["d"].dtype != typed.df["d"].dtype


def test_apply_schema_counts_values_that_are_not_dates():
    df = pd.DataFrame({"d": ["2024-01-01", "2024-01-02", "31/02/2024", None]})
    schema = sch.Schema((sch.ColumnSchema("d", "str", T.DATETIME),))
    typed = sch.apply_schema(df, schema)
    assert typed.unparsed == {"d": 1}
    assert typed.df["d"].notna().sum() == 2


def test_apply_schema_drops_timezone_as_utc():
    df = pd.DataFrame({"d": ["2024-01-02T10:00:00+02:00", "2024-01-03T10:00:00+01:00"]})
    schema = sch.Schema((sch.ColumnSchema("d", "str", T.DATETIME),))
    dates = sch.apply_schema(df, schema).df["d"]
    assert dates.dt.tz is None
    assert list(dates) == [pd.Timestamp("2024-01-02 08:00"), pd.Timestamp("2024-01-03 09:00")]


def test_apply_schema_keeps_datetime_columns_without_counting():
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-01", None])})
    schema = sch.Schema((sch.ColumnSchema("d", "datetime64[ns]", T.DATETIME),))
    assert sch.apply_schema(df, schema).unparsed == {}


def test_prepare_dataset():
    # 1 of 30 values is not a date: still a date column (at least 95% parse).
    when = [f"2024-01-{d:02d}" for d in range(1, 30)] + ["nope 1"]
    df = pd.DataFrame({"when": when, "n": range(30)})
    dataset, schema = sch.prepare_dataset(LoadedDataset(df, "f.csv", "CSV"))
    assert schema.type_of("when") is T.DATETIME
    assert dataset.unparsed == {"when": 1}
    assert dataset.name == "f.csv"
    assert pd.api.types.is_datetime64_any_dtype(dataset.df["when"])


def test_type_of_and_names_of():
    schema = sch.Schema(
        (
            sch.ColumnSchema("a", "int64", T.NUMERIC),
            sch.ColumnSchema("b", "str", T.TEXT),
            sch.ColumnSchema("c", "float64", T.NUMERIC),
        )
    )
    assert schema.type_of("b") is T.TEXT
    with pytest.raises(KeyError):
        schema.type_of("nope")
    assert schema.names_of([T.NUMERIC]) == ("a", "c")
    assert schema.names_of([]) == ()


def test_schema_lookup_cache_does_not_change_equality():
    columns = (sch.ColumnSchema("a", "int64", T.NUMERIC),)
    used, fresh = sch.Schema(columns), sch.Schema(columns)
    used.type_of("a")
    assert used == fresh and hash(used) == hash(fresh)


def test_inference_looks_at_a_random_sample_not_the_first_rows():
    # A sorted file: dates first, then free text. The first rows alone say "dates".
    values = [f"2024-01-{d % 28 + 1:02d}" for d in range(300)] + [
        f"some words {i}" for i in range(2_700)
    ]
    assert sch.infer_semantic_type(pd.Series(values)) is T.TEXT


def test_apply_schema_stores_boolean_text_as_category():
    df = pd.DataFrame({"b": ["yes", "no", None, "yes"]})
    schema = sch.Schema((sch.ColumnSchema("b", "str", T.BOOLEAN),))
    typed = sch.apply_schema(df, schema).df["b"]
    assert isinstance(typed.dtype, pd.CategoricalDtype)
    assert set(typed.cat.categories) == {"yes", "no"}
    assert typed.isna().sum() == 1


def test_apply_schema_keeps_real_booleans():
    df = pd.DataFrame({"b": [True, False]})
    schema = sch.Schema((sch.ColumnSchema("b", "bool", T.BOOLEAN),))
    assert sch.apply_schema(df, schema).df["b"].dtype == bool


@pytest.mark.parametrize(
    "values",
    [
        ["1", "2", None, "x", "2"],
        ["1", "2", "3", "2"],
        [None, None],
        [],
    ],
)
def test_by_unique_matches_converting_every_value(values):
    series = pd.Series(values, dtype="str")
    convert = lambda s: pd.to_numeric(s, errors="coerce")  # noqa: E731 (one-line helper)
    expected = convert(series)
    result = sch._by_unique(series, convert)
    assert result.dtype == expected.dtype
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_by_unique_dates_match_converting_every_value():
    series = pd.Series(["01/02/2024", None, "31/02/2024", "05/03/2024", "01/02/2024"])

    def parse(s):
        return pd.to_datetime(s, errors="coerce", format="%d/%m/%Y")

    pd.testing.assert_series_equal(sch._by_unique(series, parse), parse(series))
