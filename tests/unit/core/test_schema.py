"""Unit tests of ``core/schema.py``: semantic type of each column and conversion to its dtype.

The type decides which view profiles a column and how. Inference looks at a sample of values,
so the tests cover the values that fool it: times of day, ids that look like numbers, sorted
files whose first rows are not typical.
"""

import pandas as pd
import pytest

from litmetrics.core import schema as sch
from litmetrics.core.io.base import LoadedDataset
from litmetrics.core.schema import SemanticType as T

N = sch.IDENTIFIER_MIN_ROWS


def _schema(**columns: tuple[str, T]) -> sch.Schema:
    """A schema of ``name=(pandas dtype, semantic type)`` columns."""
    return sch.Schema(tuple(sch.ColumnSchema(n, dtype, t) for n, (dtype, t) in columns.items()))


# Inference


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
    ids=["numeric", "bool", "yes-no", "dates", "categorical", "identifier", "text"],
)
def test_infer_semantic_type(values, expected):
    """Each kind of column gets its semantic type.

    Case: "yes"/"no" stays categorical, text dates stay text, or codes look like free text.
    Affects: core/schema.py (infer_semantic_type)
    How: infers the type of one small series per kind.
    """
    assert sch.infer_semantic_type(pd.Series(values)) is expected


def test_integer_id_by_name():
    """Integers are an id only when the name says so.

    Case: every integer column with distinct values (amounts, years) is taken for an id.
    Affects: core/schema.py (infer_semantic_type)
    How: infers the same values named ``user_id`` and ``paid``.
    """
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
    ids=["hh-mm", "hh-mm-ss", "fractions", "am-pm"],
)
def test_times_of_day_are_not_dates(times):
    """Times of day are not dates.

    Case: pandas parses "09:30" as today at 09:30, inventing a date that changes every day.
    Affects: core/schema.py (infer_semantic_type)
    How: infers columns of times in four common formats.
    """
    assert sch.infer_semantic_type(pd.Series(times * 10)) is not T.DATETIME


@pytest.mark.parametrize(
    "values",
    [
        ["2024-01-02 09:30:00", "2024-01-03 10:15:00", "2024-01-04 17:45:00"],
        ["2024-01-02T10:00:00+02:00", "2024-01-03T10:00:00+01:00"],
    ],
    ids=["with-time", "mixed-offsets"],
)
def test_dates_with_time_or_offsets_are_dates(values):
    """Dates with a time, or with different UTC offsets, are still dates.

    Case: the time-of-day rule rejects full timestamps, or mixed offsets fail to parse.
    Affects: core/schema.py (infer_semantic_type)
    How: infers a column of timestamps and one of ISO dates with two offsets.
    """
    assert sch.infer_semantic_type(pd.Series(values)) is T.DATETIME


def test_inference_looks_at_a_random_sample_not_the_first_rows():
    """Inference samples the whole column, not its head.

    Case: a sorted file starting with 300 dates and then 2,700 texts is typed as dates, and
    90% of its values become "unparsed".
    Affects: core/schema.py (infer_semantic_type), core/sampling.py
    How: infers that sorted column.
    """
    values = [f"2024-01-{d % 28 + 1:02d}" for d in range(300)] + [
        f"some words {i}" for i in range(2_700)
    ]
    assert sch.infer_semantic_type(pd.Series(values)) is T.TEXT


# Conversion to the dtype of the type


def test_apply_schema_converts_dates_and_categories():
    """Text dates become datetime64 and categoricals become ``category``; input is untouched.

    Case: the argument is converted in place, changing the shared cached dataset.
    Affects: core/schema.py (apply_schema)
    How: applies a schema to a text date and a text category column; checks dtypes, the
    schema's recorded dtypes, unparsed counts and the input.
    """
    df = pd.DataFrame({"d": ["2024-01-01", "2024-01-02", "2024-01-03"], "c": ["a", "b", "a"]})
    typed = sch.apply_schema(df, _schema(d=("str", T.DATETIME), c=("str", T.CATEGORICAL)))
    assert pd.api.types.is_datetime64_any_dtype(typed.df["d"])
    assert isinstance(typed.df["c"].dtype, pd.CategoricalDtype)
    assert typed.schema.columns[0].pandas_dtype.startswith("datetime64")
    assert typed.schema.columns[1].pandas_dtype == "category"
    assert typed.unparsed == {"d": 0}
    assert df["d"].dtype != typed.df["d"].dtype


def test_apply_schema_counts_values_that_are_not_dates():
    """Values that fail to parse become missing and are counted; real missing ones are not.

    Case: an invalid date (31/02) disappears without a trace, or NaN counts as unparsed.
    Affects: core/schema.py (apply_schema)
    How: applies DATETIME to two dates, an impossible date and a missing value.
    """
    df = pd.DataFrame({"d": ["2024-01-01", "2024-01-02", "31/02/2024", None]})
    typed = sch.apply_schema(df, _schema(d=("str", T.DATETIME)))
    assert typed.unparsed == {"d": 1}
    assert typed.df["d"].notna().sum() == 2


def test_apply_schema_drops_timezone_as_utc():
    """Dates with offsets are converted to UTC and stored without a timezone.

    Case: the offset is dropped without converting, so 10:00+02:00 becomes 10:00.
    Affects: core/schema.py (apply_schema)
    How: applies DATETIME to two ISO dates with offsets and compares the UTC values.
    """
    df = pd.DataFrame({"d": ["2024-01-02T10:00:00+02:00", "2024-01-03T10:00:00+01:00"]})
    dates = sch.apply_schema(df, _schema(d=("str", T.DATETIME))).df["d"]
    assert dates.dt.tz is None
    assert list(dates) == [pd.Timestamp("2024-01-02 08:00"), pd.Timestamp("2024-01-03 09:00")]


def test_apply_schema_keeps_datetime_columns_without_counting():
    """A column that already is datetime is kept and not counted as parsed.

    Case: the loader's missing dates are reported as values that failed to parse.
    Affects: core/schema.py (apply_schema)
    How: applies DATETIME to a datetime64 column with a missing value.
    """
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-01", None])})
    assert sch.apply_schema(df, _schema(d=("datetime64[ns]", T.DATETIME))).unparsed == {}


@pytest.mark.parametrize(
    ("values", "dtype", "kept"),
    [(["yes", "no", None, "yes"], "str", False), ([True, False], "bool", True)],
    ids=["text", "real-booleans"],
)
def test_apply_schema_of_booleans(values, dtype, kept):
    """Boolean text is stored as a category of its own words; real booleans stay ``bool``.

    Case: "yes"/"no" is turned into True/False (changing the user's values), or real booleans
    become categories.
    Affects: core/schema.py (apply_schema)
    How: applies BOOLEAN to a yes/no column and to a bool column.
    """
    typed = sch.apply_schema(pd.DataFrame({"b": values}), _schema(b=(dtype, T.BOOLEAN))).df["b"]
    if kept:
        assert typed.dtype == bool
    else:
        assert isinstance(typed.dtype, pd.CategoricalDtype)
        assert set(typed.cat.categories) == {"yes", "no"}
        assert typed.isna().sum() == 1


def test_prepare_dataset():
    """A loaded dataset gets its schema and typed columns; a few bad values are tolerated.

    Case: one value in 30 that is not a date makes the whole column text.
    Affects: core/schema.py (prepare_dataset, infer_schema, apply_schema)
    How: prepares 29 dates and "nope 1", and checks type, unparsed count, name and dtype.
    """
    when = [f"2024-01-{d:02d}" for d in range(1, 30)] + ["nope 1"]
    df = pd.DataFrame({"when": when, "n": range(30)})
    dataset, schema = sch.prepare_dataset(LoadedDataset(df, "f.csv", "CSV"))
    assert schema.type_of("when") is T.DATETIME
    assert dataset.unparsed == {"when": 1}
    assert dataset.name == "f.csv"
    assert pd.api.types.is_datetime64_any_dtype(dataset.df["when"])


def test_unparsed_values_keep_their_text_and_row_position():
    """The values that became missing are kept as written, at their position in the file.

    Case: the report of an export points at the wrong row because it uses the labels of the
    index (left over from a filter) instead of positions, or lists missing values as unread.
    Affects: core/schema.py (apply_schema, _lost_values)
    How: types a text column indexed 10, 20, 30, 40 with "12 kg" second and a missing value,
    and checks the values kept, their positions and the count.
    """
    df = pd.DataFrame({"n": ["1", "12 kg", None, "4"]}, index=[10, 20, 30, 40])
    typed = sch.apply_schema(df, _schema(n=("str", T.NUMERIC)))
    lost = typed.unparsed_values["n"]
    assert lost.to_dict() == {1: "12 kg"}
    assert typed.unparsed == {"n": 1}


# Schema object


def test_type_of_and_names_of():
    """Lookups by name and by type; an unknown name is a ``KeyError``.

    Case: an unknown column silently gets a default type.
    Affects: core/schema.py (Schema.type_of, Schema.names_of)
    How: looks up a known and an unknown column, and the names of one type and of none.
    """
    schema = _schema(a=("int64", T.NUMERIC), b=("str", T.TEXT), c=("float64", T.NUMERIC))
    assert schema.type_of("b") is T.TEXT
    with pytest.raises(KeyError):
        schema.type_of("nope")
    assert schema.names_of([T.NUMERIC]) == ("a", "c")
    assert schema.names_of([]) == ()


def test_schema_lookup_cache_does_not_change_equality():
    """A schema that has been queried equals (and hashes as) a fresh one.

    Case: the lookup cache becomes part of equality, so the same schema misses the Streamlit
    cache after one lookup.
    Affects: core/schema.py (Schema)
    How: queries one of two equal schemas and compares equality and hash.
    """
    columns = (sch.ColumnSchema("a", "int64", T.NUMERIC),)
    used, fresh = sch.Schema(columns), sch.Schema(columns)
    used.type_of("a")
    assert used == fresh and hash(used) == hash(fresh)


# Converting each distinct value once


@pytest.mark.parametrize(
    "values",
    [["1", "2", None, "x", "2"], ["1", "2", "3", "2"], [None, None], []],
    ids=["mixed", "repeats", "all-missing", "empty"],
)
def test_by_unique_matches_converting_every_value(values):
    """Converting distinct values once gives the same series as converting every value.

    Case: the shortcut misaligns values with their rows, or changes the dtype (object vs float).
    Affects: core/schema.py (_by_unique)
    How: converts each series to numbers both ways and compares values and dtype.
    """
    series = pd.Series(values, dtype="str")
    convert = lambda s: pd.to_numeric(s, errors="coerce")  # noqa: E731 (one-line helper)
    expected = convert(series)
    result = sch._by_unique(series, convert)
    assert result.dtype == expected.dtype
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_by_unique_dates_match_converting_every_value():
    """The shortcut also holds for dates, with repeats, missing and impossible dates.

    Case: the date path of the shortcut maps a repeated date to the wrong row.
    Affects: core/schema.py (_by_unique)
    How: parses day/month dates both ways and compares.
    """
    series = pd.Series(["01/02/2024", None, "31/02/2024", "05/03/2024", "01/02/2024"])

    def parse(s):
        return pd.to_datetime(s, errors="coerce", format="%d/%m/%Y")

    pd.testing.assert_series_equal(sch._by_unique(series, parse), parse(series))
