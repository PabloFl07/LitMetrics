import numpy as np
import pandas as pd
import pytest

from litmetrics.core import columns as cols
from litmetrics.core.profile import base_profile
from litmetrics.core.schema import ColumnSchema, Schema, SemanticType

T = SemanticType
# "Now" of the tests that count dates in the future.
NOW = pd.Timestamp("2025-01-01")


def _schema(**types: SemanticType) -> Schema:
    return Schema(tuple(ColumnSchema(name, "", kind) for name, kind in types.items()))


@pytest.fixture
def df() -> pd.DataFrame:
    n = 40
    return pd.DataFrame(
        {
            # 1..38 plus one far outlier and one missing value.
            "value": [*range(1, 39), 1000.0, np.nan],
            "rating": [1, 2, 3, 4] * 10,
            "city": pd.Categorical(["Madrid", "madrid ", "Lugo", "Lugo"] * 10),
            "flag": [True, False] * 20,
            "when": pd.date_range("2024-01-01", periods=n, freq="D"),
            "note": ["hello world", "  ", "hi", None] * 10,
            "code": [f"A-{i:03d}" for i in range(n)],
        }
    )


@pytest.fixture
def schema() -> Schema:
    return _schema(
        value=T.NUMERIC,
        rating=T.NUMERIC,
        city=T.CATEGORICAL,
        flag=T.BOOLEAN,
        when=T.DATETIME,
        note=T.TEXT,
        code=T.IDENTIFIER,
    )


def test_names_of_keeps_dataset_order(schema):
    assert schema.names_of(cols.TEXT_TYPES) == ("note", "code")
    assert schema.names_of(cols.CATEGORICAL_TYPES) == ("city", "flag")


def test_histogram_single_value_is_one_bar():
    counts, edges = cols.histogram(np.array([5.0, 5.0]), 10, 5.0, 5.0)
    assert counts == [2]
    assert edges == [5.0, 5.0]


def test_histogram_empty():
    assert cols.histogram(np.array([]), 10, 0.0, 1.0) == ([], [])


def test_finite_values_drops_missing_and_infinite():
    values = cols.finite_values(pd.Series([1.0, np.inf, np.nan, -np.inf, 2.0]))
    assert list(values) == [1.0, 2.0]


def test_value_counts_merges_values_that_look_the_same():
    counts = cols.value_counts(pd.Series([1, "1", "a"], dtype=object))
    assert counts.to_dict() == {"1": 2, "a": 1}


# Numeric


def test_numeric_stats_outliers_and_whiskers(df):
    stats = cols.numeric_stats(df, ["value"], sample_rows=1000).set_index("column").loc["value"]
    assert stats["count"] == 39
    assert stats["outliers"] == 1
    assert stats["whisker_high"] == 38
    assert stats["max"] == 1000
    assert stats["shape"] != "Symmetric"


def test_numeric_stats_counts_infinite_apart():
    df = pd.DataFrame({"v": [1.0, 2.0, np.inf, 3.0]})
    stats = cols.numeric_stats(df, ["v"], 1000).loc[0]
    assert stats["infinite"] == 1
    assert stats["count"] == 3
    assert stats["mean"] == pytest.approx(2.0)


def test_numeric_stats_counts_unparsed_values(df):
    table = cols.numeric_stats(df, ["value", "rating"], 1000, unparsed={"value": 3})
    assert list(table["unparsed"]) == [3, 0]


def test_numeric_stats_no_columns():
    assert list(cols.numeric_stats(pd.DataFrame(), [], 1000).columns) == (
        cols.NUMERIC_STATS_COLUMNS
    )


def test_numeric_stats_quantiles_use_the_sample():
    df = pd.DataFrame({"v": np.arange(10_000, dtype=float)})
    full = cols.numeric_stats(df, ["v"], 100_000).loc[0]
    sampled = cols.numeric_stats(df, ["v"], 500).loc[0]
    # Counts stay exact; the median is an estimate close to the real one.
    assert sampled["count"] == full["count"] == 10_000
    assert sampled["median"] == pytest.approx(full["median"], rel=0.1)


def test_shape_label():
    assert cols.shape_label(0.1) == "Symmetric"
    assert cols.shape_label(0.7) == "Moderate right skew"
    assert cols.shape_label(-2) == "Strong left skew"
    assert cols.shape_label(float("nan")) == ""


def test_numeric_bars_one_bar_per_value_when_few(df):
    stats = cols.numeric_stats(df, ["rating"], 1000).loc[0]
    bars = cols.numeric_bars(df["rating"], stats, unique=4, bins=16, sample_rows=1000)
    assert bars.per_value
    assert bars.labels == ("1", "2", "3", "4")
    assert bars.counts == (10, 10, 10, 10)


def test_numeric_bars_histogram_leaves_outliers_out(df):
    stats = cols.numeric_stats(df, ["value"], 1000).loc[0]
    bars = cols.numeric_bars(df["value"], stats, unique=39, bins=5, sample_rows=1000)
    assert not bars.per_value
    assert sum(bars.counts) == 38


def test_outlier_values_most_extreme_first(df):
    stats = cols.numeric_stats(df, ["value"], 1000).loc[0]
    assert cols.outlier_values(df["value"], stats, 10, 1000) == [1000.0]


# Categorical


def test_top_bars_leaves_the_rest_out():
    bars, other_values, other_rows = cols.top_bars(pd.Series(list("aaabbc")), 2)
    assert bars.labels == ("a", "b")
    assert bars.counts == (3, 2)
    assert (other_values, other_rows) == (1, 1)


def test_value_counts_relabels_categories_and_booleans():
    counts = cols.value_counts(pd.Series([True, False, True]))
    assert counts.to_dict() == {"True": 2, "False": 1}
    categories = cols.value_counts(pd.Series(pd.Categorical(["b", "a", "b"], ["a", "b", "z"])))
    assert categories.to_dict() == {"b": 2, "a": 1}


def test_variant_groups(df):
    assert cols.variant_groups(df["city"]) == [["Madrid", "madrid "]]


def test_categorical_stats(df):
    table = cols.categorical_stats(df, ["city", "flag"]).set_index("column")
    assert table.loc["city", "unique"] == 3
    assert table.loc["city", "mode"] == "Lugo"
    assert table.loc["city", "mode_pct"] == pytest.approx(50.0)
    assert table.loc["city", "variants"] == 1
    assert table.loc["flag", "unique"] == 2


def test_categorical_stats_all_missing():
    table = cols.categorical_stats(pd.DataFrame({"c": [None, None]}), ["c"])
    assert table.loc[0, "unique"] == 0
    assert table.loc[0, "mode"] is None


# Dates


def test_datetime_values_does_not_parse_text():
    assert cols.datetime_values(pd.Series(["2024-01-01"])).empty


def test_datetime_stats(df):
    stats = cols.datetime_stats(df, ["when"], {"when": 3}).loc[0]
    assert stats["min"] == pd.Timestamp("2024-01-01")
    assert stats["span_days"] == 39
    assert stats["frequency"] == "Daily"
    assert stats["unparsed"] == 3
    assert stats["future"] == 0


def test_frequency_ignores_repeated_dates():
    dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01", "2024-03-01"]))
    assert cols.datetime_stats(pd.DataFrame({"d": dates}), ["d"], {}).loc[0, "frequency"] == (
        "Monthly"
    )


def test_datetime_bars_span_the_whole_column(df):
    bars = cols.datetime_bars(df["when"], 4, 1000)
    assert sum(bars.counts) == 40
    assert bars.labels[0].startswith("2024-01-01")
    assert bars.labels[-1].endswith("2024-02-09")


def test_datetime_bars_take_placeholder_dates_outside_the_nanosecond_range():
    # Database exports write 9999-12-31 for "no end date": it does not fit in datetime64[ns].
    dates = pd.Series(pd.to_datetime(["0001-01-01", "2024-01-01", "9999-12-31"]).as_unit("s"))
    bars = cols.datetime_bars(dates, 4, 1000)
    assert sum(bars.counts) == 3
    assert bars.labels[0].startswith("0001-01-01")
    assert bars.labels[-1].endswith("9999-12-31")


# Text


def test_text_stats(df, schema):
    base = base_profile(df)
    table = cols.text_stats(df, ["note", "code"], schema, base, 1000).set_index("column")
    assert table.loc["note", "blank"] == 10
    assert table.loc["note", "len_max"] == 11
    assert table.loc["note", "repeated"] == 30
    assert table.loc["note", "example"] == "hello world"
    assert table.loc["code", "repeated"] == 0
    assert table.loc["code", "unique_pct"] == pytest.approx(100.0)
    assert table.loc["code", "semantic_type"] == "identifier"


def test_text_stats_numeric_identifier_has_no_blank():
    df = pd.DataFrame({"user_id": [1, 2, 3]})
    schema = _schema(user_id=T.IDENTIFIER)
    table = cols.text_stats(df, ["user_id"], schema, base_profile(df), 1000)
    assert table.loc[0, "blank"] == 0
    assert table.loc[0, "len_max"] == 1


def test_text_examples_are_reproducible(df):
    first = cols.text_examples(df["code"], 3)
    assert len(first) == 3
    assert first.equals(cols.text_examples(df["code"], 3))
    assert (first["length"] == 5).all()


# One column and overview


@pytest.mark.parametrize("name", ["value", "city", "when", "note"])
def test_column_detail_has_bars(df, schema, name):
    numeric = cols.numeric_stats(df, ["value"], 1000).loc[0] if name == "value" else None
    unique = int(base_profile(df).by_column.loc[name, "unique"])
    detail = cols.column_detail(
        df,
        name,
        schema.type_of(name),
        unique=unique,
        stats=numeric,
        sample_rows=1000,
        outlier_limit=5,
        n_examples=3,
    )
    assert detail.bars.counts
    if name == "value":
        assert detail.outliers == (1000.0,)
    if name == "city":
        assert detail.variants == (("Madrid", "madrid "),)
    if name == "note":
        assert detail.examples is not None


@pytest.mark.parametrize(
    ("name", "table_of_its_type"),
    [
        ("value", lambda df, schema, base: cols.numeric_stats(df, ["value"], 1000)),
        ("city", lambda df, schema, base: cols.categorical_stats(df, ["city"])),
        ("when", lambda df, schema, base: cols.datetime_stats(df, ["when"], {}, NOW)),
        ("code", lambda df, schema, base: cols.text_stats(df, ["code"], schema, base, 1000)),
    ],
)
def test_inspect_column_gives_the_row_of_its_table(df, schema, name, table_of_its_type):
    base = base_profile(df)
    table, detail = cols.inspect_column(
        df,
        name,
        schema.type_of(name),
        base,
        sample_rows=1000,
        unparsed={},
        now=NOW,
        outlier_limit=5,
        n_examples=3,
    )
    pd.testing.assert_frame_equal(table, table_of_its_type(df, schema, base))
    assert detail.bars.counts


def test_inspect_column_counts_unparsed_values(df, schema):
    table, _ = cols.inspect_column(
        df,
        "value",
        T.NUMERIC,
        base_profile(df),
        sample_rows=1000,
        unparsed={"value": 2},
        now=None,
        outlier_limit=5,
        n_examples=3,
    )
    assert table.loc[0, "unparsed"] == 2


def test_column_overview(df, schema):
    numeric = cols.numeric_stats(df, ["value", "rating"], 1000)
    table = cols.column_overview(df, schema, base_profile(df), numeric, 1000).set_index("column")
    assert list(table.index) == list(df.columns)
    assert "summary" not in table.columns
    assert table.loc["code", "distribution"] == []
    assert sum(table.loc["rating", "distribution"]) == 40


def test_column_overview_all_missing_column():
    df = pd.DataFrame({"empty": [np.nan, np.nan]})
    schema = _schema(empty=T.NUMERIC)
    numeric = cols.numeric_stats(df, ["empty"], 1000)
    table = cols.column_overview(df, schema, base_profile(df), numeric, 1000)
    assert table.loc[0, "distribution"] == []


def test_bar_index_histogram():
    bars = cols.Bars((1, 2, 3), ("a", "b", "c"), edges=(0.0, 1.0, 2.0, 3.0))
    assert cols.bar_index(bars, 0.0) == 0
    assert cols.bar_index(bars, 1.5) == 1
    # The last edge belongs to the last bin.
    assert cols.bar_index(bars, 3.0) == 2
    assert cols.bar_index(bars, 3.5) is None
    assert cols.bar_index(bars, float("nan")) is None


def test_bar_index_per_value_picks_closest(df):
    stats = cols.numeric_stats(df, ["rating"], 1000).loc[0]
    bars = cols.numeric_bars(df["rating"], stats, unique=4, bins=16, sample_rows=1000)
    assert cols.bar_index(bars, 2.4) == 1
    assert cols.bar_index(cols.Bars(), 1.0) is None


def test_column_detail_numeric_needs_its_stats(df):
    with pytest.raises(ValueError, match="value"):
        cols.column_detail(
            df,
            "value",
            T.NUMERIC,
            unique=39,
            stats=None,
            sample_rows=1000,
            outlier_limit=5,
            n_examples=3,
        )


def test_a_table_of_one_column_gives_the_same_row(df):
    # The column inspector computes only its column: it must match the whole table.
    whole = cols.numeric_stats(df, ["value", "rating"], 10).set_index("column")
    alone = cols.numeric_stats(df, ["rating"], 10).set_index("column")
    pd.testing.assert_series_equal(whole.loc["rating"], alone.loc["rating"])


def test_numeric_stats_in_threads_matches_one_thread(df):
    names = ["value", "rating"]
    pd.testing.assert_frame_equal(
        cols.numeric_stats(df, names, 10), cols.numeric_stats(df, names, 10, workers=2)
    )


def test_numeric_stats_matches_pandas_on_every_row():
    rng = np.random.default_rng(0)
    values = pd.Series(np.where(rng.random(500) < 0.1, np.nan, rng.lognormal(size=500)))
    stats = cols.numeric_stats(pd.DataFrame({"v": values}), ["v"], 10_000).loc[0]
    assert stats["mean"] == pytest.approx(values.mean())
    assert stats["std"] == pytest.approx(values.std())
    assert stats["median"] == pytest.approx(values.median())
    assert stats["p5"] == pytest.approx(values.quantile(0.05))
    assert stats["skew"] == pytest.approx(values.skew())
    assert stats["kurtosis"] == pytest.approx(values.kurt())


def test_numeric_stats_edge_columns():
    df = pd.DataFrame(
        {
            "empty": [np.nan] * 4,
            "one": [7.0, np.nan, np.nan, np.nan],
            "nullable": pd.array([1, None, 3, 3], dtype="Int64"),
        }
    )
    table = cols.numeric_stats(df, list(df.columns), 1000).set_index("column")
    assert table.loc["empty", "count"] == 0
    assert np.isnan(table.loc["empty", "mean"])
    assert table.loc["one", "min"] == table.loc["one", "max"] == 7.0
    assert np.isnan(table.loc["one", "std"])
    assert table.loc["nullable", "count"] == 3
    assert table.loc["nullable", "median"] == 3.0


def test_frequency_of_sub_daily_dates():
    dates = pd.Series(pd.date_range("2024-01-01", periods=10, freq="h"))
    assert cols.datetime_stats(pd.DataFrame({"d": dates}), ["d"], {}).loc[0, "frequency"] == (
        "Sub-daily"
    )


def test_first_non_blank_skips_blank_and_missing_across_chunks():
    series = pd.Series([None] * 1_500 + ["  ", "found"])
    assert cols._first_non_blank(series) == "found"
    assert cols._first_non_blank(pd.Series([None, " "])) == ""
