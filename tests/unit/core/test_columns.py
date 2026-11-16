"""Unit tests of ``core/columns.py``: the statistics of the Columns tab of Profile.

Most tests use ``typed_df`` / ``typed_schema`` (``samples.typed_frame``): 40 rows with one column
of each semantic type and a known oddity in each (an outlier, values written two ways, blanks).
Statistics are checked against values worked out by hand or against pandas on every row.
"""

import numpy as np
import pandas as pd
import pytest
from fixtures.samples import schema_of

from litmetrics.core import columns as cols
from litmetrics.core.profile import base_profile
from litmetrics.core.schema import SemanticType

T = SemanticType
# "Now" of the tests that count dates in the future.
NOW = pd.Timestamp("2025-01-01")


# Shared helpers


def test_types_are_grouped_as_the_views_show_them(typed_schema):
    """Identifiers go with text and booleans with categories, in dataset order.

    Case: a type is left out of every view, so its columns are never profiled.
    Affects: core/columns.py (TEXT_TYPES, CATEGORICAL_TYPES), core/schema.py (names_of)
    How: asks ``typed_schema`` for the names of each group.
    """
    assert typed_schema.names_of(cols.TEXT_TYPES) == ("note", "code")
    assert typed_schema.names_of(cols.CATEGORICAL_TYPES) == ("city", "flag")


@pytest.mark.parametrize(
    ("values", "expected"),
    [(np.array([5.0, 5.0]), ([2], [5.0, 5.0])), (np.array([]), ([], []))],
    ids=["single-value", "empty"],
)
def test_histogram_degenerate_inputs(values, expected):
    """A column of one value is one bar; an empty one is no bars.

    Case: numpy divides by a zero range (NaN edges) or raises on an empty array.
    Affects: core/columns.py (histogram)
    How: builds the histogram of two equal values and of no values.
    """
    low = float(values.min()) if values.size else 0.0
    high = float(values.max()) if values.size else 1.0
    assert cols.histogram(values, 10, low, high) == expected


def test_finite_values_drops_missing_and_infinite():
    """Only finite numbers are kept for charts and quantiles.

    Case: an ``inf`` in a column makes every histogram edge infinite.
    Affects: core/columns.py (finite_values)
    How: filters a series with NaN, +inf and -inf.
    """
    values = cols.finite_values(pd.Series([1.0, np.inf, np.nan, -np.inf, 2.0]))
    assert list(values) == [1.0, 2.0]


def test_value_counts_merges_values_that_look_the_same():
    """``1`` and ``"1"`` count as the same value, since they are shown the same way.

    Case: a chart shows two bars both labelled "1".
    Affects: core/columns.py (value_counts)
    How: counts an object series with 1, "1" and "a".
    """
    counts = cols.value_counts(pd.Series([1, "1", "a"], dtype=object))
    assert counts.to_dict() == {"1": 2, "a": 1}


def test_value_counts_relabels_categories_and_booleans():
    """Booleans and categories are counted as text labels; unused categories are left out.

    Case: unused categories appear as bars of 0, or booleans as 0 / 1.
    Affects: core/columns.py (value_counts)
    How: counts a boolean series and a categorical one with an unused category "z".
    """
    assert cols.value_counts(pd.Series([True, False, True])).to_dict() == {"True": 2, "False": 1}
    categories = pd.Series(pd.Categorical(["b", "a", "b"], ["a", "b", "z"]))
    assert cols.value_counts(categories).to_dict() == {"b": 2, "a": 1}


# Numeric


def test_numeric_stats_outliers_and_whiskers(typed_df):
    """The far value is one outlier, outside the whisker, and makes the column skewed.

    Case: the whisker is set to the max (outliers hidden) or the missing value is counted.
    Affects: core/columns.py (numeric_stats)
    How: computes ``value`` (1..38 plus 1000 and NaN) and checks count, outliers, whisker, max.
    """
    stats = cols.numeric_stats(typed_df, ["value"], sample_rows=1000).set_index("column")
    row = stats.loc["value"]
    assert row["count"] == 39
    assert row["outliers"] == 1
    assert row["whisker_high"] == 38
    assert row["max"] == 1000
    assert row["shape"] != "Symmetric"


def test_numeric_stats_counts_infinite_apart():
    """Infinite values are counted on their own and left out of count and mean.

    Case: one ``inf`` makes the mean ``inf`` and hides the rest of the statistics.
    Affects: core/columns.py (numeric_stats)
    How: computes a column with one ``inf`` among 1, 2, 3.
    """
    stats = cols.numeric_stats(pd.DataFrame({"v": [1.0, 2.0, np.inf, 3.0]}), ["v"], 1000).loc[0]
    assert stats["infinite"] == 1
    assert stats["count"] == 3
    assert stats["mean"] == pytest.approx(2.0)


def test_numeric_stats_counts_unparsed_values(typed_df):
    """Values that could not be read as numbers are shown per column, 0 when none.

    Case: the unparsed count is shown for the wrong column, or NaN for columns without any.
    Affects: core/columns.py (numeric_stats)
    How: passes 3 unparsed values for ``value`` and none for ``rating``.
    """
    table = cols.numeric_stats(typed_df, ["value", "rating"], 1000, unparsed={"value": 3})
    assert list(table["unparsed"]) == [3, 0]


def test_numeric_stats_no_columns():
    """With no numeric columns the table is empty but keeps its columns.

    Case: an empty table without columns breaks the page that formats them.
    Affects: core/columns.py (numeric_stats, NUMERIC_STATS_COLUMNS)
    How: computes the stats of no columns and checks the column names.
    """
    assert list(cols.numeric_stats(pd.DataFrame(), [], 1000).columns) == cols.NUMERIC_STATS_COLUMNS


def test_numeric_stats_edge_columns():
    """All-missing, single-value and nullable-integer columns give defined results.

    Case: an all-missing column raises, a single value gives a fake std of 0, or ``Int64``
    with ``pd.NA`` breaks numpy.
    Affects: core/columns.py (numeric_stats)
    How: computes the three edge columns and checks count, mean, min/max, std and median.
    """
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


def test_numeric_stats_matches_pandas_on_every_row():
    """Mean, std, median, percentiles, skewness and kurtosis equal pandas' own.

    Case: the fast numpy path uses population std (ddof=0) or a different quantile method.
    Affects: core/columns.py (numeric_stats)
    How: computes 500 lognormal values with 10% missing, with a sample larger than the column.
    """
    rng = np.random.default_rng(0)
    values = pd.Series(np.where(rng.random(500) < 0.1, np.nan, rng.lognormal(size=500)))
    stats = cols.numeric_stats(pd.DataFrame({"v": values}), ["v"], 10_000).loc[0]
    assert stats["mean"] == pytest.approx(values.mean())
    assert stats["std"] == pytest.approx(values.std())
    assert stats["median"] == pytest.approx(values.median())
    assert stats["p5"] == pytest.approx(values.quantile(0.05))
    assert stats["skew"] == pytest.approx(values.skew())
    assert stats["kurtosis"] == pytest.approx(values.kurt())


def test_numeric_stats_quantiles_use_the_sample():
    """Counts stay exact on a sample; quantiles are estimated close to the real ones.

    Case: the sample is used for counts too (wrong totals), or quantiles are far off.
    Affects: core/columns.py (numeric_stats), core/sampling.py
    How: computes 10,000 values with and without a 500-row sample and compares.
    """
    df = pd.DataFrame({"v": np.arange(10_000, dtype=float)})
    full = cols.numeric_stats(df, ["v"], 100_000).loc[0]
    sampled = cols.numeric_stats(df, ["v"], 500).loc[0]
    assert sampled["count"] == full["count"] == 10_000
    assert sampled["median"] == pytest.approx(full["median"], rel=0.1)


def test_a_table_of_one_column_gives_the_same_row(typed_df):
    """A column computed alone gives the same row as in the table of every column.

    Case: the inspector (one column) and the table (all) sample different rows and disagree.
    Affects: core/columns.py (numeric_stats), core/sampling.py
    How: computes ``rating`` with ``value`` and alone, both with a 10-row sample.
    """
    whole = cols.numeric_stats(typed_df, ["value", "rating"], 10).set_index("column")
    alone = cols.numeric_stats(typed_df, ["rating"], 10).set_index("column")
    pd.testing.assert_series_equal(whole.loc["rating"], alone.loc["rating"])


def test_numeric_stats_in_threads_matches_one_thread(typed_df):
    """Computing columns in threads gives the same table as in one thread.

    Case: threads share a buffer or return rows out of order.
    Affects: core/columns.py (numeric_stats), core/parallel.py (map_columns)
    How: compares the table with 1 and 2 workers.
    """
    names = ["value", "rating"]
    pd.testing.assert_frame_equal(
        cols.numeric_stats(typed_df, names, 10), cols.numeric_stats(typed_df, names, 10, workers=2)
    )


@pytest.mark.parametrize(
    ("skew", "label"),
    [(0.1, "Symmetric"), (0.7, "Moderate right skew"), (-2, "Strong left skew"), (np.nan, "")],
)
def test_shape_label(skew, label):
    """Skewness is described in words; an undefined one gives no label.

    Case: the sign is read backwards, or NaN is shown as "Strong left skew".
    Affects: core/columns.py (shape_label)
    How: labels a symmetric, a right, a left and an undefined skewness.
    """
    assert cols.shape_label(skew) == label


def test_numeric_bars_one_bar_per_value_when_few(typed_df):
    """A column with few distinct values gets one bar per value, not a histogram.

    Case: 4 ratings drawn as 16 bins, most of them empty.
    Affects: core/columns.py (numeric_bars)
    How: builds the bars of ``rating`` (1..4, 10 each).
    """
    stats = cols.numeric_stats(typed_df, ["rating"], 1000).loc[0]
    bars = cols.numeric_bars(typed_df["rating"], stats, unique=4, bins=16, sample_rows=1000)
    assert bars.per_value
    assert bars.labels == ("1", "2", "3", "4")
    assert bars.counts == (10, 10, 10, 10)


def test_numeric_bars_histogram_leaves_outliers_out(typed_df):
    """The histogram covers the whiskers; the outlier is not in any bar.

    Case: one far value stretches the bins so every normal value falls in the first bar.
    Affects: core/columns.py (numeric_bars)
    How: builds 5 bins of ``value`` and checks they hold the 38 values that are not outliers.
    """
    stats = cols.numeric_stats(typed_df, ["value"], 1000).loc[0]
    bars = cols.numeric_bars(typed_df["value"], stats, unique=39, bins=5, sample_rows=1000)
    assert not bars.per_value
    assert sum(bars.counts) == 38


def test_outlier_values_most_extreme_first(typed_df):
    """The outliers listed are the values beyond the whiskers.

    Case: values inside the whiskers are listed as outliers.
    Affects: core/columns.py (outlier_values)
    How: lists the outliers of ``value``: only 1000.
    """
    stats = cols.numeric_stats(typed_df, ["value"], 1000).loc[0]
    assert cols.outlier_values(typed_df["value"], stats, 10, 1000) == [1000.0]


def test_bar_index_of_a_histogram():
    """A value is placed in its bin; the last edge belongs to the last bin; others give None.

    Case: the maximum falls outside every bin, so the mark of the max is not drawn.
    Affects: core/columns.py (bar_index)
    How: places values at the first edge, inside, at the last edge, beyond, and NaN.
    """
    bars = cols.Bars((1, 2, 3), ("a", "b", "c"), edges=(0.0, 1.0, 2.0, 3.0))
    assert cols.bar_index(bars, 0.0) == 0
    assert cols.bar_index(bars, 1.5) == 1
    assert cols.bar_index(bars, 3.0) == 2
    assert cols.bar_index(bars, 3.5) is None
    assert cols.bar_index(bars, float("nan")) is None


def test_bar_index_per_value_picks_closest(typed_df):
    """With one bar per value, a mean between values goes to the closest bar.

    Case: a mean of 2.4 matches no bar exactly and the mark is not drawn.
    Affects: core/columns.py (bar_index)
    How: places 2.4 in the bars of ``rating`` (bar of 2), and any value in no bars.
    """
    stats = cols.numeric_stats(typed_df, ["rating"], 1000).loc[0]
    bars = cols.numeric_bars(typed_df["rating"], stats, unique=4, bins=16, sample_rows=1000)
    assert cols.bar_index(bars, 2.4) == 1
    assert cols.bar_index(cols.Bars(), 1.0) is None


# Categorical


def test_top_bars_leaves_the_rest_out():
    """Only the most frequent values get a bar; the rest are counted apart.

    Case: a column with 10,000 distinct values sends 10,000 bars to the browser.
    Affects: core/columns.py (top_bars)
    How: keeps the top 2 of a, a, a, b, b, c and checks the count of the rest.
    """
    bars, other_values, other_rows = cols.top_bars(pd.Series(list("aaabbc")), 2)
    assert bars.labels == ("a", "b")
    assert bars.counts == (3, 2)
    assert (other_values, other_rows) == (1, 1)


def test_variant_groups(typed_df):
    """Values that differ only in case or spaces are grouped as variants.

    Case: ``Madrid`` and ``madrid `` are counted as two cities and nobody is told.
    Affects: core/columns.py (variant_groups)
    How: groups the values of ``city``.
    """
    assert cols.variant_groups(typed_df["city"]) == [["Madrid", "madrid "]]


def test_categorical_stats(typed_df):
    """Distinct values, mode, its share and variants of each categorical column.

    Case: the mode share is over all rows instead of the values, or variants are not counted.
    Affects: core/columns.py (categorical_stats)
    How: computes ``city`` and ``flag`` and compares with the values worked out by hand.
    """
    table = cols.categorical_stats(typed_df, ["city", "flag"]).set_index("column")
    assert table.loc["city", "unique"] == 3
    assert table.loc["city", "mode"] == "Lugo"
    assert table.loc["city", "mode_pct"] == pytest.approx(50.0)
    assert table.loc["city", "variants"] == 1
    assert table.loc["flag", "unique"] == 2


def test_categorical_stats_all_missing():
    """A column with no values has 0 distinct values and no mode.

    Case: ``mode()`` of an empty series raises ``IndexError``.
    Affects: core/columns.py (categorical_stats)
    How: computes a column of two missing values.
    """
    table = cols.categorical_stats(pd.DataFrame({"c": [None, None]}), ["c"])
    assert table.loc[0, "unique"] == 0
    assert table.loc[0, "mode"] is None


# Dates


def test_datetime_values_does_not_parse_text():
    """Only real datetime columns are used; text is never parsed here.

    Case: parsing every text column again on each rerun (slow, and dates the schema rejected).
    Affects: core/columns.py (datetime_values)
    How: passes a text date and checks nothing is returned.
    """
    assert cols.datetime_values(pd.Series(["2024-01-01"])).empty


def test_datetime_stats(typed_df):
    """Range, span, frequency, unparsed and future counts of a date column.

    Case: the span is off by one, or the frequency of daily data is not found.
    Affects: core/columns.py (datetime_stats)
    How: computes ``when`` (40 consecutive days) with 3 unparsed values.
    """
    stats = cols.datetime_stats(typed_df, ["when"], {"when": 3}).loc[0]
    assert stats["min"] == pd.Timestamp("2024-01-01")
    assert stats["span_days"] == 39
    assert stats["frequency"] == "Daily"
    assert stats["unparsed"] == 3
    assert stats["future"] == 0


@pytest.mark.parametrize(
    ("dates", "frequency"),
    [
        (pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01", "2024-03-01"]), "Monthly"),
        (pd.date_range("2024-01-01", periods=10, freq="h"), "Sub-daily"),
    ],
    ids=["repeated-dates", "hourly"],
)
def test_frequency(dates, frequency):
    """The frequency ignores repeated dates and recognises steps under a day.

    Case: a repeated date (gap 0) makes monthly data look irregular, or hourly data "Daily".
    Affects: core/columns.py (datetime_stats)
    How: computes the frequency of monthly dates with a repeat, and of hourly ones.
    """
    df = pd.DataFrame({"d": pd.Series(dates)})
    assert cols.datetime_stats(df, ["d"], {}).loc[0, "frequency"] == frequency


def test_datetime_bars_span_the_whole_column(typed_df):
    """The bars of a date column cover its first and last date and every row.

    Case: the last date falls outside the last bar and is lost.
    Affects: core/columns.py (datetime_bars)
    How: builds 4 bars of ``when`` and checks the total and the first and last labels.
    """
    bars = cols.datetime_bars(typed_df["when"], 4, 1000)
    assert sum(bars.counts) == 40
    assert bars.labels[0].startswith("2024-01-01")
    assert bars.labels[-1].endswith("2024-02-09")


def test_datetime_bars_take_placeholder_dates_outside_the_nanosecond_range():
    """Dates like 9999-12-31 (database exports' "no end") are binned without overflow.

    Case: converting to ``datetime64[ns]`` overflows and the Dates view crashes.
    Affects: core/columns.py (datetime_bars)
    How: bins 0001-01-01, 2024-01-01 and 9999-12-31 stored with second precision.
    """
    dates = pd.Series(pd.to_datetime(["0001-01-01", "2024-01-01", "9999-12-31"]).as_unit("s"))
    bars = cols.datetime_bars(dates, 4, 1000)
    assert sum(bars.counts) == 3
    assert bars.labels[0].startswith("0001-01-01")
    assert bars.labels[-1].endswith("9999-12-31")


# Text


def test_text_stats(typed_df, typed_schema):
    """Blanks, lengths, repeats, an example and the semantic type of each text column.

    Case: ``"  "`` is not counted as blank, or ids are reported as repeated text.
    Affects: core/columns.py (text_stats), core/profile.py (base_profile)
    How: computes ``note`` (blanks, repeats) and ``code`` (distinct ids).
    """
    base = base_profile(typed_df)
    table = cols.text_stats(typed_df, ["note", "code"], typed_schema, base, 1000)
    table = table.set_index("column")
    assert table.loc["note", "blank"] == 10
    assert table.loc["note", "len_max"] == 11
    assert table.loc["note", "repeated"] == 30
    assert table.loc["note", "example"] == "hello world"
    assert table.loc["code", "repeated"] == 0
    assert table.loc["code", "unique_pct"] == pytest.approx(100.0)
    assert table.loc["code", "semantic_type"] == "identifier"


def test_text_stats_numeric_identifier_has_no_blank():
    """A numeric id column is measured as text: no blanks, lengths of its digits.

    Case: ``.str`` on an integer column raises ``AttributeError``.
    Affects: core/columns.py (text_stats)
    How: computes an integer ``user_id`` typed as identifier.
    """
    df = pd.DataFrame({"user_id": [1, 2, 3]})
    table = cols.text_stats(
        df, ["user_id"], schema_of(user_id=T.IDENTIFIER), base_profile(df), 1000
    )
    assert table.loc[0, "blank"] == 0
    assert table.loc[0, "len_max"] == 1


def test_text_examples_are_reproducible(typed_df):
    """Examples are a fixed random pick: the same on every rerun.

    Case: the examples change on every click, which looks like the data changed.
    Affects: core/columns.py (text_examples), core/sampling.py
    How: picks 3 examples twice and compares, and checks their lengths.
    """
    first = cols.text_examples(typed_df["code"], 3)
    assert len(first) == 3
    assert first.equals(cols.text_examples(typed_df["code"], 3))
    assert (first["length"] == 5).all()


def test_first_non_blank_skips_blank_and_missing_across_chunks():
    """The example value is the first non-blank one, even after a long blank start.

    Case: the search stops at the end of its first chunk and shows no example.
    Affects: core/columns.py (_first_non_blank)
    How: looks in 1,500 missing values, a blank and "found"; then in a column with none.
    """
    assert cols._first_non_blank(pd.Series([None] * 1_500 + ["  ", "found"])) == "found"
    assert cols._first_non_blank(pd.Series([None, " "])) == ""


# One column (inspector) and overview


@pytest.mark.parametrize("name", ["value", "city", "when", "note"])
def test_column_detail_has_bars(typed_df, typed_schema, name):
    """The detail of a column of each type has bars, and the extras of its type.

    Case: the inspector of one type is drawn with an empty chart or misses its extras.
    Affects: core/columns.py (column_detail)
    How: builds the detail of one column per type; checks outliers, variants and examples.
    """
    numeric = cols.numeric_stats(typed_df, ["value"], 1000).loc[0] if name == "value" else None
    unique = int(base_profile(typed_df).by_column.loc[name, "unique"])
    detail = cols.column_detail(
        typed_df,
        name,
        typed_schema.type_of(name),
        unique=unique,
        stats=numeric,
        sample_rows=1000,
        outlier_limit=5,
        n_examples=3,
    )
    assert detail.bars.counts
    extras = {
        "value": detail.outliers == (1000.0,),
        "city": detail.variants == (("Madrid", "madrid "),),
        "when": True,
        "note": detail.examples is not None,
    }
    assert extras[name]


def test_column_detail_numeric_needs_its_stats(typed_df):
    """A numeric detail without its statistics is a ``ValueError`` naming the column.

    Case: a caller forgets the stats and the inspector silently draws no whiskers.
    Affects: core/columns.py (column_detail)
    How: asks the detail of ``value`` with ``stats=None``.
    """
    with pytest.raises(ValueError, match="value"):
        cols.column_detail(
            typed_df,
            "value",
            T.NUMERIC,
            unique=39,
            stats=None,
            sample_rows=1000,
            outlier_limit=5,
            n_examples=3,
        )


@pytest.mark.parametrize(
    ("name", "table_of_its_type"),
    [
        ("value", lambda df, schema, base: cols.numeric_stats(df, ["value"], 1000)),
        ("city", lambda df, schema, base: cols.categorical_stats(df, ["city"])),
        ("when", lambda df, schema, base: cols.datetime_stats(df, ["when"], {}, NOW)),
        ("code", lambda df, schema, base: cols.text_stats(df, ["code"], schema, base, 1000)),
    ],
)
def test_inspect_column_gives_the_row_of_its_table(typed_df, typed_schema, name, table_of_its_type):
    """The inspector of a column shows the same row as the table of its type.

    Case: the inspector picks the wrong function for a type, or other arguments, and shows
    numbers that disagree with the table above it.
    Affects: core/columns.py (inspect_column)
    How: compares ``inspect_column`` with the table function of each type.
    """
    base = base_profile(typed_df)
    table, detail = cols.inspect_column(
        typed_df,
        name,
        typed_schema.type_of(name),
        base,
        sample_rows=1000,
        unparsed={},
        now=NOW,
        outlier_limit=5,
        n_examples=3,
    )
    pd.testing.assert_frame_equal(table, table_of_its_type(typed_df, typed_schema, base))
    assert detail.bars.counts


def test_inspect_column_counts_unparsed_values(typed_df):
    """The inspector passes on the unparsed values of its column.

    Case: the inspector drops ``unparsed`` and shows 0 while the table shows 2.
    Affects: core/columns.py (inspect_column)
    How: inspects ``value`` with 2 unparsed values.
    """
    table, _ = cols.inspect_column(
        typed_df,
        "value",
        T.NUMERIC,
        base_profile(typed_df),
        sample_rows=1000,
        unparsed={"value": 2},
        now=None,
        outlier_limit=5,
        n_examples=3,
    )
    assert table.loc[0, "unparsed"] == 2


def test_column_overview(typed_df, typed_schema):
    """One row per column in dataset order, with a mini distribution where it makes sense.

    Case: columns are reordered by type, or ids get a meaningless distribution.
    Affects: core/columns.py (column_overview)
    How: builds the overview and checks order, the empty distribution of ``code`` and the
    total of ``rating``.
    """
    numeric = cols.numeric_stats(typed_df, ["value", "rating"], 1000)
    base = base_profile(typed_df)
    table = cols.column_overview(typed_df, typed_schema, base, numeric, 1000).set_index("column")
    assert list(table.index) == list(typed_df.columns)
    assert "summary" not in table.columns
    assert table.loc["code", "distribution"] == []
    assert sum(table.loc["rating", "distribution"]) == 40


def test_column_overview_all_missing_column():
    """An all-missing numeric column has an empty distribution.

    Case: the histogram of no values raises and the whole overview fails.
    Affects: core/columns.py (column_overview)
    How: builds the overview of a column of two NaN.
    """
    df = pd.DataFrame({"empty": [np.nan, np.nan]})
    numeric = cols.numeric_stats(df, ["empty"], 1000)
    table = cols.column_overview(df, schema_of(empty=T.NUMERIC), base_profile(df), numeric, 1000)
    assert table.loc[0, "distribution"] == []
