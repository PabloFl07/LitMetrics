"""Unit tests of ``core/visualize.py``: which columns a chart offers, the aggregated data of each
chart kind, row filters and the configuration saved as JSON.

Most tests run on ``chart_dataset`` (``samples.chart_frame`` typed by the Load pipeline), which
holds the columns the chart builder must tell apart: a row number, an all-missing column, free
text, a placeholder date and a correlated pair.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from fixtures import samples

from litmetrics.config import VisualizeConfig
from litmetrics.core import visualize as viz
from litmetrics.core.io.base import LoadedDataset
from litmetrics.core.schema import prepare_dataset

K = viz.ChartKind


# Column hints


def test_hints_offer_columns_in_dataset_order_and_rank_only_for_the_first_chart(chart_dataset):
    """Selectors follow the dataset; the ranking only picks the starting chart.

    Case: selectors sorted by a heuristic hide where a person's columns are, or a category
    with 60 values (free text to the schema) is offered as an axis.
    Affects: core/visualize.py (column_hints, available_kinds)
    How: reads the categorical, ranked and left-out columns of ``chart_frame``.
    """
    _, hints = chart_dataset
    assert hints.categorical == ("plan", "churn")
    assert hints.ranked[:2] == ("churn", "plan")
    assert ("city", "free text") in hints.left_out
    assert hints.datetime == ("date",)
    assert set(hints.best_pair[:2]) == {"a", "b"}
    assert set(viz.available_kinds(hints)) == set(K)


def test_all_missing_numeric_column_is_not_offered(chart_dataset):
    """A numeric column without values is left out as "no values".

    Case: ``min != max`` is True for NaN, so a column without values was offered.
    Affects: core/visualize.py (column_hints)
    How: checks ``empty`` (all NaN) is not numeric and is listed with its reason.
    """
    _, hints = chart_dataset
    assert "empty" not in hints.numeric
    assert ("empty", "no values") in hints.left_out


def test_row_counter_is_left_out_but_an_integer_measure_is_not(chart_dataset):
    """A 1..n integer is a row number even without "id" in its name; an income is a measure.

    Case: an integer measure whose values are almost all distinct (incomes) is taken for an
    id, or a row counter is offered as an axis.
    Affects: core/visualize.py (column_hints, _is_row_number)
    How: checks ``row`` and ``id`` are left out with their reason and ``income`` is numeric.
    """
    _, hints = chart_dataset
    assert ("row", "row number") in hints.left_out
    assert ("id", "identifier") in hints.left_out
    assert {"a", "b", "income"} <= set(hints.numeric)


def test_best_pair_compares_only_the_configured_columns(monkeypatch):
    """Only the first ``VisualizeConfig.pair_columns`` numeric columns are compared.

    Case: the correlation matrix of every numeric column (n²) on a wide dataset, or a
    literal limit that ignores the config.
    Affects: core/visualize.py (column_hints, _best_pair), config.py (VisualizeConfig)
    How: with ``income, a, b`` in that order, the pair is ``income, a`` when only two are
    compared and ``a, b`` (the correlated one) when three are.
    """
    frame = samples.chart_frame()[["income", "a", "b"]]
    dataset, schema = prepare_dataset(LoadedDataset(frame, "pair.csv", "CSV"))
    monkeypatch.setattr(VisualizeConfig, "pair_columns", 2)
    assert viz.column_hints(dataset.df, schema).best_pair[:2] == ("income", "a")
    monkeypatch.setattr(VisualizeConfig, "pair_columns", 3)
    assert set(viz.column_hints(dataset.df, schema).best_pair[:2]) == {"a", "b"}


def test_filters_offer_columns_the_chart_never_draws(chart_dataset):
    """Rows can be picked by any column with two values, also one that is never an axis.

    Case: a person cannot keep the rows of one city or of a range of ids because those
    columns are not offered for a chart, or an all-missing column gets a filter.
    Affects: core/visualize.py (column_hints, ColumnHints.filters, _filter_of_uncharted)
    How: reads the filter type of each column of ``chart_frame``.
    """
    _, hints = chart_dataset
    assert hints.filter_columns == (
        "id", "a", "b", "plan", "churn", "city", "date", "row", "income"
    )  # fmt: skip
    assert hints.filter_type("city") == "values" and "city" in hints.partial
    assert hints.filter_type("plan") == "values" and "plan" not in hints.partial
    assert hints.filter_type("row") == hints.filter_type("id") == "range"
    assert hints.range_of("row") == (1.0, 3_000.0)
    assert hints.filter_type("date") == "dates"
    assert hints.filter_type("empty") is None


def test_free_text_filter_lists_only_the_most_frequent_values(chart_dataset, monkeypatch):
    """A free text column lists at most ``VisualizeConfig.filter_values`` values, most
    frequent first.

    Case: an id or text column with millions of values sent whole to a multiselect.
    Affects: core/visualize.py (_filter_of_uncharted), config.py (VisualizeConfig)
    How: limits the list to 5 and compares it with the top 5 of ``city``.
    """
    df, _ = chart_dataset
    monkeypatch.setattr(VisualizeConfig, "filter_values", 5)
    hints = viz.column_hints(df, prepare_dataset(LoadedDataset(df, "c.csv", "CSV"))[1])
    top = df["city"].value_counts().index[:5]
    assert hints.values_of("city") == tuple(top)


# Chart data


@pytest.mark.parametrize("kind", list(K), ids=str)
def test_every_kind_draws_its_default_chart(chart_dataset, kind):
    """The first chart of every kind has data and a table behind it.

    Case: one kind raises or comes out empty on real columns (dates, missing values).
    Affects: core/visualize.py (default_spec, chart_data, to_frame)
    How: computes the default spec of each kind and checks series and table.
    """
    df, hints = chart_dataset
    data = viz.chart_data(df, viz.default_spec(kind, hints))
    assert data.series
    assert not viz.to_frame(data).empty


def test_bars_group_the_rest_in_other(chart_dataset):
    """Beyond ``top_k`` categories, the rest go in one "Other" bar, last, and no row is lost.

    Case: ``set_categories`` given a CategoricalIndex takes all its categories, so nothing is
    grouped (found in the PoC).
    Affects: core/visualize.py (_bars, _top_categories)
    How: bars of ``city`` (60 values) with ``top_k=10``.
    """
    df, _ = chart_dataset
    data = viz.chart_data(df, viz.DataSpec(K.BAR, x="city", top_k=10))
    assert len(data.categories) == 11
    assert data.categories[-1] == data.other == "Other (50 values)"
    assert sum(data.series[0].data) == len(df)


def test_bars_split_by_group_aggregate_the_value(chart_dataset):
    """A split bar chart has one series per group with the mean of the value.

    Case: the mean is taken over the wrong rows, or rows with missing values are dropped
    without saying so.
    Affects: core/visualize.py (_bars, _aggregate, _used)
    How: mean of ``a`` by ``plan`` split by ``churn``; ``a`` is about 50 and misses values.
    """
    df, _ = chart_dataset
    spec = viz.DataSpec(K.BAR, x="plan", value="a", agg=viz.Agg.MEAN, group="churn")
    data = viz.chart_data(df, spec)
    assert {s.name for s in data.series} == {"yes", "no"}
    assert all(40 < v < 60 for s in data.series for v in s.data)
    assert any("missing" in n for n in data.notes)


def test_line_trims_a_placeholder_date_and_coarsens_a_too_fine_unit(chart_dataset):
    """A far placeholder date is left out, and a unit with too many points is coarsened.

    Case: ``9999-12-31`` squashes eight centuries of empty buckets into the line (R-V2).
    Affects: core/visualize.py (_line, _range, _time_unit)
    How: a line of ``date`` with trimming, then by day without it, and reads notes and size.
    """
    df, _ = chart_dataset
    data = viz.chart_data(df, viz.DataSpec(K.LINE, x="date"))
    assert any("extreme dates" in n for n in data.notes)
    assert sum(v for _, v in data.series[0].data) == len(df) - 1
    daily = viz.chart_data(
        df, viz.DataSpec(K.LINE, x="date", time_unit=viz.TimeUnit.DAY, trim=False)
    )
    assert any("by year" in n for n in daily.notes)
    assert len(daily.series[0].data) <= VisualizeConfig.max_line_points


def test_scatter_is_sampled_but_its_trend_uses_every_point(chart_dataset, monkeypatch):
    """Above ``max_points`` a sample is drawn, but the correlation is of every point.

    Case: the whole column reaches the browser, or r is computed on the sample.
    Affects: core/visualize.py (_scatter), config.py (VisualizeConfig.max_points)
    How: lowers the limit to 500 and checks the elements drawn and the note.
    """
    df, _ = chart_dataset
    monkeypatch.setattr(VisualizeConfig, "max_points", 500)
    data = viz.chart_data(df, viz.DataSpec(K.SCATTER, x="a", y="b", group="plan"))
    assert data.elements == 500
    assert data.r is not None and data.r > 0.9
    assert data.notes[0].startswith("500 of ")


def test_scatter_trend_matches_numpy(chart_dataset):
    """The closed-form least squares line and r agree with numpy.

    Case: a sign or centering slip in the one-pass formula draws a wrong trend line.
    Affects: core/visualize.py (_least_squares)
    How: compares slope, intercept and r with ``np.polyfit`` and ``np.corrcoef``.
    """
    df, hints = chart_dataset
    x, y, _ = hints.best_pair
    data = viz.chart_data(df, viz.DataSpec(K.SCATTER, x=x, y=y))
    sub = df[[x, y]].dropna().to_numpy(dtype="float64")
    slope, intercept = np.polyfit(sub[:, 0], sub[:, 1], 1)
    assert data.trend == pytest.approx((slope, intercept))
    assert data.r == pytest.approx(np.corrcoef(sub[:, 0], sub[:, 1])[0, 1])


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(viz.DataSpec(K.SCATTER, x="a", y="a"), id="scatter-axes"),
        pytest.param(viz.DataSpec(K.HEATMAP, x="plan", y="plan"), id="heatmap-rows-columns"),
    ],
)
def test_a_column_cannot_be_used_twice(chart_dataset, spec):
    """A chart of a column against itself is refused with a message.

    Case: a loaded configuration puts one column in two roles and the chart is meaningless.
    Affects: core/visualize.py (chart_data)
    How: asks for each chart and expects ``ValueError``.
    """
    df, _ = chart_dataset
    with pytest.raises(ValueError, match="used twice"):
        viz.chart_data(df, spec)


def test_bars_on_intervals_of_a_numeric_x_agree_with_the_histogram(chart_dataset):
    """A numeric x is cut in the same intervals as its histogram.

    Case: bars of counts on intervals and the histogram of the same column disagree, or empty
    intervals are dropped and the axis loses its scale.
    Affects: core/visualize.py (_bars, _intervals, _histogram)
    How: mean of ``b`` by 8 intervals of ``a`` (increasing), and the counts against the histogram.
    """
    df, _ = chart_dataset
    spec = viz.DataSpec(K.BAR, x="a", value="b", agg=viz.Agg.MEAN, bins=8, trim=False)
    data = viz.chart_data(df, spec)
    hist = viz.chart_data(df, viz.DataSpec(K.HISTOGRAM, x="a", bins=8, trim=False))
    assert data.categories == hist.categories
    means = [v for v in data.series[0].data if v is not None]
    assert means == sorted(means)
    counts = viz.chart_data(df, replace(spec, value=None, agg=viz.Agg.COUNT))
    assert counts.series[0].data == hist.series[0].data


def test_line_on_a_numeric_x_has_one_point_per_interval(chart_dataset):
    """A line over a number has one point per interval, not a time axis.

    Case: a numeric x is treated as epoch dates.
    Affects: core/visualize.py (_line, _bars, to_frame)
    How: mean of ``b`` over 10 intervals of ``a``.
    """
    df, _ = chart_dataset
    data = viz.chart_data(df, viz.DataSpec(K.LINE, x="a", value="b", agg=viz.Agg.MEAN, bins=10))
    assert not data.time_axis
    assert len(data.categories) == 10
    assert len(viz.to_frame(data)) == 10


def _thin_frame() -> pd.DataFrame:
    """50 rows of group ``a`` at x = 0 and 2 rows of group ``b`` at x = 10: with 2 intervals,
    the second one holds only the 2 rows of ``b``.
    """
    return pd.DataFrame(
        {"x": [0] * 50 + [10] * 2, "y": [100.0] * 50 + [0.0] * 2, "g": ["a"] * 50 + ["b"] * 2}
    )


def test_intervals_with_too_few_rows_get_no_mean():
    """A mean of an interval with fewer than ``min_rows`` rows is not drawn, with a note.

    Case: in California Housing filtered below its cap, the top income intervals hold 1 to 3
    houses each and their means plunge, drawn as a real drop of house values.
    Affects: core/visualize.py (_bars, _aggregate, _thin_note)
    How: line of the mean of ``y`` by 2 intervals of ``x``, split by ``g``, with ``min_rows`` 5.
    """
    spec = viz.DataSpec(
        K.LINE, x="x", value="y", group="g", agg=viz.Agg.MEAN, bins=2, trim=False, min_rows=5
    )
    data = viz.chart_data(_thin_frame(), spec)
    # Group b had only the thin interval: no line left, no legend entry.
    assert [(s.name, s.data) for s in data.series] == [("a", (100, None))]
    assert any("1 point with fewer than 5 rows" in n for n in data.notes)


@pytest.mark.parametrize(
    ("agg", "min_rows"),
    [
        pytest.param(viz.Agg.COUNT, 5, id="count"),
        pytest.param(viz.Agg.SUM, 5, id="sum"),
        pytest.param(viz.Agg.MEAN, 1, id="mean-min-1"),
    ],
)
def test_counts_sums_and_a_minimum_of_one_draw_every_interval(agg, min_rows):
    """Counts and sums of a thin interval are drawn, and so is a mean when ``min_rows`` is 1.

    Case: the minimum hides small counts (2 rows is a true count), or cannot be turned off.
    Affects: core/visualize.py (_bars, _aggregate)
    How: the chart of ``_thin_frame`` with each aggregation, and a mean with ``min_rows`` 1.
    """
    spec = viz.DataSpec(
        K.LINE, x="x", value="y", group="g", agg=agg, bins=2, trim=False, min_rows=min_rows
    )
    data = viz.chart_data(_thin_frame(), spec)
    assert [s.name for s in data.series] == ["a", "b"]
    assert not any("fewer than" in n for n in data.notes)


@pytest.mark.parametrize(
    ("values", "first", "last"),
    [
        pytest.param([0.4999, 15.0], "0.50 to 1.35", "14.15 to 15.00", id="decimals"),
        pytest.param([0, 170], "0 to 10", "160 to 170", id="whole-numbers"),
        pytest.param([0, 1.7e6], "0 to 100,000", "1,600,000 to 1,700,000", id="thousands"),
    ],
)
def test_interval_labels_have_the_same_decimals(values, first, last):
    """Every end of the interval labels of an axis has the same decimals.

    Case: significant digits gave "0.4999 to 1.353" next to "14.15 to 15": labels of uneven
    length and precision along one axis.
    Affects: core/visualize.py (_interval_labels, _histogram)
    How: 17 bins between two values, reading the first and the last label.
    """
    df = pd.DataFrame({"x": values})
    data = viz.chart_data(df, viz.DataSpec(K.HISTOGRAM, x="x", bins=17, trim=False))
    assert (data.categories[0], data.categories[-1]) == (first, last)


# Filters


def test_filters_are_part_of_the_cache_key():
    """Two specs that differ only in their filters are two cache entries (R-V8).

    Case: a filter is left out of the hash and the cache returns the unfiltered chart.
    Affects: core/visualize.py (DataSpec, ValuesFilter)
    How: compares equality and hash of specs with and without the same filter.
    """
    plain = viz.DataSpec(K.BAR, x="plan")
    filtered = replace(plain, filters=(viz.ValuesFilter("churn", ("yes",)),))
    assert plain != filtered
    assert hash(replace(plain, filters=(viz.ValuesFilter("churn", ("yes",)),))) == hash(filtered)


@pytest.mark.parametrize(
    ("row_filter", "kept"),
    [
        pytest.param(
            viz.ValuesFilter("churn", ("yes",)), lambda df: df["churn"] == "yes", id="values"
        ),
        pytest.param(
            viz.RangeFilter("income", 30_000, 50_000),
            lambda df: df["income"].between(30_000, 50_000),
            id="range",
        ),
        pytest.param(
            viz.DateFilter("date", "2020-01-10", "2020-01-31"),
            lambda df: df["date"].between("2020-01-10", "2020-01-31 23:59:59"),
            id="dates-whole-last-day",
        ),
        pytest.param(
            viz.DateFilter("date", None, "2020-01-31"),
            lambda df: df["date"] <= "2020-01-31 23:59:59",
            id="dates-open-start",
        ),
        pytest.param(
            viz.ValuesFilter("city", ("city 3", "city 7", "not a city")),
            lambda df: df["city"].isin(["city 3", "city 7"]),
            id="values-of-free-text",
        ),
        pytest.param(
            viz.RangeFilter("row", None, 1_000),
            lambda df: df["row"] <= 1_000,
            id="range-of-row-number",
        ),
    ],
)
def test_filters_keep_the_rows_and_say_how_many_left(chart_dataset, row_filter, kept):
    """Each filter keeps exactly its rows, and the note counts the others.

    Case: the last day of a date range is cut at midnight, or the note miscounts.
    Affects: core/visualize.py (_filter_mask, _used)
    How: counts the rows of a bar chart of ``plan`` against a pandas mask of the same filter.
    """
    df, _ = chart_dataset
    data = viz.chart_data(df, viz.DataSpec(K.BAR, x="plan", filters=(row_filter,)))
    expected = int(kept(df).sum())
    assert sum(data.series[0].data) == expected
    assert f"{len(df) - expected:,} of {len(df):,} rows left out by the filters." in data.notes


def test_a_filter_that_leaves_no_rows_draws_nothing(chart_dataset):
    """No row left is an empty chart with a note, not an error.

    Case: an empty selection reaches ``groupby`` or ``min`` and raises.
    Affects: core/visualize.py (chart_data, _bars, _histogram)
    How: keeps a value ``plan`` never has, for a bar chart and a histogram.
    """
    df, _ = chart_dataset
    nothing = (viz.ValuesFilter("plan", ("gold",)),)
    for spec in (
        viz.DataSpec(K.BAR, x="churn", filters=nothing),
        viz.DataSpec(K.HISTOGRAM, x="a", filters=nothing),
    ):
        data = viz.chart_data(df, spec)
        assert not data.series
        assert any("left out by the filters" in n for n in data.notes)


def test_filter_on_a_column_of_the_wrong_type_is_an_error(chart_dataset):
    """A date filter on a number is refused with a message.

    Case: a configuration saved from another dataset compares numbers with dates.
    Affects: core/visualize.py (_check_filter)
    How: filters ``a`` by dates and expects ``ValueError``.
    """
    df, _ = chart_dataset
    spec = viz.DataSpec(K.BAR, x="plan", filters=(viz.DateFilter("a", "2020-01-01"),))
    with pytest.raises(ValueError, match="no dates"):
        viz.chart_data(df, spec)


# Configuration as JSON


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        pytest.param("", "chart_bar.json", id="empty"),
        pytest.param("  ..  ", "chart_bar.json", id="only-dots"),
        pytest.param("sales 2024", "sales 2024.json", id="plain"),
        pytest.param("sales.JSON", "sales.json", id="extension-typed"),
        pytest.param("a/b:c", "a_b_c.json", id="unsafe-characters"),
    ],
)
def test_config_file_name_is_safe_and_ends_in_json(typed, expected):
    """The name typed for a saved configuration becomes a file name every system accepts.

    Case: ``sales.json.json``, an empty ``.json``, or a ``/`` the browser turns into a folder.
    Affects: core/visualize.py (config_file_name)
    How: names a bar chart configuration with each typed text.
    """
    assert viz.config_file_name(typed, K.BAR) == expected


def test_data_spec_round_trips_through_json():
    """A spec written as JSON reads back the same, filters included.

    Case: an enum, a tuple or a filter type is lost on the way and the loaded chart differs.
    Affects: core/visualize.py (spec_to_dict, spec_from_dict)
    How: writes a spec with every kind of filter and reads it back.
    """
    spec = viz.DataSpec(
        K.BAR,
        x="a",
        value="b",
        group="plan",
        agg=viz.Agg.MEDIAN,
        bins=12,
        trim=False,
        min_rows=3,
        filters=(
            viz.DateFilter("date", "2020-01-01", None),
            viz.ValuesFilter("churn", ("yes", "no")),
            viz.RangeFilter("income", 1.5, None),
        ),
    )
    assert viz.spec_from_dict(viz.spec_to_dict(spec)) == spec


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param({"kind": "radar"}, id="unknown-kind"),
        pytest.param({"kind": "bar", "bins": "ten"}, id="bins-text"),
        pytest.param({"kind": "bar", "trim": 1}, id="trim-not-bool"),
        pytest.param(
            {"kind": "bar", "filters": [{"type": "dates", "column": "d", "start": "x"}]},
            id="bad-date",
        ),
        pytest.param(
            {"kind": "bar", "filters": [{"type": "near", "column": "d"}]}, id="bad-filter"
        ),
        pytest.param({"x": "a"}, id="no-kind"),
    ],
)
def test_malformed_configuration_is_rejected(raw):
    """A configuration with a wrong value raises ``ValueError``, never something else.

    Case: a hand-edited file raises ``TypeError`` or ``KeyError``, which the page does not
    catch, and the page breaks.
    Affects: core/visualize.py (spec_from_dict, _filter_from_dict)
    How: reads each malformed dict.
    """
    with pytest.raises(ValueError):
        viz.spec_from_dict(raw)


def test_fitting_a_configuration_names_what_this_dataset_lacks(chart_dataset):
    """Columns and filters this dataset cannot use are removed, each with a message.

    Case: a configuration from another dataset makes the page raise, or drops columns silently.
    Affects: core/visualize.py (fit_to_dataset)
    How: a scatter with a missing y, a stray ``value`` and two filters that cannot apply.
    """
    df, hints = chart_dataset
    spec = viz.DataSpec(
        K.SCATTER,
        x="a",
        y="price",
        group="plan",
        value="b",
        filters=(viz.ValuesFilter("region", ("north",)), viz.RangeFilter("plan", 1, 2)),
    )
    fitted, messages = viz.fit_to_dataset(spec, tuple(df.columns), hints)
    assert (fitted.x, fitted.y, fitted.group, fitted.value) == ("a", None, "plan", None)
    assert fitted.filters == ()
    assert len(messages) == 3
    assert any("'price'" in m and "no column" in m for m in messages)
    assert any("'region'" in m for m in messages)
    assert any("'plan'" in m and "cannot be filtered" in m for m in messages)


def test_fitting_keeps_a_filter_on_a_column_the_chart_never_draws(chart_dataset):
    """A saved filter on free text or a row number is kept: those columns can filter.

    Case: a loaded configuration drops a filter on ``city`` because ``city`` is no axis.
    Affects: core/visualize.py (fit_to_dataset, ColumnHints.filter_type)
    How: fits a bar chart filtered by ``city`` values and a ``row`` range.
    """
    df, hints = chart_dataset
    filters = (viz.ValuesFilter("city", ("city 3",)), viz.RangeFilter("row", 1, 10))
    spec = viz.DataSpec(K.BAR, x="plan", filters=filters)
    fitted, messages = viz.fit_to_dataset(spec, tuple(df.columns), hints)
    assert fitted.filters == filters and messages == ()


def test_fitting_says_a_column_is_skipped_not_missing(chart_dataset):
    """A skipped column exists: the message says it is skipped.

    Case: the warning tells the person the dataset lacks a column they can see in the table.
    Affects: core/visualize.py (fit_to_dataset)
    How: fits a bar chart and a filter on ``plan`` with ``plan`` skipped.
    """
    df, hints = chart_dataset
    spec = viz.DataSpec(K.BAR, x="plan", filters=(viz.ValuesFilter("plan", ("basic",)),))
    fitted, messages = viz.fit_to_dataset(spec, tuple(df.columns), hints, skipped=("plan",))
    assert fitted.x is None and fitted.filters == ()
    assert messages == (
        "X axis: 'plan' is skipped (Skip columns).",
        "Filter: 'plan' is skipped (Skip columns).",
    )
