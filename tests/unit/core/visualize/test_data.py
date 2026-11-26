"""Unit tests of ``core/visualize/data.py``: the aggregated data of each chart kind and the
row filters.

Most tests run on ``chart_dataset`` (``samples.chart_frame`` typed by the Load pipeline), which
holds the columns the chart builder must tell apart: a row number, an all-missing column, free
text, a placeholder date and a correlated pair.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from litmetrics.config import VisualizeConfig
from litmetrics.core import visualize as viz

K = viz.ChartKind


@pytest.mark.parametrize("kind", list(K), ids=str)
def test_every_kind_draws_its_default_chart(chart_dataset, kind):
    """The first chart of every kind has data and a table behind it.

    Case: one kind raises or comes out empty on real columns (dates, missing values).
    Affects: core/visualize/hints.py (default_spec), core/visualize/data.py (chart_data, to_frame)
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
    Affects: core/visualize/data.py (_bars, _top_categories)
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
    Affects: core/visualize/data.py (_bars, _aggregate, _used)
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
    Affects: core/visualize/data.py (_line, _range, _time_unit)
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
    Affects: core/visualize/data.py (_scatter), config.py (VisualizeConfig.max_points)
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
    Affects: core/visualize/data.py (_least_squares)
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
    Affects: core/visualize/data.py (chart_data)
    How: asks for each chart and expects ``ValueError``.
    """
    df, _ = chart_dataset
    with pytest.raises(ValueError, match="used twice"):
        viz.chart_data(df, spec)


def test_bars_on_intervals_of_a_numeric_x_agree_with_the_histogram(chart_dataset):
    """A numeric x is cut in the same intervals as its histogram.

    Case: bars of counts on intervals and the histogram of the same column disagree, or empty
    intervals are dropped and the axis loses its scale.
    Affects: core/visualize/data.py (_bars, _intervals, _histogram)
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
    Affects: core/visualize/data.py (_line, _bars, to_frame)
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
    Affects: core/visualize/data.py (_bars, _aggregate, _thin_note)
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
    Affects: core/visualize/data.py (_bars, _aggregate)
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
    Affects: core/visualize/data.py (_interval_labels, _histogram)
    How: 17 bins between two values, reading the first and the last label.
    """
    df = pd.DataFrame({"x": values})
    data = viz.chart_data(df, viz.DataSpec(K.HISTOGRAM, x="x", bins=17, trim=False))
    assert (data.categories[0], data.categories[-1]) == (first, last)


# Filters


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
    Affects: core/visualize/data.py (_filter_mask, _used)
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
    Affects: core/visualize/data.py (chart_data, _bars, _histogram)
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
    Affects: core/visualize/data.py (_check_filter)
    How: filters ``a`` by dates and expects ``ValueError``.
    """
    df, _ = chart_dataset
    spec = viz.DataSpec(K.BAR, x="plan", filters=(viz.DateFilter("a", "2020-01-01"),))
    with pytest.raises(ValueError, match="no dates"):
        viz.chart_data(df, spec)
