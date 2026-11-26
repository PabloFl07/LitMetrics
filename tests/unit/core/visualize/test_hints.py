"""Unit tests of ``core/visualize/hints.py``: which columns a chart offers, and what the
filters can use.

Most tests run on ``chart_dataset`` (``samples.chart_frame`` typed by the Load pipeline), which
holds the columns the chart builder must tell apart: a row number, an all-missing column, free
text, a placeholder date and a correlated pair.
"""

from fixtures import samples

from litmetrics.config import VisualizeConfig
from litmetrics.core import visualize as viz
from litmetrics.core.io.base import LoadedDataset
from litmetrics.core.schema import prepare_dataset

K = viz.ChartKind


def test_hints_offer_columns_in_dataset_order_and_rank_only_for_the_first_chart(chart_dataset):
    """Selectors follow the dataset; the ranking only picks the starting chart.

    Case: selectors sorted by a heuristic hide where a person's columns are, or a category
    with 60 values (free text to the schema) is offered as an axis.
    Affects: core/visualize/hints.py (column_hints, available_kinds)
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
    Affects: core/visualize/hints.py (column_hints)
    How: checks ``empty`` (all NaN) is not numeric and is listed with its reason.
    """
    _, hints = chart_dataset
    assert "empty" not in hints.numeric
    assert ("empty", "no values") in hints.left_out


def test_row_counter_is_left_out_but_an_integer_measure_is_not(chart_dataset):
    """A 1..n integer is a row number even without "id" in its name; an income is a measure.

    Case: an integer measure whose values are almost all distinct (incomes) is taken for an
    id, or a row counter is offered as an axis.
    Affects: core/visualize/hints.py (column_hints, _is_row_number)
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
    Affects: core/visualize/hints.py (column_hints, _best_pair), config.py (VisualizeConfig)
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
    Affects: core/visualize/hints.py (column_hints, ColumnHints.filters, _filter_of_uncharted)
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
    Affects: core/visualize/hints.py (_filter_of_uncharted), config.py (VisualizeConfig)
    How: limits the list to 5 and compares it with the top 5 of ``city``.
    """
    df, _ = chart_dataset
    monkeypatch.setattr(VisualizeConfig, "filter_values", 5)
    hints = viz.column_hints(df, prepare_dataset(LoadedDataset(df, "c.csv", "CSV"))[1])
    top = df["city"].value_counts().index[:5]
    assert hints.values_of("city") == tuple(top)
