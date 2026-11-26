"""Unit tests of ``core/visualize/saved.py``: the chart configuration saved as JSON, and how
one saved from another dataset fits the current one.
"""

import pytest

from litmetrics.core import visualize as viz

K = viz.ChartKind


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
    Affects: core/visualize/saved.py (config_file_name)
    How: names a bar chart configuration with each typed text.
    """
    assert viz.config_file_name(typed, K.BAR) == expected


def test_data_spec_round_trips_through_json():
    """A spec written as JSON reads back the same, filters included.

    Case: an enum, a tuple or a filter type is lost on the way and the loaded chart differs.
    Affects: core/visualize/saved.py (spec_to_dict, spec_from_dict)
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
    Affects: core/visualize/saved.py (spec_from_dict, _filter_from_dict)
    How: reads each malformed dict.
    """
    with pytest.raises(ValueError):
        viz.spec_from_dict(raw)


def test_fitting_a_configuration_names_what_this_dataset_lacks(chart_dataset):
    """Columns and filters this dataset cannot use are removed, each with a message.

    Case: a configuration from another dataset makes the page raise, or drops columns silently.
    Affects: core/visualize/saved.py (fit_to_dataset)
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
    Affects: core/visualize/saved.py (fit_to_dataset, ColumnHints.filter_type)
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
    Affects: core/visualize/saved.py (fit_to_dataset)
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
