"""Integration tests of the Visualization page through AppTest: the editor with its real state
(``ui/state.py``), cache (``ui/cache.py``), data (``core/visualize/``) and look
(``ui/chart_style.py``), on ``samples.chart_frame``.

AppTest does not run the chart's JavaScript: what is checked is the editor, the notes and the
table behind the chart. Elements of an earlier run are stale after ``.run()``, so they are
looked up again by key every time.
"""

import json

from fixtures.app_scripts import TIMEOUT, captions, errors, run_visualize
from streamlit.testing.v1 import AppTest

from litmetrics.core.visualize import ChartKind

K = ChartKind
KIND = "w_viz_kind"


def _start() -> AppTest:
    """The page drawn once on ``chart_frame``, failing on any exception."""
    at = AppTest.from_function(run_visualize, default_timeout=TIMEOUT).run()
    assert not at.exception, errors(at)
    return at


def _pick(at: AppTest, kind: ChartKind) -> AppTest:
    """Pick chart ``kind`` and rerun, failing on any exception."""
    at.button_group(key=KIND).set_value(kind).run()
    assert not at.exception, (kind, errors(at))
    return at


def test_every_kind_renders():
    """Every chart kind draws from the picker, with its own editor.

    Case: one kind raises in the page (an option its widgets do not have, a stale column).
    Affects: ui/pages/visualize.py, ui/cache.py (chart_data), core/visualize/,
    ui/chart_style.py
    How: picks each kind in turn and checks it stays picked.
    """
    at = _start()
    for kind in K:
        assert _pick(at, kind).button_group(key=KIND).value == kind


def test_picking_a_used_column_swaps_the_axes():
    """Picking Y's column as X swaps the axes instead of drawing a column against itself.

    Case: the chart shows a "used twice" error after a normal click.
    Affects: ui/pages/visualize.py (_swap_repeated, _data_controls)
    How: sets the scatter X to the column of Y.
    """
    at = _pick(_start(), K.SCATTER)
    x, y = at.selectbox(key="w_viz_scatter_x").value, at.selectbox(key="w_viz_scatter_y").value
    assert x != y
    at.selectbox(key="w_viz_scatter_x").set_value(y).run()
    assert not at.exception, errors(at)
    assert at.selectbox(key="w_viz_scatter_x").value == y
    assert at.selectbox(key="w_viz_scatter_y").value == x


def test_data_table_is_built_only_when_open():
    """The table behind the chart is drawn only with its expander open, without timings.

    Case: a closed expander still sends a 10,000 row table and CSV on every click.
    Affects: ui/pages/visualize.py (_chart, _data_table), config.py (show_timings)
    How: checks no dataframe is drawn, opens the expander through its key and checks again.
    """
    at = _pick(_start(), K.SCATTER)
    assert not at.dataframe
    at.session_state["viz_data_open"] = True
    at.run()
    assert not at.exception, errors(at)
    assert at.dataframe
    assert " ms" not in captions(at)


def test_range_filter_takes_exact_ends_typed_in():
    """A range can be typed exactly in "From" / "To", and the slider and the inputs follow
    each other; ends typed the wrong way round are swapped.

    Case: the slider alone cannot reach 41.237, the inputs and the slider disagree on the
    range the chart uses, or "From" above "To" leaves no row.
    Affects: ui/pages/visualize.py (_range_filter, _range_scale, _range_from_slider,
    _range_from_inputs)
    How: filters ``a`` (decimals) typing both ends, then moves the slider; filters ``income``
    (integers) and checks its inputs hold integers.
    """
    at = _pick(_start(), K.BAR)
    at.button(key="viz_filter_add").click().run()
    at.selectbox(key="w_viz_filter_0_column").set_value("a").run()
    at.number_input(key="w_viz_filter_0_value_low").set_value(41.237).run()
    at.number_input(key="w_viz_filter_0_value_high").set_value(30.5).run()
    assert not at.exception, errors(at)
    assert at.slider(key="w_viz_filter_0_value").value == (30.5, 41.237)
    assert "left out by the filters" in captions(at)
    at.slider(key="w_viz_filter_0_value").set_value((40.0, 50.0)).run()
    assert at.number_input(key="w_viz_filter_0_value_low").value == 40.0
    assert at.number_input(key="w_viz_filter_0_value_high").value == 50.0
    at.selectbox(key="w_viz_filter_0_column").set_value("income").run()
    assert not at.exception, errors(at)
    ends = (
        at.number_input(key="w_viz_filter_0_value_low"),
        at.number_input(key="w_viz_filter_0_value_high"),
    )
    assert all(isinstance(e.value, int) for e in ends)
    assert ends[0].value < 40_000 < ends[1].value


def test_unfiltered_download_is_offered_only_with_filters():
    """With filters, the data behind the chart can also be downloaded without them, while
    the table keeps showing the filtered rows.

    Case: after "N rows left out by the filters" there is no way to get the whole data, or
    the extra button makes the table drop the filters.
    Affects: ui/pages/visualize.py (_data_table, _filters)
    How: opens the table of a bar chart of ``plan``, then adds a ``churn`` filter and reads
    the buttons and the counts of the table.
    """
    at = _pick(_start(), K.BAR)
    at.session_state["viz_data_open"] = True
    at.run()
    # The first download button is "Save configuration", in the Save / load popover.
    assert [b.label for b in at.download_button][1:] == ["Download CSV"]
    at.button(key="viz_filter_add").click().run()
    at.selectbox(key="w_viz_filter_0_column").set_value("churn").run()
    at.multiselect(key="w_viz_filter_0_value").set_value(["yes"]).run()
    assert not at.exception, errors(at)
    assert [b.label for b in at.download_button][1:] == ["Download CSV", "Download unfiltered"]
    table = at.dataframe[0].value
    assert table.select_dtypes("number").sum().sum() < 3_000


def test_style_toggles_follow_the_kind():
    """Scatter has no "Value labels" toggle; bars do.

    Case: a toggle that does nothing for a kind is shown.
    Affects: ui/pages/visualize.py (_style_controls, _STYLE_KEYS; ui/chart_style.py STYLE_KINDS)
    How: looks for the labels toggle with a scatter and with bars.
    """
    at = _start()
    for kind, shown in [(K.SCATTER, False), (K.BAR, True)]:
        _pick(at, kind)
        assert any(t.key == "viz_labels" for t in at.toggle) is shown, kind


def test_switching_kinds_keeps_the_options_of_each():
    """The options of a kind come back after drawing another kind.

    Case: Streamlit drops the state of widgets not drawn, so filled lines, bins and donut
    were reset by visiting another kind.
    Affects: ui/pages/visualize.py, ui/state.py (kept_widget, restore_widget)
    How: changes an option of line, histogram and pie, then goes back to each.
    """
    at = _pick(_start(), K.LINE)
    at.toggle(key="viz_area").set_value(True).run()
    _pick(at, K.HISTOGRAM).slider(key="w_viz_histogram_bins").set_value(55).run()
    _pick(at, K.PIE).toggle(key="viz_donut").set_value(False).run()
    assert not any(t.key == "viz_area" for t in at.toggle)
    assert _pick(at, K.LINE).toggle(key="viz_area").value is True
    assert _pick(at, K.HISTOGRAM).slider(key="w_viz_histogram_bins").value == 55
    assert _pick(at, K.PIE).toggle(key="viz_donut").value is False


def test_reset_forgets_the_options_of_the_current_kind_only():
    """Reset brings the current kind back to its start and leaves the others alone.

    Case: Reset clears every kind, or misses options remembered outside the widget state.
    Affects: ui/pages/visualize.py (_reset, _kind_keys), ui/state.py (forget_widgets)
    How: changes histogram bins and the line area, resets on line, and reads both.
    """
    at = _pick(_start(), K.HISTOGRAM)
    at.slider(key="w_viz_histogram_bins").set_value(55).run()
    _pick(at, K.LINE).toggle(key="viz_area").set_value(True).run()
    at.button(key="viz_reset").click().run()
    assert not at.exception, errors(at)
    assert at.toggle(key="viz_area").value is False
    assert _pick(at, K.HISTOGRAM).slider(key="w_viz_histogram_bins").value == 55


def test_means_on_intervals_take_a_minimum_of_rows_and_never_stack():
    """A line of means on intervals offers "Min rows per interval" and no "Stack series".

    Case: series of means were stacked into sums no row has, and thin intervals could not be
    shown again once hidden.
    Affects: ui/pages/visualize.py (_data_controls, _style_controls)
    How: line of ``b`` on intervals of ``a`` split by ``plan``, with Mean and then Sum.
    """
    at = _pick(_start(), K.LINE)
    at.selectbox(key="w_viz_line_x").set_value("a").run()
    at.selectbox(key="w_viz_line_value").set_value("b").run()
    at.selectbox(key="w_viz_line_group").set_value("plan").run()
    at.button_group(key="w_viz_line_agg").set_value("mean").run()
    assert not at.exception, errors(at)
    assert at.number_input(key="w_viz_line_min_rows").value == 5
    assert not any(t.key == "viz_stack" for t in at.toggle)
    at.button_group(key="w_viz_line_agg").set_value("sum").run()
    assert not at.exception, errors(at)
    assert not any(n.key == "w_viz_line_min_rows" for n in at.number_input)
    assert any(t.key == "viz_stack" for t in at.toggle)


def test_color_picker_only_for_one_series():
    """A single color is offered only while the chart has one series.

    Case: a color picked for one series silently recolors the first group of a split chart.
    Affects: ui/pages/visualize.py (_style_controls)
    How: bars without and with "Split by".
    """
    at = _pick(_start(), K.BAR)
    assert any(c.key == "viz_color" for c in at.color_picker)
    at.selectbox(key="w_viz_bar_group").set_value("plan").run()
    assert not at.exception, errors(at)
    assert not any(c.key == "viz_color" for c in at.color_picker)


def test_row_numbers_and_ids_are_not_offered_as_axes():
    """Row numbers and ids never become an axis, a value or a split.

    Case: ``row`` / ``id`` become axes, now that the filters list them.
    Affects: ui/pages/visualize.py (_data_controls), core/visualize/ (column_hints)
    How: checks every column selector of the bar chart editor.
    """
    at = _pick(_start(), K.BAR)
    for selector in at.selectbox:
        if selector.key.startswith("w_viz_bar_"):
            assert "row" not in selector.options and "id" not in selector.options


def test_filters_are_cards_on_any_column_and_can_be_removed():
    """Several filters, each its own card, keep rows by columns the chart does not draw.

    Case: only one filter of each type fits, columns that are no axis (free text, row
    numbers) cannot filter, or removing one filter drops the others.
    Affects: ui/pages/visualize.py (_filters, _filter_card, _add_filter, _remove_filter),
    ui/state.py (chart filters), core/visualize/ (column_hints, _filter_mask)
    How: adds a filter on ``city`` values and one on a ``row`` range, then removes the first.
    """
    at = _pick(_start(), K.BAR)
    at.button(key="viz_filter_add").click().run()
    at.button(key="viz_filter_add").click().run()
    assert not at.exception, errors(at)
    at.selectbox(key="w_viz_filter_0_column").set_value("city").run()
    at.multiselect(key="w_viz_filter_0_value").set_value(["city 3"]).run()
    at.selectbox(key="w_viz_filter_1_column").set_value("row").run()
    at.slider(key="w_viz_filter_1_value").set_value((1.0, 1_000.0)).run()
    assert not at.exception, errors(at)
    assert "left out by the filters" in captions(at)
    at.button(key="viz_filter_remove_0").click().run()
    assert not at.exception, errors(at)
    assert [s.key for s in at.selectbox if "filter" in s.key] == ["w_viz_filter_1_column"]
    assert at.slider(key="w_viz_filter_1_value").value == (1.0, 1_000.0)


def test_skipped_columns_are_not_offered():
    """A column skipped with "Skip columns" (shared with Profile) leaves every selector.

    Case: the page keeps offering a column the user skipped in Profile.
    Affects: ui/skip.py, ui/pages/visualize.py, ui/state.py (skipped columns), ui/cache.py
    How: skips ``plan`` and checks the selectors and the "Not offered" caption.
    """
    at = _pick(_start(), K.BAR)
    assert "plan" in at.selectbox(key="w_viz_bar_x").options
    at.multiselect(key="w_skipped_columns").set_value(["plan"]).run()
    assert not at.exception, errors(at)
    assert all("plan" not in s.options for s in at.selectbox)
    assert "plan (skipped)" in captions(at)


def test_loading_a_configuration_sets_the_editor():
    """A saved configuration loaded from a file sets the kind, the columns and the style.

    Case: the callback sets a widget after it was drawn (Streamlit raises), or the uploader
    keeps the file and loads it again on every run.
    Affects: ui/pages/visualize.py (_load_config, _apply_config), ui/state.py (chart config)
    How: uploads a heatmap configuration with an unknown column and two filters (one card
    each), and reads the editor.
    """
    at = _start()
    config = {
        "version": 1,
        "data": {
            "kind": "heatmap",
            "x": "churn",
            "y": "plan",
            "value": "price",
            "filters": [
                {"type": "values", "column": "city", "values": ["city 3"]},
                {"type": "range", "column": "row", "low": 1, "high": 10},
            ],
        },
        "style": {"title": "Mine", "size": "large"},
    }
    at.file_uploader(key="viz_config_file_0").upload(
        "chart.json", json.dumps(config).encode(), "application/json"
    ).run()
    assert not at.exception, errors(at)
    assert at.button_group(key=KIND).value == K.HEATMAP
    assert at.selectbox(key="w_viz_heatmap_x").value == "churn"
    assert at.text_input(key="viz_title").value == "Mine"
    assert at.multiselect(key="w_viz_filter_0_value").value == ["city 3"]
    assert at.slider(key="w_viz_filter_1_value").value == (1.0, 10.0)
    assert any("'price'" in w.value for w in at.warning)


def test_a_new_dataset_forgets_column_choices_but_keeps_the_look():
    """Loading another dataset drops the remembered columns and keeps style choices.

    Case: a column remembered from the previous dataset comes back on another one that
    happens to have it, or the chosen look is lost with every file.
    Affects: ui/state.py (_reset_dependent_state, widget prefs), ui/pages/visualize.py
    How: picks 55 histogram bins and filled lines, loads another dataset, and reads both
    (the chart kind, a column choice too, starts over).
    """
    at = _pick(_start(), K.HISTOGRAM)
    at.slider(key="w_viz_histogram_bins").set_value(55).run()
    _pick(at, K.LINE).toggle(key="viz_area").set_value(True).run()
    at.session_state["other_dataset"] = True
    at.run()
    assert not at.exception, errors(at)
    assert at.button_group(key=KIND).value == K.BAR
    assert _pick(at, K.LINE).toggle(key="viz_area").value is True
    assert _pick(at, K.HISTOGRAM).slider(key="w_viz_histogram_bins").value == 30
