"""Unit tests of ``ui/chart_style.py``: the look of the Visualization charts (plain dicts, no
Streamlit).

AppTest does not run the chart's JavaScript, so a broken option is only seen in a browser.
These tests check that every style gives valid JSON and the decisions taken from the data
(heatmap scale, animation, label order), and the style saved as JSON.
"""

import json
from dataclasses import replace

import pytest
from fixtures.samples import CHART_COLORS as COLORS

from litmetrics.core import visualize as viz
from litmetrics.ui import chart_style as look

STYLE = look.StyleSpec(palette=("#2E5FD3", "#EA580C", "#059669"))
K = viz.ChartKind


def _option(data: viz.ChartData, style: look.StyleSpec = STYLE) -> dict:
    """The option of ``data`` drawn with ``style``, checked to be strict JSON (no NaN). ``Js``
    formatters are written as their code, as ``JsCode`` sends them.
    """
    option = look.build_option(data, style, COLORS)
    json.dumps(option, allow_nan=False, default=lambda o: o.code)
    return option


@pytest.mark.parametrize("kind", list(K), ids=str)
def test_every_style_gives_valid_json_for_every_kind(chart_dataset, kind):
    """Every style option builds valid JSON from the same data (R-V1).

    Case: a style (log scale, horizontal, stack...) leaves NaN, numpy values or ``None``
    keys in the option, and the chart is blank in the browser.
    Affects: ui/chart_style.py (build_option and the builder of each kind)
    How: builds the default chart of each kind with four styles that turn every option on.
    """
    df, hints = chart_dataset
    data = viz.chart_data(df, viz.default_spec(kind, hints))
    for style in (
        look.StyleSpec(horizontal=True, labels=True, log_scale=True, size="large"),
        look.StyleSpec(stack=True, legend=False, sort="label"),
        look.StyleSpec(area=True, connect_gaps=True, donut=False, title="T", size="small"),
        replace(STYLE, color="#123456", sort="desc"),
    ):
        _option(data, style)


@pytest.mark.parametrize(
    ("color_each", "expected"),
    [
        pytest.param(False, ["#2E5FD3"] * 3, id="one-color"),
        pytest.param(True, ["#2E5FD3", "#EA580C", "#059669"], id="one-color-per-box"),
    ],
)
def test_boxplot_colors_each_box_only_when_asked(chart_dataset, color_each, expected):
    """ "One color per box" gives each box the next color of the palette.

    Case: every box of a grouped boxplot is drawn with the first color, whatever is picked.
    Affects: ui/chart_style.py (_boxplot, StyleSpec.color_each)
    How: draws the boxes of ``a`` by ``plan`` (3 boxes) and reads their border colors.
    """
    df, _ = chart_dataset
    data = viz.chart_data(df, viz.DataSpec(K.BOXPLOT, value="a", x="plan"))
    boxes = _option(data, replace(STYLE, color_each=color_each))["series"][0]["data"]
    assert [b["itemStyle"]["borderColor"] for b in boxes] == expected


def test_line_on_intervals_uses_a_category_axis(chart_dataset):
    """A line over the intervals of a number has a category x axis, not a time axis.

    Case: the interval labels are drawn on a time axis at 1970.
    Affects: ui/chart_style.py (_line)
    How: builds the option of a line over 10 intervals of ``a``.
    """
    df, _ = chart_dataset
    data = viz.chart_data(df, viz.DataSpec(K.LINE, x="a", value="b", agg=viz.Agg.MEAN, bins=10))
    assert _option(data)["xAxis"]["type"] == "category"


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        pytest.param(("10", "2", "1"), ["1", "2", "10"], id="numbers"),
        pytest.param(("1.5", "-3", "1,000"), ["-3", "1.5", "1,000"], id="signs-and-thousands"),
        pytest.param(
            ("item 10", "Item 2", "item 1"), ["item 1", "Item 2", "item 10"], id="natural"
        ),
        pytest.param(
            ("1,000 to 2,000", "900 to 1,000", "100 to 900"),
            ["100 to 900", "900 to 1,000", "1,000 to 2,000"],
            id="intervals",
        ),
    ],
)
def test_order_by_label_reads_numbers_as_numbers(labels, expected):
    """ "A-Z, 1-9" sorts the numbers inside labels by value.

    Case: labels sorted as text give "1, 10, 2".
    Affects: ui/chart_style.py (label_order)
    How: sorts numbers, signed and with thousands, mixed text and interval labels.
    """
    assert [labels[i] for i in look.label_order(labels)] == expected


def test_heatmap_scale_is_diverging_only_across_zero():
    """Values that cross zero get a scale centered on it; positive ones a sequential one.

    Case: a diverging scale on counts makes half the colors meaningless, or a sequential one
    on signed values hides the sign.
    Affects: ui/chart_style.py (heat_scale, _heatmap)
    How: builds two 2-cell heatmaps, one from -1 to 3 and one from 1 to 3.
    """

    def visual_map(low: float, high: float) -> dict:
        data = viz.ChartData(
            K.HEATMAP,
            series=(viz.Series("v", ([0, 0, low], [1, 0, high])),),
            categories=("a", "b"),
            rows=("r",),
            value_range=(low, high),
        )
        return _option(data)["visualMap"]

    crossing = visual_map(-1.0, 3.0)
    assert crossing["inRange"]["color"] == list(COLORS.diverging)
    assert (crossing["min"], crossing["max"]) == (-3.0, 3.0)
    positive = visual_map(1.0, 3.0)
    assert positive["inRange"]["color"][-1] == COLORS.accent
    assert (positive["min"], positive["max"]) == (1.0, 3.0)


def test_many_elements_turn_animation_off(chart_dataset):
    """Thousands of points are drawn without animation; a few bars keep it (R-V6).

    Case: a 3,000 point scatter animates on every rerun and the page stutters.
    Affects: ui/chart_style.py (_frame)
    How: compares the option of a scatter of every row with that of a 3-bar chart.
    """
    df, _ = chart_dataset
    scatter = viz.chart_data(df, viz.DataSpec(K.SCATTER, x="a", y="b"))
    assert scatter.elements > 2_000
    assert _option(scatter)["animation"] is False
    bars = viz.chart_data(df, viz.DataSpec(K.BAR, x="plan"))
    assert _option(bars)["animation"] is True


def _value_axes(option: dict) -> list[dict]:
    """The numeric axes of ``option``: its value and log axes."""
    axes = [option.get("xAxis"), option.get("yAxis")]
    return [a for a in axes if isinstance(a, dict) and a.get("type") in ("value", "log")]


@pytest.mark.parametrize("kind", [k for k in K if k != K.PIE], ids=str)
@pytest.mark.parametrize("axis_format", ["full", "compact"])
def test_every_value_axis_follows_the_axis_format(chart_dataset, kind, axis_format):
    """Every numeric axis is written with the "Axis numbers" format, on a linear or log scale.

    Case: an axis built without the formatter (the x of a scatter, a log axis) keeps writing
    large or tiny numbers in scientific notation ("1e+21", "1e-7").
    Affects: ui/chart_style.py (_value_axis, _scatter, _heatmap, axis_formatter)
    How: builds the default chart of each kind with each format, linear and log, and compares
    the formatter of every value axis (and the heatmap scale) with ``axis_formatter``.
    """
    df, hints = chart_dataset
    data = viz.chart_data(df, viz.default_spec(kind, hints))
    expected = look.axis_formatter(axis_format)
    for log_scale in (False, True):
        option = _option(data, replace(STYLE, axis_format=axis_format, log_scale=log_scale))
        axes = _value_axes(option)
        if kind == K.HEATMAP:
            axes = [{"axisLabel": option["visualMap"]}]
        assert axes, f"{kind} has no value axis"
        assert all(a["axisLabel"]["formatter"] == expected for a in axes)


@pytest.mark.parametrize("kind", [K.BAR, K.HISTOGRAM, K.HEATMAP], ids=str)
def test_value_labels_are_rounded_to_the_decimals_asked(chart_dataset, kind):
    """The values written on the marks are rounded to "Label decimals".

    Case: a mean is written as 12.345678901 on every bar, and the labels overlap.
    Affects: ui/chart_style.py (_value_label, label_formatter, _heatmap)
    How: builds the default chart of each kind with labels on and 0 decimals, and reads the
    formatter of every series' label.
    """
    df, hints = chart_dataset
    data = viz.chart_data(df, viz.default_spec(kind, hints))
    option = _option(data, replace(STYLE, labels=True, label_decimals=0))
    shown = [s["label"] for s in option["series"] if s["label"]["show"]]
    assert shown
    assert all(label["formatter"] == look.label_formatter(0) for label in shown)
    assert "maximumFractionDigits: 0" in look.label_formatter(0).code


def test_style_round_trips_through_json_without_its_colors():
    """The style saved as JSON reads back the same, with the palette by name only.

    Case: the hex colors of light mode are saved and a loaded chart keeps them in dark mode.
    Affects: ui/chart_style.py (style_to_dict, style_from_dict)
    How: writes a style with every kind of field and reads it back.
    """
    style = look.StyleSpec(
        title="Mine",
        palette=("#000000",),
        palette_name="Contrast",
        color="#123456",
        legend=False,
        sort="label",
        size="large",
        log_scale=True,
        labels=True,
        label_decimals=0,
        axis_format="compact",
    )
    # A bar chart draws every field of this style.
    raw = json.loads(json.dumps(look.style_to_dict(style, K.BAR)))
    assert "palette" not in raw
    assert look.style_from_dict(raw) == replace(style, palette=())


def test_automatic_title_is_saved_empty():
    """A title the page wrote itself is saved empty, so a loaded chart titles itself.

    Case: "Rows by plan" is saved and stays the title after the loaded chart changes columns.
    Affects: ui/chart_style.py (style_to_dict)
    How: saves a style whose title is the automatic one.
    """
    style = look.StyleSpec(title="Rows by plan")
    assert look.style_to_dict(style, K.BAR, auto_title="Rows by plan")["title"] == ""


@pytest.mark.parametrize(
    ("kind", "saved", "not_saved"),
    [
        pytest.param(K.LINE, {"area", "connect_gaps"}, {"trend", "donut", "sort"}, id="line"),
        pytest.param(K.SCATTER, {"trend", "symbol_size"}, {"labels", "area"}, id="scatter"),
        pytest.param(K.PIE, {"donut", "labels"}, {"color", "log_scale"}, id="pie"),
    ],
)
def test_saved_style_only_has_the_options_of_its_kind(kind, saved, not_saved):
    """A saved configuration only carries the style options its chart kind draws.

    Case: the file of a line chart said ``trend: true`` and ``donut: true``, which a line
    never draws: it read as a trend line that failed to appear.
    Affects: ui/chart_style.py (style_to_dict, STYLE_KINDS, applies)
    How: saves the default style with each kind and reads the keys.
    """
    raw = look.style_to_dict(look.StyleSpec(), kind)
    assert saved <= raw.keys()
    assert not not_saved & raw.keys()


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param({"size": "huge"}, id="not-a-choice"),
        pytest.param({"legend": "yes"}, id="text-for-a-toggle"),
        pytest.param({"symbol_size": True}, id="bool-for-a-number"),
        pytest.param({"color": "red"}, id="color-not-hex"),
        pytest.param({"label_decimals": 2.5}, id="fraction-of-a-decimal"),
        pytest.param({"label_decimals": 50}, id="too-many-decimals"),
        pytest.param({"axis_format": "scientific"}, id="unknown-axis-format"),
    ],
)
def test_style_with_wrong_values_is_rejected(raw):
    """A style value of the wrong type or outside its choices raises ``ValueError``.

    Case: a hand-edited file gives a widget a value it cannot show, and the page raises.
    Affects: ui/chart_style.py (style_from_dict)
    How: reads each malformed style.
    """
    with pytest.raises(ValueError):
        look.style_from_dict(raw)
