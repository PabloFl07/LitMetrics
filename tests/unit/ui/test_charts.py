"""Unit tests of ``ui/charts.py``: ECharts option builders (plain dicts, no Streamlit).

AppTest does not run the chart's JavaScript, so a broken option is only seen in a browser.
These tests check the parts of an option that carry meaning (data, marks, zoom, colors) and
that every option is JSON, which is what streamlit-echarts sends to the browser.
"""

import json

import numpy as np
import pandas as pd
from fixtures.samples import CHART_COLORS as COLORS

from litmetrics.ui import charts

# A five-number summary (min, q1, median, q3, max) for the boxplot tests.
_BOX = (1.0, 2.0, 3.0, 4.0, 5.0)


# Heatmap


def test_heatmap_skips_undefined_cells_and_is_json():
    """Undefined correlations are left out of the data, so the option is valid JSON.

    Case: NaN reaches ``json.dumps`` (invalid JSON: the chart is blank in the browser).
    Affects: ui/charts.py (heatmap_options)
    How: builds a 2x2 matrix with NaN off the diagonal and checks data and serialization.
    """
    matrix = pd.DataFrame([[1.0, np.nan], [np.nan, 1.0]], index=["a", "b"], columns=["a", "b"])
    options = charts.heatmap_options(matrix, COLORS)
    assert options["series"][0]["data"] == [[0, 0, 1.0], [1, 1, 1.0]]
    json.dumps(options)


def test_heatmap_labels_and_height_follow_the_number_of_columns():
    """Labels shrink (to a readable minimum) and the chart grows as columns are added.

    Case: with 30 columns the numbers overlap or the cells are a few pixels high.
    Affects: ui/charts.py (heatmap_options, heatmap_label_size, heatmap_height)
    How: checks label sizes for 2..30 columns, the label of a 30-column heatmap and heights.
    """
    sizes = [charts.heatmap_label_size(n) for n in (2, 8, 9, 16, 17, 30)]
    assert sizes == sorted(sizes, reverse=True)
    assert min(sizes) >= 9
    names = [f"c{i}" for i in range(30)]
    matrix = pd.DataFrame(np.eye(30), index=names, columns=names)
    label = charts.heatmap_options(matrix, COLORS)["series"][0]["label"]
    assert label == {"show": True, "fontSize": charts.heatmap_label_size(30)}
    assert charts.heatmap_height(1) == "320px"
    assert int(charts.heatmap_height(30).removesuffix("px")) > 320


# Bars


def test_bar_options_one_value_per_label():
    """Each label gets its value, and the unit is added to the axis.

    Case: values and labels misaligned, or the unit missing from the axis.
    Affects: ui/charts.py (bar_options)
    How: builds two bars with unit "%" and checks axis, data, formatter and JSON.
    """
    options = charts.bar_options(["x", "y"], [1.5, 2.0], COLORS, unit="%")
    assert options["xAxis"]["data"] == ["x", "y"]
    assert options["series"][0]["data"] == [1.5, 2.0]
    assert options["yAxis"]["axisLabel"]["formatter"] == "{value}%"
    json.dumps(options)


def test_hbar_options_first_label_on_top():
    """Horizontal bars show the first label on top, with a height per bar.

    Case: ECharts draws the y axis bottom-up, so the most frequent value ends at the bottom.
    Affects: ui/charts.py (hbar_options, hbar_height)
    How: builds two bars and checks the reversed axis and data, and the height of 2 bars.
    """
    options = charts.hbar_options(["a", "b"], [5, 2], COLORS)
    assert options["yAxis"]["data"] == ["b", "a"]
    assert options["series"][0]["data"] == [2, 5]
    assert charts.hbar_height(2) == "88px"
    json.dumps(options)


def test_hbar_labels_use_the_text_color_without_outline():
    """Value labels use the theme's text color and no outline.

    Case: ECharts' default white outline makes labels unreadable in dark mode.
    Affects: ui/charts.py (hbar_options)
    How: reads the label style of one bar.
    """
    label = charts.hbar_options(["a"], [5], COLORS)["series"][0]["label"]
    assert label["color"] == COLORS.emphasis
    assert label["textBorderWidth"] == 0


# Histogram and boxplot


def test_histogram_options_hides_bin_labels():
    """Bin labels are hidden (shown in the tooltip instead).

    Case: long "a to b" labels overlap under narrow bars.
    Affects: ui/charts.py (histogram_options)
    How: builds two bins and checks the axis labels are off and the data kept.
    """
    options = charts.histogram_options(["0 to 1", "1 to 2"], [3, 4], COLORS)
    assert options["xAxis"]["axisLabel"]["show"] is False
    assert options["series"][0]["data"] == [3, 4]
    json.dumps(options)


def test_histogram_marks_point_at_their_bar():
    """Median and mean marks point at the label of their bar, with their value.

    Case: marks use the bar index on a category axis and land on the wrong bar.
    Affects: ui/charts.py (histogram_options, Mark)
    How: puts two marks on bar 1 and checks their axis value and label.
    """
    marks = [charts.Mark("Median", 1, "1.5"), charts.Mark("Mean", 1, "1.6")]
    options = charts.histogram_options(["0 to 1", "1 to 2"], [3, 4], COLORS, marks=marks)
    lines = options["series"][0]["markLine"]["data"]
    assert [line["xAxis"] for line in lines] == ["1 to 2", "1 to 2"]
    assert lines[0]["label"]["formatter"] == "Median 1.5"
    json.dumps(options)


def test_mark_color_follows_its_name_not_its_order():
    """The mean is always dashed and in the mean color, even when it is the only mark.

    Case: colors are given by position, so the mean takes the median's color when alone.
    Affects: ui/charts.py (histogram_options)
    How: draws only a Mean mark and checks its line style.
    """
    options = charts.histogram_options(["a"], [1], COLORS, marks=[charts.Mark("Mean", 0, "1")])
    line = options["series"][0]["markLine"]["data"][0]["lineStyle"]
    assert line["type"] == "dashed"
    assert line["color"] == COLORS.mean


def test_boxplot_zooms_on_the_box_only_with_far_outliers():
    """With a far outlier the boxplot gets a zoom; without one it does not.

    Case: one value at 100 squeezes a 1..5 box into a line; or every boxplot gets a zoom bar.
    Affects: ui/charts.py (boxplot_options)
    How: builds the boxplot with an outlier at 100, and without outliers.
    """
    zoomed = charts.boxplot_options("v", _BOX, [100.0], (1.0, 100.0), 3.0, COLORS)
    assert "dataZoom" in zoomed
    assert zoomed["series"][1]["data"] == [[100.0, 0]]
    json.dumps(zoomed)
    plain = charts.boxplot_options("v", _BOX, [], (1.0, 5.0), 3.0, COLORS)
    assert "dataZoom" not in plain


def test_boxplot_skips_marks_without_value_and_has_translucent_box():
    """Marks with no value (NaN) are skipped; the box is translucent.

    Case: a NaN mean reaches the option (invalid JSON); an opaque box hides the median line.
    Affects: ui/charts.py (boxplot_options)
    How: draws a Median and a NaN Mean and checks the mark lines and the box color.
    """
    marks = [charts.Mark("Median", 3.0, "3"), charts.Mark("Mean", float("nan"), "—")]
    options = charts.boxplot_options("v", _BOX, [], (1.0, 5.0), 3.0, COLORS, marks)
    assert [d["xAxis"] for d in options["series"][0]["markLine"]["data"]] == [3.0]
    stops = options["series"][0]["itemStyle"]["color"]["colorStops"]
    assert stops[0]["color"].startswith("rgba(")
    json.dumps(options)


# Colors and theme


def test_builders_use_the_colors_they_receive():
    """Builders take their colors from the theme, never their own.

    Case: a hard-coded hex color ignores the style variant and dark mode.
    Affects: ui/charts.py (bar_options, heatmap_options)
    How: checks the bar color and the heatmap scale come from ``COLORS``.
    """
    options = charts.bar_options(["x"], [1.0], COLORS)
    assert options["series"][0]["itemStyle"]["color"] == COLORS.accent
    matrix = pd.DataFrame([[1.0]], index=["a"], columns=["a"])
    heatmap = charts.heatmap_options(matrix, COLORS)
    assert heatmap["visualMap"]["inRange"]["color"] == list(COLORS.diverging)


def test_themed_lays_the_shared_look_under_the_option():
    """The theme fills what an option leaves unset; what the option sets wins.

    Case: the theme overwrites a chart's own colors, or changes the option it was given
    (shared between reruns).
    Affects: ui/charts.py (themed, ChartTheme)
    How: themes an option with its own color, font size and tooltip; checks the merge and
    that the input is unchanged.
    """
    theme = charts.ChartTheme(palette=("#111111", "#222222"), font="'IBM Plex Sans', sans-serif")
    option = {"tooltip": {"trigger": "axis"}, "textStyle": {"fontSize": 11}, "color": ["#333333"]}
    themed = charts.themed(option, theme)
    assert themed["color"] == ["#333333"]
    assert themed["textStyle"] == {"fontFamily": theme.font, "fontSize": 11}
    assert themed["tooltip"]["trigger"] == "axis"
    assert themed["tooltip"]["textStyle"]["fontFamily"] == theme.font
    assert themed["backgroundColor"] == "transparent"
    assert option == {
        "tooltip": {"trigger": "axis"},
        "textStyle": {"fontSize": 11},
        "color": ["#333333"],
    }


def test_themed_does_not_add_a_tooltip():
    """Theming does not add a tooltip to a chart that has none.

    Case: a tooltip appears on charts meant to have none (sparklines in tables).
    Affects: ui/charts.py (themed)
    How: themes an option without tooltip.
    """
    theme = charts.ChartTheme(palette=("#111111",), font="sans-serif")
    assert "tooltip" not in charts.themed({"series": []}, theme)


def test_mix_and_alpha():
    """Colors are mixed and made translucent correctly.

    Case: rounding or channel order errors give a visibly different color.
    Affects: ui/charts.py (mix, with_alpha)
    How: mixes black and white at 50%, mixes at 0%, and makes red half transparent.
    """
    assert charts.mix("#000000", "#FFFFFF", 0.5) == "#808080"
    assert charts.mix("#4F46E5", "#FFFFFF", 0) == "#4F46E5"
    assert charts.with_alpha("#FF0000", 0.5) == "rgba(255, 0, 0, 0.5)"
