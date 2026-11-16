import json

import numpy as np
import pandas as pd

from litmetrics.ui import charts

# Any valid colors: the builders only place them, the theme chooses them (tests/ui/test_theme.py).
COLORS = charts.ChartColors(
    accent="#4F46E5",
    accent_light="#A29DF0",
    highlight="#EA580C",
    outlier="#DC2626",
    median="#EA580C",
    mean="#059669",
    background="#FFFFFF",
    emphasis="#111827",
    diverging=(
        "#2563EB",
        "rgba(37, 99, 235, 0.45)",
        "#EEEEEE",
        "rgba(220, 38, 38, 0.45)",
        "#DC2626",
    ),
)


def test_heatmap_skips_undefined_cells_and_is_json():
    matrix = pd.DataFrame([[1.0, np.nan], [np.nan, 1.0]], index=["a", "b"], columns=["a", "b"])
    options = charts.heatmap_options(matrix, COLORS)
    assert options["series"][0]["data"] == [[0, 0, 1.0], [1, 1, 1.0]]
    json.dumps(options)


def test_bar_options_one_value_per_label():
    options = charts.bar_options(["x", "y"], [1.5, 2.0], COLORS, unit="%")
    assert options["xAxis"]["data"] == ["x", "y"]
    assert options["series"][0]["data"] == [1.5, 2.0]
    assert options["yAxis"]["axisLabel"]["formatter"] == "{value}%"
    json.dumps(options)


def test_heatmap_shows_numbers_at_any_size():
    names = [f"c{i}" for i in range(30)]
    matrix = pd.DataFrame(np.eye(30), index=names, columns=names)
    label = charts.heatmap_options(matrix, COLORS)["series"][0]["label"]
    assert label == {"show": True, "fontSize": charts.heatmap_label_size(30)}


def test_heatmap_label_size_shrinks_with_columns():
    sizes = [charts.heatmap_label_size(n) for n in (2, 8, 9, 16, 17, 30)]
    assert sizes == sorted(sizes, reverse=True)
    assert min(sizes) >= 9


def test_heatmap_height_grows_with_columns():
    assert charts.heatmap_height(1) == "320px"
    assert int(charts.heatmap_height(30).removesuffix("px")) > 320


def test_histogram_options_hides_bin_labels():
    options = charts.histogram_options(["0 to 1", "1 to 2"], [3, 4], COLORS)
    assert options["xAxis"]["axisLabel"]["show"] is False
    assert options["series"][0]["data"] == [3, 4]
    json.dumps(options)


def test_hbar_labels_use_the_text_color_without_outline():
    label = charts.hbar_options(["a"], [5], COLORS)["series"][0]["label"]
    assert label["color"] == COLORS.emphasis
    assert label["textBorderWidth"] == 0


def test_hbar_options_first_label_on_top():
    options = charts.hbar_options(["a", "b"], [5, 2], COLORS)
    assert options["yAxis"]["data"] == ["b", "a"]
    assert options["series"][0]["data"] == [2, 5]
    assert charts.hbar_height(2) == "88px"
    json.dumps(options)


def test_boxplot_zooms_on_the_box_with_far_outliers():
    options = charts.boxplot_options(
        "v", (1.0, 2.0, 3.0, 4.0, 5.0), [100.0], (1.0, 100.0), 3.0, COLORS
    )
    assert "dataZoom" in options
    assert options["series"][1]["data"] == [[100.0, 0]]
    json.dumps(options)


def test_boxplot_without_far_outliers_is_not_zoomed():
    options = charts.boxplot_options("v", (1.0, 2.0, 3.0, 4.0, 5.0), [], (1.0, 5.0), 3.0, COLORS)
    assert "dataZoom" not in options


def test_histogram_marks_point_at_their_bar():
    marks = [charts.Mark("Median", 1, "1.5"), charts.Mark("Mean", 1, "1.6")]
    options = charts.histogram_options(["0 to 1", "1 to 2"], [3, 4], COLORS, marks=marks)
    lines = options["series"][0]["markLine"]["data"]
    assert [line["xAxis"] for line in lines] == ["1 to 2", "1 to 2"]
    assert lines[0]["label"]["formatter"] == "Median 1.5"
    json.dumps(options)


def test_boxplot_mark_lines_and_translucent_box():
    marks = [charts.Mark("Median", 3.0, "3"), charts.Mark("Mean", float("nan"), "—")]
    options = charts.boxplot_options(
        "v", (1.0, 2.0, 3.0, 4.0, 5.0), [], (1.0, 5.0), 3.0, COLORS, marks
    )
    # Marks without a value are skipped.
    assert [d["xAxis"] for d in options["series"][0]["markLine"]["data"]] == [3.0]
    stops = options["series"][0]["itemStyle"]["color"]["colorStops"]
    assert stops[0]["color"].startswith("rgba(")
    json.dumps(options)


def test_mark_color_follows_its_name_not_its_order():
    options = charts.histogram_options(["a"], [1], COLORS, marks=[charts.Mark("Mean", 0, "1")])
    line = options["series"][0]["markLine"]["data"][0]["lineStyle"]
    assert line["type"] == "dashed"
    assert line["color"] == COLORS.mean


def test_builders_use_the_colors_they_receive():
    options = charts.bar_options(["x"], [1.0], COLORS)
    assert options["series"][0]["itemStyle"]["color"] == COLORS.accent
    matrix = pd.DataFrame([[1.0]], index=["a"], columns=["a"])
    heatmap = charts.heatmap_options(matrix, COLORS)
    assert heatmap["visualMap"]["inRange"]["color"] == list(COLORS.diverging)


def test_themed_lays_the_shared_look_under_the_option():
    theme = charts.ChartTheme(palette=("#111111", "#222222"), font="'IBM Plex Sans', sans-serif")
    option = {"tooltip": {"trigger": "axis"}, "textStyle": {"fontSize": 11}, "color": ["#333333"]}
    themed = charts.themed(option, theme)
    # What the option sets wins; the rest comes from the theme.
    assert themed["color"] == ["#333333"]
    assert themed["textStyle"] == {"fontFamily": theme.font, "fontSize": 11}
    assert themed["tooltip"]["trigger"] == "axis"
    assert themed["tooltip"]["textStyle"]["fontFamily"] == theme.font
    assert themed["backgroundColor"] == "transparent"
    # The input is not changed.
    assert option == {
        "tooltip": {"trigger": "axis"},
        "textStyle": {"fontSize": 11},
        "color": ["#333333"],
    }


def test_themed_does_not_add_a_tooltip():
    theme = charts.ChartTheme(palette=("#111111",), font="sans-serif")
    assert "tooltip" not in charts.themed({"series": []}, theme)


def test_mix_and_alpha():
    assert charts.mix("#000000", "#FFFFFF", 0.5) == "#808080"
    assert charts.mix("#4F46E5", "#FFFFFF", 0) == "#4F46E5"
    assert charts.with_alpha("#FF0000", 0.5) == "rgba(255, 0, 0, 0.5)"
