"""ECharts option builders. Each function returns the dict passed to ``st_echarts``.

Kept apart from the pages so they only decide what to show, not how the chart is configured.
Every builder receives data already aggregated (counts, bins, a capped list of points), never a
whole column (R-V2 in docs/performance.md).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Milliseconds of the entry animation of the inspector charts.
_ANIMATION_MS = 700
# Line type of the marks, by name; others are solid.
_MARK_LINES = {"Median": "solid", "Mean": "dashed"}
# Opacity of the box fill of the boxplot, over the page background.
_BOX_FILL_OPACITY = 0.35
# Grid lines: a translucent gray that reads on a light and on a dark background, so it does not
# depend on the mode the page had when the chart was built.
_SPLIT_LINE = {"type": "dashed", "color": "rgba(128, 128, 128, 0.25)"}


@dataclass(frozen=True)
class ChartColors:
    """Colors the builders give to their series. Built from the theme by ``ui/theme.py``.

    Attributes:
        accent: Bars and box outline (primaryColor of the theme).
        accent_light: Lighter tone of ``accent``: top of the bar gradient, box fill.
        highlight: Bars under the pointer and marks without a style of their own.
        outlier: Outlier dots of the boxplot.
        median: Line and label of the median.
        mean: Line and label of the mean.
        background: Page background: border of the outlier dots, so overlapping dots separate.
        emphasis: Border of the heatmap cell under the pointer (the text color).
        diverging: Five stops for values in [-1, 1], from negative to positive. Symmetric, with
            a neutral middle that works on the page background.
    """

    accent: str
    accent_light: str
    highlight: str
    outlier: str
    median: str
    mean: str
    background: str
    emphasis: str
    diverging: tuple[str, str, str, str, str]


@dataclass(frozen=True)
class ChartTheme:
    """Look shared by every chart, applied by ``themed`` on top of what a builder returns.

    Attributes:
        palette: Series colors, in order (``option["color"]``); the first is the accent.
        font: CSS font family of every text in the chart.
    """

    palette: tuple[str, ...]
    font: str


@dataclass(frozen=True)
class Mark:
    """Vertical line on a chart at a notable value, such as the median.

    Attributes:
        name: Short name shown on the line ("Median").
        position: Bar index (histogram) or value (boxplot) where the line goes.
        text: The value, already formatted, shown next to the name.
    """

    name: str
    position: float
    text: str


def heatmap_options(matrix: pd.DataFrame, colors: ChartColors) -> dict[str, Any]:
    """Heatmap of a square correlation matrix. Undefined cells are left empty."""
    labels = [str(c) for c in matrix.columns]
    values = matrix.to_numpy(dtype=float)
    data = [
        [j, i, round(float(values[i, j]), 2)]
        for i in range(len(labels))
        for j in range(len(labels))
        if not np.isnan(values[i, j])
    ]
    return {
        "tooltip": {"position": "top"},
        "grid": {"top": 10, "left": 10, "right": 80, "bottom": 10, "containLabel": True},
        "xAxis": {
            "type": "category",
            "data": labels,
            "axisLabel": {"rotate": 45, "interval": 0},
            "splitArea": {"show": True},
        },
        # inverse: the first column is drawn at the top, as in a table.
        "yAxis": {
            "type": "category",
            "data": labels,
            "inverse": True,
            "axisLabel": {"interval": 0},
            "splitArea": {"show": True},
        },
        "visualMap": {
            "min": -1,
            "max": 1,
            "calculable": True,
            "orient": "vertical",
            "right": 0,
            "top": "center",
            "inRange": {"color": list(colors.diverging)},
        },
        "series": [
            {
                "type": "heatmap",
                "data": data,
                # Numbers in the cells only while they fit.
                "label": {"show": len(labels) <= 12},
                "emphasis": {"itemStyle": {"borderColor": colors.emphasis, "borderWidth": 1}},
            }
        ],
    }


def heatmap_height(n_columns: int) -> str:
    """Height that keeps the heatmap cells readable."""
    return f"{max(320, 28 * n_columns + 140)}px"


def bar_options(
    labels: list[str], values: list[float], colors: ChartColors, unit: str = ""
) -> dict[str, Any]:
    """Vertical bar chart, one bar per label. ``unit`` is appended to the values ("%")."""
    return {
        "tooltip": {"trigger": "axis"},
        "grid": {"top": 16, "left": 8, "right": 16, "bottom": 8, "containLabel": True},
        "xAxis": {
            "type": "category",
            "data": labels,
            # Long names are cut with "…"; the tooltip shows them in full.
            "axisLabel": {
                "interval": 0,
                "rotate": 30 if len(labels) > 6 else 0,
                "width": 90,
                "overflow": "truncate",
            },
        },
        "yAxis": {
            "type": "value",
            "axisLabel": {"formatter": "{value}" + unit},
            "splitLine": {"lineStyle": _SPLIT_LINE},
        },
        "series": [{"type": "bar", "data": values, "itemStyle": {"color": colors.accent}}],
    }


def histogram_options(
    labels: list[str],
    counts: list[int],
    colors: ChartColors,
    show_labels: bool = False,
    marks: list[Mark] | None = None,
) -> dict[str, Any]:
    """Vertical bars from precomputed counts: a histogram, or one bar per value.

    Bin labels ("12.5 to 15") are too long for the axis, so they are only shown in the tooltip
    unless ``show_labels`` (one bar per value, short labels).

    Args:
        marks: Lines to draw over the bars, such as the median. ``Mark.position`` is the index
            of the bar the value falls in (``columns.bar_index``).
    """
    marks = marks or []
    series: dict[str, Any] = {
        "name": "Rows",
        "type": "bar",
        "data": counts,
        "barCategoryGap": "6%",
        "itemStyle": {
            "color": _vertical_gradient(colors.accent_light, colors.accent),
            "borderRadius": [4, 4, 0, 0],
        },
        "emphasis": {"itemStyle": {"color": colors.highlight}},
        "animationDuration": _ANIMATION_MS,
        "animationEasing": "cubicOut",
    }
    if marks:
        series["markLine"] = _mark_lines([labels[int(m.position)] for m in marks], marks, colors)
    return {
        "tooltip": {
            "trigger": "item",
            "formatter": "{b}<br/><b>{c}</b> rows",
        },
        "grid": {**_grid(), "top": 16 + 16 * len(marks)},
        "xAxis": {
            "type": "category",
            "data": labels,
            "axisLabel": {"show": show_labels, "hideOverlap": True},
            "axisTick": {"show": False},
        },
        "yAxis": {
            "type": "value",
            "minInterval": 1,
            "splitLine": {"lineStyle": _SPLIT_LINE},
        },
        "series": [series],
    }


def hbar_options(labels: list[str], counts: list[int], colors: ChartColors) -> dict[str, Any]:
    """Horizontal bars with the count at their end, the first label on top."""
    return {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": _grid(),
        "xAxis": {"type": "value", "minInterval": 1, "splitLine": {"lineStyle": _SPLIT_LINE}},
        "yAxis": {"type": "category", "data": labels[::-1]},
        "series": [
            {
                "name": "Rows",
                "type": "bar",
                "data": counts[::-1],
                "itemStyle": {"color": colors.accent, "borderRadius": [0, 4, 4, 0]},
                "label": {"show": True, "position": "right"},
            }
        ],
    }


def hbar_height(n_bars: int) -> str:
    """Height that gives each horizontal bar room for its label."""
    return f"{40 + 28 * n_bars}px"


def boxplot_options(
    name: str,
    box: tuple[float, float, float, float, float],
    outliers: list[float],
    data_range: tuple[float, float],
    zoom_ratio: float,
    colors: ChartColors,
    marks: list[Mark] | None = None,
) -> dict[str, Any]:
    """Horizontal boxplot from precomputed statistics, with outliers as dots.

    Args:
        box: Lower whisker, Q1, median, Q3 and upper whisker.
        outliers: Values drawn as dots (already capped by the caller).
        data_range: Lowest and highest value of the column, outliers included.
        zoom_ratio: If the data spans more than this many times the whiskers, the chart opens
            zoomed on the box, with a slider to reach the farthest outliers. Otherwise they
            squash the box into a line, and the box is what most users look at first.
        marks: Lines over the box, such as the median and the mean (``Mark.position`` is the
            value), styled as in ``histogram_options``.
    """
    low, high = data_range
    option: dict[str, Any] = {
        "tooltip": {"trigger": "item"},
        "grid": _grid(),
        "xAxis": {
            "type": "value",
            "scale": True,
            "splitLine": {"lineStyle": _SPLIT_LINE},
        },
        "yAxis": {
            "type": "category",
            "data": [name],
            "axisLabel": {"show": False},
            "axisTick": {"show": False},
        },
        "series": [
            {
                "name": name,
                "type": "boxplot",
                "data": [list(box)],
                "boxWidth": [50, 90],
                "itemStyle": {
                    "color": _vertical_gradient(
                        colors.accent_light, colors.accent, opacity=_BOX_FILL_OPACITY
                    ),
                    "borderColor": colors.accent,
                    "borderWidth": 2,
                },
                "emphasis": {"itemStyle": {"borderColor": colors.highlight, "borderWidth": 3}},
                "animationDuration": _ANIMATION_MS,
                "animationEasing": "cubicOut",
            },
            {
                "name": "Outliers",
                "type": "scatter",
                "data": [[v, 0] for v in outliers],
                "symbolSize": 9,
                "itemStyle": {
                    "color": colors.outlier,
                    "opacity": 0.6,
                    "borderColor": colors.background,
                    "borderWidth": 1,
                },
                "emphasis": {"scale": 1.6, "itemStyle": {"opacity": 1}},
            },
        ],
    }
    marks = [m for m in marks or [] if pd.notna(m.position)]
    if marks:
        option["series"][0]["markLine"] = _mark_lines(
            [float(m.position) for m in marks], marks, colors
        )
        option["grid"]["top"] = 16 + 16 * len(marks)
    span = box[-1] - box[0]
    if span > 0 and (high - low) > zoom_ratio * span:
        pad = span * 0.15
        window = {"startValue": box[0] - pad, "endValue": box[-1] + pad}
        option["dataZoom"] = [
            {"type": "inside", "xAxisIndex": 0, "filterMode": "none", **window},
            {
                "type": "slider",
                "xAxisIndex": 0,
                "filterMode": "none",
                "height": 18,
                "bottom": 4,
                **window,
            },
        ]
        option["grid"] = {**option["grid"], "bottom": 36}
    return option


def _grid() -> dict[str, Any]:
    """Tight margins; ``containLabel`` keeps the axis labels inside the chart."""
    return {"left": 8, "right": 16, "top": 16, "bottom": 8, "containLabel": True}


def _mark_lines(positions: list[Any], marks: list[Mark], colors: ChartColors) -> dict[str, Any]:
    """``markLine`` with one vertical line per mark at ``positions`` (axis values).

    Labels are stacked, so two marks at the same place do not overlap.
    """
    return {
        "silent": True,
        "symbol": "none",
        "data": [
            {
                "xAxis": position,
                "lineStyle": {"color": color, "width": 2, "type": line},
                "label": {
                    "formatter": f"{mark.name} {mark.text}",
                    "color": color,
                    "fontWeight": "bold",
                    "distance": [0, 4 + 16 * i],
                },
            }
            for i, (position, mark) in enumerate(zip(positions, marks, strict=True))
            for color, line in [_mark_style(mark.name, colors)]
        ],
    }


def _mark_style(name: str, colors: ChartColors) -> tuple[str, str]:
    """Color and line type of the mark called ``name``."""
    color = {"Median": colors.median, "Mean": colors.mean}.get(name, colors.highlight)
    return color, _MARK_LINES.get(name, "solid")


def _vertical_gradient(top: str, bottom: str, opacity: float = 1.0) -> dict[str, Any]:
    """ECharts linear gradient from ``top`` to ``bottom``, as plain JSON (no JavaScript)."""
    return {
        "type": "linear",
        "x": 0,
        "y": 0,
        "x2": 0,
        "y2": 1,
        "colorStops": [
            {"offset": 0, "color": with_alpha(top, opacity)},
            {"offset": 1, "color": with_alpha(bottom, opacity)},
        ],
    }


def with_alpha(color: str, opacity: float) -> str:
    """``#rrggbb`` as ``rgba(...)`` with ``opacity``; unchanged when fully opaque.

    Only ``#rrggbb``: ``theme.palette`` rejects any other format when the theme is read.
    """
    if opacity >= 1:
        return color
    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {opacity})"


def mix(color: str, other: str, amount: float) -> str:
    """``#rrggbb`` between ``color`` (``amount`` 0) and ``other`` (``amount`` 1)."""
    a = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    b = (int(other[i : i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(x + (y - x) * amount):02X}" for x, y in zip(a, b, strict=True))


def themed(option: dict[str, Any], theme: ChartTheme) -> dict[str, Any]:
    """``option`` with the look shared by every chart underneath: what the option sets wins.

    Text and background colors are left to the ``"streamlit"`` theme of ``st_echarts``, which
    follows light / dark mode in the browser as soon as the user switches.
    """
    defaults = {
        "color": list(theme.palette),
        "backgroundColor": "transparent",
        "textStyle": {"fontFamily": theme.font},
    }
    # Only when the chart has a tooltip: adding the key alone would turn one on.
    if isinstance(option.get("tooltip"), dict):
        defaults["tooltip"] = {"textStyle": {"fontFamily": theme.font}}
    return _merge(defaults, option)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """New dict with ``override`` on top of ``base``, merging nested dicts. Inputs untouched."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged
