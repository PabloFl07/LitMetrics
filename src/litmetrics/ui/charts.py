"""ECharts option builders. Each function returns the dict passed to ``st_echarts``.

Kept apart from the pages so they only decide what to show, not how the chart is configured.
Every builder receives data already aggregated (counts, bins, a capped list of points), never a
whole column (R-V2 in docs/performance.md).
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

# Diverging palette for values in [-1, 1]: blue for negative, red for positive.
_DIVERGING = ["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#b2182b"]
_BAR_COLOR = "#4c78a8"
# Lighter tone of _BAR_COLOR: top of the bar gradient, fill of the boxplot box.
_BAR_LIGHT = "#86b0e0"
# Bars under the pointer, outliers, and the median and mean lines. Mid tones that read on a
# light and a dark background.
_HIGHLIGHT = "#f58518"
_OUTLIER_COLOR = "#e45756"
_MEDIAN_COLOR = "#f58518"
_MEAN_COLOR = "#54a24b"
# Color and line type of the marks, by name; others are solid in _HIGHLIGHT.
_MARK_STYLES = {"Median": (_MEDIAN_COLOR, "solid"), "Mean": (_MEAN_COLOR, "dashed")}
# Milliseconds of the entry animation of the inspector charts.
_ANIMATION_MS = 700


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


def heatmap_options(matrix: pd.DataFrame) -> dict[str, Any]:
    """Heatmap of a square correlation matrix. Undefined cells are left empty."""
    labels = [str(c) for c in matrix.columns]
    data = [
        [j, i, round(float(matrix.iloc[i, j]), 2)]
        for i in range(len(labels))
        for j in range(len(labels))
        if pd.notna(matrix.iloc[i, j])
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
            "inRange": {"color": _DIVERGING},
        },
        "series": [
            {
                "type": "heatmap",
                "data": data,
                # Numbers in the cells only while they fit.
                "label": {"show": len(labels) <= 12},
                "emphasis": {"itemStyle": {"borderColor": "#333", "borderWidth": 1}},
            }
        ],
    }


def bar_options(labels: list[str], values: list[float], unit: str = "") -> dict[str, Any]:
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
        "yAxis": {"type": "value", "axisLabel": {"formatter": "{value}" + unit}},
        "series": [{"type": "bar", "data": values, "itemStyle": {"color": _BAR_COLOR}}],
    }


def heatmap_height(n_columns: int) -> str:
    """Height that keeps the heatmap cells readable."""
    return f"{max(320, 28 * n_columns + 140)}px"


def histogram_options(
    labels: list[str],
    counts: list[int],
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
            "color": _vertical_gradient(_BAR_LIGHT, _BAR_COLOR),
            "borderRadius": [4, 4, 0, 0],
        },
        "emphasis": {"itemStyle": {"color": _HIGHLIGHT}},
        "animationDuration": _ANIMATION_MS,
        "animationEasing": "cubicOut",
    }
    if marks:
        series["markLine"] = _mark_lines([labels[int(m.position)] for m in marks], marks)
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
            "splitLine": {"lineStyle": {"type": "dashed", "opacity": 0.5}},
        },
        "series": [series],
    }


def hbar_options(labels: list[str], counts: list[int]) -> dict[str, Any]:
    """Horizontal bars with the count at their end, the first label on top."""
    return {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": _grid(),
        "xAxis": {"type": "value", "minInterval": 1},
        "yAxis": {"type": "category", "data": labels[::-1]},
        "series": [
            {
                "name": "Rows",
                "type": "bar",
                "data": counts[::-1],
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
            "splitLine": {"lineStyle": {"type": "dashed", "opacity": 0.5}},
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
                    "color": _vertical_gradient(_BAR_LIGHT, _BAR_COLOR, opacity=0.35),
                    "borderColor": _BAR_COLOR,
                    "borderWidth": 2,
                },
                "emphasis": {"itemStyle": {"borderColor": _HIGHLIGHT, "borderWidth": 3}},
                "animationDuration": _ANIMATION_MS,
                "animationEasing": "cubicOut",
            },
            {
                "name": "Outliers",
                "type": "scatter",
                "data": [[v, 0] for v in outliers],
                "symbolSize": 9,
                "itemStyle": {
                    "color": _OUTLIER_COLOR,
                    "opacity": 0.6,
                    "borderColor": "#fff",
                    "borderWidth": 1,
                },
                "emphasis": {"scale": 1.6, "itemStyle": {"opacity": 1}},
            },
        ],
    }
    marks = [m for m in marks or [] if pd.notna(m.position)]
    if marks:
        option["series"][0]["markLine"] = _mark_lines([float(m.position) for m in marks], marks)
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


def _mark_lines(positions: list[Any], marks: list[Mark]) -> dict[str, Any]:
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
            for color, line in [_MARK_STYLES.get(mark.name, (_HIGHLIGHT, "solid"))]
        ],
    }


def _vertical_gradient(top: str, bottom: str, opacity: float = 1.0) -> dict[str, Any]:
    """ECharts linear gradient from ``top`` to ``bottom``, as plain JSON (no JavaScript)."""
    return {
        "type": "linear",
        "x": 0,
        "y": 0,
        "x2": 0,
        "y2": 1,
        "colorStops": [
            {"offset": 0, "color": _with_alpha(top, opacity)},
            {"offset": 1, "color": _with_alpha(bottom, opacity)},
        ],
    }


def _with_alpha(color: str, opacity: float) -> str:
    """``#rrggbb`` as ``rgba(...)`` with ``opacity``; unchanged when fully opaque."""
    if opacity >= 1:
        return color
    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {opacity})"
