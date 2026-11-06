"""ECharts option builders. Each function returns the dict passed to ``st_echarts``.

Kept apart from the pages so they only decide what to show, not how the chart is configured.
Every builder receives data already aggregated (counts, bins, a capped list of points), never a
whole column (R-V2 in docs/performance.md).
"""

from typing import Any

import pandas as pd

# Diverging palette for values in [-1, 1]: blue for negative, red for positive.
_DIVERGING = ["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#b2182b"]
_BAR_COLOR = "#4c78a8"


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
    labels: list[str], counts: list[int], show_labels: bool = False
) -> dict[str, Any]:
    """Vertical bars from precomputed counts: a histogram, or one bar per value.

    Bin labels ("12.5 to 15") are too long for the axis, so they are only shown in the tooltip
    unless ``show_labels`` (one bar per value, short labels).
    """
    return {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": _grid(),
        "xAxis": {
            "type": "category",
            "data": labels,
            "axisLabel": {"show": show_labels, "hideOverlap": True},
            "axisTick": {"show": False},
        },
        "yAxis": {"type": "value", "minInterval": 1},
        "series": [{"name": "Rows", "type": "bar", "data": counts, "barCategoryGap": "4%"}],
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
) -> dict[str, Any]:
    """Horizontal boxplot from precomputed statistics, with outliers as dots.

    Args:
        box: Lower whisker, Q1, median, Q3 and upper whisker.
        outliers: Values drawn as dots (already capped by the caller).
        data_range: Lowest and highest value of the column, outliers included.
        zoom_ratio: If the data spans more than this many times the whiskers, the chart opens
            zoomed on the box, with a slider to reach the farthest outliers. Otherwise they
            squash the box into a line, and the box is what most users look at first.
    """
    low, high = data_range
    option: dict[str, Any] = {
        "tooltip": {"trigger": "item"},
        "grid": _grid(),
        "xAxis": {"type": "value", "scale": True},
        "yAxis": {
            "type": "category",
            "data": [name],
            "axisLabel": {"show": False},
            "axisTick": {"show": False},
        },
        "series": [
            {"name": name, "type": "boxplot", "data": [list(box)], "boxWidth": [40, 70]},
            {
                "name": "Outliers",
                "type": "scatter",
                "data": [[v, 0] for v in outliers],
                "symbolSize": 7,
                "itemStyle": {"opacity": 0.45},
            },
        ],
    }
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
        option["grid"] = {**_grid(), "bottom": 36}
    return option


def _grid() -> dict[str, Any]:
    """Tight margins; ``containLabel`` keeps the axis labels inside the chart."""
    return {"left": 8, "right": 16, "top": 16, "bottom": 8, "containLabel": True}
