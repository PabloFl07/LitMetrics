"""Look of the charts of the Visualization page: ``StyleSpec`` and the ECharts option.

``build_option(data, style, colors)`` turns the aggregated ``ChartData`` into the ECharts option.
It is pure and cheap, and not cached (R-V1): changing a color, the title or the legend rebuilds
the option from the cached data without touching pandas.

Only what a person decides is a style option. The rest follows from the data: the heatmap scale
is diverging when its values cross zero, axes are always named, the legend goes on top, and
animation is off when there are many elements (R-V6).

Colors only come from the theme: the palettes of ``StyleSpec.palette`` are built from it by the
page (``palettes``), and ``ChartColors`` gives the rest. No hex literals here.
"""

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Literal, get_args

from litmetrics.core.visualize import ChartData, ChartKind, Series
from litmetrics.ui.charts import ChartColors, Js, mix, with_alpha

# The only color format ``with_alpha`` and ``mix`` read, and what ``st.color_picker`` returns.
_HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}")

type SortOrder = Literal["data", "desc", "asc", "label"]
type ChartSize = Literal["small", "medium", "large"]
# How the numbers of a value axis are written: "full" (1,250,000) or "compact" (1.25M).
type AxisFormat = Literal["full", "compact"]

# Height (px) of each chart size.
HEIGHTS: dict[ChartSize, int] = {"small": 340, "medium": 460, "large": 640}

# Grid lines: a translucent gray that reads on a light and on a dark background (as in
# ``ui/charts.py``).
_SPLIT_LINE = {"type": "dashed", "color": "rgba(128, 128, 128, 0.25)"}
# Alternate bands behind the heatmap cells, so empty cells still read as a grid. Translucent
# grays like ``_SPLIT_LINE``: the ECharts default is light gray, which glows in dark mode.
_HEAT_BANDS = {
    "show": True,
    "areaStyle": {"color": ["rgba(128, 128, 128, 0.05)", "rgba(128, 128, 128, 0.12)"]},
}
# Opacity of the "Other" bar or slice: it is a leftover, not a category.
_OTHER_OPACITY = 0.45
# Elements from which ECharts draws scatter and bars in "large" mode, and from which there is
# no animation (R-V6).
_LARGE = 2_000
# Opacity of the bars of a histogram split by group and drawn one over the other.
_OVERLAY_OPACITY = 0.55
# Height (px) taken by the title, by the legend on top, and by the axis names.
_TITLE_PX = 30
_LEGEND_PX = 28
_AXIS_NAME_TOP_PX = 16
_AXIS_NAME_BOTTOM_PX = 20
# Labels of the pie slices (px and CSS weight): a little larger and heavier than the axis labels
# (12 px, 400), since they are the only text that names each slice. The theme font loads the
# weights 400, 500 and 600, so the canvas draws them instead of faking them.
_PIE_LABEL_SIZE = 13
_PIE_LABEL_WEIGHT = 500
# The share under the name ("Percent on slices"), and the label of the slice under the pointer.
_PIE_PERCENT_SIZE = 14
_PIE_EMPHASIS_SIZE = 15
_PIE_STRONG = 600
# Text without the outline ECharts draws around labels outside a mark (pie slices, bar ends,
# line points). ECharts writes them in its own dark gray with a light stroke, and the
# "streamlit" theme does not recolor them: in dark mode the stroke was what could be read, and
# the letters looked hollow. The labels get the text color of the theme instead
# (``_pie_label``, ``_value_label``).
_NO_OUTLINE = {"textBorderWidth": 0, "textBorderColor": "transparent", "textShadowBlur": 0}
# Category labels: rotation (degrees) when there are many, width (px) where they are cut, and
# the approximate width of one character (px, 12 px font), to tell how far down they reach.
_LABEL_ROTATE = 35
_LABEL_WIDTH = 110
_LABEL_CHAR_PX = 6.5
# Numbers inside a category label, for the natural order ("item 2" before "item 10").
# Thousands separators are part of the number ("1,000 to 2,000" are interval labels).
_NUMBER = re.compile(r"(\d[\d,]*(?:\.\d+)?)")
# Most decimals a value label can be rounded to.
MAX_DECIMALS = 6
# Significant digits of an axis number below 1, and decimals of one above (enough for any tick
# ECharts picks, few enough to hide float noise such as 0.30000000000000004).
_AXIS_DIGITS = 6


@dataclass(frozen=True)
class StyleSpec:
    """How a chart looks. Never part of a cache key (R-V3).

    Attributes:
        title: Title over the chart; empty for none.
        palette: Series colors, in order, resolved by the page from ``palette_name`` and
            ``color`` for the active theme. Never saved: it changes with light and dark mode.
        palette_name: Palette chosen (a key of ``palettes``).
        color: Color picked for a chart of one series, replacing the first of the palette;
            None to keep the palette.
        legend: Show the legend (on top). Only drawn with more than one series, or for a pie.
        labels: Values written on bars, slices, points and cells.
        label_decimals: Decimals the values written on bars, points and cells are rounded to
            (0 to ``MAX_DECIMALS``); whole numbers never get trailing zeros.
        axis_format: Numbers of the value axes (and of the heatmap scale) in full with
            thousands separators, or compact (1.2K, 3.4M). Never in scientific notation.
        horizontal: Bars and boxes along the y axis.
        color_each: Boxplot: each box takes the next color of the palette, instead of all of
            them the first one.
        stack: Series of bars, lines and histograms stacked instead of side by side.
        sort: Order of the bars: as computed (most frequent first, or intervals in order), by
            value, or by label (numeric when every label is a number, natural otherwise).
        area: Lines filled down to the axis.
        connect_gaps: Lines go on across intervals or periods without a value, instead of
            stopping there.
        donut: Pie with a hole.
        symbol_size: Diameter (px) of the scatter points.
        opacity: Opacity of the scatter points.
        trend: Least squares line over the scatter.
        log_scale: Logarithmic value axis (zeros and negatives are not drawn).
        size: Height of the chart (``HEIGHTS``).
    """

    title: str = ""
    palette: tuple[str, ...] = ()
    palette_name: str = "Theme"
    color: str | None = None
    legend: bool = True
    labels: bool = False
    label_decimals: int = 2
    axis_format: AxisFormat = "full"
    horizontal: bool = False
    color_each: bool = False
    stack: bool = False
    sort: SortOrder = "data"
    area: bool = False
    connect_gaps: bool = False
    donut: bool = True
    symbol_size: int = 7
    opacity: float = 0.7
    trend: bool = True
    log_scale: bool = False
    size: ChartSize = "medium"

    @property
    def height(self) -> int:
        """Height of the chart in px."""
        return HEIGHTS[self.size]


_ALL = frozenset(ChartKind)
# Chart kinds each field of ``StyleSpec`` changes; the page only draws its widget for them, and
# a saved configuration only carries those of its kind. Fields not listed change every kind.
# Kinds of one series take a color; scatter points and boxes have no single value to label.
STYLE_KINDS: dict[str, frozenset[ChartKind]] = {
    "color": _ALL - {ChartKind.PIE, ChartKind.HEATMAP},
    "color_each": frozenset({ChartKind.BOXPLOT}),
    "horizontal": frozenset({ChartKind.BAR, ChartKind.BOXPLOT}),
    "stack": frozenset({ChartKind.BAR, ChartKind.LINE, ChartKind.HISTOGRAM}),
    "sort": frozenset({ChartKind.BAR}),
    "area": frozenset({ChartKind.LINE}),
    "connect_gaps": frozenset({ChartKind.LINE}),
    "donut": frozenset({ChartKind.PIE}),
    "symbol_size": frozenset({ChartKind.SCATTER}),
    "opacity": frozenset({ChartKind.SCATTER}),
    "trend": frozenset({ChartKind.SCATTER}),
    "labels": _ALL - {ChartKind.SCATTER, ChartKind.BOXPLOT},
    # A pie writes the share, already rounded to one decimal.
    "label_decimals": _ALL - {ChartKind.SCATTER, ChartKind.BOXPLOT, ChartKind.PIE},
    "axis_format": _ALL - {ChartKind.PIE},
    "log_scale": _ALL - {ChartKind.PIE},
}


def applies(field: str, kind: ChartKind) -> bool:
    """Whether the ``StyleSpec`` field ``field`` changes a chart of ``kind``."""
    return kind in STYLE_KINDS.get(field, _ALL)


# Fields of ``StyleSpec`` that are not saved: ``palette`` is derived from the theme.
_NOT_SAVED = frozenset({"palette"})
# Allowed values of the ``Literal`` fields, checked when a configuration is loaded.
_CHOICES: dict[str, tuple[str, ...]] = {
    "sort": get_args(SortOrder.__value__),
    "size": get_args(ChartSize.__value__),
    "axis_format": get_args(AxisFormat.__value__),
}


def style_to_dict(style: StyleSpec, kind: ChartKind, *, auto_title: str = "") -> dict[str, Any]:
    """``style`` as JSON values: only the fields that change a chart of ``kind``
    (``STYLE_KINDS``), without ``palette`` (the page rebuilds it from the theme).

    Args:
        style: The look to save.
        kind: Kind of the chart it is saved with. A "trend" or "donut" in the file of a line
            chart reads as an option the chart has and ignores.
        auto_title: The title the page writes when the person leaves it empty: saved as empty,
            so a loaded chart titles itself after its own columns.
    """
    out = {
        f.name: getattr(style, f.name)
        for f in fields(style)
        if f.name not in _NOT_SAVED and applies(f.name, kind)
    }
    if out["title"] == auto_title:
        out["title"] = ""
    return out


def style_from_dict(raw: Mapping[str, Any]) -> StyleSpec:
    """The ``StyleSpec`` that ``style_to_dict`` wrote. Unknown keys are ignored, missing ones
    take their default.

    Raises:
        ValueError: A value is not of the type of its field, or not one of its choices.
    """
    if not isinstance(raw, Mapping):
        raise ValueError("The chart style must be an object.")
    defaults = StyleSpec()
    values: dict[str, Any] = {}
    for f in fields(StyleSpec):
        if f.name in _NOT_SAVED or f.name not in raw:
            continue
        value, default = raw[f.name], getattr(defaults, f.name)
        if f.name == "color":
            # Any other text would reach ``with_alpha`` (boxplot) and the color picker, which
            # only read ``#rrggbb``.
            ok = value is None or (isinstance(value, str) and bool(_HEX_COLOR.fullmatch(value)))
        elif isinstance(default, bool):
            ok = isinstance(value, bool)
        elif isinstance(default, (int, float)):
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        else:
            ok = isinstance(value, str)
        if ok and f.name in _CHOICES:
            ok = value in _CHOICES[f.name]
        if ok and f.name == "label_decimals":
            # The browser rejects a fraction or more than 100 decimals: the chart goes blank.
            ok = isinstance(value, int) and 0 <= value <= MAX_DECIMALS
        if not ok:
            raise ValueError(f"Style {f.name!r} has a wrong value: {value!r}.")
        values[f.name] = value
    return StyleSpec(**values)


def build_option(data: ChartData, style: StyleSpec, colors: ChartColors) -> dict[str, Any]:
    """ECharts option of ``data`` drawn with ``style``."""
    match data.kind:
        case ChartKind.BAR:
            body = _bar(data, style, colors)
        case ChartKind.LINE:
            body = _line(data, style, colors)
        case ChartKind.HISTOGRAM:
            body = _histogram(data, style, colors)
        case ChartKind.SCATTER:
            body = _scatter(data, style, colors)
        case ChartKind.BOXPLOT:
            body = _boxplot(data, style, colors)
        case ChartKind.PIE:
            body = _pie(data, style, colors)
        case ChartKind.HEATMAP:
            body = _heatmap(data, style, colors)
    body["series"] = [_compact(s) for s in body["series"]]
    return {**_frame(data, style), **body}


def _compact(series: dict[str, Any]) -> dict[str, Any]:
    """``series`` without its None keys: ECharts reads a missing key as "use the default", but
    not always a ``null`` one (``barGap: null`` is not the default gap).
    """
    return {k: v for k, v in series.items() if v is not None}


def _frame(data: ChartData, style: StyleSpec) -> dict[str, Any]:
    """Parts every chart shares: colors, title, legend, toolbox and animation."""
    option: dict[str, Any] = {
        # Animation helps few marks change; thousands only slow the browser down (R-V6).
        "animation": data.elements < _LARGE,
        "toolbox": {
            "right": 8,
            "top": 0,
            # ECharts' own PNG export: with the colors of the page (light or dark mode).
            "feature": {"restore": {"title": "Reset"}, "saveAsImage": {}},
        },
    }
    if style.palette:
        option["color"] = list(style.palette)
    if style.title:
        option["title"] = {"text": style.title, "left": 0, "top": 0, "textStyle": {"fontSize": 15}}
    if _shows_legend(data, style):
        option["legend"] = {
            "type": "scroll",
            "top": _TITLE_PX if style.title else 0,
            "left": "center",
        }
    return option


def _shows_legend(data: ChartData, style: StyleSpec) -> bool:
    """Whether the legend is drawn: asked for, and there is more than one thing to name."""
    named = len(data.series) + (1 if _has_trend(data, style) else 0)
    return style.legend and (named > 1 or data.kind == ChartKind.PIE)


def _grid(data: ChartData, style: StyleSpec) -> dict[str, Any]:
    """Margins that leave room for the title, the legend and the axis names.

    The margins place the plot; ``outerBounds`` (ECharts 6) then shrinks it only as much as
    the axis labels and names need to stay inside the chart. ``containLabel`` did not count
    the axis names, nor the first and last rotated labels of an axis without boundary gap
    (lines): they were cut at the edges.
    """
    top = 16 + _AXIS_NAME_TOP_PX + (_TITLE_PX if style.title else 0)
    top += _LEGEND_PX if _shows_legend(data, style) else 0
    bottom = 8 + _AXIS_NAME_BOTTOM_PX
    return {
        "top": top,
        "bottom": bottom,
        "left": 16,
        "right": 24,
        "outerBoundsMode": "auto",
        "outerBoundsContain": "all",
    }


def _has_trend(data: ChartData, style: StyleSpec) -> bool:
    return data.kind == ChartKind.SCATTER and data.trend is not None and style.trend


def _value_axis(
    name: str, style: StyleSpec, *, scale: bool = False, vertical: bool = True
) -> dict[str, Any]:
    """Value axis: linear or logarithmic, with the dashed grid lines of the app."""
    axis: dict[str, Any] = {
        "type": "log" if style.log_scale else "value",
        "splitLine": {"lineStyle": _SPLIT_LINE},
        "axisLabel": {"formatter": axis_formatter(style.axis_format)},
        "scale": scale,
    }
    return axis | _axis_name(name, vertical=vertical)


def axis_formatter(fmt: AxisFormat) -> Js:
    """Formatter of the numbers of a value axis, written as ``fmt`` says.

    ``toLocaleString`` never falls into scientific notation, which JavaScript uses below 1e-6
    and from 1e21. Below 1 the number keeps its significant digits (0.000012), else it is
    rounded to ``_AXIS_DIGITS`` decimals. "compact" shortens from a thousand on (1.25M).
    """
    full = (
        f"Math.abs(v) < 1 ? v.toLocaleString('en-US', {{maximumSignificantDigits: {_AXIS_DIGITS}}})"
        f" : v.toLocaleString('en-US', {{maximumFractionDigits: {_AXIS_DIGITS}}})"
    )
    if fmt == "compact":
        full = (
            "Math.abs(v) >= 1000 ? v.toLocaleString('en-US', "
            f"{{notation: 'compact', maximumFractionDigits: 2}}) : {full}"
        )
    return Js(f"function (v) {{ return typeof v === 'number' ? ({full}) : v; }}")


def label_formatter(decimals: int) -> Js:
    """Formatter of the values written on marks: rounded to at most ``decimals`` decimals, with
    thousands separators. The value is the last item when a point is a list (``[time, value]``
    of a line, ``[x, y, value]`` of a heatmap cell).
    """
    return Js(
        "function (p) { var v = Array.isArray(p.value) ? p.value[p.value.length - 1] : p.value;"
        " return typeof v === 'number'"
        f" ? v.toLocaleString('en-US', {{maximumFractionDigits: {decimals}}}) : ''; }}"
    )


def _category_axis(
    labels: list[str], style: StyleSpec, *, name: str = "", show: bool = True
) -> dict[str, Any]:
    """Category axis named ``name``. Long labels are cut with "…" and the tooltip shows them
    in full. With many labels they are rotated, and the name moves down below them.
    """
    many = len(labels) > 8 and not style.horizontal
    axis = {
        "type": "category",
        "data": labels,
        "axisTick": {"alignWithLabel": True},
        "axisLabel": {
            "show": show,
            "interval": 0 if len(labels) <= 30 else "auto",
            "rotate": _LABEL_ROTATE if many else 0,
            "width": _LABEL_WIDTH,
            "overflow": "truncate",
            "hideOverlap": True,
        },
    } | _axis_name(name)
    if many and show and name:
        # Height of the longest rotated label (``_LABEL_CHAR_PX`` a character, cut at its
        # width), plus room for the name.
        longest = min(max(map(len, labels)) * _LABEL_CHAR_PX, _LABEL_WIDTH)
        axis["nameGap"] = round(longest * math.sin(math.radians(_LABEL_ROTATE)) + 30)
    return axis


def _axis_name(name: str, *, vertical: bool = False) -> dict[str, Any]:
    """Name of an axis: under a horizontal one, above a vertical one (rotated in the middle
    it would sit on the tick labels, whose width ``nameGap`` cannot know).
    """
    if not name:
        return {}
    text = {"fontSize": 12}
    if vertical:
        # Starting at the axis: centered on it, a long name ran out of the chart on the left.
        text = text | {"align": "left"}
        return {"name": name, "nameLocation": "end", "nameGap": 12, "nameTextStyle": text}
    return {"name": name, "nameLocation": "middle", "nameGap": 34, "nameTextStyle": text}


def _zoom(axis: Literal["x", "y"]) -> list[dict[str, Any]]:
    """Zoom with the mouse wheel along ``axis``; the toolbox "Reset" undoes it."""
    return [{"type": "inside", f"{axis}AxisIndex": 0, "filterMode": "none"}]


def _log_safe(value: Any, style: StyleSpec) -> Any:
    """``value``, or None on a log axis when it cannot be drawn there (zero or negative)."""
    if style.log_scale and value is not None and value <= 0:
        return None
    return value


def label_order(labels: Sequence[str]) -> list[int]:
    """Positions of ``labels`` sorted as a person reads them.

    Numerically when every label is a number ("1, 2, 10", not "1, 10, 2"); else in natural
    order, comparing the numbers inside the labels as numbers ("item 2" before "item 10",
    "900 to 1,000" before "1,000 to 2,000") and the text without case.
    """
    numbers = [_as_number(label) for label in labels]
    if all(n is not None for n in numbers):
        return sorted(range(len(labels)), key=lambda i: numbers[i])
    keys = [_natural_key(label) for label in labels]
    return sorted(range(len(labels)), key=lambda i: keys[i])


def _as_number(label: str) -> float | None:
    """``label`` as a finite number ("1,234.5" included), or None if it is not one."""
    try:
        value = float(label.replace(",", ""))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _natural_key(label: str) -> tuple[str | float, ...]:
    """Sort key of ``label``: its text and number parts alternate (text first), so two keys
    always compare text with text and numbers with numbers.
    """
    parts = _NUMBER.split(label)
    return tuple(
        float(part.replace(",", "")) if i % 2 else part.casefold() for i, part in enumerate(parts)
    )


def _order(data: ChartData, sort: SortOrder) -> list[int]:
    """Positions of the categories in the order asked; "Other" always last."""
    n = len(data.categories)
    totals = [sum(s.data[i] or 0 for s in data.series) for i in range(n)]
    match sort:
        case "desc":
            order = sorted(range(n), key=lambda i: -totals[i])
        case "asc":
            order = sorted(range(n), key=lambda i: totals[i])
        case "label":
            order = label_order(data.categories)
        case _:
            order = list(range(n))
    if data.other in data.categories:
        last = data.categories.index(data.other)
        order = [i for i in order if i != last] + [last]
    return order


def _bar(data: ChartData, style: StyleSpec, colors: ChartColors) -> dict[str, Any]:
    order = _order(data, style.sort)
    labels = [data.categories[i] for i in order]
    stacked = style.stack and len(data.series) > 1
    radius = [0, 4, 4, 0] if style.horizontal else [4, 4, 0, 0]
    series = []
    for s in data.series:
        items = []
        for i in order:
            item: dict[str, Any] = {"value": _log_safe(s.data[i], style)}
            if data.categories[i] == data.other:
                item["itemStyle"] = {"opacity": _OTHER_OPACITY}
            items.append(item)
        series.append(
            {
                "type": "bar",
                "name": s.name,
                "data": items,
                "stack": "total" if stacked else None,
                "itemStyle": {"borderRadius": 0 if stacked else radius},
                "label": _value_label(
                    style.labels,
                    "inside" if stacked else ("right" if style.horizontal else "top"),
                    style,
                    colors,
                ),
                "emphasis": {"focus": "series"},
                "large": len(items) > _LARGE,
            }
        )
    cat_axis = _category_axis(labels, style, name="" if style.horizontal else data.x_name)
    val_axis = _value_axis(data.y_name, style, vertical=not style.horizontal)
    option: dict[str, Any] = {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": _grid(data, style),
        "series": series,
    }
    if style.horizontal:
        cat_axis["inverse"] = True
        option |= {"xAxis": val_axis, "yAxis": cat_axis, "dataZoom": _zoom("y")}
    else:
        option |= {"xAxis": cat_axis, "yAxis": val_axis, "dataZoom": _zoom("x")}
    return option


def _value_label(
    show: bool, position: str, style: StyleSpec, colors: ChartColors
) -> dict[str, Any]:
    """Value written next to a bar or point, rounded to ``style.label_decimals``. Outside the
    mark it takes the text color of the theme without an outline (see ``_NO_OUTLINE``); inside
    a stacked bar ECharts already picks a color that contrasts with the bar.
    """
    label: dict[str, Any] = {"show": show, "position": position}
    if show:
        label["formatter"] = label_formatter(style.label_decimals)
    if position != "inside":
        label |= {"color": colors.emphasis, **_NO_OUTLINE}
    return label


def _line(data: ChartData, style: StyleSpec, colors: ChartColors) -> dict[str, Any]:
    """Lines over time (``[ms, value]`` pairs), or over the intervals of a numeric x (values
    aligned with ``data.categories``).
    """
    stacked = style.stack and len(data.series) > 1
    points = max((len(s.data) for s in data.series), default=0)

    def values(s: Series) -> list[Any]:
        if data.time_axis:
            return [[t, _log_safe(v, style)] for t, v in s.data]
        return [_log_safe(v, style) for v in s.data]

    series = [
        {
            "type": "line",
            "name": s.name,
            "data": values(s),
            "stack": "total" if stacked else None,
            "areaStyle": {"opacity": 0.25} if style.area or stacked else None,
            "showSymbol": points <= 60,
            "symbolSize": 5,
            "sampling": "lttb",
            "connectNulls": style.connect_gaps,
            "label": _value_label(style.labels and points <= 40, "top", style, colors),
            "emphasis": {"focus": "series"},
        }
        for s in data.series
    ]
    if data.time_axis:
        x_axis: dict[str, Any] = {"type": "time", "splitLine": {"show": False}}
    else:
        x_axis = _category_axis(list(data.categories), style, name=data.x_name) | {
            "boundaryGap": False
        }
    return {
        "tooltip": {"trigger": "axis"},
        "grid": _grid(data, style),
        "xAxis": x_axis if not data.time_axis else x_axis | _axis_name(data.x_name),
        "yAxis": _value_axis(data.y_name, style),
        "dataZoom": _zoom("x"),
        "series": series,
    }


def _histogram(data: ChartData, style: StyleSpec, colors: ChartColors) -> dict[str, Any]:
    split = len(data.series) > 1
    overlay = split and not style.stack
    series = [
        {
            "type": "bar",
            "name": s.name,
            "data": [_log_safe(v, style) for v in s.data],
            "stack": "total" if split and style.stack else None,
            "barGap": "-100%" if overlay else None,
            "barCategoryGap": "4%",
            "itemStyle": {
                "opacity": _OVERLAY_OPACITY if overlay else 1,
                "borderRadius": 0 if split else [3, 3, 0, 0],
            },
            "label": _value_label(style.labels and len(s.data) <= 30, "top", style, colors),
            "emphasis": {"focus": "series"},
        }
        for s in data.series
    ]
    # Bin labels ("12.5 to 15") are too long for the axis: shown in the tooltip only.
    x_axis = _category_axis(list(data.categories), style, name=data.x_name, show=False)
    return {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": _grid(data, style),
        "xAxis": x_axis | {"nameGap": 12},
        "yAxis": _value_axis(data.y_name, style) | {"minInterval": 1},
        "dataZoom": _zoom("x"),
        "series": series,
    }


def _trend_name(data: ChartData) -> str:
    return f"Trend (r = {data.r:.2f})" if data.r is not None else "Trend"


def _scatter(data: ChartData, style: StyleSpec, colors: ChartColors) -> dict[str, Any]:
    large = data.elements > _LARGE
    series: list[dict[str, Any]] = [
        {
            "type": "scatter",
            "name": s.name,
            "data": [[x, _log_safe(y, style)] for x, y in s.data],
            "symbolSize": style.symbol_size,
            "itemStyle": {"opacity": style.opacity},
            "large": large,
            "largeThreshold": _LARGE,
            "progressive": 4_000 if large else 0,
            "emphasis": {"focus": "series"},
        }
        for s in data.series
    ]
    if _has_trend(data, style) and data.x_range and data.trend:
        slope, intercept = data.trend
        ends = [[x, slope * x + intercept] for x in data.x_range]
        series.append(
            {
                "type": "line",
                "name": _trend_name(data),
                "data": ends,
                "symbol": "none",
                "silent": True,
                "lineStyle": {"color": colors.highlight, "width": 2, "type": "dashed"},
                "itemStyle": {"color": colors.highlight},
                "z": 10,
            }
        )
    return {
        "tooltip": {"trigger": "item"},
        "grid": _grid(data, style),
        "xAxis": {
            "type": "value",
            "scale": True,
            "splitLine": {"lineStyle": _SPLIT_LINE},
            "axisLabel": {"formatter": axis_formatter(style.axis_format)},
        }
        | _axis_name(data.x_name),
        "yAxis": _value_axis(data.y_name, style, scale=True),
        "dataZoom": [*_zoom("x"), *_zoom("y")],
        "series": series,
    }


def _boxplot(data: ChartData, style: StyleSpec, colors: ChartColors) -> dict[str, Any]:
    palette = style.palette or (colors.accent,)
    boxes = data.series[0].data if data.series else ()
    items = []
    for i, box in enumerate(boxes):
        # Past the end of the palette the colors repeat, as ECharts does with series.
        color = palette[i % len(palette)] if style.color_each else palette[0]
        items.append(
            {
                "value": list(box),
                "itemStyle": {
                    "color": with_alpha(color, 0.3),
                    "borderColor": color,
                    "borderWidth": 2,
                    "opacity": _OTHER_OPACITY if data.categories[i] == data.other else 1,
                },
            }
        )
    horizontal = style.horizontal
    outliers = [[v, i] if horizontal else [i, v] for i, v in data.outliers]
    series = [
        {"type": "boxplot", "name": data.y_name, "data": items, "boxWidth": [12, 60]},
        {
            "type": "scatter",
            "name": "Outliers",
            "data": outliers,
            "symbolSize": 7,
            "itemStyle": {
                "color": colors.outlier,
                "opacity": 0.6,
                "borderColor": colors.background,
                "borderWidth": 1,
            },
        },
    ]
    cat_axis = _category_axis(list(data.categories), style, name="" if horizontal else data.x_name)
    val_axis = _value_axis(data.y_name, style, scale=True, vertical=not horizontal)
    option: dict[str, Any] = {
        "tooltip": {"trigger": "item"},
        "grid": _grid(data, style),
        "series": series,
    }
    if horizontal:
        option |= {"xAxis": val_axis, "yAxis": cat_axis | {"inverse": True}, "dataZoom": _zoom("x")}
    else:
        option |= {"xAxis": cat_axis, "yAxis": val_axis, "dataZoom": _zoom("y")}
    return option


def _pie(data: ChartData, style: StyleSpec, colors: ChartColors) -> dict[str, Any]:
    values = data.series[0].data if data.series else ()
    items = []
    for name, value in zip(data.categories, values, strict=True):
        if value is None or value <= 0:
            continue
        item: dict[str, Any] = {"name": name, "value": value}
        if name == data.other:
            item["itemStyle"] = {"opacity": _OTHER_OPACITY}
        items.append(item)
    top = (_TITLE_PX if style.title else 0) + (_LEGEND_PX if _shows_legend(data, style) else 0)
    return {
        "tooltip": {"trigger": "item", "formatter": "{b}<br/><b>{c}</b> ({d}%)"},
        "series": [
            {
                "type": "pie",
                "name": data.y_name,
                "data": items,
                "radius": ["42%", "70%"] if style.donut else [0, "70%"],
                "center": ["50%", "55%" if top else "50%"],
                "avoidLabelOverlap": True,
                # One decimal: 14.4% reads at a glance, 14.36% is noise in a pie.
                "percentPrecision": 1,
                "itemStyle": {
                    "borderRadius": 4,
                    "borderColor": colors.background,
                    "borderWidth": 2,
                },
                "label": _pie_label(style.labels, colors.emphasis),
                "labelLine": {"length": 12, "length2": 14},
                "emphasis": {
                    "scaleSize": 6,
                    "label": {
                        "fontSize": _PIE_EMPHASIS_SIZE,
                        "fontWeight": _PIE_STRONG,
                        "color": colors.emphasis,
                        **_NO_OUTLINE,
                    },
                },
            }
        ],
    }


def _pie_label(percent: bool, color: str) -> dict[str, Any]:
    """Labels of the slices in ``color`` (the text color of the theme): the name, and the share
    under it in bold when ``percent``.

    Rich text styles each part: ``{name|...}`` and ``{pct|...}`` refer to ``rich``.
    """
    name = {
        "fontSize": _PIE_LABEL_SIZE,
        "fontWeight": _PIE_LABEL_WEIGHT,
        "lineHeight": 18,
        "color": color,
        **_NO_OUTLINE,
    }
    label: dict[str, Any] = {"show": True, **name, "formatter": "{b}"}
    if percent:
        pct = {
            "fontSize": _PIE_PERCENT_SIZE,
            "fontWeight": _PIE_STRONG,
            "lineHeight": 20,
            "color": color,
            **_NO_OUTLINE,
        }
        label |= {"formatter": "{name|{b}}\n{pct|{d}%}", "rich": {"name": name, "pct": pct}}
    return label


def heat_scale(low: float, high: float, colors: ChartColors) -> tuple[float, float, list[str]]:
    """Color scale of a heatmap whose cells go from ``low`` to ``high``.

    Diverging around zero (two hues, the middle color at 0) when the values cross it, so the
    sign reads at a glance; else sequential, from the background to the accent.

    Returns:
        The ends of the scale and its colors.
    """
    if low < 0 < high:
        bound = max(abs(low), abs(high))
        return -bound, bound, list(colors.diverging)
    sequential = [mix(colors.accent, colors.background, 0.88), colors.accent_light, colors.accent]
    return low, high if high > low else low + 1, sequential


def _heatmap(data: ChartData, style: StyleSpec, colors: ChartColors) -> dict[str, Any]:
    low, high, scale = heat_scale(*data.value_range, colors)
    cells = data.series[0].data if data.series else ()
    show_labels = style.labels or len(cells) <= 120
    # Room for the color scale and its labels on the right.
    grid = _grid(data, style) | {"right": 110}
    return {
        "tooltip": {"position": "top"},
        "grid": grid,
        "xAxis": _category_axis(list(data.categories), style, name=data.x_name)
        | {"splitArea": _HEAT_BANDS},
        "yAxis": {
            "type": "category",
            "data": list(data.rows),
            "inverse": True,
            "splitArea": _HEAT_BANDS,
            "axisLabel": {"width": 110, "overflow": "truncate"},
        }
        # The axis is inverted (first row on top): its "start" is the top.
        | _axis_name(data.y_name, vertical=True)
        | {"nameLocation": "start"},
        "visualMap": {
            "min": low,
            "max": high,
            "calculable": True,
            "orient": "vertical",
            "right": 8,
            "top": "middle",
            "inRange": {"color": scale},
            "formatter": axis_formatter(style.axis_format),
        },
        "series": [
            {
                "type": "heatmap",
                "name": data.series[0].name if data.series else "",
                "data": [list(c) for c in cells],
                "label": {
                    "show": show_labels,
                    "formatter": label_formatter(style.label_decimals),
                },
                "itemStyle": {"borderColor": colors.background, "borderWidth": 1},
                "emphasis": {"itemStyle": {"borderColor": colors.emphasis, "borderWidth": 1}},
            }
        ],
    }


# Palettes


def palettes(theme_palette: tuple[str, ...], colors: ChartColors) -> dict[str, tuple[str, ...]]:
    """Named palettes, all built from the active theme (no color is written here).

    Args:
        theme_palette: Series colors of the theme (``theme.chart_theme().palette``).
        colors: Chart colors of the theme (``theme.chart_colors()``).
    """
    accent, back = colors.accent, colors.background
    blue, _, middle, _, red = colors.diverging
    return {
        "Theme": theme_palette,
        "Accent shades": tuple(mix(accent, back, t) for t in (0, 0.25, 0.45, 0.6, 0.72, 0.82)),
        "Contrast": (accent, colors.highlight, colors.mean, colors.outlier, *theme_palette[4:]),
        "Cold to warm": (blue, mix(blue, middle, 0.5), middle, mix(red, middle, 0.5), red),
    }
