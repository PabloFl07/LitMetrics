"""What a chart is, independent of any dataset: chart kinds, the columns each one takes
(``ROLES``), row filters and ``DataSpec``, the cache key of a chart's data (R-V3, R-V8), and
``ChartData``, what ``chart_data`` returns.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd
from pandas.api import types as ptypes

from litmetrics.config import VisualizeConfig
from litmetrics.core.schema import SemanticType


class ChartKind(StrEnum):
    """Kinds of chart the page draws."""

    BAR = "bar"
    LINE = "line"
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    BOXPLOT = "boxplot"
    PIE = "pie"
    HEATMAP = "heatmap"


class Agg(StrEnum):
    """How the value column is aggregated in each group; ``COUNT`` counts rows."""

    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"


class TimeUnit(StrEnum):
    """Bucket of a time series; ``AUTO`` picks the finest readable one."""

    AUTO = "auto"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


# What kind of column each role accepts.
type Accepts = frozenset[SemanticType]
NUMERIC: Accepts = frozenset({SemanticType.NUMERIC})
CATEGORICAL: Accepts = frozenset({SemanticType.CATEGORICAL, SemanticType.BOOLEAN})
DATETIME: Accepts = frozenset({SemanticType.DATETIME})


@dataclass(frozen=True)
class Role:
    """One column a chart kind takes.

    Attributes:
        field: Attribute of ``DataSpec`` it fills: ``x``, ``y``, ``value`` or ``group``.
        label: Label of its selector.
        accepts: Semantic types of the columns it takes.
        required: False if the chart works without it ("Split by").
        help: Tooltip of its selector.
    """

    field: str
    label: str
    accepts: Accepts
    required: bool = True
    help: str = ""


_VALUE_HELP = "Without a value, the chart counts rows."
_GROUP_HELP = "One color per value of this column (the most frequent ones)."
_INTERVALS_HELP = "A numeric column is grouped in intervals."

# Columns each chart kind takes, in the order the editor shows them.
ROLES: dict[ChartKind, tuple[Role, ...]] = {
    ChartKind.BAR: (
        Role("x", "X axis", CATEGORICAL | NUMERIC, help=_INTERVALS_HELP),
        Role("value", "Value", NUMERIC, False, _VALUE_HELP),
        Role("group", "Split by", CATEGORICAL, False, _GROUP_HELP),
    ),
    ChartKind.LINE: (
        Role("x", "X axis", DATETIME | NUMERIC, help="Dates by a time unit; " + _INTERVALS_HELP),
        Role("value", "Value", NUMERIC, False, _VALUE_HELP),
        Role("group", "One line per", CATEGORICAL, False, _GROUP_HELP),
    ),
    ChartKind.HISTOGRAM: (
        Role("x", "Column", NUMERIC | DATETIME),
        Role("group", "Split by", CATEGORICAL, False, _GROUP_HELP),
    ),
    ChartKind.SCATTER: (
        Role("x", "X axis", NUMERIC),
        Role("y", "Y axis", NUMERIC),
        Role("group", "Color by", CATEGORICAL, False, _GROUP_HELP),
    ),
    ChartKind.BOXPLOT: (
        Role("value", "Value", NUMERIC),
        Role("x", "Group by", CATEGORICAL, False, "One box per value of this column."),
    ),
    ChartKind.PIE: (
        Role("x", "Category", CATEGORICAL),
        Role("value", "Value", NUMERIC, False, "Without a value, slices are rows."),
    ),
    ChartKind.HEATMAP: (
        Role("x", "Columns", CATEGORICAL),
        Role("y", "Rows", CATEGORICAL),
        Role("value", "Value", NUMERIC, False, _VALUE_HELP),
    ),
}

# Aggregations offered when a chart has a value column. A pie only adds up parts of a whole.
AGGREGATIONS: dict[ChartKind, tuple[Agg, ...]] = {
    ChartKind.PIE: (Agg.SUM,),
    # A boxplot draws every value: there is nothing to aggregate.
    ChartKind.BOXPLOT: (),
}
DEFAULT_AGGREGATIONS = (Agg.MEAN, Agg.SUM, Agg.MEDIAN, Agg.MIN, Agg.MAX)


# Row filters: part of the cache key (R-V8), so frozen and JSON-friendly (ISO dates, floats).


@dataclass(frozen=True)
class DateFilter:
    """Rows whose date in ``column`` is between ``start`` and ``end``, both days included.

    Attributes:
        column: Date column.
        start, end: ISO dates (``"2024-01-31"``); None leaves that side open.
    """

    column: str
    start: str | None = None
    end: str | None = None


@dataclass(frozen=True)
class ValuesFilter:
    """Rows whose value of ``column``, as text, is one of ``values``.

    Attributes:
        column: Categorical, boolean, free text or identifier column.
        values: Values kept, as shown in the editor (``str`` of each value).
    """

    column: str
    values: tuple[str, ...] = ()


@dataclass(frozen=True)
class RangeFilter:
    """Rows whose number in ``column`` is between ``low`` and ``high``, both included.

    Attributes:
        column: Numeric column.
        low, high: Ends of the range; None leaves that side open.
    """

    column: str
    low: float | None = None
    high: float | None = None


type RowFilter = DateFilter | ValuesFilter | RangeFilter

# Name of each filter type in the JSON of a configuration.
FILTER_TYPES: dict[str, type[RowFilter]] = {
    "dates": DateFilter,
    "values": ValuesFilter,
    "range": RangeFilter,
}


@dataclass(frozen=True)
class DataSpec:
    """What a chart shows: everything that changes its data, and nothing else (R-V3).

    Attributes:
        kind: Chart kind.
        x, y, value, group: Columns of each role (see ``ROLES``); None when not used.
        agg: How ``value`` is aggregated. ``COUNT`` (rows) when there is no value.
        bins: Bars of a histogram, or intervals of a numeric x axis of a bar or line chart.
        top_k: Categories kept on the category axis; the rest are grouped in "Other".
        time_unit: Bucket of a time series.
        trim: Leave extreme values out of histograms and numeric or time axes, so a few
            placeholder dates (9999-12-31) or wrong numbers do not squash the chart.
        min_rows: Bar or line on intervals of a numeric x: rows an interval needs for its
            mean, median, min or max to be drawn (1 draws every one).
        filters: Rows kept: those that pass every filter.
    """

    kind: ChartKind
    x: str | None = None
    y: str | None = None
    value: str | None = None
    group: str | None = None
    agg: Agg = Agg.COUNT
    bins: int = 30
    top_k: int = 20
    time_unit: TimeUnit = TimeUnit.AUTO
    trim: bool = True
    min_rows: int = VisualizeConfig.min_rows_per_point
    filters: tuple[RowFilter, ...] = ()


@dataclass(frozen=True)
class Series:
    """One series of a chart.

    Attributes:
        name: Shown in the legend and the tooltip.
        data: Values aligned with ``ChartData.categories`` (bar, histogram, pie, line on
            intervals), ``[x, y]`` pairs (time line, scatter), ``[low, q1, median, q3, high]``
            rows (boxplot) or ``[column, row, value]`` cells (heatmap). Missing values are None.
    """

    name: str
    data: tuple[Any, ...]


@dataclass(frozen=True)
class ChartData:
    """Aggregated data of one chart: small, JSON-friendly, independent of the look.

    Attributes:
        kind: Chart kind it was computed for.
        series: Series to draw. Empty when nothing is left to draw.
        categories: Labels of the category axis (bar, histogram, boxplot, pie, heatmap
            columns, line on intervals).
        rows: Labels of the heatmap rows.
        x_name, y_name: Names of the axes ("Mean of income").
        time_axis: True if the x of the series are dates in epoch milliseconds.
        other: Label of the "Other" category, if some categories were grouped.
        outliers: Boxplot: ``[box index, value]`` of the outliers drawn.
        trend: Scatter: slope and intercept of the least squares line, on every point.
        x_range: Scatter: lowest and highest x, where the trend line starts and ends.
        r: Scatter: Pearson correlation on every point.
        value_range: Heatmap: lowest and highest cell.
        notes: What was cut or left out, for the caption under the chart.
    """

    kind: ChartKind
    series: tuple[Series, ...] = ()
    categories: tuple[str, ...] = ()
    rows: tuple[str, ...] = ()
    x_name: str = ""
    y_name: str = ""
    time_axis: bool = False
    other: str | None = None
    outliers: tuple[tuple[int, float], ...] = ()
    trend: tuple[float, float] | None = None
    x_range: tuple[float, float] | None = None
    r: float | None = None
    value_range: tuple[float, float] = (0.0, 0.0)
    notes: tuple[str, ...] = field(default=())

    @property
    def elements(self) -> int:
        """Points, bars or cells sent to the browser."""
        return sum(len(s.data) for s in self.series) + len(self.outliers)


def describe(spec: DataSpec, *, over_time: bool = True) -> str:
    """Default title of a chart: what it shows in a few words.

    Args:
        spec: The chart.
        over_time: Whether the x of a line chart is a date (else it is a numeric column).
    """
    what = value_name(spec)
    match spec.kind:
        case ChartKind.BAR | ChartKind.PIE:
            title = f"{what} by {spec.x}"
        case ChartKind.LINE:
            title = f"{what} over time" if over_time else f"{what} by {spec.x}"
        case ChartKind.HISTOGRAM:
            title = f"Distribution of {spec.x}"
        case ChartKind.SCATTER:
            title = f"{spec.y} vs {spec.x}"
        case ChartKind.BOXPLOT:
            title = f"{spec.value} by {spec.x}" if spec.x else f"Spread of {spec.value}"
        case ChartKind.HEATMAP:
            title = f"{what} by {spec.x} and {spec.y}"
    if spec.group and spec.kind not in (ChartKind.PIE, ChartKind.HEATMAP):
        title += f", by {spec.group}"
    return title


def value_name(spec: DataSpec) -> str:
    """Name of the measured value: "Rows" or "Mean of income"."""
    if spec.value is None or spec.agg == Agg.COUNT:
        return "Rows"
    return f"{spec.agg.capitalize()} of {spec.value}"


def is_number(column: pd.Series) -> bool:
    """Whether ``column`` holds numbers (booleans are numeric to pandas, not here)."""
    return ptypes.is_numeric_dtype(column) and not ptypes.is_bool_dtype(column)
