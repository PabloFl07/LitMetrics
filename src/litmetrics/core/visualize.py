"""Data side of the Visualization page: what a chart shows, aggregated from the dataset.

Pure: no Streamlit. ``chart_data(df, spec)`` turns the dataset into the small, aggregated
``ChartData`` that a chart draws (R-V1, R-V2 in docs/performance.md). Everything that changes
the numbers is in ``DataSpec``, a frozen dataclass that is the cache key (R-V3), row filters
included (R-V8); nothing about the look (colors, title, legend) is there, so changing the look
never touches pandas.

The page is a chart builder: the person picks the kind, the columns and the look. This module
only gives it what to offer and where to start:

- ``ROLES``: for each chart kind, which columns it takes (x, y, value, group) and of which
  semantic types. The editor is generated from this table.
- ``column_hints``: the columns that can go in a chart, in dataset order, without the ones that
  never make a useful chart (constant, empty, row numbers), plus what the filters need: any
  column with two values or more, drawn or not.
- ``default_spec``: a reasonable first chart of a kind, never a recommendation.
- ``spec_to_dict`` / ``spec_from_dict`` / ``fit_to_dataset``: a ``DataSpec`` as JSON, and how
  one saved from another dataset is adapted to the current one.
"""

import datetime as dt
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from enum import StrEnum
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from litmetrics.config import VisualizeConfig
from litmetrics.core.columns import histogram, value_labels
from litmetrics.core.sampling import positions, sample, sample_array
from litmetrics.core.schema import IDENTIFIER_MIN_DISTINCT, Schema, SemanticType, mostly_distinct


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


# Approximate days of each time unit, to estimate how many points a line will have.
_UNIT_DAYS = {
    TimeUnit.DAY: 1.0,
    TimeUnit.WEEK: 7.0,
    TimeUnit.MONTH: 30.44,
    TimeUnit.QUARTER: 91.31,
    TimeUnit.YEAR: 365.25,
}
# pandas period frequency of each time unit.
_UNIT_FREQ = {
    TimeUnit.DAY: "D",
    TimeUnit.WEEK: "W",
    TimeUnit.MONTH: "M",
    TimeUnit.QUARTER: "Q",
    TimeUnit.YEAR: "Y",
}

# What kind of column each role accepts.
type Accepts = frozenset[SemanticType]
_NUMERIC: Accepts = frozenset({SemanticType.NUMERIC})
_CATEGORICAL: Accepts = frozenset({SemanticType.CATEGORICAL, SemanticType.BOOLEAN})
_DATETIME: Accepts = frozenset({SemanticType.DATETIME})


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
        Role("x", "X axis", _CATEGORICAL | _NUMERIC, help=_INTERVALS_HELP),
        Role("value", "Value", _NUMERIC, False, _VALUE_HELP),
        Role("group", "Split by", _CATEGORICAL, False, _GROUP_HELP),
    ),
    ChartKind.LINE: (
        Role("x", "X axis", _DATETIME | _NUMERIC, help="Dates by a time unit; " + _INTERVALS_HELP),
        Role("value", "Value", _NUMERIC, False, _VALUE_HELP),
        Role("group", "One line per", _CATEGORICAL, False, _GROUP_HELP),
    ),
    ChartKind.HISTOGRAM: (
        Role("x", "Column", _NUMERIC | _DATETIME),
        Role("group", "Split by", _CATEGORICAL, False, _GROUP_HELP),
    ),
    ChartKind.SCATTER: (
        Role("x", "X axis", _NUMERIC),
        Role("y", "Y axis", _NUMERIC),
        Role("group", "Color by", _CATEGORICAL, False, _GROUP_HELP),
    ),
    ChartKind.BOXPLOT: (
        Role("value", "Value", _NUMERIC),
        Role("x", "Group by", _CATEGORICAL, False, "One box per value of this column."),
    ),
    ChartKind.PIE: (
        Role("x", "Category", _CATEGORICAL),
        Role("value", "Value", _NUMERIC, False, "Without a value, slices are rows."),
    ),
    ChartKind.HEATMAP: (
        Role("x", "Columns", _CATEGORICAL),
        Role("y", "Rows", _CATEGORICAL),
        Role("value", "Value", _NUMERIC, False, _VALUE_HELP),
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
_FILTER_TYPES: dict[str, type[RowFilter]] = {
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


# Column hints: which columns can go in a chart


@dataclass(frozen=True)
class ColumnHints:
    """Columns that can go in a chart, by kind, in dataset order, and what filters need.

    Attributes:
        order: Every usable column, in dataset order: the order of the selectors.
        numeric: Numeric columns with at least two values that are not a row number.
        categorical: Categorical and boolean columns with at least two values.
        datetime: Date columns with at least two values.
        ranked: The categorical columns, best first, only to pick the first chart
            (``default_spec``): those with 2 to 12 values first (they read well as bars and
            colors), then by number of values; those where one value is almost every row
            ("España" in 99% of rows) last, since a chart of them is one bar.
        distinct: Distinct values of each categorical column.
        best_pair: The two numeric columns with the highest absolute correlation, and it.
        left_out: Columns not offered and why ("constant", "row number"...).
        values: Values of each column filtered by value as text, the most frequent first.
        ranges: Lowest and highest value of each column filtered by range.
        date_ranges: First and last day of each date column, as ISO dates.
        filters: Every column a row filter can use, in dataset order, with its filter type
            (a key of ``_FILTER_TYPES``): the chart columns and also those never offered for
            a chart (identifiers, free text, row numbers), since rows can be picked by a
            column that is not drawn. Only columns with fewer than two values are missing.
        partial: Columns whose ``values`` are only the most frequent ones of a sample (free
            text, identifiers): other values exist and can be typed in.
    """

    order: tuple[str, ...] = ()
    numeric: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()
    datetime: tuple[str, ...] = ()
    ranked: tuple[str, ...] = ()
    distinct: tuple[tuple[str, int], ...] = ()
    best_pair: tuple[str, str, float] | None = None
    left_out: tuple[tuple[str, str], ...] = ()
    values: tuple[tuple[str, tuple[str, ...]], ...] = ()
    ranges: tuple[tuple[str, float, float], ...] = ()
    date_ranges: tuple[tuple[str, str, str], ...] = ()
    filters: tuple[tuple[str, str], ...] = ()
    partial: tuple[str, ...] = ()

    def columns_for(self, accepts: Accepts) -> tuple[str, ...]:
        """Columns of the types ``accepts``, in dataset order."""
        wanted: set[str] = set()
        if accepts & _CATEGORICAL:
            wanted.update(self.categorical)
        if accepts & _NUMERIC:
            wanted.update(self.numeric)
        if accepts & _DATETIME:
            wanted.update(self.datetime)
        return tuple(n for n in self.order if n in wanted)

    @property
    def filter_columns(self) -> tuple[str, ...]:
        """Columns a row filter can use, in dataset order."""
        return tuple(name for name, _ in self.filters)

    def filter_type(self, name: str) -> str | None:
        """Filter type of column ``name`` (``"dates"``, ``"values"``, ``"range"``), or None
        if it cannot be filtered.
        """
        return dict(self.filters).get(name)

    def n_distinct(self, name: str) -> int:
        """Distinct values of categorical column ``name`` (0 if it is not one)."""
        return dict(self.distinct).get(name, 0)

    def values_of(self, name: str) -> tuple[str, ...]:
        """Values of column ``name`` filtered by value, as text (empty if it is not one)."""
        return dict(self.values).get(name, ())

    def range_of(self, name: str) -> tuple[float, float]:
        """Lowest and highest value of column ``name``, filtered by range.

        Raises:
            KeyError: ``name`` is not filtered by range.
        """
        low, high = {n: (lo, hi) for n, lo, hi in self.ranges}[name]
        return low, high

    def dates_of(self, name: str) -> tuple[dt.date, dt.date]:
        """First and last day of date column ``name``.

        Raises:
            KeyError: ``name`` is not a usable date column.
        """
        start, end = {n: (a, b) for n, a, b in self.date_ranges}[name]
        return dt.date.fromisoformat(start), dt.date.fromisoformat(end)


# Categorical columns with this many values or fewer read well as bars and colors.
_FEW_VALUES = 12
# A categorical column where one value has a larger share of the rows than this says little.
_DOMINANT_SHARE = 0.9
# Why the columns of these semantic types are never offered.
_NOT_CHARTED = {
    SemanticType.IDENTIFIER: "identifier",
    SemanticType.TEXT: "free text",
}


def column_hints(df: pd.DataFrame, schema: Schema) -> ColumnHints:
    """The columns of ``df`` that can go in a chart. Cheap: one pass per column, once.

    Only leaves columns out, never ranks the selectors: a person knows which columns mean
    something (latitude, a postal code) better than any rule. Left out are columns with fewer
    than two values (constant, or all missing) and integer columns that count rows (1, 2, 3...,
    almost all distinct over a range as long as the column), which the schema does not mark as
    identifiers when their name has no "id".
    """
    left_out: list[tuple[str, str]] = []
    numeric: list[str] = []
    ranges: list[tuple[str, float, float]] = []
    for name in schema.names_of([SemanticType.NUMERIC]):
        column = df[name]
        # min / max skip missing values: an all-NaN column gives NaN < NaN, False (never
        # ``min != max``, which is True for NaN). Cheaper than ``nunique``, and needed anyway
        # for the range of the filter.
        low, high = column.min(), column.max()
        if not low < high:
            left_out.append((name, "constant" if low == high else "no values"))
            continue
        # A row number is no axis, but a range of rows is a filter ("rows 1 to 1,000").
        ranges.append((name, float(low), float(high)))
        if _is_row_number(column, float(low), float(high)):
            left_out.append((name, "row number"))
        else:
            numeric.append(name)
    categorical: list[str] = []
    distinct: dict[str, int] = {}
    dominated: dict[str, bool] = {}
    values: list[tuple[str, tuple[str, ...]]] = []
    for name in schema.names_of(_CATEGORICAL):
        counts = df[name].value_counts()
        counts = counts[counts > 0]
        if len(counts) < 2:
            left_out.append((name, "constant" if len(counts) else "no values"))
            continue
        categorical.append(name)
        distinct[name] = len(counts)
        dominated[name] = bool(counts.iloc[0] > _DOMINANT_SHARE * counts.sum())
        values.append((name, tuple(dict.fromkeys(str(v) for v in counts.index))))
    datetime: list[str] = []
    date_ranges: list[tuple[str, str, str]] = []
    for name in schema.names_of(_DATETIME):
        first, last = df[name].min(), df[name].max()
        if pd.isna(first) or first == last:
            left_out.append((name, "no values" if pd.isna(first) else "constant"))
            continue
        datetime.append(name)
        date_ranges.append((name, first.date().isoformat(), last.date().isoformat()))
    left_out += [
        (c.name, _NOT_CHARTED[c.semantic_type])
        for c in schema.columns
        if c.semantic_type in _NOT_CHARTED
    ]
    partial: list[str] = []
    filter_types = {
        **dict.fromkeys(datetime, "dates"),
        **dict.fromkeys(categorical, "values"),
        **{name: "range" for name, _, _ in ranges},
    }
    for name in schema.names_of(_NOT_CHARTED):
        found = _filter_of_uncharted(df[name])
        if found is None:
            continue
        filter_types[name], extent = found
        if filter_types[name] == "range":
            ranges.append((name, *extent))
        else:
            values.append((name, extent))
            partial.append(name)
    usable = {*numeric, *categorical, *datetime}
    position = {c.name: i for i, c in enumerate(schema.columns)}
    ranked = sorted(
        categorical, key=lambda n: (dominated[n], distinct[n] > _FEW_VALUES, distinct[n])
    )
    return ColumnHints(
        order=tuple(c.name for c in schema.columns if c.name in usable),
        numeric=tuple(numeric),
        categorical=tuple(categorical),
        datetime=tuple(datetime),
        ranked=tuple(ranked),
        distinct=tuple(distinct.items()),
        best_pair=_best_pair(df, numeric),
        left_out=tuple(sorted(left_out, key=lambda item: position[item[0]])),
        values=tuple(values),
        ranges=tuple(ranges),
        date_ranges=tuple(date_ranges),
        filters=tuple(
            (c.name, filter_types[c.name]) for c in schema.columns if c.name in filter_types
        ),
        partial=tuple(partial),
    )


def _filter_of_uncharted(column: pd.Series) -> tuple[str, Any] | None:
    """How a column never offered for a chart (identifier, free text) is filtered: by
    ``"range"`` with its lowest and highest number when it holds numbers, else by
    ``"values"`` with the most frequent ones of a sample (``VisualizeConfig.filter_values``).
    None when it has fewer than two values.

    Counting the values of a sample, not of the column: an id column of 3 M rows has 3 M of
    them, and the list only helps to pick; any value can be typed in.
    """
    if _is_number(column):
        low, high = column.min(), column.max()
        return ("range", (float(low), float(high))) if low < high else None
    counts = sample(column, VisualizeConfig.filter_sample).value_counts()
    counts = counts[counts > 0]
    if len(counts) < 2:
        return None
    top = counts.index[: VisualizeConfig.filter_values]
    return "values", tuple(dict.fromkeys(str(v) for v in top))


def _is_row_number(column: pd.Series, low: float, high: float) -> bool:
    """Whether integer ``column`` numbers rows (1..n, 1001..2000): almost all its values are
    distinct and they fill a range barely longer than the column.

    The span is checked first: it is free, and a measure (an income, a count of purchases)
    almost never fills its range so tightly. Only then are the distinct values counted.
    """
    if not ptypes.is_integer_dtype(column):
        return False
    present = int(column.notna().sum())
    if high - low + 1 > present / IDENTIFIER_MIN_DISTINCT:
        return False
    return mostly_distinct(column.nunique(), present)


def _best_pair(df: pd.DataFrame, numeric: Sequence[str]) -> tuple[str, str, float] | None:
    """The two most correlated numeric columns, on a sample (R-P2). None if fewer than two."""
    if len(numeric) < 2:
        return None
    names = list(numeric[: VisualizeConfig.pair_columns])
    corr = sample(df[names], VisualizeConfig.correlation_sample).corr().abs().to_numpy(copy=True)
    np.fill_diagonal(corr, np.nan)
    if np.isnan(corr).all():
        return (names[0], names[1], 0.0)
    i, j = np.unravel_index(np.nanargmax(corr), corr.shape)
    return (names[int(i)], names[int(j)], float(corr[i, j]))


def available_kinds(hints: ColumnHints) -> tuple[ChartKind, ...]:
    """Chart kinds the dataset has columns for, in the order of ``ChartKind``."""
    return tuple(k for k in ChartKind if default_spec(k, hints) is not None)


def default_spec(kind: ChartKind, hints: ColumnHints) -> DataSpec | None:
    """A reasonable first chart of ``kind``, or None if the dataset has no columns for it.

    Only the starting point when a kind is opened: the person changes it from there.
    """
    num, cat, dates = hints.numeric, hints.ranked, hints.datetime
    first_num = num[0] if num else None
    match kind:
        case ChartKind.BAR if cat:
            return DataSpec(kind, x=cat[0])
        case ChartKind.BAR if num:
            return DataSpec(kind, x=num[0], bins=VisualizeConfig.x_bins)
        case ChartKind.LINE if dates:
            agg = Agg.MEAN if first_num else Agg.COUNT
            return DataSpec(kind, x=dates[0], value=first_num, agg=agg)
        case ChartKind.LINE if len(num) >= 2:
            return DataSpec(kind, x=num[0], value=num[1], agg=Agg.MEAN, bins=VisualizeConfig.x_bins)
        case ChartKind.HISTOGRAM if num or dates:
            return DataSpec(kind, x=(num or dates)[0], bins=VisualizeConfig.histogram_bins)
        case ChartKind.SCATTER if hints.best_pair:
            a, b, _ = hints.best_pair
            return DataSpec(kind, x=a, y=b)
        case ChartKind.BOXPLOT if num:
            group = cat[0] if cat and hints.n_distinct(cat[0]) <= _FEW_VALUES else None
            return DataSpec(kind, value=num[0], x=group)
        case ChartKind.PIE if cat:
            return DataSpec(kind, x=cat[0], top_k=VisualizeConfig.max_slices)
        case ChartKind.HEATMAP if len(cat) >= 2:
            return DataSpec(kind, x=cat[0], y=cat[1])
    return None


def describe(spec: DataSpec, *, over_time: bool = True) -> str:
    """Default title of a chart: what it shows in a few words.

    Args:
        spec: The chart.
        over_time: Whether the x of a line chart is a date (else it is a numeric column).
    """
    what = _value_name(spec)
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


def _value_name(spec: DataSpec) -> str:
    """Name of the measured value: "Rows" or "Mean of income"."""
    if spec.value is None or spec.agg == Agg.COUNT:
        return "Rows"
    return f"{spec.agg.capitalize()} of {spec.value}"


# Configuration as JSON


def spec_to_dict(spec: DataSpec) -> dict[str, Any]:
    """``spec`` as plain JSON values; each filter carries its ``type`` (``_FILTER_TYPES``)."""
    out: dict[str, Any] = {
        f.name: _plain(getattr(spec, f.name)) for f in fields(spec) if f.name != "filters"
    }
    names = {cls: name for name, cls in _FILTER_TYPES.items()}
    out["filters"] = [
        {"type": names[type(f)], **{k: _plain(v) for k, v in asdict(f).items()}}
        for f in spec.filters
    ]
    return out


def _plain(value: Any) -> Any:
    """``value`` as a JSON value: enums as their text, tuples as lists."""
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def spec_from_dict(raw: Mapping[str, Any]) -> DataSpec:
    """The ``DataSpec`` that ``spec_to_dict`` wrote. Unknown keys are ignored.

    Raises:
        ValueError: A value has the wrong type or is not one of the allowed ones (unknown
            chart kind, aggregation, filter type, malformed date...).
    """
    if not isinstance(raw, Mapping):
        raise ValueError("The chart data must be an object.")
    try:
        kind = ChartKind(raw["kind"])
    except KeyError:
        raise ValueError("The configuration has no chart kind.") from None
    columns = {name: _optional(raw.get(name), str, name) for name in ("x", "y", "value", "group")}
    filters = raw.get("filters", [])
    if not isinstance(filters, list):
        raise ValueError("'filters' must be a list.")
    return DataSpec(
        kind,
        **columns,
        agg=Agg(raw.get("agg", Agg.COUNT)),
        bins=_bounded(raw.get("bins", VisualizeConfig.histogram_bins), "bins"),
        top_k=_bounded(raw.get("top_k", 20), "top_k"),
        time_unit=TimeUnit(raw.get("time_unit", TimeUnit.AUTO)),
        trim=_typed(raw.get("trim", True), bool, "trim"),
        filters=tuple(_filter_from_dict(f) for f in filters),
    )


def _filter_from_dict(raw: Any) -> RowFilter:
    """One filter of ``spec_from_dict``.

    Raises:
        ValueError: Unknown type, missing column or values of the wrong type.
    """
    if not isinstance(raw, Mapping) or raw.get("type") not in _FILTER_TYPES:
        raise ValueError(f"Unknown filter: {raw!r}.")
    column = _typed(raw.get("column"), str, "column")
    cls = _FILTER_TYPES[raw["type"]]
    if cls is DateFilter:
        start, end = (_optional(raw.get(k), str, k) for k in ("start", "end"))
        for day in (start, end):
            if day is not None:
                dt.date.fromisoformat(day)
        return DateFilter(column, start, end)
    if cls is ValuesFilter:
        values = raw.get("values", [])
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError("The values of a filter must be a list of text.")
        return ValuesFilter(column, tuple(values))
    low, high = (_optional(raw.get(k), (int, float), k) for k in ("low", "high"))
    return RangeFilter(
        column, None if low is None else float(low), None if high is None else float(high)
    )


def _typed(value: Any, kind: type | tuple[type, ...], name: str) -> Any:
    """``value`` if it is a ``kind`` (``bool`` is never taken for a number).

    Raises:
        ValueError: It is not.
    """
    if not isinstance(value, kind) or (isinstance(value, bool) and kind is not bool):
        raise ValueError(f"{name!r} has a wrong value: {value!r}.")
    return value


def _optional(value: Any, kind: type | tuple[type, ...], name: str) -> Any:
    """``value`` if it is None or a ``kind``.

    Raises:
        ValueError: It is neither.
    """
    return None if value is None else _typed(value, kind, name)


def _bounded(value: Any, name: str) -> int:
    """``value`` as a count from 1 to 1,000 (bins, categories shown).

    Raises:
        ValueError: It is not an integer in that range.
    """
    count = _typed(value, int, name)
    if not 1 <= count <= 1_000:
        raise ValueError(f"{name!r} is out of range: {count}.")
    return count


def fit_to_dataset(
    spec: DataSpec,
    columns: Collection[str],
    hints: ColumnHints,
    skipped: Collection[str] = (),
) -> tuple[DataSpec, tuple[str, ...]]:
    """``spec`` (saved from any dataset) with only what the current one can draw.

    A column that does not exist, or cannot take its role here, is removed (a required role is
    then filled by the editor with another column), and so is a filter on such a column.
    Columns of roles the kind does not have are dropped silently.

    Args:
        spec: Loaded configuration.
        columns: Every column of the current dataset.
        hints: Its usable columns (``column_hints`` of the dataset without ``skipped``).
        skipped: Columns the person chose to skip ("Skip columns", shared with Profile):
            they exist but are not offered, and the message says so.

    Returns:
        The adapted spec, and one message per column or filter that was removed.
    """
    messages: list[str] = []
    changes: dict[str, str | None] = {}
    roles = {role.field: role for role in ROLES[spec.kind]}
    for name in ("x", "y", "value", "group"):
        column = getattr(spec, name)
        if column is None:
            continue
        role = roles.get(name)
        if role is None:
            changes[name] = None
        elif column not in columns:
            messages.append(f"{role.label}: there is no column {column!r} in this dataset.")
            changes[name] = None
        elif column in skipped:
            messages.append(f"{role.label}: {column!r} is skipped (Skip columns).")
            changes[name] = None
        elif column not in hints.columns_for(role.accepts):
            messages.append(f"{role.label}: {column!r} cannot be used here.")
            changes[name] = None
    type_names = {cls: name for name, cls in _FILTER_TYPES.items()}
    filters = []
    for f in spec.filters:
        if f.column not in columns:
            messages.append(f"Filter: there is no column {f.column!r} in this dataset.")
        elif f.column in skipped:
            messages.append(f"Filter: {f.column!r} is skipped (Skip columns).")
        elif hints.filter_type(f.column) != type_names[type(f)]:
            messages.append(f"Filter: {f.column!r} cannot be filtered this way.")
        else:
            filters.append(f)
    return replace(spec, **changes, filters=tuple(filters)), tuple(messages)


# Chart data


def chart_data(df: pd.DataFrame, spec: DataSpec) -> ChartData:
    """Aggregated data of the chart ``spec`` describes. ``df`` is not modified.

    Rows the filters leave out, and rows with a missing value in any column the chart uses,
    are not drawn (and counted in a note). Filter masks live only during this call (R-V8).

    Raises:
        ValueError: ``spec`` lacks a column its kind requires, names a missing column, uses the
            same column in two roles (a scatter of a column against itself), or filters a
            column of the wrong type.
    """
    used: set[str] = set()
    for role in ROLES[spec.kind]:
        name = getattr(spec, role.field)
        if role.required and name is None:
            raise ValueError(f"A {spec.kind} chart needs a column for {role.label!r}.")
        if name is not None and name not in df.columns:
            raise ValueError(f"There is no column {name!r}.")
        if name is not None and name in used:
            raise ValueError(f"{name!r} is used twice: pick another column for {role.label!r}.")
        if name is not None:
            used.add(name)
    for f in spec.filters:
        _check_filter(df, f)
    match spec.kind:
        case ChartKind.BAR | ChartKind.PIE:
            return _bars(df, spec)
        case ChartKind.LINE:
            return _line(df, spec)
        case ChartKind.HISTOGRAM:
            return _histogram(df, spec)
        case ChartKind.SCATTER:
            return _scatter(df, spec)
        case ChartKind.BOXPLOT:
            return _boxplot(df, spec)
        case ChartKind.HEATMAP:
            return _heatmap(df, spec)


def _check_filter(df: pd.DataFrame, f: RowFilter) -> None:
    """Raise if filter ``f`` cannot apply to ``df``.

    Raises:
        ValueError: Its column does not exist or is not of the type the filter compares.
    """
    if f.column not in df.columns:
        raise ValueError(f"There is no column {f.column!r} to filter.")
    column = df[f.column]
    if isinstance(f, DateFilter) and not ptypes.is_datetime64_any_dtype(column):
        raise ValueError(f"{f.column!r} has no dates to filter by.")
    if isinstance(f, RangeFilter) and not _is_number(column):
        raise ValueError(f"{f.column!r} has no numbers to filter by.")


def _is_number(column: pd.Series) -> bool:
    """Whether ``column`` holds numbers (booleans are numeric to pandas, not here)."""
    return ptypes.is_numeric_dtype(column) and not ptypes.is_bool_dtype(column)


def _filter_mask(df: pd.DataFrame, filters: Sequence[RowFilter]) -> np.ndarray:
    """Rows of ``df`` that pass every filter. A missing value never passes a filter on it.

    Each mask has the length of the dataset: it is built here and dropped by the caller once
    the rows are picked, never cached (R-V8).
    """
    keep = np.ones(len(df), dtype=bool)
    for f in filters:
        column = df[f.column]
        match f:
            case DateFilter(start=start, end=end):
                if start is not None:
                    keep &= (column >= pd.Timestamp(start)).to_numpy(dtype=bool, na_value=False)
                if end is not None:
                    # Whole last day: before the next midnight.
                    limit = pd.Timestamp(end) + pd.Timedelta(days=1)
                    keep &= (column < limit).to_numpy(dtype=bool, na_value=False)
            case ValuesFilter(values=values):
                if ptypes.is_string_dtype(column) and not isinstance(
                    column.dtype, pd.CategoricalDtype
                ):
                    # Text is its own label: no need to list the distinct values of an id
                    # column with millions of them.
                    kept: Collection[Any] = values
                else:
                    # Compares the few distinct values as text, not every row.
                    wanted = set(values)
                    kept = [v for v in column.dropna().unique() if str(v) in wanted]
                keep &= column.isin(kept).to_numpy(dtype=bool)
            case RangeFilter(low=low, high=high):
                if low is not None:
                    keep &= (column >= low).to_numpy(dtype=bool, na_value=False)
                if high is not None:
                    keep &= (column <= high).to_numpy(dtype=bool, na_value=False)
    return keep


def _used(
    df: pd.DataFrame, spec: DataSpec, names: Sequence[str | None]
) -> tuple[pd.DataFrame, list[str]]:
    """Columns ``names`` (None skipped) of the rows that pass the filters of ``spec`` and have
    no missing value in them, and the notes saying how many rows were left out.

    Only the used columns are copied, never the whole DataFrame: the filter columns are read
    to build the mask, not carried along.
    """
    cols = list(dict.fromkeys(n for n in names if n is not None))
    sub = df[cols]
    present = sub.notna().all(axis=1).to_numpy()
    notes: list[str] = []
    if spec.filters:
        passed = _filter_mask(df, spec.filters)
        if out := int((~passed).sum()):
            notes.append(f"{out:,} of {len(df):,} rows left out by the filters.")
        missing = int((passed & ~present).sum())
        keep = passed & present
    else:
        missing = int((~present).sum())
        keep = present
    if missing:
        notes.append(f"{missing:,} rows with missing values left out.")
    return (sub if keep.all() else sub[keep]), notes


def _least_squares(
    xs: np.ndarray, ys: np.ndarray
) -> tuple[tuple[float, float] | None, float | None]:
    """Slope and intercept of the least squares line of ``ys`` on ``xs``, and Pearson's r.

    Closed form from the centered sums, in one pass: on 3 M points it takes 20 ms, where
    ``np.polyfit`` (a least squares solve) plus ``np.corrcoef`` took 155 ms.

    Returns:
        ``(None, None)`` with fewer than 3 points or when either column is constant.
    """
    if xs.size < 3:
        return None, None
    dx, dy = xs - xs.mean(), ys - ys.mean()
    sxx, syy, sxy = float(dx @ dx), float(dy @ dy), float(dx @ dy)
    if sxx <= 0 or syy <= 0:
        return None, None
    slope = sxy / sxx
    return (slope, float(ys.mean()) - slope * float(xs.mean())), sxy / float(np.sqrt(sxx * syy))


# Steps the visible range of a scatter axis is cut into when its coordinates are rounded:
# 1,000x finer than a 1,000 px chart, so even zoomed in with the mouse wheel points stay put.
_SCREEN_STEPS = 1_000_000


def _screen_round(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """``values`` rounded to the decimals that ``_SCREEN_STEPS`` of ``[low, high]`` need.

    ``1234.5678912`` on a range of 10,000 is sent as ``1234.57``: same pixel, a third of the
    characters. Never rounds to tens or more, so whole numbers stay exact.
    """
    span = high - low or abs(high) or 1.0
    decimals = max(0, int(np.ceil(-np.log10(span / _SCREEN_STEPS))))
    return np.round(values, decimals)


def _top_categories(series: pd.Series, k: int) -> tuple[pd.Series, str | None, str | None]:
    """``series`` as a category with its ``k`` most frequent values, the rest as "Other".

    Categories are ordered by frequency and relabeled as text. ``series`` has no missing values.

    Returns:
        The new series, the label of "Other" (None if nothing was grouped) and a note.
    """
    cat = series if isinstance(series.dtype, pd.CategoricalDtype) else series.astype("category")
    counts = cat.value_counts()
    counts = counts[counts > 0]
    other = note = None
    if len(counts) > k:
        rest = len(counts) - k
        other = f"Other ({rest:,} values)"
        top = list(counts.index[:k])
        cat = cat.cat.set_categories(top).cat.add_categories([other]).fillna(other)
        note = f"{series.name}: the {k} most frequent of {len(counts):,} values; the rest in Other."
    else:
        # A list: given a CategoricalIndex, set_categories would take all its categories.
        cat = cat.cat.set_categories(list(counts.index))
    labels = _unique_labels(cat.cat.categories)
    return cat.cat.rename_categories(labels), other, note


def _unique_labels(values: pd.Index) -> list[str]:
    """Values as text, made unique when two look the same (1 and "1")."""
    labels = [str(v) for v in values]
    if len(set(labels)) == len(labels):
        return labels
    seen: dict[str, int] = {}
    out = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f"{label} ({seen[label]})")
    return out


def _intervals(sub: pd.DataFrame, spec: DataSpec) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Numeric x of ``sub`` cut in ``spec.bins`` intervals of the same width.

    The edges come from ``columns.histogram``, the same as a histogram of the column, so a
    bar chart of counts on intervals and the histogram agree. Extreme values are left out when
    ``spec.trim`` (``_range``).

    Returns:
        The rows of ``sub`` kept (finite x, inside the range), the interval of each as a
        category, and the labels of every interval in order ("10 to 20"), empty ones included.
    """
    values = sub[spec.x].to_numpy(dtype="float64", na_value=np.nan)
    keep = np.isfinite(values)
    if not keep.any():
        return sub.iloc[:0], pd.Series(dtype="category"), []
    low, high, _ = _range(values[keep], spec.trim)
    keep &= (values >= low) & (values <= high)
    sub, values = sub[keep], values[keep]
    _, edges = histogram(values, spec.bins, low, high)
    ends = value_labels(edges)
    labels = list(dict.fromkeys(f"{a} to {b}" for a, b in pairwise(ends)))
    # The last interval includes its upper end, as in ``np.histogram``.
    codes = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(labels) - 1)
    keys = pd.Series(
        pd.Categorical.from_codes(codes, categories=labels), index=sub.index, name=spec.x
    )
    return sub, keys, labels


def _aggregate(
    frame: pd.DataFrame, keys: list[pd.Series], value: str | None, agg: Agg
) -> pd.Series:
    """``agg`` of column ``value`` by ``keys`` (rows when ``value`` is None or ``agg`` COUNT).

    Groups on categories with ``observed=True`` (R-V9): only combinations that exist.
    """
    grouped = frame.groupby(keys, observed=True, sort=True)
    if value is None or agg == Agg.COUNT:
        return grouped.size()
    return grouped[value].agg(str(agg))


def _clean(value: Any) -> float | int | None:
    """A number as a plain ``int`` / ``float`` for JSON, None if missing or infinite."""
    if value is None or not np.isfinite(value):
        return None
    if isinstance(value, (int, np.integer)) or float(value).is_integer():
        return int(value)
    return float(value)


def _bars(df: pd.DataFrame, spec: DataSpec) -> ChartData:
    """Bar and pie: one bar (slice) per category or interval of a numeric x, optionally split
    in one series per group. Also a line over a numeric x (one point per interval).
    """
    is_pie = spec.kind == ChartKind.PIE
    agg = Agg.SUM if is_pie and spec.value else spec.agg
    group = None if is_pie else spec.group
    sub, notes = _used(df, spec, [spec.x, spec.value, group])
    if sub.empty:
        return ChartData(spec.kind, notes=tuple(notes))
    intervals: list[str] | None = None
    other = None
    if not is_pie and _is_number(sub[spec.x]):
        rows = len(sub)
        sub, keys, intervals = _intervals(sub, spec)
        if outside := rows - len(sub):
            notes.append(f"{outside:,} rows with extreme values of {spec.x} left out (trimmed).")
        if sub.empty:
            return ChartData(spec.kind, notes=tuple(notes))
    else:
        k = min(
            spec.top_k, VisualizeConfig.max_slices if is_pie else VisualizeConfig.max_categories
        )
        keys, other, note = _top_categories(sub[spec.x], k)
        notes += [note] if note else []
    if group is None:
        table = _aggregate(sub, [keys], spec.value, agg).to_frame(_value_name(spec))
    else:
        groups, _, gnote = _top_categories(sub[group], VisualizeConfig.max_series)
        notes += [gnote] if gnote else []
        # unstack of an already aggregated, small series: pivot_table would group again.
        table = _aggregate(sub, [keys, groups], spec.value, agg).unstack()  # noqa: PD010
    if intervals is not None:
        # Every interval, also the empty ones, so the axis keeps its scale.
        table = table.reindex(intervals)
    if spec.value is None or agg in (Agg.COUNT, Agg.SUM):
        table = table.fillna(0)
    return ChartData(
        spec.kind,
        series=tuple(
            Series(str(name), tuple(_clean(v) for v in table[name])) for name in table.columns
        ),
        categories=tuple(str(c) for c in table.index),
        x_name=str(spec.x),
        y_name=_value_name(spec),
        other=other,
        notes=tuple(notes),
    )


def _range(values: np.ndarray, trim: bool) -> tuple[float, float, int]:
    """Range of ``values`` to draw: all of it, or without the few extreme values that stretch
    it ``VisualizeConfig.trim_ratio`` times its central part (when ``trim``).

    Only far values are cut: the kept range is the central one widened by its own width on each
    side, so the ordinary tails of the data stay in.

    Returns:
        Low and high end, and how many values fall outside.
    """
    low, high = float(values.min()), float(values.max())
    if not trim or values.size < 20:
        return low, high, 0
    q = VisualizeConfig.trim_quantile
    part = sample_array(values, VisualizeConfig.quantile_sample)
    c_low, c_high = (float(v) for v in np.quantile(part, [q, 1 - q]))
    width = c_high - c_low
    if width <= 0 or (high - low) <= VisualizeConfig.trim_ratio * width:
        return low, high, 0
    kept = values[(values >= c_low - width) & (values <= c_high + width)]
    return float(kept.min()), float(kept.max()), values.size - kept.size


def _date_ticks(series: pd.Series) -> tuple[np.ndarray, str]:
    """Dates as integers in the unit the column is stored in, and that unit.

    Never converted to nanoseconds: placeholder dates such as 9999-12-31 only fit in the
    coarser units (same as ``columns.datetime_bars``).
    """
    return series.astype("int64").to_numpy(), series.dt.unit


def _histogram(df: pd.DataFrame, spec: DataSpec) -> ChartData:
    """Histogram of a numeric or date column, with the same bins for every group."""
    sub, notes = _used(df, spec, [spec.x, spec.group])
    column = sub[spec.x]
    is_date = ptypes.is_datetime64_any_dtype(column)
    if is_date:
        values, unit = _date_ticks(column)
    else:
        values = column.to_numpy(dtype="float64", na_value=np.nan)
        finite = np.isfinite(values)
        sub, values = sub[finite], values[finite]
    if values.size == 0:
        return ChartData(spec.kind, notes=tuple(notes))
    low, high, outside = _range(values, spec.trim)
    if outside:
        notes.append(f"{outside:,} extreme values outside the chart (trimmed).")
    counts, edges = histogram(values, spec.bins, low, high)
    if is_date:
        dates = [pd.Timestamp(np.datetime64(round(e), unit)) for e in edges]
        labels = tuple(f"{a:%Y-%m-%d} to {b:%Y-%m-%d}" for a, b in pairwise(dates))
    else:
        ends = value_labels(edges)
        labels = tuple(f"{a} to {b}" for a, b in pairwise(ends))
    if spec.group is None:
        series = (Series("Rows", tuple(counts)),)
    else:
        groups, _, gnote = _top_categories(sub[spec.group], VisualizeConfig.max_series)
        notes += [gnote] if gnote else []
        codes = groups.cat.codes.to_numpy()
        series = tuple(
            Series(name, tuple(histogram(values[codes == i], spec.bins, low, high)[0]))
            for i, name in enumerate(groups.cat.categories)
        )
    return ChartData(
        spec.kind,
        series=series,
        categories=labels,
        x_name=str(spec.x),
        y_name="Rows",
        notes=tuple(notes),
    )


def _time_unit(spec: DataSpec, span_days: float) -> tuple[TimeUnit, str | None]:
    """Time unit of a line over ``span_days``: the chosen one, coarsened if it would give more
    than ``VisualizeConfig.max_line_points`` points (R-V2), or the finest readable one for AUTO.
    """
    units = list(_UNIT_DAYS)
    if spec.time_unit == TimeUnit.AUTO:
        for unit in units:
            if span_days / _UNIT_DAYS[unit] <= VisualizeConfig.auto_line_points:
                return unit, None
        return TimeUnit.YEAR, None
    start = units.index(spec.time_unit)
    for unit in units[start:]:
        if span_days / _UNIT_DAYS[unit] <= VisualizeConfig.max_line_points:
            if unit != spec.time_unit:
                return unit, f"Shown by {unit}: by {spec.time_unit} it has too many points."
            return unit, None
    if spec.time_unit != TimeUnit.YEAR:
        return TimeUnit.YEAR, f"Shown by year: by {spec.time_unit} it has too many points."
    return TimeUnit.YEAR, None


def _line(df: pd.DataFrame, spec: DataSpec) -> ChartData:
    """``agg`` of ``value`` (or rows) per time bucket, one line per group. Over a numeric x,
    one point per interval (same data as a bar chart on intervals).
    """
    if _is_number(df[spec.x]):
        return _bars(df, spec)
    sub, notes = _used(df, spec, [spec.x, spec.value, spec.group])
    if sub.empty:
        return ChartData(spec.kind, notes=tuple(notes))
    ticks, _ = _date_ticks(sub[spec.x])
    low, high, outside = _range(ticks, spec.trim)
    if outside:
        sub = sub[(ticks >= low) & (ticks <= high)]
        notes.append(f"{outside:,} rows with extreme dates left out (trimmed).")
    dates = sub[spec.x]
    span_days = (dates.max() - dates.min()) / pd.Timedelta(days=1)
    unit, unit_note = _time_unit(spec, span_days)
    notes += [unit_note] if unit_note else []
    periods = dates.dt.to_period(_UNIT_FREQ[unit])
    full = pd.period_range(periods.min(), periods.max(), freq=_UNIT_FREQ[unit])
    filled = spec.value is None or spec.agg in (Agg.COUNT, Agg.SUM)
    if spec.group is None:
        table = _aggregate(sub, [periods], spec.value, spec.agg).to_frame(_value_name(spec))
    else:
        groups, _, gnote = _top_categories(sub[spec.group], VisualizeConfig.max_series)
        notes += [gnote] if gnote else []
        table = _aggregate(sub, [periods, groups], spec.value, spec.agg).unstack()  # noqa: PD010 (aggregated)
    # Every bucket, also the empty ones: no rows is 0 rows, but a mean of nothing is a gap.
    # Unless even by year there are too many (untrimmed placeholder dates): then only the
    # buckets with rows, so the browser never gets thousands of empty years (R-V2).
    if len(full) <= VisualizeConfig.max_line_points:
        table = table.reindex(full)
        if filled:
            table = table.fillna(0)
    else:
        notes.append("Periods without rows are not drawn: the dates span too long.")
    millis = table.index.start_time.as_unit("ms").asi8
    series = tuple(
        Series(
            str(name),
            tuple([int(t), _clean(v)] for t, v in zip(millis, table[name], strict=True)),
        )
        for name in table.columns
    )
    return ChartData(
        spec.kind,
        series=series,
        x_name=f"{spec.x} (by {unit})",
        y_name=_value_name(spec),
        time_axis=True,
        notes=tuple(notes),
    )


def _scatter(df: pd.DataFrame, spec: DataSpec) -> ChartData:
    """Points of two numeric columns: a sample (R-V2), trend and correlation on every point.

    Works on numpy positions instead of filtering the DataFrame: only the sampled rows of the
    group column are read, and coordinates are rounded to what the screen can show
    (``_screen_round``), which halves the option sent on every rerun.
    """
    sub, notes = _used(df, spec, [spec.x, spec.y, spec.group])
    xs = sub[spec.x].to_numpy(dtype="float64", na_value=np.nan)
    ys = sub[spec.y].to_numpy(dtype="float64", na_value=np.nan)
    keep = np.flatnonzero(np.isfinite(xs) & np.isfinite(ys))
    if keep.size == 0:
        return ChartData(spec.kind, notes=tuple(notes))
    if keep.size < xs.size:
        xs, ys = xs[keep], ys[keep]
    trend, r = _least_squares(xs, ys)
    x_lo, x_hi = float(xs.min()), float(xs.max())
    # Same rows as sampling the filtered DataFrame: positions only depend on the length.
    picked = np.arange(keep.size)
    if keep.size > VisualizeConfig.max_points:
        picked = positions(keep.size, VisualizeConfig.max_points)
        notes.insert(0, f"{picked.size:,} of {keep.size:,} points (random sample).")
    px = _screen_round(xs[picked], x_lo, x_hi)
    py = _screen_round(ys[picked], float(ys.min()), float(ys.max()))
    if spec.group is None:
        groups = [(str(spec.y), slice(None))]
    else:
        group = sub[spec.group].iloc[keep[picked]]
        keys, _, gnote = _top_categories(group, VisualizeConfig.max_series)
        notes += [gnote] if gnote else []
        codes = keys.cat.codes.to_numpy()
        groups = [(str(name), codes == i) for i, name in enumerate(keys.cat.categories)]
    series = tuple(
        Series(name, tuple(np.column_stack((px[rows], py[rows])).tolist())) for name, rows in groups
    )
    return ChartData(
        spec.kind,
        series=series,
        x_name=str(spec.x),
        y_name=str(spec.y),
        trend=trend,
        x_range=(x_lo, x_hi),
        r=r,
        notes=tuple(notes),
    )


def _boxplot(df: pd.DataFrame, spec: DataSpec) -> ChartData:
    """One box per group (Tukey whiskers), quartiles on a sample, the most extreme outliers."""
    sub, notes = _used(df, spec, [spec.value, spec.x])
    values = sub[spec.value].to_numpy(dtype="float64", na_value=np.nan)
    finite = np.isfinite(values)
    sub, values = sub[finite], values[finite]
    if values.size == 0:
        return ChartData(spec.kind, notes=tuple(notes))
    if spec.x is None:
        parts = [(str(spec.value), values)]
        other = None
    else:
        k = min(spec.top_k, VisualizeConfig.max_shown_categories)
        keys, other, note = _top_categories(sub[spec.x], k)
        notes += [note] if note else []
        codes = keys.cat.codes.to_numpy()
        parts = [(str(name), values[codes == i]) for i, name in enumerate(keys.cat.categories)]
    per_box = max(1, VisualizeConfig.max_outliers // len(parts))
    boxes, outliers = [], []
    for index, (_, part) in enumerate(parts):
        q1, median, q3 = np.quantile(
            sample_array(part, VisualizeConfig.quantile_sample), [0.25, 0.5, 0.75]
        )
        fence_low, fence_high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        inside = part[(part >= fence_low) & (part <= fence_high)]
        low, high = (inside.min(), inside.max()) if inside.size else (q1, q3)
        boxes.append(tuple(_clean(v) for v in (low, q1, median, q3, high)))
        out = part[(part < fence_low) | (part > fence_high)]
        if out.size > per_box:
            out = out[np.argsort(-np.abs(out - median))[:per_box]]
        outliers += [(index, float(v)) for v in out]
    if values.size > VisualizeConfig.quantile_sample:
        notes.append(
            f"Quartiles from a sample of {VisualizeConfig.quantile_sample:,} values per box."
        )
    return ChartData(
        spec.kind,
        series=(Series(str(spec.value), tuple(boxes)),),
        categories=tuple(name for name, _ in parts),
        x_name=str(spec.x or ""),
        y_name=str(spec.value),
        other=other,
        outliers=tuple(outliers),
        notes=tuple(notes),
    )


def _heatmap(df: pd.DataFrame, spec: DataSpec) -> ChartData:
    """``agg`` of ``value`` (or rows) for each pair of categories of ``x`` and ``y``."""
    sub, notes = _used(df, spec, [spec.x, spec.y, spec.value])
    if sub.empty:
        return ChartData(spec.kind, notes=tuple(notes))
    k = min(spec.top_k, VisualizeConfig.max_shown_categories)
    xs, _, xnote = _top_categories(sub[spec.x], k)
    ys, _, ynote = _top_categories(sub[spec.y], k)
    notes += [n for n in (xnote, ynote) if n]
    table = _aggregate(sub, [xs, ys], spec.value, spec.agg)
    columns = tuple(str(c) for c in xs.cat.categories)
    rows = tuple(str(c) for c in ys.cat.categories)
    col_index = {c: i for i, c in enumerate(columns)}
    row_index = {r: i for i, r in enumerate(rows)}
    cells = tuple(
        [col_index[str(a)], row_index[str(b)], _clean(v)]
        for (a, b), v in table.items()
        if _clean(v) is not None
    )
    finite = [c[2] for c in cells]
    return ChartData(
        spec.kind,
        series=(Series(_value_name(spec), cells),),
        categories=columns,
        rows=rows,
        x_name=str(spec.x),
        y_name=str(spec.y),
        value_range=(float(min(finite)), float(max(finite))) if finite else (0.0, 0.0),
        notes=tuple(notes),
    )


def to_frame(data: ChartData) -> pd.DataFrame:
    """The data of a chart as a table, to show or download what the chart draws."""
    match data.kind:
        case ChartKind.LINE if data.time_axis:
            frames = [
                pd.DataFrame(list(s.data), columns=["date", s.name]).set_index("date")
                for s in data.series
            ]
            table = pd.concat(frames, axis=1) if frames else pd.DataFrame()
            table.index = pd.to_datetime(table.index, unit="ms")
            return table.reset_index()
        case ChartKind.BAR | ChartKind.PIE | ChartKind.HISTOGRAM | ChartKind.LINE:
            return pd.DataFrame(
                {s.name: list(s.data) for s in data.series},
                index=pd.Index(data.categories, name=data.x_name),
            ).reset_index()
        case ChartKind.SCATTER:
            return pd.DataFrame(
                [(s.name, x, y) for s in data.series for x, y in s.data],
                columns=["series", data.x_name, data.y_name],
            )
        case ChartKind.BOXPLOT:
            stats = ["low", "q1", "median", "q3", "high"]
            boxes = data.series[0].data if data.series else ()
            return pd.DataFrame(list(boxes), columns=stats).assign(group=list(data.categories))[
                ["group", *stats]
            ]
        case ChartKind.HEATMAP:
            cells = data.series[0].data if data.series else ()
            return pd.DataFrame(
                [(data.categories[i], data.rows[j], v) for i, j, v in cells],
                columns=[data.x_name, data.y_name, data.series[0].name if data.series else ""],
            )
