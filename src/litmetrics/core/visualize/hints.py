"""Which columns of a dataset can go in a chart (``column_hints``), which chart kinds it can
draw, and a reasonable first chart of each kind (``default_spec``).
"""

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from litmetrics.config import VisualizeConfig
from litmetrics.core.sampling import sample
from litmetrics.core.schema import IDENTIFIER_MIN_DISTINCT, Schema, SemanticType, mostly_distinct
from litmetrics.core.visualize.spec import (
    CATEGORICAL,
    DATETIME,
    NUMERIC,
    Accepts,
    Agg,
    ChartKind,
    DataSpec,
    is_number,
)

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
            (a key of ``FILTER_TYPES``): the chart columns and also those never offered for
            a chart (identifiers, free text, row numbers), since rows can be picked by a
            column that is not drawn. Only columns with fewer than two values are missing.
        partial: Columns whose ``values`` are only the most frequent ones of a sample (free
            text, identifiers): other values exist and can be typed in.
        integer: Columns filtered by range that hold integers: their filter steps by 1.
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
    integer: tuple[str, ...] = ()

    def columns_for(self, accepts: Accepts) -> tuple[str, ...]:
        """Columns of the types ``accepts``, in dataset order."""
        wanted: set[str] = set()
        if accepts & CATEGORICAL:
            wanted.update(self.categorical)
        if accepts & NUMERIC:
            wanted.update(self.numeric)
        if accepts & DATETIME:
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
    integer: list[str] = []
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
        if ptypes.is_integer_dtype(column):
            integer.append(name)
        if _is_row_number(column, float(low), float(high)):
            left_out.append((name, "row number"))
        else:
            numeric.append(name)
    categorical: list[str] = []
    distinct: dict[str, int] = {}
    dominated: dict[str, bool] = {}
    values: list[tuple[str, tuple[str, ...]]] = []
    for name in schema.names_of(CATEGORICAL):
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
    for name in schema.names_of(DATETIME):
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
            if ptypes.is_integer_dtype(df[name]):
                integer.append(name)
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
        integer=tuple(integer),
    )


def _filter_of_uncharted(column: pd.Series) -> tuple[str, Any] | None:
    """How a column never offered for a chart (identifier, free text) is filtered: by
    ``"range"`` with its lowest and highest number when it holds numbers, else by
    ``"values"`` with the most frequent ones of a sample (``VisualizeConfig.filter_values``).
    None when it has fewer than two values.

    Counting the values of a sample, not of the column: an id column of 3 M rows has 3 M of
    them, and the list only helps to pick; any value can be typed in.
    """
    if is_number(column):
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
