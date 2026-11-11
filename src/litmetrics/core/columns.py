"""Column table of the Profile page: what each column holds, described for its semantic type.

Two levels:
- ``column_overview``: one short row per column (type, missing, distinct values and the bars
  of a small chart);
- per-type tables (``numeric_stats``, ``categorical_stats``, ``datetime_stats``,
  ``text_stats``), each with only the measures that make sense for its type, and
  ``column_detail`` with the chart data of one column.

Every table is computed column by column, so a table of one column (the column inspector) costs
only that column and gives the same row as the whole table.

Cost on large datasets (docs/performance.md):
- Counts are exact; quantiles, histograms, skewness, text lengths and the outliers drawn come
  from a sample of the non-missing values of each column (R-P2). Results computed on a sample
  say so in the page.
- Numeric columns are summarised with numpy on their finite values, one column at a time
  (R-P8): 4x faster than the same measures on a DataFrame, with the memory of one column
  instead of several copies of all of them, and in threads when ``workers`` > 1.
- Missing and distinct values come from ``profile.BaseProfile`` instead of being counted again
  (R-P1).
- Date columns are already ``datetime64`` (``schema.apply_schema``): nothing here parses
  dates (R-L2).
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from litmetrics.core.parallel import map_columns
from litmetrics.core.profile import BaseProfile
from litmetrics.core.sampling import positions, sample, sample_array
from litmetrics.core.schema import Schema, SemanticType

# Semantic types described by each per-type table.
NUMERIC_TYPES = (SemanticType.NUMERIC,)
CATEGORICAL_TYPES = (SemanticType.CATEGORICAL, SemanticType.BOOLEAN)
DATE_TYPES = (SemanticType.DATETIME,)
TEXT_TYPES = (SemanticType.TEXT, SemanticType.IDENTIFIER)

# Bars of the small distribution drawn in each row of the overview.
OVERVIEW_BINS = 16
# Bars of the histograms of a single column.
DETAIL_BINS = 30
# Categories listed for a single column.
TOP_K = 10
# Tukey's fences: values beyond TUKEY_K * IQR from the quartiles are outliers.
TUKEY_K = 1.5
# A category with a smaller share (%) of the non-missing values is rare.
RARE_PCT = 1.0
# Percentiles of the numeric table (quartiles included).
PERCENTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
# Characters of the example value shown in the text table.
EXAMPLE_WIDTH = 60
# Values read at a time when looking for the first non-blank one.
_EXAMPLE_CHUNK = 1_000

NUMERIC_STATS_COLUMNS = [
    "column", "count", "mean", "std", "cv", "min", "p5", "q1", "median", "q3", "p95", "max",
    "iqr", "whisker_low", "whisker_high", "outliers", "outliers_pct", "zeros", "negatives",
    "infinite", "unparsed", "skew", "kurtosis", "shape",
]  # fmt: skip
CATEGORICAL_STATS_COLUMNS = [
    "column", "unique", "mode", "mode_pct", "top", "rare", "variants",
]  # fmt: skip
DATETIME_STATS_COLUMNS = [
    "column", "min", "max", "span_days", "frequency", "unparsed", "future",
]  # fmt: skip
TEXT_STATS_COLUMNS = [
    "column", "semantic_type", "len_min", "len_mean", "len_max", "blank", "unique_pct",
    "repeated", "example",
]  # fmt: skip
OVERVIEW_COLUMNS = ["column", "semantic_type", "missing_pct", "unique_pct", "distribution"]


@dataclass(frozen=True)
class Bars:
    """Data of a small bar chart, already aggregated.

    Attributes:
        counts: Height of each bar.
        labels: Label of each bar: a value, or a range such as ``"10 to 12.5"``.
        per_value: True if each bar is one value (short labels), False if it is a range.
        edges: Numeric columns only, to place a value on the chart (``bar_index``): the value
            of each bar if ``per_value``, else the bin edges (one more than ``counts``).
    """

    counts: tuple[int, ...] = ()
    labels: tuple[str, ...] = ()
    per_value: bool = False
    edges: tuple[float, ...] = ()


def bar_index(bars: Bars, value: float) -> int | None:
    """Position of the bar where ``value`` falls, or None if it is outside the chart.

    With one bar per value, the bar of the closest value. Without ``edges``, always None.
    """
    if not bars.edges or pd.isna(value):
        return None
    edges = np.asarray(bars.edges)
    if bars.per_value:
        return int(np.abs(edges - value).argmin())
    if not edges[0] <= value <= edges[-1]:
        return None
    # The last bin includes its right edge, as in np.histogram.
    return int(min(np.searchsorted(edges, value, side="right") - 1, len(bars.counts) - 1))


@dataclass(frozen=True, eq=False)
class ColumnDetail:
    """Chart data of one column, for the inspector of the Columns tab.

    Attributes:
        bars: Main chart: values or histogram (numeric), most common values (categorical),
            rows over time (dates) or length of the values (text).
        outliers: Numeric only: the most extreme outliers, to draw as dots.
        variants: Categorical only: values written in different ways (see ``variant_groups``).
        examples: Text only: a few values with their length (``value``, ``length``).
        other_values: Categorical only: distinct values left out of ``bars``.
        other_rows: Categorical only: rows with those values.
    """

    bars: Bars = field(default_factory=Bars)
    outliers: tuple[float, ...] = ()
    variants: tuple[tuple[str, ...], ...] = ()
    examples: pd.DataFrame | None = None
    other_values: int = 0
    other_rows: int = 0


# Shared helpers


def histogram(
    values: np.ndarray, bins: int, low: float, high: float
) -> tuple[list[int], list[float]]:
    """Counts and bin edges of ``values`` between ``low`` and ``high`` (values outside ignored)."""
    if values.size == 0:
        return [], []
    if low == high:
        # A single value: one bar, so the chart is not empty.
        return [int(np.count_nonzero(values == low))], [low, high]
    counts, edges = np.histogram(values, bins=bins, range=(low, high))
    return [int(c) for c in counts], [float(e) for e in edges]


def sample_values(series: pd.Series, n: int) -> pd.Series:
    """Up to ``n`` random non-missing values of a column (fixed seed).

    Missing values are dropped before sampling, not after: a sample of rows of a sparse column
    could keep one value or none, and every measure computed on it would be wrong.
    """
    return sample(series.dropna(), n)


def value_labels(values: Iterable[float]) -> tuple[str, ...]:
    """Short labels for distinct numbers, with as many digits as needed to tell them apart.

    ``,.4g`` alone writes 123451 and 123452 as "1.235e+05" both.
    """
    values = list(values)
    for digits in (4, 6, 10):
        labels = tuple(f"{v:,.{digits}g}" for v in values)
        if len(set(labels)) == len(labels):
            return labels
    return tuple(repr(v) for v in values)


def finite_values(series: pd.Series) -> np.ndarray:
    """Finite, non-missing values of a numeric column, as a ``float64`` array."""
    values = series.to_numpy(dtype="float64", na_value=np.nan)
    return values[np.isfinite(values)]


def value_counts(series: pd.Series) -> pd.Series:
    """Rows of each distinct value, as text, the most frequent first. Counted on every row."""
    counts = series.value_counts()
    # A category column also lists the categories that never appear.
    counts = counts[counts > 0]
    if not ptypes.is_string_dtype(counts.index):
        labels = counts.index.astype(str)
        # Values that look the same as text (1 and "1") are one value for the user. Usually
        # (categories, booleans, numbers) there is nothing to merge and relabelling is enough.
        counts = counts.set_axis(labels) if labels.is_unique else counts.groupby(labels).sum()
    return counts.sort_values(ascending=False, kind="stable")


# Numeric


def numeric_stats(
    df: pd.DataFrame,
    columns: Sequence[str],
    sample_rows: int,
    workers: int = 1,
    *,
    unparsed: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Position, spread, outliers and shape of every numeric column in ``columns``.

    Missing values are ignored and infinite ones are counted apart (``infinite``) and then
    ignored too, so they do not turn the mean into ``inf``.

    Quantiles, skewness and kurtosis come from a sample of the finite values of each column
    (``numeric_sample``); the rest (counts, mean, min, max, outliers) is exact. The *whiskers*
    are the most extreme values that are not outliers by Tukey's rule (inside
    ``[Q1 - 1.5 IQR, Q3 + 1.5 IQR]``): the range of the column without its outliers, the same
    one a boxplot draws.

    Args:
        workers: Threads, one column each (``parallel.map_columns``). Same result.
        unparsed: Values of each column that were not numbers when the file was loaded
            (``LoadedDataset.unparsed``), now missing. Columns not listed count 0.
    """
    unparsed = unparsed or {}

    def row(name: str) -> dict[str, Any]:
        return _numeric_row(name, df[name], sample_rows, int(unparsed.get(name, 0)))

    return pd.DataFrame(map_columns(row, columns, workers), columns=NUMERIC_STATS_COLUMNS)


def numeric_sample(series: pd.Series, n: int) -> np.ndarray:
    """Up to ``n`` random finite values of a numeric column (fixed seed).

    The quantiles of ``numeric_stats``, the histogram of ``numeric_bars`` and the outliers of
    ``outlier_values`` use this same sample, so they agree with each other.
    """
    return sample_array(finite_values(series), n)


def _numeric_row(name: str, series: pd.Series, sample_rows: int, unparsed: int) -> dict[str, Any]:
    """Row of ``numeric_stats`` for one column. Same definitions as pandas' ``describe``:
    ``std`` with one degree of freedom, linear interpolation between quantiles, and pandas'
    bias-corrected skewness and kurtosis.
    """
    raw = series.to_numpy(dtype="float64", na_value=np.nan)
    finite_mask = np.isfinite(raw)
    values = raw[finite_mask]
    count = int(values.size)
    picked = sample_array(values, sample_rows)
    nan = float("nan")
    if count:
        quantiles = np.quantile(picked, PERCENTILES)
        p5, q1, median, q3, p95 = (float(q) for q in quantiles)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if count > 1 else nan
        low, high = q1 - TUKEY_K * (q3 - q1), q3 + TUKEY_K * (q3 - q1)
        inside = values[(values >= low) & (values <= high)]
        outliers = count - int(inside.size)
        whisker_low = float(inside.min()) if inside.size else nan
        whisker_high = float(inside.max()) if inside.size else nan
        minimum, maximum = float(values.min()), float(values.max())
        # pandas' estimators on the sample: a 200,000-value Series, a few milliseconds.
        sampled = pd.Series(picked)
        skew, kurtosis = float(sampled.skew()), float(sampled.kurt())
    else:
        p5 = q1 = median = q3 = p95 = mean = std = nan
        whisker_low = whisker_high = minimum = maximum = skew = kurtosis = nan
        outliers = 0
    return {
        "column": name,
        "count": count,
        "mean": mean,
        "std": std,
        # Coefficient of variation: spread relative to the size of the values.
        "cv": std / abs(mean) if count and mean != 0 else nan,
        "min": minimum,
        "p5": p5,
        "q1": q1,
        "median": median,
        "q3": q3,
        "p95": p95,
        "max": maximum,
        "iqr": q3 - q1,
        "whisker_low": whisker_low,
        "whisker_high": whisker_high,
        "outliers": outliers,
        "outliers_pct": outliers / count * 100 if count else 0.0,
        "zeros": int(np.count_nonzero(values == 0)),
        "negatives": int(np.count_nonzero(values < 0)),
        "infinite": int(np.count_nonzero(~finite_mask & ~np.isnan(raw))),
        "unparsed": unparsed,
        "skew": skew,
        "kurtosis": kurtosis,
        "shape": shape_label(skew),
    }


def shape_label(skew: float) -> str:
    """Plain-language reading of the skewness of a distribution."""
    if pd.isna(skew):
        return ""
    size = abs(skew)
    if size < 0.5:
        return "Symmetric"
    side = "right" if skew > 0 else "left"
    if size < 1:
        return f"Moderate {side} skew"
    return f"Strong {side} skew"


def numeric_bars(
    series: pd.Series, stats: pd.Series, unique: int, bins: int, sample_rows: int
) -> Bars:
    """Bars of the distribution of a numeric column.

    With at most ``bins`` distinct values (counts, ratings, codes), one bar per value, outliers
    included, counted on every row. Otherwise a histogram of the sample between the whiskers:
    outliers are left out so they do not flatten the bars.

    Args:
        stats: The row of ``numeric_stats`` for this column.
        unique: Distinct values of the column (from ``BaseProfile``), to choose between both.
    """
    if unique <= bins:
        values, counts = np.unique(finite_values(series), return_counts=True)
        return Bars(
            tuple(int(c) for c in counts),
            value_labels(values.tolist()),
            per_value=True,
            edges=tuple(float(v) for v in values),
        )
    values = numeric_sample(series, sample_rows)
    heights, edges = histogram(values, bins, stats["whisker_low"], stats["whisker_high"])
    labels = [f"{a:,.4g} to {b:,.4g}" for a, b in pairwise(edges)]
    return Bars(tuple(heights), tuple(labels), edges=tuple(edges))


def outlier_values(
    series: pd.Series, stats: pd.Series, limit: int, sample_rows: int
) -> list[float]:
    """Up to ``limit`` outliers of the sample, the most extreme first (R-P2, R-V2)."""
    values = numeric_sample(series, sample_rows)
    outside = values[(values < stats["whisker_low"]) | (values > stats["whisker_high"])]
    # Stable, so equal distances keep the order of the sample.
    order = np.argsort(-np.abs(outside - stats["median"]), kind="stable")[:limit]
    return [float(v) for v in outside[order]]


# Categorical and boolean


def top_bars(series: pd.Series, k: int) -> tuple[Bars, int, int]:
    """Bars of the ``k`` most frequent values, and the distinct values and rows left out.

    The rest is not a bar: it usually adds up to far more rows than any single value and would
    flatten the others.
    """
    return _top_bars(value_counts(series), k)


def _top_bars(counts: pd.Series, k: int) -> tuple[Bars, int, int]:
    """``top_bars`` from the output of ``value_counts``."""
    top, rest = counts.head(k), counts.iloc[k:]
    bars = Bars(tuple(int(c) for c in top), tuple(str(v) for v in top.index), True)
    return bars, len(rest), int(rest.sum())


def variant_groups(series: pd.Series) -> list[list[str]]:
    """Values that only differ in case or surrounding spaces ("Madrid", "madrid ").

    Each group lists its spellings, the most frequent first. Groups with more rows first.
    """
    return _variants(value_counts(series))


def _variants(counts: pd.Series) -> list[list[str]]:
    """``variant_groups`` from the output of ``value_counts``: loops over values, not rows."""
    normalized = counts.index.str.strip().str.casefold()
    groups: dict[str, list[str]] = {}
    totals: dict[str, int] = {}
    for value, key, count in zip(counts.index, normalized, counts.to_numpy(), strict=True):
        groups.setdefault(key, []).append(value)
        totals[key] = totals.get(key, 0) + int(count)
    found = [key for key, values in groups.items() if len(values) > 1]
    found.sort(key=lambda key: totals[key], reverse=True)
    return [groups[key] for key in found]


def categorical_stats(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Distinct values, most common one, rare categories and spelling variants of each column.

    Everything is counted on every row: one ``value_counts`` per column, and the rest is
    computed from it.
    """
    rows = []
    for name in columns:
        counts = value_counts(df[name])
        total = counts.sum()
        share = counts / total * 100 if total else counts.astype(float)
        rows.append(
            {
                "column": name,
                "unique": len(counts),
                "mode": counts.index[0] if total else None,
                "mode_pct": float(share.iloc[0]) if total else float("nan"),
                "top": ", ".join(f"{v} ({p:.0f}%)" for v, p in share.head(3).items()),
                "rare": int((share < RARE_PCT).sum()),
                "variants": len(_variants(counts)),
            }
        )
    return pd.DataFrame(rows, columns=CATEGORICAL_STATS_COLUMNS)


# Dates


def datetime_values(series: pd.Series) -> pd.Series:
    """Non-missing values of a date column. Empty if the column is not ``datetime64``.

    Columns are converted when the file is loaded (``schema.apply_schema``), so a date column
    that is still text here could not be read as dates; it is not parsed again (R-L2).
    """
    if not ptypes.is_datetime64_any_dtype(series):
        return pd.Series([], dtype="datetime64[ns]")
    return series.dropna()


def datetime_stats(
    df: pd.DataFrame,
    columns: Sequence[str],
    unparsed: Mapping[str, int],
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """First and last date, span, usual spacing between dates and suspicious values.

    Args:
        unparsed: Values of each column that were not dates when the file was loaded
            (``LoadedDataset.unparsed``).
        now: Dates after it are counted in ``future``. Timezone-naive, like the dates (local
            clock time). ``None`` is the current time; a cached caller passes it, so the count
            does not stay frozen at the moment it was cached. Dates that had a timezone in the
            file were converted to UTC (``schema.apply_schema``), so for them the count can be
            off by the difference between UTC and local time.
    """
    now = pd.Timestamp.now() if now is None else now
    rows = []
    for name in columns:
        values = datetime_values(df[name])
        has_values = not values.empty
        first, last = (values.min(), values.max()) if has_values else (pd.NaT, pd.NaT)
        rows.append(
            {
                "column": name,
                "min": first,
                "max": last,
                "span_days": (last - first).days if has_values else None,
                "frequency": _frequency(values),
                "unparsed": int(unparsed.get(name, 0)),
                "future": int((values > now).sum()),
            }
        )
    return pd.DataFrame(rows, columns=DATETIME_STATS_COLUMNS)


def _frequency(values: pd.Series) -> str:
    """Usual spacing between consecutive distinct dates, in words. Exact, on every row."""
    # Gaps between the sorted distinct dates. ``unique`` (a hash table) first: dates repeat,
    # so sorting the distinct ones took 30 ms on 3 M rows against 234 ms sorting every row.
    distinct = np.sort(pd.unique(values.to_numpy()))
    gaps = np.diff(distinct)
    if len(gaps) < 2:
        return ""
    days = np.median(gaps) / np.timedelta64(1, "D")
    for low, high, label in (
        (0, 0.99, "Sub-daily"),
        (0.99, 1.5, "Daily"),
        (6, 8, "Weekly"),
        (27, 32, "Monthly"),
        (88, 93, "Quarterly"),
        (360, 370, "Yearly"),
    ):
        if low <= days < high:
            return label
    return f"Every ~{days:.0f} days"


def datetime_bars(series: pd.Series, bins: int, sample_rows: int) -> Bars:
    """Rows over time: histogram of the sample between the first and the last date."""
    values = datetime_values(series)
    if values.empty:
        return Bars()
    # Integers in the unit the column is stored in (s, ms, us or ns), never converted to ns:
    # placeholder dates such as 9999-12-31 or 0001-01-01 only fit in the coarser units.
    unit = values.dt.unit
    ticks = values.astype("int64")
    # Exact bounds, so the chart spans the whole column; the heights come from the sample.
    low, high = float(ticks.min()), float(ticks.max())
    counts, edges = histogram(sample(ticks, sample_rows).to_numpy(), bins, low, high)
    dates = [pd.Timestamp(np.datetime64(round(e), unit)) for e in edges]
    labels = [f"{a:%Y-%m-%d} to {b:%Y-%m-%d}" for a, b in pairwise(dates)]
    return Bars(tuple(counts), tuple(labels))


# Text and identifiers


def text_lengths(series: pd.Series) -> pd.Series:
    """Characters of each non-missing value."""
    return series.dropna().astype(str).str.len()


def text_stats(
    df: pd.DataFrame, columns: Sequence[str], schema: Schema, base: BaseProfile, sample_rows: int
) -> pd.DataFrame:
    """Length of the values, blank ones and repeats, for text and identifier columns.

    Lengths come from ``sample_values`` of each column; blank values are exact, and distinct and
    repeated values come from ``base`` (R-P1).
    """
    rows = [_text_row(df[name], schema.type_of(name), base, sample_rows) for name in columns]
    return pd.DataFrame(rows, columns=TEXT_STATS_COLUMNS)


def _text_row(
    series: pd.Series, kind: SemanticType, base: BaseProfile, sample_rows: int
) -> dict[str, Any]:
    """Row of ``text_stats`` for one column of semantic type ``kind``."""
    name = series.name
    lengths = text_lengths(sample_values(series, sample_rows))
    has_values = not lengths.empty
    return {
        "column": name,
        "semantic_type": kind.value,
        "len_min": int(lengths.min()) if has_values else None,
        "len_mean": float(lengths.mean()) if has_values else float("nan"),
        "len_max": int(lengths.max()) if has_values else None,
        "blank": _count_blank(series),
        "unique_pct": float(base.by_column.loc[name, "unique_pct"]),
        "repeated": int(base.by_column.loc[name, "repeated"]),
        "example": _first_non_blank(series),
    }


def _count_blank(series: pd.Series) -> int:
    """Values that are "" or only spaces: not missing for pandas, but with no information."""
    if not ptypes.is_string_dtype(series):
        # Numbers stored in an id column are never blank.
        return 0
    return int(series.dropna().astype(str).str.strip().eq("").sum())


def _first_non_blank(series: pd.Series) -> str:
    """First value with some text, cut to ``EXAMPLE_WIDTH`` characters.

    Reads the column in chunks and stops at the first chunk with such a value: usually the
    first one, instead of copying the whole column without its missing values.
    """
    for start in range(0, len(series), _EXAMPLE_CHUNK):
        for value in series.iloc[start : start + _EXAMPLE_CHUNK].dropna():
            text = str(value)
            if text.strip():
                return text if len(text) <= EXAMPLE_WIDTH else text[: EXAMPLE_WIDTH - 1] + "…"
    return ""


def text_bars(series: pd.Series, bins: int, sample_rows: int) -> Bars:
    """Histogram of the length of the values, on the sample."""
    return _length_bars(text_lengths(sample_values(series, sample_rows)), bins)


def _length_bars(lengths: pd.Series, bins: int) -> Bars:
    """Histogram of ``lengths`` (output of ``text_lengths``). No bars if there are none."""
    if lengths.empty:
        return Bars()
    counts, edges = histogram(lengths.to_numpy(), bins, float(lengths.min()), float(lengths.max()))
    labels = [f"{a:.0f} to {b:.0f}" for a, b in pairwise(edges)]
    return Bars(tuple(counts), tuple(labels))


def text_examples(series: pd.Series, n: int) -> pd.DataFrame:
    """``n`` random values (fixed seed) and their length, as ``value`` and ``length`` (R-P7)."""
    # Pick first and convert after: converting a whole column to text is slow.
    values = series.dropna()
    picked = values.iloc[positions(len(values), min(n, len(values)))].astype(str)
    return pd.DataFrame({"value": picked.to_numpy(), "length": picked.str.len().to_numpy()})


# One column


def column_detail(
    df: pd.DataFrame,
    name: str,
    kind: SemanticType,
    *,
    unique: int,
    stats: pd.Series | None,
    sample_rows: int,
    outlier_limit: int,
    n_examples: int,
) -> ColumnDetail:
    """Chart data of column ``name``, according to its semantic type ``kind``.

    Args:
        unique: Distinct values of the column (from ``BaseProfile``).
        stats: Numeric columns only: the row of ``numeric_stats`` for this column.
        outlier_limit: Most outliers returned, to keep the chart light.
        n_examples: Example values returned for text and identifier columns.
    """
    series = df[name]
    if kind in NUMERIC_TYPES:
        if stats is None:
            raise ValueError(f"Column '{name}' is numeric: its stats are needed.")
        return ColumnDetail(
            bars=numeric_bars(series, stats, unique, DETAIL_BINS, sample_rows),
            outliers=tuple(outlier_values(series, stats, outlier_limit, sample_rows)),
        )
    if kind in CATEGORICAL_TYPES:
        # One count for the bars and the variants.
        counts = value_counts(series)
        bars, other_values, other_rows = _top_bars(counts, TOP_K)
        return ColumnDetail(
            bars=bars,
            variants=tuple(tuple(g) for g in _variants(counts)),
            other_values=other_values,
            other_rows=other_rows,
        )
    if kind in DATE_TYPES:
        return ColumnDetail(bars=datetime_bars(series, DETAIL_BINS, sample_rows))
    return ColumnDetail(
        bars=text_bars(series, DETAIL_BINS, sample_rows),
        examples=text_examples(series, n_examples),
    )


def inspect_column(
    df: pd.DataFrame,
    name: str,
    kind: SemanticType,
    base: BaseProfile,
    *,
    sample_rows: int,
    unparsed: Mapping[str, int],
    now: pd.Timestamp | None,
    outlier_limit: int,
    n_examples: int,
) -> tuple[pd.DataFrame, ColumnDetail]:
    """Measures and chart data of column ``name``, for the column inspector.

    Only that column is computed: the row is the same as in the table of its type, without
    describing every column of that type.

    Args:
        base: Output of ``profile.base_profile(df)``.
        unparsed: ``LoadedDataset.unparsed``, for numeric and date columns.
        now: Date columns only: see ``datetime_stats``.
        outlier_limit: Most outliers returned for a numeric column, to keep the chart light.
        n_examples: Example values returned for text and identifier columns.

    Returns:
        The one-row table of the type of the column (a row of ``numeric_stats``,
        ``categorical_stats``, ``datetime_stats`` or ``text_stats``) and its chart data.
    """
    stats: pd.Series | None = None
    if kind in NUMERIC_TYPES:
        table = numeric_stats(df, (name,), sample_rows, unparsed=unparsed)
        stats = table.iloc[0]
    elif kind in CATEGORICAL_TYPES:
        table = categorical_stats(df, (name,))
    elif kind in DATE_TYPES:
        table = datetime_stats(df, (name,), unparsed, now)
    else:
        row = _text_row(df[name], kind, base, sample_rows)
        table = pd.DataFrame([row], columns=TEXT_STATS_COLUMNS)
    detail = column_detail(
        df,
        name,
        kind,
        unique=int(base.by_column.loc[name, "unique"]),
        stats=stats,
        sample_rows=sample_rows,
        outlier_limit=outlier_limit,
        n_examples=n_examples,
    )
    return table, detail


# Overview


def column_overview(
    df: pd.DataFrame,
    schema: Schema,
    base: BaseProfile,
    numeric: pd.DataFrame,
    sample_rows: int,
    workers: int = 1,
) -> pd.DataFrame:
    """One short row per column: type, missing, distinct values and a distribution.

    Columns: ``column``, ``semantic_type``, ``missing_pct``, ``unique_pct`` (from ``base``) and
    ``distribution`` (the bar heights of a small chart of its values):
    - numeric: one bar per value if there are few, else histogram without outliers;
    - date: histogram over time;
    - categorical / boolean: frequency of the most common values;
    - text: histogram of the length of the values;
    - identifier: empty (every value is different, there is nothing to draw).

    Args:
        base: Output of ``profile.base_profile(df)``.
        numeric: Output of ``numeric_stats`` for the numeric columns.
        workers: Threads, one column each. Same result.
    """
    numeric = numeric.set_index("column")
    # Read before the threads start: they all share it.
    by_column = base.by_column

    def row(name: str, kind: SemanticType) -> dict[str, Any]:
        stats = by_column.loc[name]
        has_values = int(stats["missing"]) < base.n_rows
        bars = (
            _overview_bars(df[name], kind, int(stats["unique"]), numeric, sample_rows)
            if has_values
            else Bars()
        )
        return {
            "column": name,
            "semantic_type": kind.value,
            "missing_pct": float(stats["missing_pct"]),
            "unique_pct": float(stats["unique_pct"]),
            "distribution": list(bars.counts),
        }

    rows = map_columns(lambda c: row(c.name, c.semantic_type), schema.columns, workers)
    return pd.DataFrame(rows, columns=OVERVIEW_COLUMNS)


def _overview_bars(
    series: pd.Series,
    kind: SemanticType,
    unique: int,
    numeric: pd.DataFrame,
    sample_rows: int,
) -> Bars:
    """Small chart of a column with at least one value, for the overview."""
    if kind in NUMERIC_TYPES:
        stats = numeric.loc[series.name]
        if not stats["count"]:
            return Bars()
        return numeric_bars(series, stats, unique, OVERVIEW_BINS, sample_rows)
    if kind in CATEGORICAL_TYPES:
        return top_bars(series, TOP_K)[0]
    if kind in DATE_TYPES:
        return datetime_bars(series, OVERVIEW_BINS, sample_rows)
    if kind is SemanticType.TEXT:
        return text_bars(series, OVERVIEW_BINS, sample_rows)
    # Identifier: every value is different, there is nothing to draw.
    return Bars()
