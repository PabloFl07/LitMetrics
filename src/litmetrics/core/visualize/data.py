"""Aggregated data of one chart: ``chart_data(df, spec)`` turns the dataset into the small
``ChartData`` a chart draws (R-V1, R-V2), and ``to_frame`` gives it back as a table.

Every row is read here: the rules of docs/performance.md apply (R-V1 to R-V9).
"""

from collections.abc import Collection, Sequence
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from litmetrics.config import VisualizeConfig
from litmetrics.core.columns import histogram, value_labels
from litmetrics.core.sampling import positions, sample_array
from litmetrics.core.visualize.spec import (
    ROLES,
    Agg,
    ChartData,
    ChartKind,
    DataSpec,
    DateFilter,
    RangeFilter,
    RowFilter,
    Series,
    TimeUnit,
    ValuesFilter,
    is_number,
    value_name,
)

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

# Aggregations that summarize the rows of a point instead of adding them up: from a handful of
# rows they are noise (``VisualizeConfig.min_rows_per_point``).
_SUMMARIES = frozenset({Agg.MEAN, Agg.MEDIAN, Agg.MIN, Agg.MAX})


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
    if isinstance(f, RangeFilter) and not is_number(column):
        raise ValueError(f"{f.column!r} has no numbers to filter by.")


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
    labels = list(dict.fromkeys(_interval_labels(edges)))
    # The last interval includes its upper end, as in ``np.histogram``.
    codes = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(labels) - 1)
    keys = pd.Series(
        pd.Categorical.from_codes(codes, categories=labels), index=sub.index, name=spec.x
    )
    return sub, keys, labels


# Interval labels are written with fixed decimals up to these: past them (tiny widths, huge
# values) the decimals or digits are noise, and ``value_labels`` writes them shorter.
_MAX_LABEL_DECIMALS = 6
_MAX_LABEL_NUMBER = 1e9


def _interval_labels(edges: np.ndarray) -> list[str]:
    """Labels of the intervals between ``edges`` ("0.50 to 1.35"), every end with the same
    decimals: whole numbers when every edge is one, else enough to tell apart a tenth of the
    width of an interval. Ends that would still look alike, or numbers too long to write out,
    take the shortest distinct form of ``columns.value_labels`` instead.
    """
    width = float(edges[1] - edges[0]) if len(edges) > 1 else 0.0
    if np.all(edges == np.round(edges)):
        decimals = 0
    elif width > 0:
        decimals = max(0, int(np.ceil(-np.log10(width))) + 1)
    else:
        decimals = _MAX_LABEL_DECIMALS + 1
    ends = [f"{e:,.{decimals}f}" for e in edges]
    too_long = decimals > _MAX_LABEL_DECIMALS or float(np.abs(edges).max()) >= _MAX_LABEL_NUMBER
    if too_long or len(set(ends)) < len(ends):
        ends = list(value_labels(edges))
    return [f"{a} to {b}" for a, b in pairwise(ends)]


def _aggregate(
    frame: pd.DataFrame,
    keys: list[pd.Series],
    value: str | None,
    agg: Agg,
    min_rows: int = 1,
) -> tuple[pd.Series, int]:
    """``agg`` of column ``value`` by ``keys`` (rows when ``value`` is None or ``agg`` COUNT).

    Groups on categories with ``observed=True`` (R-V9): only combinations that exist.

    Args:
        min_rows: Rows a group needs for a mean, median, min or max (``_SUMMARIES``); with
            fewer it is missing (NaN). Counts and sums are never left out.

    Returns:
        The aggregated values, and how many groups were left out for having too few rows.
    """
    grouped = frame.groupby(keys, observed=True, sort=True)
    if value is None or agg == Agg.COUNT:
        return grouped.size(), 0
    result = grouped[value].agg(str(agg))
    if min_rows <= 1 or agg not in _SUMMARIES:
        return result, 0
    thin = grouped.size() < min_rows
    return result.mask(thin), int(thin.sum())


def _thin_note(thin: int, spec: DataSpec) -> list[str]:
    """Note of the points ``_aggregate`` left out (none when ``thin`` is 0)."""
    if not thin:
        return []
    what = "point" if thin == 1 else "points"
    return [
        f"{thin:,} {what} with fewer than {spec.min_rows:,} rows not drawn "
        f"(Min rows per interval): a {spec.agg} of so few rows is noise."
    ]


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

    Intervals with fewer than ``spec.min_rows`` rows get no mean, median, min or max, and a
    group with no interval left is not drawn.
    """
    is_pie = spec.kind == ChartKind.PIE
    agg = Agg.SUM if is_pie and spec.value else spec.agg
    group = None if is_pie else spec.group
    sub, notes = _used(df, spec, [spec.x, spec.value, group])
    if sub.empty:
        return ChartData(spec.kind, notes=tuple(notes))
    intervals: list[str] | None = None
    other = None
    if not is_pie and is_number(sub[spec.x]):
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
    # Intervals of a numeric x read as a trend: a point of a few rows is a false spike.
    # Categories are named one by one, and the person sees what each one is.
    least = spec.min_rows if intervals is not None else 1
    if group is None:
        values, thin = _aggregate(sub, [keys], spec.value, agg, least)
        table = values.to_frame(value_name(spec))
    else:
        groups, _, gnote = _top_categories(sub[group], VisualizeConfig.max_series)
        notes += [gnote] if gnote else []
        values, thin = _aggregate(sub, [keys, groups], spec.value, agg, least)
        table = values.unstack()  # noqa: PD010 (aggregated: pivot_table would group again)
    notes += _thin_note(thin, spec)
    if thin:
        # A group whose every interval was too thin has no line or bar left: no legend entry.
        table = table.dropna(axis=1, how="all")
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
        y_name=value_name(spec),
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
        labels = tuple(_interval_labels(edges))
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
    if is_number(df[spec.x]):
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
    # No minimum of rows per period: a daily series often has one row a day.
    if spec.group is None:
        table = _aggregate(sub, [periods], spec.value, spec.agg)[0].to_frame(value_name(spec))
    else:
        groups, _, gnote = _top_categories(sub[spec.group], VisualizeConfig.max_series)
        notes += [gnote] if gnote else []
        table = _aggregate(sub, [periods, groups], spec.value, spec.agg)[0].unstack()  # noqa: PD010 (aggregated)
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
        y_name=value_name(spec),
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
    table, _ = _aggregate(sub, [xs, ys], spec.value, spec.agg)
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
        series=(Series(value_name(spec), cells),),
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
