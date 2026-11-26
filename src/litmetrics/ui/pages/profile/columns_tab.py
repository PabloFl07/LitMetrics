"""Columns tab of the Profile page: an overview with one row per column, one table per
semantic type, and the inspector of the column whose row is selected.
"""

from typing import Any

import pandas as pd
import streamlit as st

from litmetrics.config import ProfileConfig
from litmetrics.core import columns as cols
from litmetrics.core.schema import SemanticType
from litmetrics.ui import cache, charts, echarts, state, text, theme
from litmetrics.ui.pages.profile import shared

_CFG = ProfileConfig

# Views of the Columns tab: the semantic types each one covers, and its Material icon.
_VIEWS: dict[str, tuple[tuple[SemanticType, ...], str]] = {
    "Overview": (tuple(SemanticType), "table_rows"),
    "Numeric": (cols.NUMERIC_TYPES, "123"),
    "Categorical": (cols.CATEGORICAL_TYPES, "category"),
    "Dates": (cols.DATE_TYPES, "calendar_month"),
    "Text & ids": (cols.TEXT_TYPES, "notes"),
}

# Groups of measures of the numeric table. The user picks which ones to show.
_NUMERIC_GROUPS: dict[str, list[str]] = {
    "Position": ["min", "p5", "q1", "median", "q3", "p95", "max"],
    "Spread": ["mean", "std", "cv", "iqr"],
    "Outliers": ["whisker_low", "whisker_high", "outliers_label"],
    "Special values": ["count", "zeros", "negatives", "infinite", "unparsed"],
    "Shape": ["skew", "kurtosis", "shape"],
}
_NUMERIC_DEFAULT_GROUPS = ["Position", "Outliers"]

# Height (px) of the header of a table, to size tables to their content. Rows have a fixed
# height (``ProfileConfig.table_row_height``) passed to every table sized this way.
_HEADER_PX = 36
# Height of the small charts of the date and text column inspectors.
_CHART_HEIGHT = "200px"
# Height of the histogram / boxplot of the numeric column inspector.
_INSPECT_CHART_HEIGHT = "240px"
# Material icon of each chart of the numeric column inspector.
_NUMERIC_CHART_ICONS = {"Histogram": "bar_chart", "Boxplot": "candlestick_chart"}

# Label, number format and explanation of every measure of the numeric table. The table's
# ``column_config`` and the "All measures" cards (``_format_stat``) are both built from it.
_NUMERIC_MEASURES: dict[str, tuple[str, str | None, str | None]] = {
    "count": ("Values", None, "Finite, non-missing values"),
    "min": ("Min", "%.4g", None),
    "p5": ("P5", "%.4g", "5% of the values are lower"),
    "q1": ("Q1", "%.4g", "25% of the values are lower"),
    "median": ("Median", "%.4g", None),
    "q3": ("Q3", "%.4g", "75% of the values are lower"),
    "p95": ("P95", "%.4g", "95% of the values are lower"),
    "max": ("Max", "%.4g", None),
    "mean": ("Mean", "%.4g", None),
    "std": ("Std dev", "%.4g", None),
    "cv": ("CV", "%.2f", "Coefficient of variation: std dev / |mean|"),
    "iqr": ("IQR", "%.4g", "Q3 - Q1: width of the middle half of the values"),
    "whisker_low": (
        "Low (no outliers)",
        "%.4g",
        "Lowest value that is not an outlier (lower whisker of the boxplot)",
    ),
    "whisker_high": (
        "High (no outliers)",
        "%.4g",
        "Highest value that is not an outlier (upper whisker of the boxplot)",
    ),
    # Count and share in one text column, built by ``_with_labels``.
    "outliers_label": (
        "Outliers",
        None,
        f"Values beyond {cols.TUKEY_K:g} times the IQR from the quartiles (Tukey), and their "
        "share of the values",
    ),
    "zeros": ("Zeros", None, None),
    "negatives": ("Negatives", None, None),
    "infinite": ("Infinite", None, None),
    "unparsed": (
        "Not numbers",
        None,
        'Values that were not numbers in the file ("cincuenta", "41 años"). They are counted '
        "as missing.",
    ),
    "skew": ("Skewness", "%+.2f", "0: symmetric. > 0: long tail to the right"),
    "kurtosis": ("Kurtosis", "%+.2f", "> 0: heavier tails than a normal distribution"),
    "shape": ("Shape", None, None),
}
# Measures shown as text in the table; the rest are numbers.
_TEXT_MEASURES = {"outliers_label", "shape"}

# Explanations shown both in a per-type table and in the inspector of one column.
_BLANK_HELP = 'Empty or only spaces: "" or "   "'
_REPEATED_HELP = "Rows whose value appears more than once"
# First column of every per-type table.
_NAME_COLUMN = "column"


# Columns


@st.fragment
def tab() -> None:
    """Overview of every column, or the table of one semantic type."""
    active = shared.active()
    dataset_id, dataset, schema = active.dataset_id, active.dataset, active.schema
    sizes = {view: len(schema.names_of(kinds)) for view, (kinds, _) in _VIEWS.items()}
    # Only views that have columns in this dataset.
    views = [view for view in _VIEWS if sizes[view]]
    view = st.segmented_control(
        "View",
        views,
        default="Overview",
        format_func=lambda v: f":material/{_VIEWS[v][1]}: {v} ({sizes[v]})",
        label_visibility="collapsed",
        key=state.widget_key("columns_view"),
    )
    # Clicking the selected option again deselects it: fall back to the overview.
    view = view or "Overview"
    names = schema.names_of(_VIEWS[view][0])
    df = dataset.df

    # One table key per view and set of skipped columns: a selected row never points to a
    # column of another table.
    key = state.widget_key(f"columns_{view}_{active.view_tag}")
    if view == "Overview":
        selected = _overview(active)
    elif view == "Numeric":
        table = cache.numeric_stats(dataset_id, df, names, _CFG.sample_rows, dataset.unparsed)
        selected = _numeric_table(_with_labels(table), key)
        shared.sample_note(dataset, "Percentiles, IQR, skewness and kurtosis are")
    elif view == "Categorical":
        selected = _categorical_table(cache.categorical_stats(dataset_id, df, names), key)
    elif view == "Dates":
        table = cache.datetime_stats(dataset_id, df, names, dataset.unparsed, _now())
        selected = _dates_table(table, key)
    else:
        table = cache.text_stats(dataset_id, df, names, schema, _CFG.sample_rows)
        selected = _text_table(table, key)
        shared.sample_note(dataset, "Lengths are")
    if selected is None:
        st.caption(":material/touch_app: Select a row to inspect that column.")
    else:
        _inspector(active, selected)
    shared.about(_columns_help())


def _overview(active: shared.Active) -> str | None:
    """One short row per column.

    Returns:
        The name of the column of the selected row, or ``None`` when no row is selected.
    """
    dataset = active.dataset
    table = cache.column_overview(
        active.dataset_id, dataset.df, active.schema, _CFG.sample_rows, dataset.unparsed
    )
    counts = table["semantic_type"].value_counts()
    kinds = [kind for kind in SemanticType if kind.value in counts]
    picked = st.pills(
        "Filter by type",
        kinds,
        selection_mode="multi",
        format_func=lambda k: f"{k.value} ({counts[k.value]})",
        label_visibility="collapsed",
        key=state.widget_key("overview_types"),
    )
    if picked:
        table = table[table["semantic_type"].isin([k.value for k in picked])]
        table = table.reset_index(drop=True)

    event = st.dataframe(
        table,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        # A new key when the filter or the skipped columns change, so a selected row never
        # points to another column (or past the end of a shorter table).
        key=state.widget_key(
            f"overview_{active.view_tag}_" + "-".join(sorted(k.value for k in picked))
        ),
        height=_table_height(len(table)),
        row_height=_CFG.table_row_height,
        column_order=cols.OVERVIEW_COLUMNS,
        column_config={
            "column": st.column_config.TextColumn("Column", pinned=True),
            "semantic_type": st.column_config.TextColumn("Type"),
            "missing_pct": st.column_config.ProgressColumn(
                "Missing", format="%.1f%%", min_value=0, max_value=100
            ),
            "unique_pct": st.column_config.NumberColumn(
                "Distinct", format="%.0f%%", help="Distinct values among the non-missing ones"
            ),
            "distribution": st.column_config.BarChartColumn(
                "Distribution",
                y_min=0,
                help=(
                    "Numeric: one bar per value if there are few, else a histogram without "
                    "outliers. Dates: rows over time. Categorical: most common values. "
                    "Text: length of the values."
                ),
            ),
        },
    )
    shared.sample_note(dataset, "Histograms and text lengths are")
    return _selected_column(table, event)


def _inspector(active: shared.Active, name: str) -> None:
    """Everything about one column, laid out for its semantic type.

    Only that column is computed (``cache.inspect_column``), not the table of its type.
    """
    dataset_id, dataset = active.dataset_id, active.dataset
    df = dataset.df
    kind = active.schema.type_of(name)
    base = cache.base_profile(dataset_id, df).by_column.loc[name]
    # Keyed: styles.css gives the inspector a shadow.
    with st.container(border=True, key="inspector"):
        st.subheader(text.md(name), anchor=False)
        st.caption(
            f"{kind.value} · pandas type `{df[name].dtype}` · {base['missing_pct']:.1f}% missing"
        )
        if base["missing"] == dataset.n_rows:
            st.info("The column has no values.")
            return
        now = _now() if kind in cols.DATE_TYPES else None
        table, detail = cache.inspect_column(
            dataset_id, df, name, kind, _CFG.sample_rows, dataset.unparsed, now
        )
        if kind in cols.NUMERIC_TYPES:
            stats = _with_labels(table).iloc[0]
            # Infinite values are not missing, but every measure ignores them.
            if not stats["count"]:
                st.info("The column has no finite values.")
                return
            _inspect_numeric(name, stats, detail)
        elif kind in cols.CATEGORICAL_TYPES:
            _inspect_categorical(table.iloc[0])
            _top_values(name, detail)
        elif kind in cols.DATE_TYPES:
            _inspect_dates(name, table.iloc[0], detail)
        else:
            _inspect_text(name, kind, table.iloc[0], detail)
        if kind not in cols.CATEGORICAL_TYPES:
            shared.sample_note(dataset, "Charts are")


def _inspect_numeric(name: str, stats: pd.Series, detail: cols.ColumnDetail) -> None:
    """Headline measures, histogram or boxplot, and every measure in cards."""
    if stats["unparsed"]:
        st.caption(
            f":material/warning: {stats['unparsed']:,} values were not numbers in the file "
            "(words, units...). They are counted as missing."
        )
    median, mean, iqr, outliers, skew = st.columns(5)
    _metric(median, "Median", _num(stats["median"]))
    _metric(mean, "Mean", _num(stats["mean"]))
    _metric(iqr, "IQR", _num(stats["iqr"]), help_text="Width of the middle half of the values")
    _metric(
        outliers,
        "Outliers",
        f"{stats['outliers']:,}",
        f"{stats['outliers_pct']:.1f}% of the values",
    )
    _metric(skew, "Skewness", _format_stat("skew", stats["skew"]), stats["shape"] or None)

    chart = st.segmented_control(
        "Chart",
        list(_NUMERIC_CHART_ICONS),
        default="Histogram",
        format_func=lambda c: f":material/{_NUMERIC_CHART_ICONS[c]}: {c}",
        label_visibility="collapsed",
        # One key for every column: the chosen chart stays when another column is inspected.
        key=state.widget_key("inspect_numeric_chart"),
    )
    # Clicking the selected option again deselects it: fall back to the histogram.
    if chart == "Boxplot":
        _boxplot(name, stats, detail)
    else:
        _histogram(name, stats, detail.bars)

    # One card per group, side by side; they wrap to new rows on narrow screens.
    with st.expander("All measures"), st.container(horizontal=True, gap="small"):
        for group, keys in _NUMERIC_GROUPS.items():
            card_key = "measure_card_" + group.lower().replace(" ", "_")
            with st.container(border=True, width="stretch", gap=None, key=card_key):
                st.markdown(f"**{group}**")
                for key in keys:
                    # Label on the left, value on the right. The value never moves to a line
                    # of its own: a long label wraps instead.
                    with st.container(
                        horizontal=True, horizontal_alignment="distribute", wrap=False
                    ):
                        label, _, help_text = _NUMERIC_MEASURES[key]
                        st.markdown(f":gray[{label}]", width="stretch", help=help_text)
                        st.markdown(f"**{_format_stat(key, stats[key])}**", width="content")


def _numeric_chart_key(name: str) -> str:
    """Key shared by the histogram and the boxplot of column ``name``.

    With one key, switching chart updates the same component instead of removing it and
    mounting another: the page does not shrink for a moment, so the scroll does not jump.
    """
    return f"profile_numeric_{name}"


def _histogram(name: str, stats: pd.Series, bars: cols.Bars) -> None:
    """Histogram (or one bar per value) with lines at the median and the mean."""
    if bars.per_value:
        st.caption("Few distinct values: one bar per value, outliers included.")
    else:
        st.caption("Between the whiskers: outliers are left out so they do not flatten the bars.")
    marks = [
        charts.Mark(label, index, _num(stats[key]))
        for label, key in (("Median", "median"), ("Mean", "mean"))
        if (index := cols.bar_index(bars, stats[key])) is not None
    ]
    echarts.show(
        charts.histogram_options(
            list(bars.labels),
            list(bars.counts),
            theme.chart_colors(),
            show_labels=bars.per_value,
            marks=marks,
        ),
        height=_INSPECT_CHART_HEIGHT,
        key=_numeric_chart_key(name),
    )


def _boxplot(name: str, stats: pd.Series, detail: cols.ColumnDetail) -> None:
    """Horizontal boxplot with the outliers as dots and a line at the mean."""
    option = charts.boxplot_options(
        name,
        (
            float(stats["whisker_low"]),
            float(stats["q1"]),
            float(stats["median"]),
            float(stats["q3"]),
            float(stats["whisker_high"]),
        ),
        list(detail.outliers),
        (float(stats["min"]), float(stats["max"])),
        _CFG.box_zoom_ratio,
        theme.chart_colors(),
        marks=[
            charts.Mark(label, float(stats[key]), _num(stats[key]))
            for label, key in (("Median", "median"), ("Mean", "mean"))
        ],
    )
    caption = (
        f"Box: Q1 to Q3, split at the median. Whiskers: {_num(stats['whisker_low'])} to "
        f"{_num(stats['whisker_high'])}. Dots: outliers."
    )
    if "dataZoom" in option:
        caption += " Zoomed on the box: drag the slider to reach the farthest outliers."
    st.caption(caption)
    echarts.show(option, height=_INSPECT_CHART_HEIGHT, key=_numeric_chart_key(name))


def _inspect_categorical(stats: pd.Series) -> None:
    """Headline measures of a categorical or boolean column."""
    unique, mode, rare, variants = st.columns(4)
    _metric(unique, "Distinct values", f"{stats['unique']:,}")
    _metric(mode, "Most common", str(stats["mode"]), f"{stats['mode_pct']:.1f}% of the rows")
    _metric(
        rare,
        "Rare categories",
        stats["rare"],
        help_text=f"Less than {cols.RARE_PCT:g}% of the rows",
    )
    _metric(
        variants,
        "Spelling variants",
        stats["variants"],
        help_text="Same value with other case or spaces",
    )


def _top_values(name: str, detail: cols.ColumnDetail) -> None:
    """Spelling variants, if any, and the most common values as horizontal bars."""
    if detail.variants:
        listed = "\n".join(
            f"- {' / '.join(text.code(repr(v)) for v in group)}" for group in detail.variants[:5]
        )
        st.warning(f"Same value written in different ways:\n{listed}", icon=":material/spellcheck:")
    bars = detail.bars
    st.markdown("**Most common values**")
    if detail.other_values:
        st.caption(
            f"Not drawn: {detail.other_values:,} other values in {detail.other_rows:,} rows."
        )
    echarts.show(
        charts.hbar_options(list(bars.labels), list(bars.counts), theme.chart_colors()),
        height=charts.hbar_height(len(bars.counts)),
        key=f"profile_top_{name}",
    )


def _inspect_dates(name: str, stats: pd.Series, detail: cols.ColumnDetail) -> None:
    """Headline measures of a date column and its rows over time."""
    # Two rows of three: six metrics in one row cut the dates short.
    first, last, span = st.columns(3)
    frequency, unparsed, future = st.columns(3)
    _metric(first, "From", f"{stats['min']:%Y-%m-%d}")
    _metric(last, "To", f"{stats['max']:%Y-%m-%d}")
    _metric(span, "Span", f"{stats['span_days']:,} days")
    _metric(
        frequency,
        "Spacing",
        stats["frequency"] or "—",
        help_text="Usual gap between consecutive dates",
    )
    _metric(
        unparsed,
        "Not dates",
        stats["unparsed"],
        help_text="Values that could not be read as dates when the file was loaded",
    )
    _metric(future, "In the future", stats["future"])

    bars = detail.bars
    st.markdown("**Rows over time**")
    echarts.show(
        charts.histogram_options(list(bars.labels), list(bars.counts), theme.chart_colors()),
        height=_CHART_HEIGHT,
        key=f"profile_dates_{name}",
    )


def _inspect_text(
    name: str, kind: SemanticType, stats: pd.Series, detail: cols.ColumnDetail
) -> None:
    """Headline measures of a text or identifier column, its lengths and some examples."""
    if kind is SemanticType.IDENTIFIER:
        st.caption(
            ":material/key: Identifier: (almost) one distinct value per row. Useful to join "
            "tables, not for statistics or charts."
        )
    length, blank, unique, repeated = st.columns(4)
    _metric(
        length,
        "Length",
        f"{stats['len_min']} to {stats['len_max']}",
        f"{stats['len_mean']:.0f} on average",
    )
    _metric(blank, "Blank", stats["blank"], help_text=_BLANK_HELP)
    _metric(unique, "Distinct", f"{stats['unique_pct']:.1f}%")
    _metric(repeated, "Repeated values", f"{stats['repeated']:,}", help_text=_REPEATED_HELP)

    chart, examples = st.columns([3, 2])
    with chart:
        bars = detail.bars
        st.markdown("**Length of the values**")
        echarts.show(
            charts.histogram_options(list(bars.labels), list(bars.counts), theme.chart_colors()),
            height=_CHART_HEIGHT,
            key=f"profile_lengths_{name}",
        )
    with examples:
        st.markdown("**Examples**")
        st.dataframe(detail.examples, hide_index=True)


def _numeric_table(table: pd.DataFrame, key: str) -> str | None:
    """Numeric table, with the groups of measures the user picks.

    Returns:
        The column of the selected row, or ``None``.
    """
    groups = st.pills(
        "Measures",
        list(_NUMERIC_GROUPS),
        selection_mode="multi",
        default=_NUMERIC_DEFAULT_GROUPS,
        key=state.widget_key("numeric_groups"),
    )
    order = [_NAME_COLUMN] + [
        key for group in _NUMERIC_GROUPS if group in groups for key in _NUMERIC_GROUPS[group]
    ]
    config: dict[str, Any] = {}
    for key, (label, fmt, help_text) in _NUMERIC_MEASURES.items():
        if key in _TEXT_MEASURES:
            config[key] = st.column_config.TextColumn(label, help=help_text)
        else:
            config[key] = st.column_config.NumberColumn(label, format=fmt, help=help_text)
    return _stats_table(table, key, config, column_order=order)


def _categorical_table(table: pd.DataFrame, key: str) -> str | None:
    """Categorical and boolean table. Returns the column of the selected row, or ``None``."""
    return _stats_table(
        table,
        key,
        {
            "unique": st.column_config.NumberColumn("Distinct"),
            "mode": st.column_config.TextColumn("Most common"),
            "mode_pct": st.column_config.ProgressColumn(
                "Its share", format="%.0f%%", min_value=0, max_value=100
            ),
            "top": st.column_config.TextColumn("Top 3", width="large"),
            "rare": st.column_config.NumberColumn(
                "Rare", help=f"Categories with less than {cols.RARE_PCT:g}% of the rows"
            ),
            "variants": st.column_config.NumberColumn(
                "Variants", help='Values that only differ in case or spaces ("Madrid", "madrid ")'
            ),
        },
    )


def _dates_table(table: pd.DataFrame, key: str) -> str | None:
    """Date table. Returns the column of the selected row, or ``None``."""
    return _stats_table(
        table,
        key,
        {
            "min": st.column_config.DatetimeColumn("From", format="YYYY-MM-DD"),
            "max": st.column_config.DatetimeColumn("To", format="YYYY-MM-DD"),
            "span_days": st.column_config.NumberColumn("Span (days)"),
            "frequency": st.column_config.TextColumn(
                "Spacing", help="Usual gap between consecutive distinct dates"
            ),
            "unparsed": st.column_config.NumberColumn(
                "Not dates", help="Values that could not be read as dates when loading"
            ),
            "future": st.column_config.NumberColumn("In the future"),
        },
    )


def _text_table(table: pd.DataFrame, key: str) -> str | None:
    """Text and identifier table. Returns the column of the selected row, or ``None``."""
    return _stats_table(
        table,
        key,
        {
            "semantic_type": st.column_config.TextColumn("Type"),
            "len_min": st.column_config.NumberColumn("Min length"),
            "len_mean": st.column_config.NumberColumn("Avg length", format="%.1f"),
            "len_max": st.column_config.NumberColumn("Max length"),
            "blank": st.column_config.NumberColumn("Blank", help=_BLANK_HELP),
            "unique_pct": st.column_config.NumberColumn("Distinct", format="%.1f%%"),
            "repeated": st.column_config.NumberColumn("Repeated", help=_REPEATED_HELP),
            "example": st.column_config.TextColumn("Example", width="large"),
        },
    )


def _stats_table(
    table: pd.DataFrame,
    key: str,
    column_config: dict[str, Any],
    column_order: list[str] | None = None,
) -> str | None:
    """A per-type table: one row per column, its name pinned, sized to its rows. One row can
    be selected, to inspect that column.

    Args:
        key: Widget key of the table, which holds the selected row.
        column_config: ``column_config`` of the measures; the name column is added here.
        column_order: Columns shown, in order. None shows every column of ``table``.

    Returns:
        The column of the selected row, or ``None`` when no row is selected.
    """
    event = st.dataframe(
        table,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=key,
        column_order=column_order,
        column_config={
            _NAME_COLUMN: st.column_config.TextColumn("Column", pinned=True),
            **column_config,
        },
        height=_table_height(len(table)),
        row_height=_CFG.table_row_height,
    )
    return _selected_column(table, event)


def _selected_column(table: pd.DataFrame, event: Any) -> str | None:
    """Name of the column in the selected row of ``table`` (``event`` of its
    ``st.dataframe``), or ``None`` when no row is selected."""
    if not event.selection.rows:
        return None
    return str(table.iloc[event.selection.rows[0]][_NAME_COLUMN])


def _columns_help() -> str:
    return f"""
**Overview**: one row per column. *Type* is what the values mean (numeric, categorical, date,
boolean, text, id), guessed from how they are stored and what they look like. *Distinct* is
the share of different values among the non-missing ones: low means the values repeat
(categories), 100% means every value is different. *Distribution* is a small chart of the
values. Select a row to see that column in detail.

**Per-type tables**: only the measures that make sense for each type. Here too, select a row
to see that column in detail.

- *Numeric*: the range without outliers is the one a boxplot draws: values beyond
  {cols.TUKEY_K:g} times the IQR from the quartiles (Tukey's rule) are outliers. If the mean and the
  median are far apart, the distribution is skewed or has outliers.
- *Categorical*: rare categories have less than {cols.RARE_PCT:g}% of the rows. Spelling
  variants are the same value written with other case or spaces ("Madrid", "madrid ").
- *Dates*: values that were not dates were left empty when the file was loaded.
- *Text & ids*: blank values are "" or only spaces, which pandas does not count as missing.

With more than {_CFG.sample_rows:,} rows, percentiles, histograms and text lengths are
computed on a random sample of up to that many values of each column. Counts are always
exact.
"""


# Helpers


def _now() -> pd.Timestamp:
    """Current local time, to the hour: dates after it are "in the future".

    Rounded so the cached date table is reused within the hour and recomputed after it.
    """
    return pd.Timestamp.now().floor("h")


def _with_labels(table: pd.DataFrame) -> pd.DataFrame:
    """Numeric table plus ``outliers_label``: outliers and their share in one text column."""
    labels = [
        f"{int(n):,} ({pct:.1f}%)"
        for n, pct in zip(table["outliers"], table["outliers_pct"], strict=True)
    ]
    return table.assign(outliers_label=labels)


def _table_height(n_rows: int) -> int:
    """Height that fits ``n_rows`` rows of ``ProfileConfig.table_row_height``, up to
    ``ProfileConfig.table_max_height``. Tables sized with it pass that row height.
    """
    return min(_HEADER_PX + _CFG.table_row_height * n_rows, _CFG.table_max_height)


def _metric(
    where: Any, label: str, value: Any, note: str | None = None, help_text: str | None = None
) -> None:
    """A metric whose ``note`` is a neutral remark under the value, not a rise or a fall."""
    where.metric(label, value, note, delta_color="off", delta_arrow="off", help=help_text)


def _num(value: float) -> str:
    """A measure in the unit of the column (median, whiskers...), or a dash if undefined."""
    return "—" if pd.isna(value) else f"{value:,.4g}"


def _format_stat(key: str, value: Any) -> str:
    """Measure ``key`` of the numeric table as text, in the format ``_NUMERIC_MEASURES`` gives
    it in the table, plus thousands separators. A dash if it is undefined.
    """
    if key in _TEXT_MEASURES:
        return str(value) or "—"
    if pd.isna(value):
        return "—"
    fmt = _NUMERIC_MEASURES[key][1]
    if fmt is None:
        # Measures without a number format are counts.
        return f"{int(value):,}"
    # "%+.2f" -> "+,.2f": printf format to format spec, sign first.
    spec = fmt.removeprefix("%")
    sign = "+" if spec.startswith("+") else ""
    return format(value, f"{sign},{spec.removeprefix('+')}")
