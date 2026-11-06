"""Profile page: how the active dataset looks, and what to check before analysing it.

Headline quality numbers on top, then four tabs:
- Issues: problems found, with a filter by kind.
- Columns: an overview with one row per column (select one to inspect it) and one table per
  semantic type, with only the measures that make sense for it.
- Missing values: by column, by row and a sample of incomplete rows.
- Correlation: heatmap of the numeric columns and the strongest pairs.

How it stays fast on large datasets (docs/performance.md):
- Only the open tab runs (R-E1), and every tab with widgets is an ``st.fragment``, so a widget
  reruns its tab and nothing else (R-E2). Fragments read the dataset from ``state``.
- Everything is computed through ``ui/cache.py``, once per dataset. Expensive measures use a
  sample, and the page says so where it applies (R-P2).
- Tables and charts receive aggregates or capped rows, never the whole dataset (R-E3, R-V2).
"""

import dataclasses
from typing import Any

import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts

from litmetrics.config import ProfileConfig
from litmetrics.core import columns as cols
from litmetrics.core.io.base import LoadedDataset
from litmetrics.core.profile import (
    CorrelationMethod,
    IssueKind,
    detect_issues,
    missing_by_column,
    top_correlated_pairs,
)
from litmetrics.core.schema import IDENTIFIER_MIN_DISTINCT, Schema, SemanticType, drop_columns
from litmetrics.ui import cache, charts, state

_CFG = ProfileConfig

_ISSUES_TAB = ":material/warning: Issues"
_COLUMNS_TAB = ":material/view_column: Columns"
_MISSING_TAB = ":material/block: Missing values"
_CORRELATION_TAB = ":material/link: Correlation"

_METHODS: dict[str, CorrelationMethod] = {"Pearson": "pearson", "Spearman": "spearman"}

# How each kind of issue is shown: label, Streamlit badge color and Material icon.
_ISSUE_STYLE: dict[IssueKind, tuple[str, str, str]] = {
    IssueKind.HIGH_MISSING: ("Missing values", "red", "block"),
    IssueKind.DUPLICATE_ROWS: ("Duplicates", "orange", "content_copy"),
    IssueKind.CONSTANT: ("Constant", "gray", "horizontal_rule"),
    IssueKind.POSSIBLE_ID: ("Possible id", "violet", "key"),
    IssueKind.HIGH_CORRELATION: ("Correlated", "blue", "link"),
}

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
    "Outliers": ["whisker_low", "whisker_high", "outliers", "outliers_pct"],
    "Special values": ["count", "zeros", "negatives", "infinite", "unparsed"],
    "Shape": ["skew", "kurtosis", "shape"],
}
_NUMERIC_DEFAULT_GROUPS = ["Position", "Outliers"]

# Height (px) of a table row and of its header, to size tables to their content.
_ROW_PX = 35
_HEADER_PX = 38
_CHART_HEIGHT = "240px"

# Labels, formats and explanations of every measure of the numeric table.
_NUMERIC_CONFIG: dict[str, Any] = {
    "column": st.column_config.TextColumn("Column", pinned=True),
    "count": st.column_config.NumberColumn("Values", help="Finite, non-missing values"),
    "min": st.column_config.NumberColumn("Min", format="%.4g"),
    "p5": st.column_config.NumberColumn("P5", format="%.4g", help="5% of the values are lower"),
    "q1": st.column_config.NumberColumn("Q1", format="%.4g", help="25% of the values are lower"),
    "median": st.column_config.NumberColumn("Median", format="%.4g"),
    "q3": st.column_config.NumberColumn("Q3", format="%.4g", help="75% of the values are lower"),
    "p95": st.column_config.NumberColumn("P95", format="%.4g", help="95% of the values are lower"),
    "max": st.column_config.NumberColumn("Max", format="%.4g"),
    "mean": st.column_config.NumberColumn("Mean", format="%.4g"),
    "std": st.column_config.NumberColumn("Std dev", format="%.4g"),
    "cv": st.column_config.NumberColumn(
        "CV", format="%.2f", help="Coefficient of variation: std dev / |mean|"
    ),
    "iqr": st.column_config.NumberColumn(
        "IQR", format="%.4g", help="Q3 - Q1: width of the middle half of the values"
    ),
    "whisker_low": st.column_config.NumberColumn(
        "Low (no outliers)",
        format="%.4g",
        help="Lowest value that is not an outlier (lower whisker of the boxplot)",
    ),
    "whisker_high": st.column_config.NumberColumn(
        "High (no outliers)",
        format="%.4g",
        help="Highest value that is not an outlier (upper whisker of the boxplot)",
    ),
    "outliers": st.column_config.NumberColumn(
        "Outliers", help=f"Values beyond {cols.TUKEY_K:g} times the IQR from the quartiles (Tukey)"
    ),
    "outliers_pct": st.column_config.NumberColumn("Outliers %", format="%.1f%%"),
    "zeros": st.column_config.NumberColumn("Zeros"),
    "negatives": st.column_config.NumberColumn("Negatives"),
    "infinite": st.column_config.NumberColumn("Infinite"),
    "unparsed": st.column_config.NumberColumn(
        "Not numbers",
        help='Values that were not numbers in the file ("cincuenta", "41 años"). They are '
        "counted as missing.",
    ),
    "skew": st.column_config.NumberColumn(
        "Skewness", format="%+.2f", help="0: symmetric. > 0: long tail to the right"
    ),
    "kurtosis": st.column_config.NumberColumn(
        "Kurtosis", format="%+.2f", help="> 0: heavier tails than a normal distribution"
    ),
    "shape": st.column_config.TextColumn("Shape"),
}


def profile_page() -> None:
    """Entry point registered in app.py."""
    dataset = state.get_dataset()
    dataset_id = state.get_dataset_id()
    if dataset is None or dataset_id is None:
        st.info("Load a file in the Load page to see its profile.")
        return

    _skip_selector(state.require_schema())
    # From here on, the dataset without the skipped columns.
    dataset_id, dataset, schema = _active()
    if not schema.columns:
        st.info("Every column is skipped. Remove some from the list to see their profile.")
        return
    _summary(dataset_id, dataset)
    # on_change="rerun": only the open tab runs (R-E1).
    issues, columns, missing, correlation = st.tabs(
        [_ISSUES_TAB, _COLUMNS_TAB, _MISSING_TAB, _CORRELATION_TAB],
        on_change="rerun",
        key=state.widget_key("profile_tab"),
    )
    if issues.open:
        with issues:
            _issues()
    if columns.open:
        with columns:
            _columns()
    if missing.open:
        with missing:
            _missing(dataset_id, dataset)
    if correlation.open:
        with correlation:
            _correlation()


# COMPONENTS


def _skip_selector(schema: Schema) -> None:
    """Columns to leave out of every tab, such as codes stored as numbers."""
    key = state.widget_key("skipped_columns")
    st.multiselect(
        "Skip columns",
        [c.name for c in schema.columns],
        key=key,
        on_change=lambda: state.set_skipped_columns(state.widget_value(key)),
        # Kept when the user visits another page and comes back.
        persist_state="session",
        placeholder="None: every column is profiled",
        help=(
            "Columns left out of every tab and of the numbers above. Useful for codes stored "
            "as numbers (phone numbers, postal codes): statistics on them mean nothing."
        ),
    )


def _summary(dataset_id: str, dataset: LoadedDataset) -> None:
    """Headline quality numbers, from the base pass every tab starts from."""
    base = cache.base_profile(dataset_id, dataset.df)
    columns = f"{dataset.n_cols} columns"
    if skipped := len(state.get_skipped_columns()):
        columns += f" ({skipped} skipped)"
    st.caption(f"{dataset.name} · {dataset.n_rows:,} rows · {columns}")
    missing, rows, duplicates, constant, memory = st.columns(5)
    missing.metric("Missing cells", f"{base.missing_cells_pct:.1f}%")
    rows.metric("Rows with missing", f"{base.rows_with_missing:,}")
    duplicates.metric("Duplicate rows", f"{base.duplicate_rows:,}")
    constant.metric("Constant columns", base.constant_columns)
    memory.metric("Memory", _format_bytes(base.memory_bytes))


# Issues


@st.fragment
def _issues() -> None:
    """Problems worth looking at before analysing the data, with a filter by kind."""
    dataset_id, dataset, schema = _active()
    df = dataset.df
    # Always Pearson, whatever the method chosen in Correlation, so the list does not change
    # when the user explores that tab. Same cache entry as Pearson there (R-P6).
    corr = cache.correlation(
        dataset_id,
        df,
        cols.columns_of(schema, cols.NUMERIC_TYPES),
        "pearson",
        _CFG.max_heatmap_columns,
        _CFG.sample_rows,
    )
    issues = detect_issues(
        df,
        cache.base_profile(dataset_id, df),
        schema,
        # Every pair, not only the top ones: none above the threshold may be left out.
        top_correlated_pairs(corr.matrix),
        high_missing_pct=_CFG.high_missing_pct,
        high_correlation=_CFG.high_correlation,
    )
    if not issues:
        st.success("No issues found.")
    else:
        # Legend and filter: one pill per kind found, with how many issues of that kind.
        counts: dict[IssueKind, int] = {}
        for issue in issues:
            counts[issue.kind] = counts.get(issue.kind, 0) + 1
        kinds = [kind for kind in _ISSUE_STYLE if kind in counts]
        selected = st.pills(
            "Filter issues",
            kinds,
            selection_mode="multi",
            format_func=lambda kind: (
                f":material/{_ISSUE_STYLE[kind][2]}: {_ISSUE_STYLE[kind][0]} ({counts[kind]})"
            ),
            label_visibility="collapsed",
            key=state.widget_key("issue_filter"),
        )
        # Nothing selected shows everything. Grouped by kind, in the order of the legend.
        with st.container(border=True):
            for kind in selected or kinds:
                for issue in (i for i in issues if i.kind == kind):
                    where = " · ".join(f"`{c}`" for c in issue.columns)
                    text = f"{where} — {issue.message}" if where else issue.message
                    st.markdown(f"{_badge(kind)} {text}")
    _about(_issues_help())


# Columns


@st.fragment
def _columns() -> None:
    """Overview of every column, or the table of one semantic type."""
    dataset_id, dataset, schema = _active()
    sizes = {view: len(cols.columns_of(schema, kinds)) for view, (kinds, _) in _VIEWS.items()}
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
    names = cols.columns_of(schema, _VIEWS[view][0])
    df = dataset.df

    if view == "Overview":
        _overview(dataset_id, dataset, schema)
    elif view == "Numeric":
        table = cache.numeric_stats(dataset_id, df, names, _CFG.sample_rows)
        _numeric_table(cols.with_unparsed(table, dataset.unparsed))
        _sample_note(dataset, "Percentiles, IQR, skewness and kurtosis are")
    elif view == "Categorical":
        _categorical_table(cache.categorical_stats(dataset_id, df, names))
    elif view == "Dates":
        _dates_table(cache.datetime_stats(dataset_id, df, names, dataset.unparsed, _now()))
    else:
        _text_table(cache.text_stats(dataset_id, df, names, schema, _CFG.sample_rows))
        _sample_note(dataset, "Lengths are")
    _about(_columns_help())


def _overview(dataset_id: str, dataset: LoadedDataset, schema: Schema) -> None:
    """One short row per column. Selecting a row opens the inspector of that column."""
    table = cache.column_overview(dataset_id, dataset.df, schema, _CFG.sample_rows)
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
        # A new key when the filter changes, so a selected row never points to another column.
        key=state.widget_key("overview_" + "-".join(sorted(k.value for k in picked))),
        height=_table_height(len(table)),
        column_order=[
            "column",
            "semantic_type",
            "missing_pct",
            "unique_pct",
            "distribution",
            "summary",
        ],
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
            "summary": st.column_config.TextColumn("Summary", width="large"),
        },
    )
    _sample_note(dataset, "Histograms and text lengths are")
    if not event.selection.rows:
        st.caption(":material/touch_app: Select a row to inspect that column.")
        return
    _inspector(dataset_id, dataset, schema, str(table.iloc[event.selection.rows[0]]["column"]))


def _inspector(dataset_id: str, dataset: LoadedDataset, schema: Schema, name: str) -> None:
    """Everything about one column, laid out for its semantic type."""
    df = dataset.df
    kind = schema.type_of(name)
    base = cache.base_profile(dataset_id, df).columns.set_index("column").loc[name]
    with st.container(border=True):
        st.markdown(f"#### `{name}`")
        st.caption(
            f"{kind.value} · pandas type `{df[name].dtype}` · {base['missing_pct']:.1f}% missing"
        )
        if base["missing"] == dataset.n_rows:
            st.info("The column has no values.")
            return
        if kind in cols.NUMERIC_TYPES:
            names = cols.columns_of(schema, cols.NUMERIC_TYPES)
            table = cache.numeric_stats(dataset_id, df, names, _CFG.sample_rows)
            stats = _row(cols.with_unparsed(table, dataset.unparsed), name)
            # Infinite values are not missing, but every measure ignores them.
            if not stats["count"]:
                st.info("The column has no finite values.")
                return
        detail = cache.column_detail(dataset_id, df, name, kind, schema, _CFG.sample_rows)
        if kind in cols.NUMERIC_TYPES:
            _inspect_numeric(name, stats, detail)
        elif kind in cols.CATEGORICAL_TYPES:
            names = cols.columns_of(schema, cols.CATEGORICAL_TYPES)
            _inspect_categorical(name, _row(cache.categorical_stats(dataset_id, df, names), name))
            _top_values(name, detail)
        elif kind in cols.DATE_TYPES:
            names = cols.columns_of(schema, cols.DATE_TYPES)
            table = cache.datetime_stats(dataset_id, df, names, dataset.unparsed, _now())
            _inspect_dates(name, _row(table, name), detail)
        else:
            names = cols.columns_of(schema, cols.TEXT_TYPES)
            table = cache.text_stats(dataset_id, df, names, schema, _CFG.sample_rows)
            _inspect_text(name, kind, _row(table, name), detail)
        if kind not in cols.CATEGORICAL_TYPES:
            _sample_note(dataset, "Charts are")


def _inspect_numeric(name: str, stats: pd.Series, detail: cols.ColumnDetail) -> None:
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
    _metric(skew, "Skewness", f"{stats['skew']:+.2f}", stats["shape"] or None)

    box, hist = st.columns(2)
    with box:
        st.markdown("**Boxplot**")
        caption = (
            f"Box: Q1 to Q3. Whiskers: {_num(stats['whisker_low'])} to "
            f"{_num(stats['whisker_high'])}. Dots: outliers."
        )
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
        )
        if "dataZoom" in option:
            caption += " Zoomed on the box: drag the slider to reach the farthest outliers."
        st.caption(caption)
        st_echarts(option, height="230px", key=f"profile_box_{name}")
    with hist:
        bars = detail.bars
        if bars.per_value:
            st.markdown("**Values**")
            st.caption("Few distinct values: one bar per value, outliers included.")
        else:
            st.markdown("**Histogram**")
            st.caption(
                "Between the whiskers: outliers are left out so they do not flatten the bars."
            )
        st_echarts(
            charts.histogram_options(
                list(bars.labels), list(bars.counts), show_labels=bars.per_value
            ),
            height="230px",
            key=f"profile_hist_{name}",
        )

    with st.expander("All measures"):
        rows = [
            (group, _NUMERIC_CONFIG[key]["label"], _format_stat(key, stats[key]))
            for group, keys in _NUMERIC_GROUPS.items()
            for key in keys
        ]
        st.dataframe(
            pd.DataFrame(rows, columns=["Group", "Measure", "Value"]),
            hide_index=True,
            height=_HEADER_PX + _ROW_PX * len(rows),
        )


def _inspect_categorical(name: str, stats: pd.Series) -> None:
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
    if detail.variants:
        listed = "\n".join(
            f"- {' / '.join(f'`{v!r}`' for v in group)}" for group in detail.variants[:5]
        )
        st.warning(f"Same value written in different ways:\n{listed}", icon=":material/spellcheck:")
    bars = detail.bars
    st.markdown("**Most common values**")
    if detail.other_values:
        st.caption(
            f"Not drawn: {detail.other_values:,} other values in {detail.other_rows:,} rows."
        )
    st_echarts(
        charts.hbar_options(list(bars.labels), list(bars.counts)),
        height=charts.hbar_height(len(bars.counts)),
        key=f"profile_top_{name}",
    )


def _inspect_dates(name: str, stats: pd.Series, detail: cols.ColumnDetail) -> None:
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
    st_echarts(
        charts.histogram_options(list(bars.labels), list(bars.counts)),
        height=_CHART_HEIGHT,
        key=f"profile_dates_{name}",
    )


def _inspect_text(
    name: str, kind: SemanticType, stats: pd.Series, detail: cols.ColumnDetail
) -> None:
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
    _metric(blank, "Blank", stats["blank"], help_text='Empty or only spaces: "" or "   "')
    _metric(unique, "Distinct", f"{stats['unique_pct']:.1f}%")
    _metric(
        repeated,
        "Repeated values",
        f"{stats['repeated']:,}",
        help_text="Rows whose value appears more than once",
    )

    chart, examples = st.columns([3, 2])
    with chart:
        bars = detail.bars
        st.markdown("**Length of the values**")
        st_echarts(
            charts.histogram_options(list(bars.labels), list(bars.counts)),
            height=_CHART_HEIGHT,
            key=f"profile_lengths_{name}",
        )
    with examples:
        st.markdown("**Examples**")
        st.dataframe(detail.examples, hide_index=True)


def _numeric_table(table: pd.DataFrame) -> None:
    groups = st.pills(
        "Measures",
        list(_NUMERIC_GROUPS),
        selection_mode="multi",
        default=_NUMERIC_DEFAULT_GROUPS,
        key=state.widget_key("numeric_groups"),
    )
    order = ["column"] + [
        key for group in _NUMERIC_GROUPS if group in groups for key in _NUMERIC_GROUPS[group]
    ]
    st.dataframe(
        table,
        hide_index=True,
        column_order=order,
        column_config=_NUMERIC_CONFIG,
        height=_table_height(len(table)),
    )


def _categorical_table(table: pd.DataFrame) -> None:
    st.dataframe(
        table,
        hide_index=True,
        height=_table_height(len(table)),
        column_config={
            "column": st.column_config.TextColumn("Column", pinned=True),
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
            "variant_example": st.column_config.TextColumn("Example of variant", width="medium"),
        },
    )


def _dates_table(table: pd.DataFrame) -> None:
    st.dataframe(
        table,
        hide_index=True,
        height=_table_height(len(table)),
        column_config={
            "column": st.column_config.TextColumn("Column", pinned=True),
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


def _text_table(table: pd.DataFrame) -> None:
    st.dataframe(
        table,
        hide_index=True,
        height=_table_height(len(table)),
        column_config={
            "column": st.column_config.TextColumn("Column", pinned=True),
            "semantic_type": st.column_config.TextColumn("Type"),
            "len_min": st.column_config.NumberColumn("Min length"),
            "len_mean": st.column_config.NumberColumn("Avg length", format="%.1f"),
            "len_max": st.column_config.NumberColumn("Max length"),
            "blank": st.column_config.NumberColumn(
                "Blank", help='Empty or only spaces: "" or "   "'
            ),
            "unique_pct": st.column_config.NumberColumn("Distinct", format="%.1f%%"),
            "repeated": st.column_config.NumberColumn(
                "Repeated", help="Rows whose value appears more than once"
            ),
            "example": st.column_config.TextColumn("Example", width="large"),
        },
    )


# Missing values


def _missing(dataset_id: str, dataset: LoadedDataset) -> None:
    """Where the missing values are: by column, by row and a sample of incomplete rows."""
    base = cache.base_profile(dataset_id, dataset.df)
    by_column = missing_by_column(base)
    if by_column.empty:
        st.success("No missing values.")
        _about(_MISSING_HELP)
        return

    by_col, by_row = st.columns(2)
    with by_col:
        st.markdown("**By column**")
        # Sorted by missing values: the cap keeps the most incomplete columns.
        shown = by_column.head(_CFG.missing_chart_columns)
        caption = "% of missing values"
        if len(shown) < len(by_column):
            caption += f". The {len(shown)} most incomplete of {len(by_column)} columns."
        st.caption(caption)
        st_echarts(
            charts.bar_options(
                [str(c) for c in shown["column"]],
                [round(float(v), 2) for v in shown["missing_pct"]],
                unit="%",
            ),
            key="profile_missing_by_column",
        )
    with by_row:
        st.markdown("**Per row**")
        st.caption("Rows by number of missing values")
        per_row = base.missing_per_row
        st_echarts(
            charts.bar_options(
                [str(n) for n in per_row["missing_in_row"]],
                [int(n) for n in per_row["rows"]],
            ),
            key="profile_missing_per_row",
        )

    n = _CFG.missing_rows_preview
    with st.expander(f"Rows with missing values (first {n})"):
        st.dataframe(cache.rows_with_missing(dataset_id, dataset.df, n))
    _about(_MISSING_HELP)


# Correlation


@st.fragment
def _correlation() -> None:
    """Correlation heatmap of the numeric columns and the strongest pairs."""
    dataset_id, dataset, schema = _active()
    label = st.radio("Method", list(_METHODS), horizontal=True, key=state.widget_key("corr_method"))
    result = cache.correlation(
        dataset_id,
        dataset.df,
        cols.columns_of(schema, cols.NUMERIC_TYPES),
        _METHODS[label],
        _CFG.max_heatmap_columns,
        _CFG.sample_rows,
    )
    if result.matrix.empty:
        st.info("At least two numeric columns with more than one value are needed.")
        _about(_CORRELATION_HELP)
        return
    if result.numeric_columns > len(result.matrix):
        st.caption(
            f"Showing {len(result.matrix)} of {result.numeric_columns} numeric columns "
            "(the ones with fewest missing values)."
        )
    if result.rows_used < dataset.n_rows:
        st.caption(
            f":material/info: {label} computed on {result.rows_used:,} of "
            f"{dataset.n_rows:,} rows (random sample)."
        )

    heatmap, pairs = st.columns([3, 2])
    with heatmap:
        st_echarts(
            charts.heatmap_options(result.matrix),
            height=charts.heatmap_height(len(result.matrix)),
            key="profile_correlation_heatmap",
        )
    with pairs:
        st.markdown("**Most correlated pairs**")
        st.dataframe(
            top_correlated_pairs(result.matrix, _CFG.top_pairs),
            hide_index=True,
            column_config={
                "column_a": st.column_config.TextColumn("Column A"),
                "column_b": st.column_config.TextColumn("Column B"),
                "correlation": st.column_config.NumberColumn("Correlation", format="%+.2f"),
            },
        )
    _about(_CORRELATION_HELP)


# HELP TEXTS


def _issues_help() -> str:
    return f"""
Problems worth checking before analysing the data. Each one has a colored badge; click the
buttons above the list to show only some kinds.

- {_badge(IssueKind.HIGH_MISSING)} More than {_CFG.high_missing_pct:g}% of the values of a
  column are missing. The column carries little information, and any result computed from it
  rests on a small part of the rows.
- {_badge(IssueKind.DUPLICATE_ROWS)} Some rows are exact copies of others. They can be a
  loading error and they inflate counts, averages and the weight of those rows in a model.
- {_badge(IssueKind.CONSTANT)} The column has a single value. It cannot explain any
  difference between rows, so it can usually be dropped.
- {_badge(IssueKind.POSSIBLE_ID)} (Almost) every value is different, at least
  {IDENTIFIER_MIN_DISTINCT:.0%}: integers, or codes without spaces. Duplicated rows in a dirty
  file repeat some ids, so a few repeats are allowed. It is probably a key such as a user or
  order id: useful to join tables, not to compute statistics or train a model, where it would
  only add noise. Dates and free text are not flagged, even if all distinct.
- {_badge(IssueKind.HIGH_CORRELATION)} Two numeric columns have a Pearson correlation above
  {_CFG.high_correlation:g} in absolute value. They probably measure the same thing, so one of
  them may be redundant.
"""


def _columns_help() -> str:
    return f"""
**Overview**: one row per column. *Type* is what the values mean (numeric, categorical, date,
boolean, text, id), guessed from how they are stored and what they look like. *Distinct* is
the share of different values among the non-missing ones: low means the values repeat
(categories), 100% means every value is different. *Distribution* is a small chart of the
values. Select a row to see that column in detail.

**Per-type tables**: only the measures that make sense for each type.

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


_MISSING_HELP = """
- **By column**: which columns are incomplete and how much.
- **Per row**: whether the missing values are spread out or concentrated in a few rows.
- **Rows with missing values**: a sample, to look for a pattern (same source, same dates...).

Most statistics ignore missing values, but most models cannot use them. Before training a
model, decide what to do with them: drop the rows or columns, or fill them in (with the
median, the most frequent value, or a model). Each choice changes the results, so it is worth
understanding first why the values are missing.
"""

_CORRELATION_HELP = """
Measures how much two numeric columns move together, from -1 to +1.

- **+1**: when one grows, the other always grows. **-1**: when one grows, the other always
  falls. **0**: no relation of that kind.
- **Pearson** measures straight-line relations and is sensitive to outliers. **Spearman**
  works on the order of the values: it also catches relations that are not straight lines
  (as long as they always go in the same direction) and is more robust to outliers. On large
  datasets it is computed on a random sample.
- Only numeric columns are compared. Identifiers (such as ``user_id``) are left out: they
  often grow with the row order and would look correlated with anything that does too.
- Correlation is not causation: two columns can move together because a third one drives
  both, or by chance.
- A correlation near 0 does not mean the columns are unrelated: a U-shaped relation gives
  about 0 with both methods. Plot them in Visualization to check.
- Strongly correlated inputs make some models (such as linear regression) unstable and hard
  to interpret. Keeping only one of each pair is a common fix.
"""


# HELPERS


def _active() -> tuple[str, LoadedDataset, Schema]:
    """Id, dataset and schema of the active dataset without the skipped columns.

    The id is a cache key of its own (``cache.view_id``), since the data differs from the full
    dataset. Fragments call it instead of receiving the data as arguments, so a fragment rerun
    always sees the current dataset (R-E2).
    """
    dataset_id, dataset, schema = (
        state.require_dataset_id(),
        state.require_dataset(),
        state.require_schema(),
    )
    skipped = state.get_skipped_columns()
    if not skipped:
        return dataset_id, dataset, schema
    df, schema = drop_columns(dataset.df, schema, skipped)
    return cache.view_id(dataset_id, skipped), dataclasses.replace(dataset, df=df), schema


def _about(text: str) -> None:
    """Collapsible explanation of a tab, for users new to data profiling."""
    with st.expander("About this tab", icon=":material/help:"):
        st.markdown(text)


def _sample_note(dataset: LoadedDataset, what: str) -> None:
    """Say that ``what`` was computed on a sample, if the dataset is larger than the sample."""
    n = _CFG.sample_rows
    if dataset.n_rows > n:
        st.caption(
            f":material/info: {what} computed on a random sample of up to {n:,} values "
            f"per column ({dataset.n_rows:,} rows in the dataset)."
        )


def _now() -> pd.Timestamp:
    """Current local time, to the hour: dates after it are "in the future".

    Rounded so the cached date table is reused within the hour and recomputed after it.
    """
    return pd.Timestamp.now().floor("h")


def _row(table: pd.DataFrame, name: str) -> pd.Series:
    """Row of a per-type table for column ``name``."""
    return table.set_index("column").loc[name]


def _table_height(n_rows: int) -> int:
    """Height that fits ``n_rows`` rows, up to ``ProfileConfig.table_max_height``."""
    return min(_HEADER_PX + _ROW_PX * n_rows, _CFG.table_max_height)


def _metric(
    where: Any, label: str, value: Any, note: str | None = None, help_text: str | None = None
) -> None:
    """A metric whose ``note`` is a neutral remark under the value, not a rise or a fall."""
    where.metric(label, value, note, delta_color="off", delta_arrow="off", help=help_text)


def _num(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:,.4g}"


def _format_stat(key: str, value: Any) -> str:
    if key == "shape":
        return str(value) or "—"
    if pd.isna(value):
        return "—"
    if key in ("count", "outliers", "zeros", "negatives", "infinite", "unparsed"):
        return f"{int(value):,}"
    if key == "outliers_pct":
        return f"{value:.1f}%"
    if key in ("skew", "kurtosis"):
        return f"{value:+.2f}"
    return f"{value:,.4g}"


def _badge(kind: IssueKind) -> str:
    """Markdown for the colored badge of an issue kind."""
    label, color, icon = _ISSUE_STYLE[kind]
    return f":{color}-badge[:material/{icon}: {label}]"


def _format_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
