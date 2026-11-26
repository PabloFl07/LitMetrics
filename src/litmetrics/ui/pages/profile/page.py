"""Profile page: the entry point, the headline numbers, and the Issues, Missing values and
Correlation tabs. The Columns tab is in ``columns_tab``.
"""

import importlib.metadata
from datetime import datetime

import streamlit as st

from litmetrics.config import ProfileConfig
from litmetrics.core import columns as cols
from litmetrics.core import report
from litmetrics.core.profile import (
    BaseProfile,
    CorrelationMethod,
    CorrelationResult,
    Issue,
    IssueKind,
    detect_issues,
    missing_by_column,
    top_correlated_pairs,
)
from litmetrics.core.sampling import SAMPLE_SEED
from litmetrics.core.schema import IDENTIFIER_MIN_DISTINCT
from litmetrics.ui import cache, charts, echarts, reading, skip, state, style, tables, text, theme
from litmetrics.ui.pages.profile import columns_tab, shared

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

# Height of the charts of the Missing values tab.
_MISSING_CHART_HEIGHT = "300px"


def profile_page() -> None:
    """Entry point registered in app.py."""
    style.page_header("Profile", "An overall description of the data.")
    dataset = state.get_dataset()
    dataset_id = state.get_dataset_id()
    if dataset is None or dataset_id is None:
        st.info("Load a file in the Load page to see its profile.")
        return

    skip.skip_selector(state.require_schema())
    # From here on, the dataset without the skipped columns.
    active = shared.active()
    if not active.schema.columns:
        st.info("Every column is skipped. Remove some from the list to see their profile.")
        return
    _summary(active)
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
            columns_tab.tab()
    if missing.open:
        with missing:
            _missing(active)
    if correlation.open:
        with correlation:
            _correlation()


# Components


def _summary(active: shared.Active) -> None:
    """Headline quality numbers, from the base pass every tab starts from."""
    dataset = active.view
    base = cache.base_profile(active.view_id, dataset.df)
    columns = f"{dataset.n_cols} columns"
    if skipped := len(state.get_skipped_columns()):
        columns += f" ({skipped} skipped)"
    st.caption(f"{text.md(dataset.name)} · {dataset.n_rows:,} rows · {columns}")
    # Keyed container: styles.css turns these metrics into cards.
    with st.container(key="headline"):
        missing, rows, duplicates, constant, memory = st.columns(5)
    for where, label, value, icon in (
        (missing, "Missing cells", f"{base.missing_cells_pct:.1f}%", "block"),
        (rows, "Rows with missing", f"{base.rows_with_missing:,}", "table_rows"),
        (duplicates, "Duplicate rows", f"{base.duplicate_rows:,}", "content_copy"),
        (constant, "Constant columns", base.constant_columns, "horizontal_rule"),
        (memory, "Memory", text.format_bytes(base.memory_bytes), "memory"),
    ):
        where.metric(label, value, icon=f":material/{icon}:", border=True)


# Issues


@st.fragment
def _issues() -> None:
    """Problems worth looking at before analysing the data, with a filter by kind."""
    active = shared.active()
    schema = active.schema
    # Always Pearson, whatever the method chosen in Correlation, so the list does not change
    # when the user explores that tab. Same cache entry as Pearson there (R-P6).
    corr = cache.correlation(
        active.dataset_id,
        active.dataset.df,
        schema.names_of(cols.NUMERIC_TYPES),
        "pearson",
        _CFG.max_heatmap_columns,
        _CFG.sample_rows,
    )
    # Duplicate rows depend on the columns left: the base pass of the view.
    base = cache.base_profile(active.view_id, active.view.df)
    issues = detect_issues(
        base,
        schema,
        # Every pair, not only the top ones: none above the threshold may be left out.
        top_correlated_pairs(corr.matrix),
        high_missing_pct=_CFG.high_missing_pct,
        high_correlation=_CFG.high_correlation,
    )
    _report_button(active, base, issues, corr)
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
        with style.card("issues"):
            for kind in selected or kinds:
                for issue in (i for i in issues if i.kind == kind):
                    where = " · ".join(text.code(c) for c in issue.columns)
                    line = f"{where} — {issue.message}" if where else issue.message
                    st.markdown(f"{_badge(kind)} {line}")
    shared.about(_issues_help())


def _report_button(
    active: shared.Active, base: BaseProfile, issues: list[Issue], corr: CorrelationResult
) -> None:
    """Download of the issues as a Markdown report, with why each one matters and what to do.

    The report is written only on click, in another thread (no rerun), from results already
    computed: every value it needs is taken here, none is read from the session there.
    """
    view, skipped = active.view, state.get_skipped_columns()
    part = reading.active_part()
    settings = report.ReportSettings(
        high_missing_pct=_CFG.high_missing_pct,
        high_correlation=_CFG.high_correlation,
        correlation_method="pearson",
        sample_rows=corr.rows_used if corr.rows_used < view.n_rows else None,
        seed=SAMPLE_SEED,
        correlation_columns=len(corr.matrix),
        numeric_columns=corr.numeric_columns,
    )

    def write() -> str:
        return report.issues_report(
            view,
            base,
            active.schema,
            issues,
            settings,
            memory=text.format_bytes(base.memory_bytes),
            generated_at=datetime.now(),
            version=importlib.metadata.version("litmetrics"),
            part=part,
            skipped=skipped,
        )

    st.download_button(
        "Download report",
        data=write,
        file_name=report.report_file_name(view.name, part),
        mime=report.MIME,
        # Downloading changes nothing in the app: no rerun.
        on_click="ignore",
        icon=":material/download:",
        help="The issues as a Markdown file, with why each one matters and what to do about "
        "it, and a summary of the dataset and its columns.",
    )


# Missing values


def _missing(active: shared.Active) -> None:
    """Where the missing values are: by column, by row and a sample of incomplete rows."""
    dataset_id, dataset = active.view_id, active.view
    base = cache.base_profile(dataset_id, dataset.df)
    by_column = missing_by_column(base)
    if by_column.empty:
        st.success("No missing values.")
        shared.about(_MISSING_HELP)
        return

    by_col, by_row = st.columns(2)
    with by_col, style.card("missing_by_column"):
        st.markdown("**By column**")
        # Sorted by missing values: the cap keeps the most incomplete columns.
        shown = by_column.head(_CFG.missing_chart_columns)
        caption = "% of missing values"
        if len(shown) < len(by_column):
            caption += f". The {len(shown)} most incomplete of {len(by_column)} columns."
        st.caption(caption)
        echarts.show(
            charts.bar_options(
                [str(c) for c in shown["column"]],
                [round(float(v), 2) for v in shown["missing_pct"]],
                theme.chart_colors(),
                unit="%",
            ),
            height=_MISSING_CHART_HEIGHT,
            key="profile_missing_by_column",
        )
    with by_row, style.card("missing_per_row"):
        st.markdown("**Per row**")
        st.caption("Rows by number of missing values")
        per_row = base.missing_per_row
        echarts.show(
            charts.bar_options(
                [str(n) for n in per_row["missing_in_row"]],
                [int(n) for n in per_row["rows"]],
                theme.chart_colors(),
            ),
            height=_MISSING_CHART_HEIGHT,
            key="profile_missing_per_row",
        )

    n = _CFG.missing_rows_preview
    with st.expander(f"Rows with missing values (first {n})"):
        st.dataframe(tables.for_display(cache.rows_with_missing(dataset_id, dataset.df, n)))
    shared.about(_MISSING_HELP)


# Correlation


@st.fragment
def _correlation() -> None:
    """Correlation heatmap of the numeric columns and the strongest pairs."""
    active = shared.active()
    dataset = active.dataset
    label = st.radio("Method", list(_METHODS), horizontal=True, key=state.widget_key("corr_method"))
    result = cache.correlation(
        active.dataset_id,
        dataset.df,
        active.schema.names_of(cols.NUMERIC_TYPES),
        _METHODS[label],
        _CFG.max_heatmap_columns,
        _CFG.sample_rows,
    )
    if result.matrix.empty:
        st.info("At least two numeric columns with more than one value are needed.")
        shared.about(_CORRELATION_HELP)
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

    if len(result.matrix) >= _CFG.heatmap_full_width_columns:
        # Many columns: full width, so the numbers fit in the cells; the pairs go below.
        heatmap, pairs = st.container(), st.container()
    else:
        heatmap, pairs = st.columns([3, 2])
    with heatmap, style.card("correlation_heatmap"):
        echarts.show(
            charts.heatmap_options(result.matrix, theme.chart_colors()),
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
    shared.about(_CORRELATION_HELP)


# Help texts


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


# Helpers


def _badge(kind: IssueKind) -> str:
    """Markdown for the colored badge of an issue kind."""
    label, color, icon = _ISSUE_STYLE[kind]
    return f":{color}-badge[:material/{icon}: {label}]"
