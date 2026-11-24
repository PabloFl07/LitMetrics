"""Visualization page: build a chart of the active dataset. The person picks the kind, the
columns, the rows and the look; the page never suggests charts.

From top to bottom: "Skip columns" (shared with Profile), the chart kinds the dataset has
columns for, and the editor (Data, Filters and Style tabs, Reset, Save / load) next to the
chart, its notes and the data behind it. Filters are a list the person adds to, one card each,
on any column with two values or more, drawn in the chart or not.

How it stays fast (docs/performance.md, section Visualization):
- Data and look are apart (R-V1): ``core/visualize.py`` aggregates what ``DataSpec`` says,
  cached in ``cache.chart_data`` (R-V3, filters included: R-V8); ``ui/chart_style.py`` builds
  the option from it and ``StyleSpec`` on every run, without touching pandas.
- The editor and the chart are one ``st.fragment`` (R-V4), drawn with one chart key (R-V5).
- Only aggregated or sampled data reaches the browser (R-V2, limits in ``VisualizeConfig``).

Streamlit drops the state of a widget that is not drawn, so the options of each chart kind are
kept with ``state.kept_widget`` and come back when the kind is drawn again.
"""

import datetime as dt
import json
import time
from collections.abc import Collection, Iterable
from dataclasses import dataclass, replace
from typing import Any

import pandas as pd
import streamlit as st

from litmetrics.config import VisualizeConfig
from litmetrics.core import visualize as viz
from litmetrics.core.schema import drop_columns
from litmetrics.ui import cache, echarts, skip, state, style, text, theme
from litmetrics.ui import chart_style as look

_KIND_LABELS = {
    viz.ChartKind.BAR: ":material/bar_chart: Bar",
    viz.ChartKind.LINE: ":material/show_chart: Line",
    viz.ChartKind.HISTOGRAM: ":material/equalizer: Histogram",
    viz.ChartKind.SCATTER: ":material/scatter_plot: Scatter",
    viz.ChartKind.BOXPLOT: ":material/candlestick_chart: Boxplot",
    viz.ChartKind.PIE: ":material/donut_large: Pie",
    viz.ChartKind.HEATMAP: ":material/grid_on: Heatmap",
}
# When "Value labels" is not drawn although it is on: more marks would overlap (limits in
# ``ui/chart_style.py``).
_LABELS_HELP = {
    viz.ChartKind.LINE: "Drawn when each line has 40 points or fewer.",
    viz.ChartKind.HISTOGRAM: "Drawn with 30 bins or fewer.",
    viz.ChartKind.HEATMAP: "Up to 120 cells they are always drawn; this adds them to larger ones.",
}
# Ends of the "Bins" / "Intervals" slider, and lowest "Categories shown".
_BINS = (5, 100)
_MIN_TOP_K = 3
_SORT_LABELS = {"data": "Default", "desc": "Value ↓", "asc": "Value ↑", "label": "A-Z, 1-9"}

_ALL = frozenset(viz.ChartKind)
_K = viz.ChartKind


@dataclass(frozen=True)
class _StyleWidget:
    """Widget of one ``StyleSpec`` field.

    Attributes:
        key: Its session key. Not ``state.widget_key``: the look does not depend on the
            columns, so it stays when another dataset is loaded.
        kinds: Chart kinds that draw it (``_reset`` clears it for them).
    """

    key: str
    kinds: frozenset[viz.ChartKind]


# Style widgets by ``StyleSpec`` field. Kinds of one series (no "Split by") take a color.
_STYLE_WIDGETS: dict[str, _StyleWidget] = {
    "title": _StyleWidget("viz_title", _ALL),
    "palette_name": _StyleWidget("viz_palette", _ALL),
    "color": _StyleWidget("viz_color", _ALL - {_K.PIE, _K.HEATMAP}),
    "color_each": _StyleWidget("viz_color_each", frozenset({_K.BOXPLOT})),
    "horizontal": _StyleWidget("viz_horizontal", frozenset({_K.BAR, _K.BOXPLOT})),
    "stack": _StyleWidget("viz_stack", frozenset({_K.BAR, _K.LINE, _K.HISTOGRAM})),
    "sort": _StyleWidget("viz_sort", frozenset({_K.BAR})),
    "smooth": _StyleWidget("viz_smooth", frozenset({_K.LINE})),
    "area": _StyleWidget("viz_area", frozenset({_K.LINE})),
    "trend": _StyleWidget("viz_trend", frozenset({_K.SCATTER})),
    "donut": _StyleWidget("viz_donut", frozenset({_K.PIE})),
    # Scatter points and boxes have no single value to write: no toggle, rather than one that
    # does nothing.
    "labels": _StyleWidget("viz_labels", _ALL - {_K.SCATTER, _K.BOXPLOT}),
    "log_scale": _StyleWidget("viz_log", _ALL - {_K.PIE}),
    "legend": _StyleWidget("viz_legend", _ALL),
    "size": _StyleWidget("viz_size", _ALL),
}
_STYLE_DEFAULTS = look.StyleSpec()


def _key(name: str) -> str:
    """Key of a widget that depends on the columns: cleared with the dataset."""
    return state.widget_key(f"viz_{name}")


def _role_key(kind: viz.ChartKind, field: str) -> str:
    """Each kind keeps its own columns, so switching kinds and back keeps the choice."""
    return _key(f"{kind}_{field}")


def _filter_keys(filter_id: int) -> tuple[str, str]:
    """Keys of the column and of the values of filter ``filter_id``. Filters are shared by
    every kind: the rows are the same for all.
    """
    return _key(f"filter_{filter_id}_column"), _key(f"filter_{filter_id}_value")


def _kind_keys(kind: viz.ChartKind) -> list[str]:
    """Keys of every widget drawn for ``kind``: its columns and data options, and the style
    widgets it draws (also those other kinds share). Filters are not here (``_drop_filters``).
    """
    data = [_role_key(kind, role.field) for role in viz.ROLES[kind]]
    data += [_role_key(kind, "agg"), *(_key(f"{kind}_{n}") for n in ("bins", "unit", "top_k"))]
    data += [_key(f"{kind}_trim")]
    looks = [w.key for w in _STYLE_WIDGETS.values() if kind in w.kinds]
    return data + looks


def _reset(kind: viz.ChartKind) -> None:
    """Callback of "Reset": the options of ``kind`` back to where the page starts, no filter."""
    state.forget_widgets(_kind_keys(kind))
    _drop_filters()


# Page


def visualize_page() -> None:
    """Entry point registered in app.py."""
    style.page_header(
        "Visualization",
        "Build a chart: pick its kind, its columns, the rows it uses and its look.",
    )
    if state.get_dataset() is None or state.get_dataset_id() is None:
        st.info("Load a file in the Load page to draw charts.", icon=":material/upload_file:")
        return
    _workspace()


@st.fragment
def _workspace() -> None:
    """Chart picker, editor and chart. A fragment: editing reruns only this (R-V4)."""
    dataset = state.require_dataset()
    dataset_id = state.require_dataset_id()
    schema = state.require_schema()
    skip.skip_selector(schema)
    skipped = state.get_skipped_columns()
    # Only the offered columns depend on what is skipped: the hints are computed on the view
    # without those columns, under its own id (``cache.view_id``), as in Profile. The chart
    # data only reads the columns of the chart, never skipped ones: it keeps the id of the
    # whole dataset, so skipping a column does not throw its cache away.
    view, view_schema = drop_columns(dataset.df, schema, skipped)
    hints = cache.column_hints(cache.view_id(dataset_id, skipped), view, view_schema)
    kinds = viz.available_kinds(hints)
    if not kinds:
        what = "the columns not skipped have no" if skipped else "the dataset has no"
        st.warning(f"Nothing to draw: {what} numbers, dates or categories.")
        return
    kind = _kind_picker(kinds)
    editor, canvas = st.columns([1, 2.7], gap="medium")
    with editor:
        # Filled once the spec and the style are known, but drawn above the tabs.
        actions = st.container(horizontal=True, vertical_alignment="center")
        notices = st.container()
        data_tab, filter_tab, style_tab = st.tabs(
            [":material/dataset: Data", ":material/filter_alt: Filters", ":material/palette: Style"]
        )
        with data_tab:
            spec = _data_controls(kind, hints, skipped)
        with filter_tab:
            spec = replace(spec, filters=_filters(hints))
        with style_tab:
            chart_style = _style_controls(spec, hints)
        with actions:
            _config_actions(spec, chart_style, hints, tuple(dataset.df.columns))
        with notices:
            _config_messages()
    with canvas:
        _chart(dataset_id, dataset.df, spec, chart_style)


def _kind_picker(kinds: tuple[viz.ChartKind, ...]) -> viz.ChartKind:
    """Chart kinds the dataset has columns for."""
    key = _key("kind")
    state.restore_widget(key)
    if state.get_widget(key) not in kinds:
        state.set_widget(key, kinds[0])
    kind = st.pills(
        "Chart",
        kinds,
        format_func=_KIND_LABELS.__getitem__,
        key=key,
        required=True,
        label_visibility="collapsed",
    )
    state.remember_widget(key, kind)
    return viz.ChartKind(kind)


def _swap_repeated(kind: viz.ChartKind, field: str, old: str | None) -> None:
    """Callback of a column selector: a chart uses each column once (X and Y of a scatter,
    rows and columns of a heatmap, category and split of a bar chart).

    If the new column is already in another role, that role takes the old column of this
    one: picking Y's column as X swaps the axes.
    """
    new = state.get_widget(_role_key(kind, field))
    if new is None:
        return
    for role in viz.ROLES[kind]:
        key = _role_key(kind, role.field)
        if role.field != field and state.get_widget(key) == new:
            # None in a required role is refilled with a free column on the next run.
            state.set_widget(key, old)


def _free_column(
    preferred: str | None, options: list[Any], used: set[str], required: bool
) -> str | None:
    """``preferred`` if it fits and is not used yet; else the first free option of a required
    role, or None for an optional one.
    """
    if preferred in options and preferred not in used:
        return preferred
    if not required:
        return None
    return next((o for o in options if o not in used), None)


def _data_controls(
    kind: viz.ChartKind, hints: viz.ColumnHints, skipped: tuple[str, ...]
) -> viz.DataSpec:
    """Columns of each role of ``kind`` and the options that change the data (no filters).

    ``skipped`` are the columns of "Skip columns": not in ``hints``, only listed at the end.
    """
    default = viz.default_spec(kind, hints) or viz.DataSpec(kind)
    picked: dict[str, Any] = {}
    for role in viz.ROLES[kind]:
        # Dataset order: the person knows where their columns are; no ranking (README).
        columns = hints.columns_for(role.accepts)
        options = list(columns) if role.required else [None, *columns]
        key = _role_key(kind, role.field)
        used = {v for v in picked.values() if v is not None}
        state.restore_widget(key)
        value = state.get_widget(key) if state.has_widget(key) else getattr(default, role.field)
        # A column that no longer fits, or that an earlier role already uses (never left by
        # the swap of ``_swap_repeated``, but a dataset or a loaded configuration can).
        if value not in options or value in used:
            value = _free_column(getattr(default, role.field), options, used, role.required)
        if not state.has_widget(key) or state.get_widget(key) != value:
            state.set_widget(key, value)
        picked[role.field] = st.selectbox(
            role.label,
            options,
            key=key,
            format_func=str,
            # A None value shows the placeholder instead of an option.
            placeholder="None",
            help=role.help or None,
            on_change=_swap_repeated,
            # The current column, read when the user changes it: the callback runs after.
            args=(kind, role.field, value),
        )
        state.remember_widget(key, picked[role.field])
    spec = viz.DataSpec(kind, **picked)
    aggs = viz.AGGREGATIONS.get(kind, viz.DEFAULT_AGGREGATIONS)
    if spec.value is not None and aggs:
        key = _role_key(kind, "agg")
        state.restore_widget(key)
        if state.get_widget(key) not in aggs:
            state.set_widget(key, default.agg if default.agg in aggs else aggs[0])
        if len(aggs) > 1:
            st.segmented_control(
                "Aggregation", aggs, key=key, format_func=str.capitalize, required=True
            )
        state.remember_widget(key, state.get_widget(key))
        spec = replace(spec, agg=viz.Agg(state.get_widget(key)))
    numeric_x = spec.x in hints.numeric
    if kind == _K.HISTOGRAM or (kind in (_K.BAR, _K.LINE) and numeric_x):
        bins = state.kept_widget(
            st.slider,
            "Bins" if kind == _K.HISTOGRAM else "Intervals",
            *_BINS,
            key=_key(f"{kind}_bins"),
            default=VisualizeConfig.histogram_bins
            if kind == _K.HISTOGRAM
            else VisualizeConfig.x_bins,
            help=None
            if kind == _K.HISTOGRAM
            else "Intervals of the same width the x axis is cut in.",
        )
        spec = replace(spec, bins=bins)
    if kind == _K.LINE and not numeric_x:
        unit = state.kept_widget(
            st.selectbox,
            "Time unit",
            list(viz.TimeUnit),
            key=_key(f"{kind}_unit"),
            format_func=str.capitalize,
        )
        spec = replace(spec, time_unit=viz.TimeUnit(unit))
    if kind in (_K.PIE, _K.BOXPLOT, _K.HEATMAP) or (kind == _K.BAR and not numeric_x):
        most = (
            VisualizeConfig.max_slices if kind == _K.PIE else VisualizeConfig.max_shown_categories
        )
        top = state.kept_widget(
            st.slider,
            "Categories shown",
            _MIN_TOP_K,
            most,
            key=_key(f"{kind}_top_k"),
            default=min(default.top_k, most),
            help="The most frequent; the rest are grouped in “Other”.",
        )
        spec = replace(spec, top_k=top)
    if kind in (_K.HISTOGRAM, _K.LINE) or (kind == _K.BAR and numeric_x):
        trim = state.kept_widget(
            st.toggle,
            "Trim extreme values",
            key=_key(f"{kind}_trim"),
            default=True,
            help="Leave out a few extreme values (placeholder dates, typos) that squash the chart.",
        )
        spec = replace(spec, trim=trim)
    left_out = [*hints.left_out, *((name, "skipped") for name in skipped)]
    if left_out:
        names = ", ".join(f"{text.md(name)} ({why})" for name, why in left_out)
        st.caption(f"Not offered: {names}.")
    return spec


# Filters


def _filters(hints: viz.ColumnHints) -> tuple[viz.RowFilter, ...]:
    """Rows the chart uses: those that pass every filter of the list, one card each.

    Any column with two values or more can filter, also one the chart does not draw (an id,
    free text, a row number): ``hints.filters``. They go in ``DataSpec`` and so in the cache
    key; the masks are built inside ``chart_data`` (R-V8). A filter without a column, or left
    at its full range, is no filter, so it does not split the cache.
    """
    ids = state.get_chart_filters()
    if not ids:
        st.caption(
            "Every row is drawn. Add a filter to keep only some rows: by any column, "
            "also one the chart does not show."
        )
    found: list[viz.RowFilter] = []
    for filter_id in ids:
        found += _filter_card(filter_id, hints)
    with st.container(horizontal=True):
        st.button(
            "Add filter",
            icon=":material/add:",
            key="viz_filter_add",
            on_click=_add_filter,
            disabled=not hints.filters,
        )
        if len(ids) > 1:
            st.button(
                "Remove all",
                icon=":material/delete_sweep:",
                key="viz_filter_clear",
                on_click=_drop_filters,
                type="tertiary",
            )
    return tuple(found)


def _add_filter() -> None:
    """Callback of "Add filter": a new card, without a column yet."""
    ids = state.get_chart_filters()
    new = max(ids, default=-1) + 1
    state.forget_widgets(_filter_keys(new))
    state.set_chart_filters((*ids, new))


def _remove_filter(filter_id: int) -> None:
    """Callback of the close button of a card: the filter and its widgets are gone."""
    state.forget_widgets(_filter_keys(filter_id))
    state.set_chart_filters(i for i in state.get_chart_filters() if i != filter_id)


def _drop_filters() -> None:
    """Every filter and its widgets are gone. From a callback."""
    for filter_id in state.get_chart_filters():
        state.forget_widgets(_filter_keys(filter_id))
    state.set_chart_filters(())


def _filter_card(filter_id: int, hints: viz.ColumnHints) -> list[viz.RowFilter]:
    """One filter in a card of its own: its column, a button to remove it and, once a column
    is picked, the control of its type (dates, values or range).
    """
    column_key, value_key = _filter_keys(filter_id)
    with st.container(border=True, key=f"viz_filter_{filter_id}"):
        with st.container(horizontal=True, vertical_alignment="bottom"):
            column = _filter_column(hints.filter_columns, column_key, value_key)
            st.button(
                "",
                icon=":material/close:",
                key=f"viz_filter_remove_{filter_id}",
                help="Remove this filter",
                on_click=_remove_filter,
                args=(filter_id,),
                type="tertiary",
            )
        if column is None:
            return []
        match hints.filter_type(column):
            case "dates":
                return _date_filter(hints, column, value_key)
            case "values":
                return _values_filter(hints, column, value_key)
            case "range":
                return _range_filter(hints, column, value_key)
    return []


def _filter_column(columns: tuple[str, ...], key: str, bound_key: str) -> str | None:
    """Selector of the column a filter applies to (None: no filter yet). Changing it drops the
    values picked for the previous column (``bound_key``).
    """
    state.restore_widget(key)
    if state.get_widget(key) not in (None, *columns):
        state.set_widget(key, None)
    column = st.selectbox(
        "Column",
        [None, *columns],
        key=key,
        format_func=lambda v: "Pick a column" if v is None else str(v),
        placeholder="Pick a column",
        on_change=state.forget_widgets,
        args=([bound_key],),
    )
    state.remember_widget(key, column)
    return column


def _date_filter(hints: viz.ColumnHints, column: str, range_key: str) -> list[viz.RowFilter]:
    first, last = hints.dates_of(column)
    picked = state.kept_widget(
        st.date_input,
        "Between",
        key=range_key,
        default=(first, last),
        min_value=first,
        max_value=last,
    )
    # While the second day is being picked the widget returns one date.
    start = picked[0] if picked else first
    end = picked[1] if len(picked) > 1 else last
    if (start, end) == (first, last):
        return []
    return [
        viz.DateFilter(
            column,
            start.isoformat() if start > first else None,
            end.isoformat() if end < last else None,
        )
    ]


def _values_filter(hints: viz.ColumnHints, column: str, values_key: str) -> list[viz.RowFilter]:
    options = hints.values_of(column)
    # Free text and ids only list their most frequent values: others can be typed in.
    partial = column in hints.partial
    state.restore_widget(values_key)
    if state.has_widget(values_key) and not partial:
        # Values remembered for this column that it no longer has would make the widget raise.
        state.set_widget(values_key, [v for v in state.get_widget(values_key) if v in options])
    values = state.kept_widget(
        st.multiselect,
        "Keep only",
        options,
        key=values_key,
        placeholder="Pick or type values" if partial else "Every value",
        accept_new_options=partial,
        help=f"Lists the {len(options):,} most frequent values; type any other one."
        if partial
        else None,
    )
    # None or every value picked is no filter (as a full range): rows are not left out
    # and the cache is not split.
    if not values or (not partial and set(values) == set(options)):
        return []
    return [viz.ValuesFilter(column, tuple(str(v) for v in values))]


def _range_filter(hints: viz.ColumnHints, column: str, range_key: str) -> list[viz.RowFilter]:
    low, high = hints.range_of(column)
    picked_low, picked_high = state.kept_widget(
        st.slider, "Between", low, high, key=range_key, default=(low, high)
    )
    if (picked_low, picked_high) == (low, high):
        return []
    return [
        viz.RangeFilter(
            column,
            picked_low if picked_low > low else None,
            picked_high if picked_high < high else None,
        )
    ]


# Style


def _draws(field: str, kind: viz.ChartKind) -> bool:
    """Whether the style widget of ``field`` is drawn for ``kind``."""
    return kind in _STYLE_WIDGETS[field].kinds


def _auto_title(spec: viz.DataSpec, hints: viz.ColumnHints) -> str:
    """Title of a chart whose title is left empty."""
    return viz.describe(spec, over_time=spec.x in hints.datetime)


def _style_controls(spec: viz.DataSpec, hints: viz.ColumnHints) -> look.StyleSpec:
    """Look of the chart. Nothing here changes the data, so no widget touches pandas (R-V1).

    Only what a person decides: the heatmap scale, axis names, legend position and animation
    follow from the data (``ui/chart_style.py``).
    """
    kind = spec.kind
    one_series = spec.group is None
    keys = {field: w.key for field, w in _STYLE_WIDGETS.items()}
    auto_title = _auto_title(spec, hints)
    title = state.kept_widget(st.text_input, "Title", key=keys["title"], placeholder=auto_title)
    named = look.palettes(theme.chart_theme().palette, theme.chart_colors())
    if state.get_widget(keys["palette_name"]) not in (None, *named):
        state.set_widget(keys["palette_name"], _STYLE_DEFAULTS.palette_name)
    palette_name = state.kept_widget(
        st.selectbox,
        "Colors",
        list(named),
        key=keys["palette_name"],
        default=_STYLE_DEFAULTS.palette_name,
        # A color picked for the previous palette would hide the new one.
        on_change=state.forget_widgets,
        args=([keys["color"]],),
    )
    palette = named[palette_name]
    options: dict[str, Any] = {}
    # A boxplot is one series: its boxes share a color unless each takes one of the palette.
    if _draws("color_each", kind) and spec.x is not None:
        options["color_each"] = state.kept_widget(
            st.toggle,
            "One color per box",
            key=keys["color_each"],
            default=_STYLE_DEFAULTS.color_each,
            help="Each box takes the next color of the palette.",
        )
    color = None
    if _draws("color", kind) and one_series and not options.get("color_each"):
        picked = state.kept_widget(st.color_picker, "Color", key=keys["color"], default=palette[0])
        if picked != palette[0]:
            color = picked
            palette = (picked, *palette[1:])

    def toggle(field: str, label: str, *, when: bool = True, help: str | None = None) -> None:
        if _draws(field, kind) and when:
            default = getattr(_STYLE_DEFAULTS, field)
            options[field] = state.kept_widget(
                st.toggle, label, key=keys[field], default=default, help=help
            )

    toggle("horizontal", "Horizontal")
    toggle("stack", "Stack series", when=not one_series)
    if _draws("sort", kind):
        options["sort"] = state.kept_widget(
            st.segmented_control,
            "Order",
            list(_SORT_LABELS),
            format_func=_SORT_LABELS.__getitem__,
            key=keys["sort"],
            default=_STYLE_DEFAULTS.sort,
            required=True,
        )
    toggle("smooth", "Smooth lines")
    toggle("area", "Fill area")
    toggle("trend", "Trend line")
    toggle("donut", "Donut")
    label = "Percent on slices" if kind == _K.PIE else "Value labels"
    toggle("labels", label, help=_LABELS_HELP.get(kind))
    toggle("log_scale", "Log scale", help="Zeros and negative values are not drawn.")
    # Only when there is more than one thing to name: series, a trend line or slices.
    toggle("legend", "Legend", when=not one_series or kind in (_K.PIE, _K.SCATTER))
    options["size"] = state.kept_widget(
        st.segmented_control,
        "Size",
        list(look.HEIGHTS),
        format_func=str.capitalize,
        key=keys["size"],
        default=_STYLE_DEFAULTS.size,
        required=True,
    )
    return look.StyleSpec(
        title=title or auto_title,
        palette=palette,
        palette_name=palette_name,
        color=color,
        **options,
    )


# Configuration: reset, save and load


def _config_actions(
    spec: viz.DataSpec,
    chart_style: look.StyleSpec,
    hints: viz.ColumnHints,
    columns: tuple[str, ...],
) -> None:
    """Reset the options of this kind; save or load the whole chart as JSON."""
    st.button(
        "Reset",
        icon=":material/restart_alt:",
        key="viz_reset",
        on_click=_reset,
        args=(spec.kind,),
        help=f"Back to the starting options of the {spec.kind} chart, filters included.",
    )
    # Stretched: next to "Reset" in a narrow editor, a content-width label was cut short.
    with st.popover("Save / load", icon=":material/save:", width="stretch"):
        name = state.kept_widget(
            st.text_input,
            "File name",
            key="viz_config_name",
            placeholder=f"chart_{spec.kind}",
            help="Saved as JSON; “.json” is added if missing.",
        )
        st.download_button(
            "Save configuration",
            _config_json(spec, chart_style, _auto_title(spec, hints)),
            file_name=viz.config_file_name(name, spec.kind),
            mime="application/json",
            icon=":material/download:",
            on_click="ignore",
        )
        key = f"viz_config_file_{state.chart_config_uploads()}"
        st.file_uploader(
            "Load a configuration",
            type=["json"],
            key=key,
            on_change=_load_config,
            args=(key, hints, columns),
        )


def _config_json(spec: viz.DataSpec, chart_style: look.StyleSpec, auto_title: str) -> str:
    """The chart as JSON: its ``DataSpec`` and its ``StyleSpec`` (palette by name)."""
    config = {
        "version": 1,
        "data": viz.spec_to_dict(spec),
        "style": look.style_to_dict(chart_style, auto_title=auto_title),
    }
    return json.dumps(config, indent=2, ensure_ascii=False)


def _load_config(key: str, hints: viz.ColumnHints, columns: Collection[str]) -> None:
    """Callback of the configuration uploader: check the file and set every widget from it.

    Columns that do not exist or do not fit are left out with a warning (``fit_to_dataset``);
    an unreadable file changes nothing.
    """
    file = state.get_widget(key)
    # A new uploader next run: this one would keep showing the file.
    state.count_chart_config_upload()
    if file is None:
        return
    try:
        raw = json.loads(file.getvalue())
        if not isinstance(raw, dict):
            raise ValueError("it is not a chart configuration")
        spec = viz.spec_from_dict(raw.get("data"))
        chart_style = look.style_from_dict(raw.get("style", {}))
    except ValueError as e:
        state.set_chart_config_messages([("error", f"Could not load {file.name}: {e}")])
        return
    spec, problems = viz.fit_to_dataset(spec, columns, hints, state.get_skipped_columns())
    if spec.kind not in viz.available_kinds(hints):
        state.set_chart_config_messages(
            [("error", f"This dataset has no columns for a {spec.kind} chart.")]
        )
        return
    _apply_config(spec, chart_style, hints)
    messages = [("warning", p) for p in problems] or [("success", f"Loaded {file.name}.")]
    state.set_chart_config_messages(messages)


def _apply_config(spec: viz.DataSpec, chart_style: look.StyleSpec, hints: viz.ColumnHints) -> None:
    """Set the widgets of ``spec.kind`` to ``spec`` and ``chart_style``. From a callback.

    Starts from a reset kind, so whatever the file does not say takes its default. Values are
    brought inside the bounds of the widgets (bins, dates, ranges), which raise otherwise.
    """
    kind = spec.kind
    state.forget_widgets(_kind_keys(kind))
    state.put_widget(_key("kind"), kind)
    for role in viz.ROLES[kind]:
        value = getattr(spec, role.field)
        # A required role without a column is filled by the editor.
        if value is not None or not role.required:
            state.put_widget(_role_key(kind, role.field), value)
    state.put_widget(_role_key(kind, "agg"), spec.agg)
    state.put_widget(_key(f"{kind}_bins"), min(max(spec.bins, _BINS[0]), _BINS[1]))
    state.put_widget(_key(f"{kind}_unit"), spec.time_unit)
    most = VisualizeConfig.max_slices if kind == _K.PIE else VisualizeConfig.max_shown_categories
    state.put_widget(_key(f"{kind}_top_k"), min(max(spec.top_k, _MIN_TOP_K), most))
    state.put_widget(_key(f"{kind}_trim"), spec.trim)
    _apply_filters(spec.filters, hints)
    for field, widget in _STYLE_WIDGETS.items():
        value = getattr(chart_style, field)
        if field == "color" and value is None:
            continue
        state.put_widget(widget.key, value)


def _apply_filters(filters: Iterable[viz.RowFilter], hints: viz.ColumnHints) -> None:
    """Replace the filter cards with one per filter of ``filters``, clamped to what the column
    holds (``fit_to_dataset`` already left out those this dataset cannot use).
    """
    _drop_filters()
    ids = []
    for filter_id, f in enumerate(filters):
        column_key, value_key = _filter_keys(filter_id)
        ids.append(filter_id)
        state.put_widget(column_key, f.column)
        match f:
            case viz.DateFilter(column=column, start=start, end=end):
                first, last = hints.dates_of(column)
                begin = max(first, _iso(start, first))
                finish = min(last, _iso(end, last))
                state.put_widget(value_key, (min(begin, finish), max(begin, finish)))
            case viz.ValuesFilter(column=column, values=values):
                options = set(hints.values_of(column))
                partial = column in hints.partial
                state.put_widget(value_key, [v for v in values if partial or v in options])
            case viz.RangeFilter(column=column, low=low, high=high):
                lo, hi = hints.range_of(column)
                begin = lo if low is None else min(max(low, lo), hi)
                finish = hi if high is None else min(max(high, lo), hi)
                state.put_widget(value_key, (min(begin, finish), max(begin, finish)))
    state.set_chart_filters(ids)


def _iso(day: str | None, default: dt.date) -> dt.date:
    """``day`` (ISO) as a date, or ``default`` when it is None."""
    return default if day is None else dt.date.fromisoformat(day)


def _config_messages() -> None:
    """What the last configuration load said: shown once."""
    show = {"error": st.error, "warning": st.warning, "success": st.success}
    for level, message in state.pop_chart_config_messages():
        show[level](message)


# Chart


def _chart(
    dataset_id: str, df: pd.DataFrame, spec: viz.DataSpec, chart_style: look.StyleSpec
) -> None:
    """Draw the chart, say what was cut, and show the data it was drawn from."""
    start = time.perf_counter()
    try:
        data = cache.chart_data(dataset_id, df, spec)
    except ValueError as e:
        st.warning(str(e))
        return
    data_ms = (time.perf_counter() - start) * 1000
    if not data.series:
        st.info("Nothing to draw: no row is left with a value in every column chosen.")
        if data.notes:
            st.caption(" ".join(f":material/info: {n}" for n in data.notes))
        return
    start = time.perf_counter()
    option = look.build_option(data, chart_style, theme.chart_colors())
    option_ms = (time.perf_counter() - start) * 1000
    # One key for every chart (R-V5): changing parameters updates it instead of remounting it.
    echarts.show(option, key="viz_chart", height=f"{chart_style.height}px")
    if data.notes:
        st.caption(" ".join(f":material/info: {n}" for n in data.notes))
    # The expander reports whether it is open (``on_change="rerun"``) and only builds its
    # content then. Collapsed, Streamlit still sends and renders what is inside: a 10,000 point
    # scatter was a 390 KB CSV and its table on every click of any control.
    table_box = st.expander(
        "Data behind the chart", icon=":material/table:", key="viz_data_open", on_change="rerun"
    )
    if table_box.open:
        with table_box:
            _data_table(dataset_id, df, spec, data, data_ms, option_ms)


def _data_table(
    dataset_id: str,
    df: pd.DataFrame,
    spec: viz.DataSpec,
    data: viz.ChartData,
    data_ms: float,
    option_ms: float,
) -> None:
    """The rows the chart was drawn from, and a CSV built only when it is downloaded. With
    filters, also the same chart data without them: the table stays the filtered one.
    """
    table = viz.to_frame(data)
    if VisualizeConfig.show_timings:
        st.caption(
            f"{data.elements:,} elements sent · data {data_ms:.0f} ms (cached after the first "
            f"time) · option {option_ms:.1f} ms"
        )
    st.dataframe(table, height=260, hide_index=True)
    stem = f"{spec.kind}_{spec.x or spec.value}"
    with st.container(horizontal=True):
        st.download_button(
            "Download CSV",
            # A callable runs only on click, in its own thread: no CSV is built on every rerun.
            lambda: table.to_csv(index=False).encode(),
            file_name=f"{stem}.csv",
            mime="text/csv",
            icon=":material/download:",
            on_click="ignore",
        )
        if spec.filters:
            unfiltered = replace(spec, filters=())
            st.download_button(
                "Download unfiltered",
                # Aggregated only on click (cached like any chart): every row, as if the
                # filters were removed, while the table and the chart keep them.
                lambda: (
                    viz.to_frame(cache.chart_data(dataset_id, df, unfiltered))
                    .to_csv(index=False)
                    .encode()
                ),
                file_name=f"{stem}_unfiltered.csv",
                mime="text/csv",
                icon=":material/filter_alt_off:",
                on_click="ignore",
                help="The same chart data without the filters: every row is counted.",
            )
