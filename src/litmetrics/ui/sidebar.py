"""Sidebar shown on every page: controls that are not tied to one page.

For now it holds the selector of the active dataset, which lists the datasets kept in memory
(``cache.cached_datasets``), so the user can switch between them from any page without
uploading the file again. Removing them is done in the Load page.

When the active dataset is one of the datasets of a file with several (a table of a
database), a second selector below lists the others. Opening such a dataset from memory opens
its file again (``reading.open_kept``), so the selector comes back with it. Picking one only
records the choice: ``reading.read_chosen_part``, run by app.py before the page, reads it.

Below them, an Export button opens a dialog that downloads the active dataset in another
format (``core/io/export.py``). The file is only written when the user clicks the download
button: ``st.download_button`` gets a function, not the bytes, so no rerun converts the
dataset. Above the button, a warning lists what the file will not hold as in the original
(``export.export_warnings``) and, when values will be written empty, a report lists each one
with its row (``export.unparsed_report``), to preview and download. Right above the button,
for a slow format and a large dataset, how long the file takes to write
(``export.slow_export``): a warning, or an error when the browser may give up waiting.
"""

import streamlit as st

from litmetrics.config import ExportConfig
from litmetrics.core.io import export
from litmetrics.core.io.base import LoadedDataset
from litmetrics.core.io.library import LibraryEntry
from litmetrics.ui import cache, reading, state, text


def draw() -> None:
    """Draw the sidebar. Called by app.py after the page, so it shows what the page loaded."""
    entries = cache.cached_datasets()
    active_id = state.get_dataset_id()
    if not entries and active_id is None:
        # Nothing to choose: no sidebar at all.
        return
    with st.sidebar:
        _dataset_switcher(entries, active_id)
        _part_switcher()
        _export()


def _dataset_switcher(entries: tuple[LibraryEntry, ...], active_id: str | None) -> None:
    """Selector of the active dataset among the ones kept in memory."""
    labels = {entry.dataset_id: entry.label for entry in entries}
    ids = list(labels)
    if active_id is not None and active_id not in labels:
        # Dropped from memory (by another session, or as the oldest) but still open here.
        dataset = state.require_dataset()
        labels[active_id] = dataset.name
        ids.insert(0, active_id)
    state.sync_switcher()
    st.selectbox(
        "Dataset",
        ids,
        format_func=labels.__getitem__,
        key=state.SWITCHER_WIDGET,
        on_change=_on_switch,
        placeholder="Choose a dataset",
        help="Datasets read in this app, kept in memory. Manage them in the Load page.",
    )


def _on_switch() -> None:
    """Make the dataset picked in the selector the active one, if it is still in memory."""
    dataset_id = state.widget_value(state.SWITCHER_WIDGET)
    if dataset_id is None or dataset_id == state.get_dataset_id():
        return
    entry = cache.cached_dataset(dataset_id)
    if entry is None:
        st.toast("That dataset is no longer in memory. Upload the file again.")
        return
    reading.open_kept(entry)


def _part_switcher() -> None:
    """Selector of the dataset of the active file with several, if the active one is of it."""
    if (multipart := state.get_multipart()) is None:
        return
    parts = multipart[1]
    if (active := reading.active_part()) is None:
        # The active dataset is of another file: this one's list would not be about it.
        return
    state.sync_part_switcher(active)
    st.selectbox(
        parts.label,
        parts.names,
        # The description (its size) is written by the format: shown as is.
        format_func=lambda name: _part_label(name, parts.descriptions.get(name)),
        key=state.PART_SWITCHER_WIDGET,
        on_change=_on_part_switch,
    )
    if (error := reading.part_error()) is not None:
        st.error(error)


def _part_label(name: str, description: str | None) -> str:
    """Option of the selector of tables: its name and, when the format gives one, its size."""
    return name if not description else f"{name} — {description}"


def _on_part_switch() -> None:
    """Record the dataset picked; app.py reads it before drawing the page."""
    name = state.widget_value(state.PART_SWITCHER_WIDGET)
    if name is not None and name != reading.active_part():
        state.choose_part(name)


def _export() -> None:
    """Button that opens the Export dialog, when a dataset is active."""
    if state.get_dataset() is None:
        return
    if st.button("Export", icon=":material/download:", width="stretch"):
        _export_dialog()


# A dialog is a fragment: changing the format reruns only the dialog, not the page behind it.
@st.dialog("Export dataset", width="medium", icon=":material/download:")
def _export_dialog() -> None:
    """Download the active dataset as a file of another format, warned of what it loses."""
    if (dataset := state.get_dataset()) is None:
        st.info("Load a file in the Load page to export it.")
        return
    names = [fmt.name for fmt in export.EXPORT_FORMATS]
    # Offer another format than the one read first: same format is rarely what is wanted.
    name = st.segmented_control(
        "Format",
        names,
        default=next((n for n in names if n != dataset.format), names[0]),
        required=True,
    )
    file_name = export.export_file_name(dataset.name, reading.active_part(), name)
    warnings = export.export_warnings(dataset, file_name)
    if warnings:
        # Messages quote column and file names: escaped, so they show as written.
        st.warning("\n".join(f"- {text.md(message)}" for message in warnings))
    _problem_report(dataset, file_name)
    _slow_export(dataset, name)
    df = dataset.df
    st.download_button(
        "Download",
        # Called only on click, in another thread: no rerun converts the dataset.
        data=lambda: export.export(df, name),
        file_name=file_name,
        mime=export.export_format(name).mime,
        # Downloading changes nothing in the app: no rerun.
        on_click="ignore",
        icon=":material/download:",
        type="primary",
        width="stretch",
    )


def _slow_export(dataset: LoadedDataset, name: str) -> None:
    """How long the file takes to write, when it is long; an error when it may not finish."""
    slow = export.slow_export(
        dataset, name, ExportConfig.slow_seconds, ExportConfig.download_timeout_seconds
    )
    if slow is None:
        return
    show = st.error if slow.may_fail else st.warning
    show(text.md(slow.message), icon=":material/schedule:")


def _problem_report(dataset: LoadedDataset, file_name: str) -> None:
    """Values the export writes empty, with their row: a preview and a CSV to download."""
    report = export.unparsed_report(dataset)
    if report.empty:
        return
    rows = export.affected_rows(report)
    label = f"Problem report: {rows:,} {'row' if rows == 1 else 'rows'} affected"
    with st.expander(label, icon=":material/report:"):
        preview = ExportConfig.report_preview_rows
        shown = "" if len(report) <= preview else f" The first {preview:,} are shown."
        st.caption(
            f"{len(report):,} values could not be read and will be empty in the file."
            f" Row 1 is the first row of data, below the header.{shown}"
        )
        # Capped: the report can have as many lines as bad values (R-E3).
        st.dataframe(
            report.head(preview),
            hide_index=True,
            width="stretch",
            column_config={
                "row": st.column_config.NumberColumn("Row", format="%d"),
                "column": "Column",
                "value": st.column_config.TextColumn("Value in the file"),
                "expected": "Expected",
            },
        )
        st.download_button(
            "Download report",
            data=lambda: export.write_report(report),
            file_name=export.report_file_name(file_name),
            mime="text/csv",
            on_click="ignore",
            icon=":material/table_view:",
            width="stretch",
        )
