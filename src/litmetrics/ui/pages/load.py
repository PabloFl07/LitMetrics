"""Load page: upload a file, read it and show what was loaded.

A new upload is processed once. Its format is chosen by its extension (``core/io/formats.py``),
the read options of that format are detected (for a CSV: separator, encoding, header, decimal
mark), the file is read through a cache keyed by its content and those options, its
columns are converted to the dtype of their semantic type (dates, categories), and the result
becomes the active dataset in the session state. Uploading the same file again keeps the
current one.

On every other rerun nothing is read, not even the bytes of the upload: the page only draws the
active dataset. It stays there after the file is removed from the uploader or the user moves to
another page. The page shows a summary (rows, columns, format, how the file was read) and a
preview of the first rows. The profile of each column is in the Profile page.

Every dataset read is kept in memory (``cache.cached_datasets``, shared by all sessions). The
page lists them, to open one again without its file, remove one, or clear the whole cache;
the sidebar (``ui/sidebar.py``) only switches between them, from any page.

If a file cannot be read, the error is shown and the previous dataset, if any, stays active.
The error is remembered for that upload, so later reruns show it without reading the file
again.
"""

import streamlit as st
from streamlit.typing import UploadedFile

from litmetrics.config import LoadConfig
from litmetrics.core.io import formats
from litmetrics.core.io.base import LoadedDataset, ReadError, Source, compute_dataset_id
from litmetrics.core.io.library import LibraryEntry
from litmetrics.ui import cache, state, style, tables


def load_page() -> None:
    """Entry point registered in app.py."""
    names = " or ".join(fmt.name for fmt in formats.FORMATS)
    style.page_header("Load data", f"Upload a {names} file. How to read it is detected for you.")
    uploaded = st.file_uploader(
        "Upload a file", type=list(formats.extensions()), label_visibility="collapsed"
    )
    if uploaded is not None:
        _load(uploaded)
    _part()
    _cached_datasets()

    # The dataset lives in the session state, so it stays after the file is removed from the
    # uploader or the user goes to another page.
    dataset = state.get_dataset()
    if dataset is None:
        st.info("Upload a file to start.")
        return
    _summary(dataset)
    _preview(dataset)


# Data sourcing


def _load(uploaded: UploadedFile) -> None:
    """Read ``uploaded`` and make it the active dataset. Show an error if it cannot be read.

    Steps: skip if this upload was already processed, or show its error if it already failed.
    A file with several datasets is only listed and kept: ``_part`` reads them. Any other is
    read here.
    """
    source_id = uploaded.file_id
    # Checked before ``getvalue``, which copies the whole file on every call.
    if not state.source_changed(source_id) or _is_active_multipart(source_id):
        return
    if (error := state.get_read_error(source_id)) is not None:
        st.error(error)
        return
    source = _source(uploaded)
    try:
        parts = formats.list_parts(source)
    except ReadError as e:
        state.set_read_error(source_id, str(e))
        st.error(str(e))
        return
    if parts is not None:
        state.set_multipart(source, parts)
    elif _read(source, None, source_id):
        # A file of one dataset replaces the file with several: its selector goes away.
        state.clear_multipart()


def _part() -> None:
    """Draw the selector of the active file with several datasets, and read the chosen one.

    The file and its datasets come from the session state, not from the uploader, so the
    selector stays and keeps its choice after visiting another page (the uploader forgets
    its file there). Each dataset of the file is a read of its own, processed once.
    """
    if (multipart := state.get_multipart()) is None:
        return
    source, parts = multipart
    state.restore_part_widget()
    part = st.selectbox(parts.label, parts.names, key=state.PART_WIDGET)
    state.set_part(part)
    read_id = f"{source.source_id}/{part}"
    if not state.source_changed(read_id):
        return
    if (error := state.get_read_error(read_id)) is not None:
        st.error(error)
        return
    _read(source, part, read_id)


def _read(source: Source, part: str | None, read_id: str) -> bool:
    """Read ``part`` of ``source`` (``None``: its only dataset) and make it the active dataset.

    Guess how to read it, read it and convert its columns to their semantic types (cached by
    content), and store it. The schema only replaces the current one when the content is new.
    On a ``ReadError`` the message is shown and remembered for ``read_id``.

    Returns:
        Whether the dataset was read.
    """
    try:
        options = formats.detect_options(source, part)
        dataset_id = compute_dataset_id(source.data, options)
        dataset, schema = cache.load_dataset(dataset_id, source, options, part)
    except ReadError as e:
        # Exceptions are not cached: without this, every rerun would read the file again.
        state.set_read_error(read_id, str(e))
        st.error(str(e))
        return False
    if state.set_dataset(dataset, dataset_id, read_id):
        state.set_schema(schema)
    return True


def _is_active_multipart(source_id: str) -> bool:
    """Whether upload ``source_id`` is the file with several datasets already kept."""
    multipart = state.get_multipart()
    return multipart is not None and multipart[0].source_id == source_id


def _source(uploaded: UploadedFile) -> Source:
    """The upload as a ``Source``. Copies the whole file: call it only when it will be read."""
    return Source(uploaded.getvalue(), uploaded.name, uploaded.file_id)


# Datasets kept in memory


def _cached_datasets() -> None:
    """List the datasets kept in memory, with buttons to open or remove each one."""
    entries = cache.cached_datasets()
    if not entries:
        return
    active_id = state.get_dataset_id()
    # A fixed label: one that changes (a count) makes the expander close after a removal.
    with st.expander("Datasets in memory", icon=":material/history:"):
        st.caption(
            f"{len(entries)} of {LoadConfig.cache_max_datasets} kept, most recent first. "
            "Open one to use it again without its file."
        )
        for entry in entries:
            _cached_row(entry, entry.dataset_id == active_id)
        # Opened from the script, not from ``on_click``: a dialog needs the rerun to draw it.
        if st.button(
            "Clear cache",
            icon=":material/delete_sweep:",
            key="clear_cache",
            help="Remove every dataset and every computed result from memory.",
        ):
            _confirm_clear()


def _cached_row(entry: LibraryEntry, active: bool) -> None:
    """One dataset kept in memory: name, size, when it was read, and its buttons."""
    dataset = entry.dataset
    with st.container(
        horizontal=True, vertical_alignment="center", key=f"cached_{entry.dataset_id}"
    ):
        with st.container(gap=None):
            badge = " :green-badge[Active]" if active else ""
            st.markdown(f"**{entry.label}**{badge}")
            st.caption(
                f"{dataset.format} · {dataset.n_rows:,} rows · {dataset.n_cols} columns · "
                f"read at {entry.loaded_at:%H:%M}"
            )
        st.button(
            "Open",
            icon=":material/open_in_new:",
            key=f"open_{entry.dataset_id}",
            disabled=active,
            on_click=_open_cached,
            args=(entry.dataset_id,),
        )
        st.button(
            "Remove",
            icon=":material/delete:",
            type="tertiary",
            key=f"remove_{entry.dataset_id}",
            on_click=_remove_cached,
            args=(entry.dataset_id,),
        )


def _open_cached(dataset_id: str) -> None:
    """Make a dataset kept in memory the active one (callback of its Open button)."""
    entry = cache.cached_dataset(dataset_id)
    if entry is None:
        st.toast("That dataset is no longer in memory. Upload the file again.")
        return
    state.switch_dataset(entry.dataset, entry.dataset_id, entry.schema)


def _remove_cached(dataset_id: str) -> None:
    """Drop a dataset from memory; if it is the active one, the session is left without one."""
    cache.forget_dataset(dataset_id)
    if state.get_dataset_id() == dataset_id:
        state.clear_dataset()


@st.dialog("Clear cache", icon=":material/delete_sweep:")
def _confirm_clear() -> None:
    """Ask before removing every dataset and result from memory."""
    st.markdown(
        "Every dataset in memory and every computed result is removed, also for other open "
        "tabs. To use a dataset again you will have to upload its file."
    )
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("Cancel", key="clear_cache_cancel"):
            st.rerun()
        # Cleared in the callback, which runs on any rerun; ``st.rerun`` only closes the dialog.
        if st.button(
            "Clear cache", type="primary", key="clear_cache_confirm", on_click=_clear_cache
        ):
            st.rerun()


def _clear_cache() -> None:
    """Drop every dataset and result from memory; the session is left without a dataset."""
    cache.clear_all()
    state.clear_dataset()


# Components


def _summary(dataset: LoadedDataset) -> None:
    """Rows, columns and format of the dataset, and how the file was read."""
    # Keyed container: styles.css turns these metrics into cards.
    with st.container(key="headline"):
        rows, columns, file_format = st.columns(3)
    rows.metric("Rows", f"{dataset.n_rows:,}", icon=":material/table_rows:", border=True)
    columns.metric("Columns", dataset.n_cols, icon=":material/view_column:", border=True)
    file_format.metric("Format", dataset.format, icon=":material/description:", border=True)
    if dataset.details:
        # Written by the loader of each format: the page does not know what they mean.
        details = ", ".join(f"{label} **{value}**" for label, value in dataset.details.items())
        st.caption(f":material/tune: Read with {details}.")


def _preview(dataset: LoadedDataset) -> None:
    """First rows of the dataset. Only those rows are sent to the browser (R-E3)."""
    st.markdown("**Preview**")
    st.caption(f"First {LoadConfig.preview_rows} rows")
    rows = tables.for_display(dataset.df.head(LoadConfig.preview_rows))
    st.dataframe(rows, height=LoadConfig.preview_height)
