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
the sidebar (``ui/sidebar.py``) only switches between them, from any page. For a file with
several datasets (the tables of a database), the list of its datasets goes above the preview;
the sidebar has a selector of them too.

If a file cannot be read, the error is shown and the previous dataset, if any, stays active.
The error is remembered for that upload, so later reruns show it without reading the file
again.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import streamlit as st
from streamlit.typing import UploadedFile

from litmetrics.config import LoadConfig
from litmetrics.core.io import formats
from litmetrics.core.io.base import LoadedDataset, ReadError, Source
from litmetrics.core.io.library import LibraryEntry
from litmetrics.ui import cache, reading, state, style, tables, text


def load_page() -> None:
    """Entry point registered in app.py."""
    names = " or ".join(fmt.name for fmt in formats.FORMATS)
    style.page_header("Load data", f"Upload a {names} file. How to read it is detected for you.")
    uploaded = st.file_uploader(
        "Upload a file", type=list(formats.extensions()), label_visibility="collapsed"
    )
    if uploaded is not None:
        _load(uploaded)
    # After the upload: a database just uploaded starts with its first table.
    reading.read_chosen_part()
    # The list is drawn after the reads, so a dataset read on this rerun is already in it.
    _cached_datasets()

    # The dataset lives in the session state, so it stays after the file is removed from the
    # uploader or the user goes to another page.
    dataset = state.get_dataset()
    if dataset is None:
        if not _parts():
            st.info("Upload a file to start.")
        return
    _summary(dataset)
    _parts()
    _preview(dataset)


# Data sourcing


def _load(uploaded: UploadedFile) -> None:
    """Read ``uploaded`` and make it the active dataset. Show an error if it cannot be read.

    Steps: skip if this upload was already processed, or show its error if it already failed.
    A file with several datasets is only listed and kept: ``reading.read_chosen_part`` reads
    them. Any other is read here.
    """
    source_id = uploaded.file_id
    # Checked before ``getvalue``, which copies the whole file on every call.
    if not state.upload_changed(source_id):
        if (error := state.get_read_error(source_id)) is not None:
            st.error(text.md(error))
        return
    state.mark_upload(source_id)
    source = _source(uploaded)
    try:
        parts = formats.list_parts(source)
    except ReadError as e:
        state.set_read_error(source_id, str(e))
        st.error(text.md(e))
        return
    if parts is not None:
        state.set_multipart(source, parts)
    elif (error := reading.read(source, None, source_id)) is not None:
        st.error(text.md(error))
    else:
        # A file of one dataset replaces the file with several: its list goes away.
        state.clear_multipart()


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
            f"{len(entries)} of {LoadConfig.cache_max_datasets} kept, "
            f"{text.format_bytes(cache.cached_bytes())} in all, most recent first. "
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
    caption = (
        f"{dataset.format} · {dataset.n_rows:,} rows · {dataset.n_cols} columns · "
        f"{text.format_bytes(entry.memory_bytes)} · read at {entry.loaded_at:%H:%M}"
    )
    with _row(f"cached_{entry.dataset_id}", entry.label, caption, active):
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


@contextmanager
def _row(key: str, title: str, caption: str, active: bool) -> Iterator[None]:
    """Row of a list of datasets: title, Active badge and caption; buttons go inside the block.

    Shared by the datasets of a file and the datasets in memory, so both lists look the same.
    ``title`` is a name from the user's file, shown as written; ``caption`` is Markdown.
    """
    with st.container(horizontal=True, vertical_alignment="center", key=key):
        with st.container(gap=None):
            badge = " :green-badge[Active]" if active else ""
            st.markdown(f"**{text.md(title)}**{badge}")
            if caption:
                st.caption(caption)
        yield


def _open_cached(dataset_id: str) -> None:
    """Make a dataset kept in memory the active one (callback of its Open button)."""
    entry = cache.cached_dataset(dataset_id)
    if entry is None:
        st.toast("That dataset is no longer in memory. Upload the file again.")
        return
    reading.open_kept(entry)


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
        "tabs, and the open database or workbook is closed. To use a dataset again you will "
        "have to upload its file."
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
    """Drop every dataset and result from memory, and close the open file with several.

    The session is left without a dataset and without the bytes of that file.
    """
    cache.clear_all()
    state.clear_dataset()
    state.close_multipart()


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
        # Values hold names from the file (tables, columns): shown as written.
        details = ", ".join(
            f"{label} **{text.md(value)}**" for label, value in dataset.details.items()
        )
        st.caption(f":material/tune: Read with {details}.")


def _parts() -> bool:
    """List the datasets of the active file with several, with an Open button each.

    Drawn like the datasets in memory. The file and its choice come from the session state,
    not from the uploader, so the list stays after visiting another page (the uploader forgets
    its file there). Picking one only records the choice: ``reading.read_chosen_part`` reads
    it on the next rerun, as for the selector of the sidebar.

    Returns:
        Whether there is such a file (the list was drawn).
    """
    if (multipart := state.get_multipart()) is None:
        return False
    source, parts = multipart
    if (error := reading.part_error()) is not None:
        st.error(text.md(error))
    active = reading.active_part()
    # The label names the active one, so it shows while collapsed. It changes with the
    # choice, which collapses the expander after picking a dataset.
    label = f"{parts.label}: {text.md(active or reading.chosen_part(parts))}"
    with st.expander(label, icon=":material/table_view:"):
        st.caption(f"{len(parts.names)} in {text.md(source.name)}. Open one to read it.")
        for i, name in enumerate(parts.names):
            # Positions, not names, in keys: names may hold any character.
            with _row(f"part_{i}", name, parts.descriptions.get(name, ""), name == active):
                st.button(
                    "Open",
                    icon=":material/open_in_new:",
                    key=f"open_part_{i}",
                    disabled=name == active,
                    on_click=state.choose_part,
                    args=(name,),
                )
        # The file stays in the session so its other datasets can be read: closing it frees
        # its bytes. The datasets already read stay in memory.
        st.button(
            "Close file",
            icon=":material/close:",
            key="close_file",
            on_click=state.close_multipart,
            help=f"Close {text.md(source.name)}. The {parts.label.lower()}s already read stay "
            "in memory, and opening one of them opens the file again while it is kept.",
        )
    return True


def _preview(dataset: LoadedDataset) -> None:
    """First rows of the dataset. Only those rows are sent to the browser (R-E3)."""
    st.markdown("**Preview**")
    st.caption(f"First {LoadConfig.preview_rows} rows")
    rows = tables.for_display(dataset.df.head(LoadConfig.preview_rows))
    st.dataframe(rows, height=LoadConfig.preview_height)
