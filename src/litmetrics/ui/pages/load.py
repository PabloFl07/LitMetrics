"""Load page: upload a CSV file, read it and show what was loaded.

A new upload is processed once. Its read options (separator, encoding, header, decimal mark)
are detected, the file is read through a cache keyed by its content and those options, its
columns are converted to the dtype of their semantic type (dates, categories), and the result
becomes the active dataset in the session state. Uploading the same file again keeps the
current one.

On every other rerun nothing is read, not even the bytes of the upload: the page only draws the
active dataset. It stays there after the file is removed from the uploader or the user moves to
another page. The page shows a summary (rows, columns, encoding, how the file was read) and a
preview of the first rows. The profile of each column is in the Profile page.

If a file cannot be read, the error is shown and the previous dataset, if any, stays active.
The error is remembered for that upload, so later reruns show it without reading the file
again.
"""

import streamlit as st
from streamlit.typing import UploadedFile

from litmetrics.config import LoadConfig
from litmetrics.core.io.base import LoadedDataset, ReadError, Source, compute_dataset_id
from litmetrics.core.io.sniffing import SEPARATORS, detect_options
from litmetrics.ui import cache, state, style, tables


def load_page() -> None:
    """Entry point registered in app.py."""
    style.page_header(
        "Load data",
        "Upload a CSV file. Separator, encoding, header and decimal mark are detected for you.",
    )
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"], label_visibility="collapsed")
    if uploaded is not None:
        _load(uploaded)

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

    Steps: skip if this upload was already processed, or show its error if it already failed;
    guess how to read it, read it and convert its columns to their semantic types (cached by
    content), and store it. The schema only replaces the current one when the content is new.
    """
    source_id = uploaded.file_id
    # Checked before ``getvalue``, which copies the whole file on every call.
    if not state.source_changed(source_id):
        return
    if (error := state.get_read_error(source_id)) is not None:
        st.error(error)
        return
    source = Source(uploaded.getvalue(), uploaded.name, source_id)
    options = detect_options(source.data)
    dataset_id = compute_dataset_id(source.data, options)
    try:
        dataset, schema = cache.load_dataset(dataset_id, source, options)
    except ReadError as e:
        # Exceptions are not cached: without this, every rerun would read the file again.
        state.set_read_error(source_id, str(e))
        st.error(str(e))
        return
    if state.set_dataset(dataset, dataset_id, source_id):
        state.set_schema(schema)


# Components


def _summary(dataset: LoadedDataset) -> None:
    """Rows, columns and encoding of the dataset, and how the file was read."""
    # Keyed container: styles.css turns these metrics into cards.
    with st.container(key="headline"):
        rows, columns, encoding = st.columns(3)
    rows.metric("Rows", f"{dataset.n_rows:,}", icon=":material/table_rows:", border=True)
    columns.metric("Columns", dataset.n_cols, icon=":material/view_column:", border=True)
    encoding.metric("Encoding", dataset.encoding, icon=":material/translate:", border=True)
    separator = SEPARATORS.get(dataset.separator, repr(dataset.separator))
    decimal = "Comma (,)" if dataset.decimal == "," else "Dot (.)"
    header = "first row" if dataset.has_header else "none (columns numbered)"
    st.caption(
        f":material/tune: Read with separator **{separator}**, decimal mark **{decimal}**, "
        f"column names: **{header}**."
    )


def _preview(dataset: LoadedDataset) -> None:
    """First rows of the dataset. Only those rows are sent to the browser (R-E3)."""
    st.markdown("**Preview**")
    st.caption(f"First {LoadConfig.preview_rows} rows")
    rows = tables.for_display(dataset.df.head(LoadConfig.preview_rows))
    st.dataframe(rows, height=LoadConfig.preview_height)
