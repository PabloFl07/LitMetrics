"""Load page: upload a CSV file, read it and show what was loaded.

A new upload is processed once. Its read options (separator, encoding, header) are detected,
the file is read through a cache keyed by its content and those options, its columns are
converted to the dtype of their semantic type (dates, categories), and the result becomes the
active dataset in the session state. Uploading the same file again keeps the current one.

On every other rerun nothing is read: the page only draws the active dataset. It stays there
after the file is removed from the uploader or the user moves to another page. The page shows
a summary (rows, columns, encoding) and a preview of the first rows. The profile of each
column is in the Profile page.

If a file cannot be read, the error is shown and the previous dataset, if any, stays active.
"""

import streamlit as st

from litmetrics.config import LoadConfig
from litmetrics.core.io.base import LoadedDataset, ReadError, Source, compute_dataset_id
from litmetrics.core.io.sniffing import detect_options
from litmetrics.ui import cache, state, style


def load_page() -> None:
    """Entry point registered in app.py."""
    style.page_header(
        "Load data", "Upload a CSV file. Separator, encoding and header are detected for you."
    )
    source = _upload_source()
    if source is not None:
        _load(source)

    # The dataset lives in the session state, so it stays after the file is removed from the
    # uploader or the user goes to another page.
    dataset = state.get_dataset()
    if dataset is None:
        st.info("Upload a file to start.")
        return
    _summary(dataset)
    _preview(dataset)


# Data sourcing


def _upload_source() -> Source | None:
    """Show the uploader and return the chosen file, or None if there is none."""
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"], label_visibility="collapsed")
    if uploaded is None:
        return None
    return Source(uploaded.getvalue(), uploaded.name, uploaded.file_id)


def _load(source: Source) -> None:
    """Read ``source`` and make it the active dataset. Show an error if it cannot be read.

    Steps: skip if this upload was already processed, guess how to read it, read it and convert
    its columns to their semantic types (cached by content), and store it. The schema only
    replaces the current one when the content is new.
    """
    if not state.source_changed(source.source_id):
        return
    options = detect_options(source.data)
    dataset_id = compute_dataset_id(source.data, options)
    try:
        dataset, schema = cache.load_dataset(dataset_id, source, options)
    except ReadError as e:
        # The upload is not marked as processed, so the error stays visible on later reruns.
        st.error(str(e))
        return
    if state.set_dataset(dataset, dataset_id, source.source_id):
        state.set_schema(schema)


# COMPONENTS


def _summary(dataset: LoadedDataset) -> None:
    """Basic information (Rows,columns,encoding) about the dataset"""
    # Keyed container: styles.css turns these metrics into cards.
    with st.container(key="headline"):
        rows, columns, encoding = st.columns(3)
    rows.metric("Rows", f"{dataset.n_rows:,}", icon=":material/table_rows:", border=True)
    columns.metric("Columns", dataset.n_cols, icon=":material/view_column:", border=True)
    encoding.metric("Encoding", dataset.encoding, icon=":material/translate:", border=True)


def _preview(dataset: LoadedDataset) -> None:
    """
    First rows of the dataset.
    These are separated from the whole dataset before being sent to the browser
    """
    st.markdown("**Preview**")
    st.caption(f"First {LoadConfig.preview_rows} rows")
    st.dataframe(dataset.df.head(LoadConfig.preview_rows), height=500)
