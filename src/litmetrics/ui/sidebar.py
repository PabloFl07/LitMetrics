"""Sidebar shown on every page: controls that are not tied to one page.

For now it holds the selector of the active dataset, which lists the datasets kept in memory
(``cache.cached_datasets``), so the user can switch between them from any page without
uploading the file again. Removing them is done in the Load page.
"""

import streamlit as st

from litmetrics.core.io.library import LibraryEntry
from litmetrics.ui import cache, state


def draw() -> None:
    """Draw the sidebar. Called by app.py after the page, so it shows what the page loaded."""
    entries = cache.cached_datasets()
    active_id = state.get_dataset_id()
    if not entries and active_id is None:
        # Nothing to choose: no sidebar at all.
        return
    with st.sidebar:
        _dataset_switcher(entries, active_id)


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
    state.switch_dataset(entry.dataset, entry.dataset_id, entry.schema)
