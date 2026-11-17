"""Sidebar shown on every page: controls that are not tied to one page.

For now it holds the selector of the active dataset, which lists the datasets kept in memory
(``cache.cached_datasets``), so the user can switch between them from any page without
uploading the file again. Removing them is done in the Load page.

When the active dataset is one of the datasets of a file with several (a table of a
database), a second selector below lists the others. Opening such a dataset from memory opens
its file again (``reading.open_kept``), so the selector comes back with it. Picking one only
records the choice: ``reading.read_chosen_part``, run by app.py before the page, reads it.
"""

import streamlit as st

from litmetrics.core.io.library import LibraryEntry
from litmetrics.ui import cache, reading, state


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
