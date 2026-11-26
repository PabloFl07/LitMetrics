"""What every tab of the Profile page shares: the active dataset, whole and without the
skipped columns (``active``), the "About this tab" expander and the note on sampled measures.
"""

import dataclasses

import streamlit as st

from litmetrics.config import ProfileConfig
from litmetrics.core.io.base import LoadedDataset
from litmetrics.core.schema import Schema, drop_columns
from litmetrics.ui import cache, state

_CFG = ProfileConfig


@dataclasses.dataclass(frozen=True, eq=False)
class Active:
    """The active dataset, whole and without the skipped columns.

    Attributes:
        dataset_id: Id of the whole dataset. Key of the measures of one column, which do not
            depend on the other columns: computed with ``dataset.df`` and reused whatever the
            user skips.
        dataset: The whole dataset.
        view_id: Cache key of the dataset without the skipped columns (``cache.view_id``). Key
            of what depends on the set of columns, such as duplicate rows.
        view: The dataset without the skipped columns.
        schema: Schema of ``view``: the columns the page describes.
        view_tag: Short text that changes with the skipped columns, for widget keys.
    """

    dataset_id: str
    dataset: LoadedDataset
    view_id: str
    view: LoadedDataset
    schema: Schema
    view_tag: str


def active() -> Active:
    """The active dataset, whole and without the skipped columns.

    Fragments call it instead of receiving the data as arguments, so a fragment rerun always
    sees the current dataset (R-E2).
    """
    dataset_id, dataset, schema = (
        state.require_dataset_id(),
        state.require_dataset(),
        state.require_schema(),
    )
    skipped = state.get_skipped_columns()
    if not skipped:
        return Active(dataset_id, dataset, dataset_id, dataset, schema, "all")
    # No copy: the view shares the columns of the dataset.
    df, view_schema = drop_columns(dataset.df, schema, skipped)
    view_id = cache.view_id(dataset_id, skipped)
    return Active(
        dataset_id,
        dataset,
        view_id,
        dataclasses.replace(dataset, df=df),
        view_schema,
        view_id.removeprefix(f"{dataset_id}-skip-"),
    )


def about(body: str) -> None:
    """Collapsible explanation of a tab, for users new to data profiling."""
    with st.expander("About this tab", icon=":material/help:"):
        st.markdown(body)


def sample_note(dataset: LoadedDataset, what: str) -> None:
    """Say that ``what`` was computed on a sample, if the dataset is larger than the sample."""
    n = _CFG.sample_rows
    if dataset.n_rows > n:
        st.caption(
            f":material/info: {what} computed on a random sample of up to {n:,} values "
            f"per column ({dataset.n_rows:,} rows in the dataset)."
        )
