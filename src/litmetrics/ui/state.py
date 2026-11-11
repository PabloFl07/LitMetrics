"""Single access point to st.session_state. No other module reads or writes it directly.

Keeping every key here means the rest of the app uses typed functions instead of strings, and
the rules about what is cleared when the dataset changes live in one place.

State kept per browser session:
    dataset     The active ``LoadedDataset``, or None.
    dataset_id  Content hash of the active dataset (see ``compute_dataset_id``).
    source_id   Id of the last upload that was processed, to skip it on later reruns.
    read_error  Id of the last upload that could not be read, and its message.
    schema      Semantic types of the active dataset. Cleared when the dataset changes.
    skipped     Columns the user left out of the profile. Cleared when the dataset changes.
"""

from collections.abc import Iterable
from typing import Any

import streamlit as st

from litmetrics.core.io.base import LoadedDataset
from litmetrics.core.schema import Schema

# Keys

_DATASET = "dataset"
_DATASET_ID = "dataset_id"
_SOURCE_ID = "source_id"
_READ_ERROR = "read_error"
_SCHEMA = "schema"
_SKIPPED = "skipped_columns"

# Widgets whose value depends on the columns of the dataset use this prefix,
# so they are cleared when a new dataset is loaded.
WIDGET_PREFIX = "w_"


# Initialization


def init_state() -> None:
    """Fill the missing keys. Called on every rerun from app.py.

    Existing values are kept, so it is safe to call it on every rerun.
    """
    for key, value in _defaults().items():
        if key not in st.session_state:
            st.session_state[key] = value


def _defaults() -> dict[str, Any]:
    return {
        _DATASET: None,
        _DATASET_ID: None,
        _SOURCE_ID: None,
        _READ_ERROR: None,
        _SCHEMA: None,
        _SKIPPED: (),
    }


# Dataset: changing it (used by the load page)


def source_changed(source_id: str) -> bool:
    """Cheap check, done on every rerun, to know whether the file has to be read again.

    Streamlit reruns the page on every interaction while the file stays in the uploader, so
    without this check the file would be hashed and parsed again each time.
    """
    return st.session_state[_SOURCE_ID] != source_id


def set_dataset(dataset: LoadedDataset, dataset_id: str, source_id: str) -> bool:
    """Make ``dataset`` the active one. Return False if it was already active (same content).

    The upload is always marked as processed. The state that depends on the dataset (schema,
    column widgets) is only reset when the content is new, so uploading the same file again
    keeps the user's choices.
    """
    st.session_state[_SOURCE_ID] = source_id
    if st.session_state[_DATASET_ID] == dataset_id:
        return False
    _reset_dependent_state()
    st.session_state[_DATASET] = dataset
    st.session_state[_DATASET_ID] = dataset_id
    return True


def get_read_error(source_id: str) -> str | None:
    """Message of the error of upload ``source_id``, if reading it already failed."""
    failed = st.session_state[_READ_ERROR]
    return failed[1] if failed is not None and failed[0] == source_id else None


def set_read_error(source_id: str, message: str) -> None:
    """Remember that upload ``source_id`` could not be read, so it is not read again."""
    st.session_state[_READ_ERROR] = (source_id, message)


# Dataset: reading it (used by every page)


def get_dataset() -> LoadedDataset | None:
    return st.session_state[_DATASET]


def require_dataset() -> LoadedDataset:
    """Active dataset, for pages that cannot work without one.

    Raises:
        RuntimeError: No dataset is loaded. Pages should check ``get_dataset()`` first and
            show a message instead.
    """
    dataset = st.session_state[_DATASET]
    if dataset is None:
        raise RuntimeError("No dataset loaded")
    return dataset


def get_dataset_id() -> str | None:
    return st.session_state[_DATASET_ID]


def require_dataset_id() -> str:
    """Id of the active dataset, for code that runs only when one is loaded.

    Raises:
        RuntimeError: No dataset is loaded.
    """
    dataset_id = st.session_state[_DATASET_ID]
    if dataset_id is None:
        raise RuntimeError("No dataset loaded")
    return dataset_id


# Schema


def get_schema() -> Schema | None:
    return st.session_state[_SCHEMA]


def require_schema() -> Schema:
    """Schema of the active dataset, for code that runs only when one is loaded.

    Raises:
        RuntimeError: No schema was set (the Load page sets it with the dataset).
    """
    schema = st.session_state[_SCHEMA]
    if schema is None:
        raise RuntimeError("No schema for the active dataset")
    return schema


def set_schema(schema: Schema) -> None:
    st.session_state[_SCHEMA] = schema


# Columns left out of the profile


def get_skipped_columns() -> tuple[str, ...]:
    """Columns the user chose not to profile (phone numbers, codes...), in dataset order."""
    return st.session_state[_SKIPPED]


def set_skipped_columns(names: Iterable[str]) -> None:
    st.session_state[_SKIPPED] = tuple(names)


# Widgets


def widget_key(name: str) -> str:
    """Key for a widget that depends on the dataset columns (e.g. a column selector).

    Use it as ``key=state.widget_key("x_axis")`` so the widget is cleared with the dataset.
    """
    return WIDGET_PREFIX + name


def widget_value(key: str) -> Any:
    """Current value of widget ``key``, for its ``on_change`` callback."""
    return st.session_state[key]


# Internal helpers


def _reset_dependent_state() -> None:
    """Clear everything derived from the active dataset."""
    st.session_state[_SCHEMA] = None
    st.session_state[_SKIPPED] = ()
    # list(): the keys cannot be deleted while iterating over them.
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith(WIDGET_PREFIX):
            del st.session_state[key]
