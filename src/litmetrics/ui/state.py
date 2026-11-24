"""Single access point to st.session_state. No other module reads or writes it directly.

Keeping every key here means the rest of the app uses typed functions instead of strings, and
the rules about what is cleared when the dataset changes live in one place.

State kept per browser session:
    dataset     The active ``LoadedDataset``, or None.
    dataset_id  Content hash of the active dataset (see ``compute_dataset_id``).
    upload_id   Id of the last upload of the Load page that was processed (Streamlit's
                ``file_id``), so the file still in the uploader is not read again on reruns.
    source_id   Id of the last read that was done (an upload, or the upload plus the part read
                for files with several datasets), to skip it on later reruns.
    read_error  Reads that could not be done: read id -> message. One per read, so two failed
                reads (a broken upload and an empty table) do not overwrite each other and
                both stay unread.
    multipart   The active file with several datasets (``Source``, ``Parts``), or None. Its
                bytes stay in the session so another dataset can be read after the uploader
                forgot the file (it does when the user visits another page). Opening one of
                its datasets from memory opens it again (``reopen_multipart``).
    part        Dataset of that file picked by the user (None: the first one).
    part_ids    Datasets of that file read in this session: dataset id -> name of the part.
                Tells which part the active dataset is, even with another version of a file
                of the same name in memory.
    schema      Semantic types of the active dataset. Cleared when the dataset changes.
    skipped     Columns the user left out of Profile and Visualization. Cleared when the
                dataset changes.
    widget_prefs  Last value of the remembered widgets (``kept_widget``), by widget key.
                Streamlit drops the state of a widget that is not drawn in a run; these come
                back when it is drawn again (the options of a chart kind after drawing another
                kind, or after visiting another page). Those of ``WIDGET_PREFIX`` widgets are
                cleared when the dataset changes, like the widgets.
    chart_uploads  Chart configurations loaded so far in Visualization: part of the key of
                its uploader, which gets a new key after each load so it empties.
    chart_messages  What the last chart configuration load said, shown once.

The active dataset changes on a new read (``set_dataset``), when the user picks one kept in
memory (``switch_dataset``) or removes it (``clear_dataset``).
"""

from collections.abc import Callable, Iterable
from typing import Any

import streamlit as st

from litmetrics.core.io.base import LoadedDataset, Source
from litmetrics.core.io.formats import Parts
from litmetrics.core.schema import Schema

# Keys

_DATASET = "dataset"
_DATASET_ID = "dataset_id"
_UPLOAD_ID = "upload_id"
_SOURCE_ID = "source_id"
_READ_ERROR = "read_error"
_SCHEMA = "schema"
_SKIPPED = "skipped_columns"
_WIDGET_PREFS = "widget_prefs"
_CHART_UPLOADS = "chart_config_uploads"
_CHART_MESSAGES = "chart_config_messages"

_MULTIPART = "multipart"
_PART = "part"
_PART_IDS = "part_ids"

# Key of the sidebar selector of the active dataset. Not WIDGET_PREFIX: it lists datasets, not
# columns, and is set to the active dataset before every draw (``sync_switcher``).
SWITCHER_WIDGET = "dataset_switcher"

# Key of the sidebar selector of the dataset of a file with several (a table). Like
# SWITCHER_WIDGET, set to the active one before every draw (``sync_part_switcher``).
PART_SWITCHER_WIDGET = "part_switcher"

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
        _UPLOAD_ID: None,
        _SOURCE_ID: None,
        _READ_ERROR: {},
        _MULTIPART: None,
        _PART: None,
        _PART_IDS: {},
        _SCHEMA: None,
        _SKIPPED: (),
        _WIDGET_PREFS: {},
        _CHART_UPLOADS: 0,
        _CHART_MESSAGES: (),
    }


# Dataset: changing it (used by the load page)


def upload_changed(upload_id: str) -> bool:
    """Whether upload ``upload_id`` (the file in the uploader) was not processed yet.

    Kept apart from the reads (``source_changed``): reading a table, or opening a dataset
    from memory, must not make the file still in the uploader look new and be read again.
    """
    return st.session_state[_UPLOAD_ID] != upload_id


def mark_upload(upload_id: str) -> None:
    """Record that upload ``upload_id`` was processed (read, listed, or failed)."""
    st.session_state[_UPLOAD_ID] = upload_id


def mark_read(source_id: str) -> None:
    """Record read ``source_id`` as done without reading: its dataset is already active."""
    st.session_state[_SOURCE_ID] = source_id


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


def switch_dataset(dataset: LoadedDataset, dataset_id: str, schema: Schema) -> None:
    """Make a dataset kept in memory the active one, without reading any file.

    ``source_id`` is kept: the last read stays done, so ``reading.read_chosen_part`` does not
    read the table picked again and take the place of the dataset chosen. Does nothing if it
    is already active.
    """
    if st.session_state[_DATASET_ID] == dataset_id:
        return
    _reset_dependent_state()
    st.session_state[_DATASET] = dataset
    st.session_state[_DATASET_ID] = dataset_id
    st.session_state[_SCHEMA] = schema


def clear_dataset() -> None:
    """Leave the session with no active dataset (it was dropped from memory).

    The upload stays processed, so the file still in the uploader is not read again.
    """
    _reset_dependent_state()
    st.session_state[_DATASET] = None
    st.session_state[_DATASET_ID] = None


def get_read_error(source_id: str) -> str | None:
    """Message of the error of read ``source_id``, if reading it already failed."""
    return st.session_state[_READ_ERROR].get(source_id)


def set_read_error(source_id: str, message: str) -> None:
    """Remember that read ``source_id`` could not be done, so it is not tried again.

    Kept per read: a later failure of another read does not make this one be tried again.
    """
    st.session_state[_READ_ERROR] = {**st.session_state[_READ_ERROR], source_id: message}


# File with several datasets (tables of a database): kept so its selector outlives the uploader


def get_multipart() -> tuple[Source, Parts] | None:
    """The active file with several datasets and its datasets, or None (``set_multipart``)."""
    return st.session_state[_MULTIPART]


def set_multipart(source: Source, parts: Parts) -> None:
    """Keep ``source`` and its datasets, so another one can be read after leaving the page.

    The uploader forgets its file when another page is shown; without the bytes kept here,
    picking another table would need the file uploaded again. The previous choice of dataset
    is forgotten: the new file starts from its first one.
    """
    st.session_state[_MULTIPART] = (source, parts)
    st.session_state[_PART] = None
    st.session_state[_PART_IDS] = {}


def clear_multipart() -> None:
    """Forget the file with several datasets, and its bytes (a single-dataset file replaced it)."""
    st.session_state[_MULTIPART] = None
    st.session_state[_PART] = None
    st.session_state[_PART_IDS] = {}


def close_multipart() -> None:
    """Close the file with several datasets on the user's request, freeing its bytes.

    Its upload is marked as processed: if the file is still in the uploader, it is not listed
    and read again (which would open it back). The active dataset, if it is one of its
    datasets, stays. The datasets read from it still keep the file in memory
    (``LibraryEntry.file``): it is freed when they leave it.
    """
    if (multipart := st.session_state[_MULTIPART]) is None:
        return
    mark_upload(multipart[0].source_id)
    clear_multipart()


def reopen_multipart(source: Source, parts: Parts, part: str, dataset_id: str) -> None:
    """Open again ``source``, kept in memory with dataset ``dataset_id`` (its part ``part``).

    The file becomes the active file with several, with ``part`` picked. The parts already
    known stay when it is the file already open (the same upload).
    """
    current = st.session_state[_MULTIPART]
    if current is None or current[0].source_id != source.source_id:
        st.session_state[_PART_IDS] = {}
    st.session_state[_MULTIPART] = (source, parts)
    st.session_state[_PART] = part
    set_part_id(part, dataset_id)


def get_part() -> str | None:
    """Dataset of the active file with several picked by the user, or None (the first one)."""
    return st.session_state[_PART]


def set_part_id(name: str, dataset_id: str) -> None:
    """Remember that part ``name`` of the active file with several is dataset ``dataset_id``."""
    st.session_state[_PART_IDS] = {**st.session_state[_PART_IDS], dataset_id: name}


def part_of(dataset_id: str | None) -> str | None:
    """Part of the active file with several that is dataset ``dataset_id``, or None.

    None as well for a dataset of another file, even one with the same name: only the parts
    read from the file kept now are known (``set_part_id``).
    """
    return st.session_state[_PART_IDS].get(dataset_id)


def choose_part(name: str) -> None:
    """Pick dataset ``name`` of the active file with several, to read it on the next rerun.

    The last read is forgotten, so the dataset is read even if it was the last one read (the
    user may have switched to another dataset since): the read is a cache hit. The upload
    itself is not read again: it has its own marker (``upload_changed``).
    """
    st.session_state[_PART] = name
    st.session_state[_SOURCE_ID] = None


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


# Columns left out of Profile and Visualization


def get_skipped_columns() -> tuple[str, ...]:
    """Columns the user chose to skip (phone numbers, codes...), in dataset order."""
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


def has_widget(key: str) -> bool:
    """Whether widget ``key`` has a value in this run."""
    return key in st.session_state


def get_widget(key: str) -> Any:
    """Value of widget ``key``, or None if it has none."""
    return st.session_state.get(key)


def set_widget(key: str, value: Any) -> None:
    """Change widget ``key``. Only from a callback, or before the widget is drawn in this run."""
    st.session_state[key] = value


# Remembered widgets: their value survives runs where they are not drawn


def _prefs() -> dict[str, Any]:
    return st.session_state[_WIDGET_PREFS]


def restore_widget(key: str, default: Any = None) -> None:
    """Before drawing widget ``key``: give it its remembered value if Streamlit dropped it,
    else ``default`` the first time (None: leave it to the widget).
    """
    if key in st.session_state:
        return
    prefs = _prefs()
    if key in prefs:
        st.session_state[key] = prefs[key]
    elif default is not None:
        st.session_state[key] = default


def remember_widget(key: str, value: Any) -> None:
    """Keep the value widget ``key`` was drawn with, for when it is drawn again."""
    _prefs()[key] = value


def put_widget(key: str, value: Any) -> None:
    """Give widget ``key`` the value ``value`` now and when it is drawn again. Only from a
    callback, or before the widget is drawn.
    """
    st.session_state[key] = value
    _prefs()[key] = value


def forget_widgets(keys: Iterable[str]) -> None:
    """Drop widgets ``keys`` and their remembered values: drawn again, they take their
    defaults. Only from a callback (or for widgets not drawn yet in this run).
    """
    prefs = _prefs()
    for key in keys:
        st.session_state.pop(key, None)
        prefs.pop(key, None)


def kept_widget[T](
    draw: Callable[..., T], label: str, *args: Any, key: str, default: Any = None, **kwargs: Any
) -> T:
    """``draw(label, *args, key=key, **kwargs)`` with the value it had the last time it was
    drawn, or ``default`` the first time.

    Never pass the default to the widget itself (``value=``, ``default=``): with a value
    already in its state Streamlit warns.
    """
    restore_widget(key, default)
    value = draw(label, *args, key=key, **kwargs)
    remember_widget(key, value)
    return value


# Chart configurations loaded in Visualization


def chart_config_uploads() -> int:
    """Chart configurations loaded so far: part of the key of the uploader."""
    return st.session_state[_CHART_UPLOADS]


def count_chart_config_upload() -> None:
    """One more configuration loaded: the uploader gets a new key, and so empties."""
    st.session_state[_CHART_UPLOADS] += 1


def set_chart_config_messages(messages: Iterable[tuple[str, str]]) -> None:
    """What the last configuration load said, as ``(level, text)``: error, warning, success."""
    st.session_state[_CHART_MESSAGES] = tuple(messages)


def pop_chart_config_messages() -> tuple[tuple[str, str], ...]:
    """The messages of the last configuration load, once: they are cleared."""
    messages = st.session_state[_CHART_MESSAGES]
    st.session_state[_CHART_MESSAGES] = ()
    return messages


def sync_switcher() -> None:
    """Point the sidebar selector at the active dataset. Call it before drawing the selector.

    The active dataset also changes from the Load page; without this the selector would keep
    showing the one picked before.
    """
    st.session_state[SWITCHER_WIDGET] = st.session_state[_DATASET_ID]


def sync_part_switcher(part: str) -> None:
    """Point the sidebar selector of tables at ``part``, the active one. Call it before drawing.

    The active table also changes from the Load page and from the datasets in memory.
    """
    st.session_state[PART_SWITCHER_WIDGET] = part


# Internal helpers


def _reset_dependent_state() -> None:
    """Clear everything derived from the active dataset."""
    st.session_state[_SCHEMA] = None
    st.session_state[_SKIPPED] = ()
    st.session_state[_WIDGET_PREFS] = {
        k: v for k, v in _prefs().items() if not k.startswith(WIDGET_PREFIX)
    }
    # list(): the keys cannot be deleted while iterating over them.
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith(WIDGET_PREFIX):
            del st.session_state[key]
