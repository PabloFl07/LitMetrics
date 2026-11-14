"""Reading a dataset into the session: shared by the Load page and the sidebar.

A file read from the uploader is handled by the Load page. A file with several datasets (the
tables of a database) is kept in the session state, and the user picks its dataset from the
Load page or from the sidebar of any page; ``read_chosen_part`` reads that choice. app.py calls
it before drawing the page, so a table picked in the sidebar is the active dataset of the page
drawn in the same rerun; the Load page calls it again after an upload.

Every read goes through ``read``: cached by content (``cache.load_dataset``), done once per
``read_id``, and a failed one is remembered so it is not tried again on every rerun.
"""

from litmetrics.core.io import formats
from litmetrics.core.io.base import ReadError, Source, compute_dataset_id
from litmetrics.core.io.formats import Parts
from litmetrics.ui import cache, state


def read(source: Source, part: str | None, read_id: str) -> str | None:
    """Read ``part`` of ``source`` (``None``: its only dataset) and make it the active dataset.

    Guess how to read it, read it and convert its columns to their semantic types (cached by
    content), and store it. The schema only replaces the current one when the content is new.

    Returns:
        None if it was read, or the message of the ``ReadError``, also remembered for
        ``read_id`` (``state.get_read_error``) so the read is not tried again.
    """
    try:
        options = formats.detect_options(source, part)
        dataset_id = compute_dataset_id(source.data, options)
        dataset, schema = cache.load_dataset(dataset_id, source, options, part)
    except ReadError as e:
        # Exceptions are not cached: without this, every rerun would read the file again.
        state.set_read_error(read_id, str(e))
        return str(e)
    if state.set_dataset(dataset, dataset_id, read_id):
        state.set_schema(schema)
    return None


def read_chosen_part() -> None:
    """Read the dataset picked of the file with several, if it was not read yet.

    Cheap when there is nothing to do (no such file, or the choice was already read or
    failed), so it runs on every rerun.
    """
    if (multipart := state.get_multipart()) is None:
        return
    source, parts = multipart
    part = chosen_part(parts)
    read_id = _read_id(source, part)
    if state.source_changed(read_id) and state.get_read_error(read_id) is None:
        read(source, part, read_id)


def chosen_part(parts: Parts) -> str:
    """Dataset of the file with several picked by the user; the first one until they pick."""
    return state.get_part() or parts.names[0]


def part_error() -> str | None:
    """Message of the error of reading the dataset picked, or None if it was read."""
    if (multipart := state.get_multipart()) is None:
        return None
    source, parts = multipart
    return state.get_read_error(_read_id(source, chosen_part(parts)))


def active_part() -> str | None:
    """Dataset of the file with several that is the active dataset, or None.

    None when there is no such file, or the active dataset is another (one of another file
    opened from the datasets in memory). Looked up in the library, which knows the file and
    part of each dataset kept.
    """
    if (multipart := state.get_multipart()) is None:
        return None
    source = multipart[0]
    active_id = state.get_dataset_id()
    for entry in cache.cached_datasets():
        if entry.dataset_id == active_id:
            return entry.part if entry.dataset.name == source.name else None
    return None


def _read_id(source: Source, part: str) -> str:
    """Id of the read of ``part`` of ``source``: each dataset of a file is a read of its own."""
    return f"{source.source_id}/{part}"
