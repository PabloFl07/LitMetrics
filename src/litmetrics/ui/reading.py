"""Reading a dataset into the session: shared by the Load page and the sidebar.

A file read from the uploader is handled by the Load page. A file with several datasets (the
tables of a database) is kept in the session state, and the user picks its dataset from the
Load page or from the sidebar of any page; ``read_chosen_part`` reads that choice. app.py calls
it before drawing the page, so a table picked in the sidebar is the active dataset of the page
drawn in the same rerun; the Load page calls it again after an upload.

A dataset kept in memory is opened with ``open_kept``: for a table, its file comes back too.

Every read goes through ``read``: cached by content (``cache.load_dataset``), done once per
``read_id``, and a failed one is remembered so it is not tried again on every rerun.
"""

from litmetrics.core.io import formats
from litmetrics.core.io.base import ReadError, Source, compute_dataset_id
from litmetrics.core.io.formats import Parts
from litmetrics.core.io.library import LibraryEntry
from litmetrics.ui import cache, state


def read(source: Source, part: str | None, read_id: str, parts: Parts | None = None) -> str | None:
    """Read ``part`` of ``source`` (``None``: its only dataset) and make it the active dataset.

    Guess how to read it, read it and convert its columns to their semantic types (cached by
    content), and store it. The schema only replaces the current one when the content is new.
    ``parts`` are the datasets of ``source``, kept with the one read (``cache.load_dataset``).

    Returns:
        None if it was read, or the message of the ``ReadError``, also remembered for
        ``read_id`` (``state.get_read_error``) so the read is not tried again.
    """
    try:
        options = formats.detect_options(source, part)
        dataset_id = compute_dataset_id(source.data, options)
        dataset, schema = cache.load_dataset(dataset_id, source, options, part, parts)
    except ReadError as e:
        # Exceptions are not cached: without this, every rerun would read the file again.
        state.set_read_error(read_id, str(e))
        return str(e)
    if state.set_dataset(dataset, dataset_id, read_id):
        state.set_schema(schema)
    if part is not None:
        state.set_part_id(part, dataset_id)
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
        read(source, part, read_id, parts)


def open_kept(entry: LibraryEntry) -> None:
    """Make ``entry``, a dataset kept in memory, the active one, without reading anything.

    For a dataset of a file with several kept with it (``LibraryEntry.file``), that file is
    opened again with it as the dataset picked, so the user can switch to its others.
    """
    state.switch_dataset(entry.dataset, entry.dataset_id, entry.schema)
    if entry.file is None or entry.part is None:
        return
    source, parts = entry.file
    state.reopen_multipart(source, parts, entry.part, entry.dataset_id)
    # Marked as read: ``read_chosen_part`` must not read it (or the first one) again.
    state.mark_read(_read_id(source, entry.part))


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

    None when there is no such file, or the active dataset is another: one of another file
    opened from the datasets in memory, even a file of the same name (an older version).
    """
    return state.part_of(state.get_dataset_id())


def _read_id(source: Source, part: str) -> str:
    """Id of the read of ``part`` of ``source``: each dataset of a file is a read of its own."""
    return f"{source.source_id}/{part}"
