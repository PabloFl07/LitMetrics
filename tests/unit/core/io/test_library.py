"""Unit tests of ``core/io/library.py``: the datasets kept in memory (an LRU list).

The Load page and the sidebar list these entries; when the list is full the least recently
used one is dropped. These tests check that order, the limit and the sizes shown.
"""

from datetime import datetime

import pandas as pd
import pytest

from litmetrics.core.io.base import LoadedDataset
from litmetrics.core.io.library import DatasetLibrary, LibraryEntry
from litmetrics.core.schema import Schema


def _entry(dataset_id: str, part: str | None = None) -> LibraryEntry:
    """An entry of a 2-row table read from ``data.db``, optionally from table ``part``."""
    dataset = LoadedDataset(pd.DataFrame({"a": [1, 2]}), "data.db", "SQLite")
    return LibraryEntry.create(dataset_id, dataset, Schema(()), part, datetime(2026, 1, 1))


def _library(*ids: str, size: int = 3) -> DatasetLibrary:
    """A library of ``size`` entries holding ``ids``, put in that order."""
    library = DatasetLibrary(size)
    for dataset_id in ids:
        library.put(_entry(dataset_id))
    return library


def _ids(library: DatasetLibrary) -> list[str]:
    return [e.dataset_id for e in library.entries()]


def test_get_returns_the_entry_put():
    """An entry put can be got back by its id, and an unknown id gives None.

    Case: entries are copied (memory doubles) or lookups of a removed id raise ``KeyError``.
    Affects: core/io/library.py (put, get)
    How: puts an entry, gets it by id (same object), and asks for an id never put.
    """
    library = DatasetLibrary(3)
    entry = _entry("a")
    library.put(entry)
    assert library.get("a") is entry
    assert library.get("missing") is None


def test_entries_are_most_recent_first():
    """Entries are listed from the most recently put.

    Case: the list of datasets in memory shows the oldest file on top.
    Affects: core/io/library.py (entries)
    How: puts a, b, c and checks the order c, b, a.
    """
    assert _ids(_library("a", "b", "c")) == ["c", "b", "a"]


def test_get_marks_entry_as_recent():
    """Getting an entry moves it to the top.

    Case: a dataset the user keeps opening is still the first one dropped.
    Affects: core/io/library.py (get)
    How: gets the oldest of three entries and checks it comes first.
    """
    library = _library("a", "b", "c")
    library.get("a")
    assert _ids(library) == ["a", "c", "b"]


def test_put_over_limit_drops_least_recently_used():
    """A put over the limit drops the least recently used entry, not the oldest put.

    Case: plain FIFO drops a dataset that was used a moment ago.
    Affects: core/io/library.py (put)
    How: in a library of 2, puts a and b, uses a, puts c; b must be the one dropped.
    """
    library = _library("a", "b", size=2)
    library.get("a")
    library.put(_entry("c"))
    assert _ids(library) == ["c", "a"]


def test_put_same_id_replaces_without_duplicating():
    """Putting an id again replaces its entry.

    Case: the same file read twice is listed twice and counted twice in memory.
    Affects: core/io/library.py (put)
    How: puts two entries with id a and checks the length and the entry kept.
    """
    library = _library("a")
    newer = _entry("a")
    library.put(newer)
    assert len(library) == 1
    assert library.get("a") is newer


def test_remove_forgets_entry_and_says_whether_it_was_there():
    """``remove`` drops an entry and returns whether it existed.

    Case: removing a dataset already dropped (by another session) raises.
    Affects: core/io/library.py (remove)
    How: removes a kept id (True, then gone) and an unknown one (False).
    """
    library = _library("a")
    assert library.remove("a") is True
    assert library.get("a") is None
    assert library.remove("missing") is False


def test_clear_forgets_everything():
    """``clear`` empties the library.

    Case: "Clear cache" leaves some datasets listed.
    Affects: core/io/library.py (clear)
    How: clears a library of two entries and checks it lists none.
    """
    library = _library("a", "b")
    library.clear()
    assert library.entries() == ()


def test_max_entries_below_one_raises():
    """A library must hold at least one entry.

    Case: a config value of 0 makes every read drop itself at once, silently.
    Affects: core/io/library.py (DatasetLibrary), config.py (cache_max_datasets)
    How: creates a library of size 0.
    """
    with pytest.raises(ValueError, match="at least 1"):
        DatasetLibrary(0)


@pytest.mark.parametrize(
    ("part", "label"),
    [(None, "data.db"), ("customers", "data.db / customers")],
    ids=["whole-file", "table"],
)
def test_label_names_the_file_and_its_part(part, label):
    """The label is the file name, plus the table or sheet when there is one.

    Case: two tables of one database are listed with the same label and cannot be told apart.
    Affects: core/io/library.py (LibraryEntry.label), shown by the sidebar
    How: builds an entry with and without a part and compares the label.
    """
    assert _entry("a", part).label == label


def test_memory_of_an_entry_and_in_all():
    """Each entry measures its DataFrame, and the total follows puts and removes.

    Case: memory is measured shallow (text columns count as 8 bytes a value), or a removed
    entry is still counted in the total shown to the user.
    Affects: core/io/library.py (LibraryEntry.create, total_bytes)
    How: compares one entry with ``memory_usage(deep=True)``, then the total with 2 and 1 entries.
    """
    entry = _entry("x")
    assert entry.memory_bytes == int(entry.dataset.df.memory_usage(deep=True).sum()) > 0
    library = _library("a", "b")
    assert library.total_bytes() == 2 * entry.memory_bytes
    library.remove("a")
    assert library.total_bytes() == entry.memory_bytes
