from datetime import datetime

import pandas as pd
import pytest

from litmetrics.core.io.base import LoadedDataset
from litmetrics.core.io.library import DatasetLibrary, LibraryEntry
from litmetrics.core.schema import Schema


def _entry(dataset_id: str, part: str | None = None) -> LibraryEntry:
    dataset = LoadedDataset(pd.DataFrame({"a": [1, 2]}), "data.db", "SQLite")
    return LibraryEntry.create(dataset_id, dataset, Schema(()), part, datetime(2026, 1, 1))


def test_get_returns_kept_entry():
    library = DatasetLibrary(3)
    entry = _entry("a")
    library.put(entry)
    assert library.get("a") is entry


def test_get_unknown_returns_none():
    assert DatasetLibrary(3).get("missing") is None


def test_entries_most_recent_first():
    library = DatasetLibrary(3)
    for dataset_id in "abc":
        library.put(_entry(dataset_id))
    assert [e.dataset_id for e in library.entries()] == ["c", "b", "a"]


def test_get_marks_entry_as_recent():
    library = DatasetLibrary(3)
    for dataset_id in "abc":
        library.put(_entry(dataset_id))
    library.get("a")
    assert [e.dataset_id for e in library.entries()] == ["a", "c", "b"]


def test_put_over_limit_drops_least_recently_used():
    library = DatasetLibrary(2)
    library.put(_entry("a"))
    library.put(_entry("b"))
    library.get("a")
    library.put(_entry("c"))
    assert [e.dataset_id for e in library.entries()] == ["c", "a"]


def test_put_same_id_replaces_without_duplicating():
    library = DatasetLibrary(3)
    library.put(_entry("a"))
    newer = _entry("a")
    library.put(newer)
    assert len(library) == 1
    assert library.get("a") is newer


def test_remove_forgets_entry():
    library = DatasetLibrary(3)
    library.put(_entry("a"))
    assert library.remove("a") is True
    assert library.get("a") is None


def test_remove_unknown_returns_false():
    assert DatasetLibrary(3).remove("missing") is False


def test_clear_forgets_everything():
    library = DatasetLibrary(3)
    library.put(_entry("a"))
    library.put(_entry("b"))
    library.clear()
    assert library.entries() == ()


def test_max_entries_below_one_raises():
    with pytest.raises(ValueError, match="at least 1"):
        DatasetLibrary(0)


def test_label_is_file_name_without_part():
    assert _entry("a").label == "data.db"


def test_label_adds_part_of_file_with_several():
    assert _entry("a", "customers").label == "data.db / customers"


def test_create_measures_the_memory_of_the_dataset():
    entry = _entry("a")
    assert entry.memory_bytes == int(entry.dataset.df.memory_usage(deep=True).sum())
    assert entry.memory_bytes > 0


def test_total_bytes_adds_every_entry_and_forgets_removed_ones():
    library = DatasetLibrary(3)
    library.put(_entry("a"))
    library.put(_entry("b"))
    one = _entry("x").memory_bytes
    assert library.total_bytes() == 2 * one
    library.remove("a")
    assert library.total_bytes() == one
