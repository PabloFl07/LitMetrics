"""Datasets kept in memory after reading them, so the user can go back to one without the file.

``DatasetLibrary`` is a store of the last datasets read, oldest dropped first. Unlike a hidden
function cache, it can list what it holds and forget one dataset or all of them, which is what
the user manages from the app (``ui/cache.py`` keeps one library shared by every session).

It is used from several threads (one per browser session), so every method takes a lock.
"""

import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime

from litmetrics.core.io.base import LoadedDataset, Source
from litmetrics.core.io.formats import Parts
from litmetrics.core.schema import Schema

# A file with several datasets (a database) and its datasets: what reading another one needs.
type MultipartFile = tuple[Source, Parts]


@dataclass(frozen=True, eq=False)
class LibraryEntry:
    """A dataset kept in the library.

    Attributes:
        dataset_id: Content id of the dataset (``compute_dataset_id``).
        dataset: The dataset, typed. Shared: never modify it.
        schema: Semantic types of its columns.
        part: Dataset of a file with several (a table of a database), or None.
        loaded_at: When it was read.
        memory_bytes: Memory its DataFrame takes (``memory_usage(deep=True)``), measured
            once by ``create``: deep counts the text of object columns, value by value.
        file: For a dataset of a file with several, that file (its bytes included) and its
            datasets, so opening the dataset again can offer the others; None when not kept.
            The datasets of one file share the same objects: the bytes are kept once.
    """

    dataset_id: str
    dataset: LoadedDataset
    schema: Schema
    part: str | None
    loaded_at: datetime
    memory_bytes: int
    file: MultipartFile | None = None

    @classmethod
    def create(
        cls,
        dataset_id: str,
        dataset: LoadedDataset,
        schema: Schema,
        part: str | None,
        loaded_at: datetime,
        file: MultipartFile | None = None,
    ) -> LibraryEntry:
        """Entry for ``dataset``, with the memory it takes measured now."""
        memory = int(dataset.df.memory_usage(deep=True).sum())
        return cls(dataset_id, dataset, schema, part, loaded_at, memory, file)

    @property
    def label(self) -> str:
        """Name shown to the user: the file name, and the dataset of the file if it has several."""
        name = self.dataset.name
        return name if self.part is None else f"{name} / {self.part}"


class DatasetLibrary:
    """The last ``max_entries`` datasets read, most recently used first.

    Args:
        max_entries: Datasets kept at most. Adding one more drops the least recently used.

    Raises:
        ValueError: ``max_entries`` is lower than 1.
    """

    def __init__(self, max_entries: int) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._max_entries = max_entries
        # Least recently used first, so eviction pops from the front.
        self._entries: OrderedDict[str, LibraryEntry] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, dataset_id: str) -> LibraryEntry | None:
        """The entry of ``dataset_id``, marked as just used, or None if it is not kept."""
        with self._lock:
            entry = self._entries.get(dataset_id)
            if entry is not None:
                self._entries.move_to_end(dataset_id)
            return entry

    def put(self, entry: LibraryEntry) -> None:
        """Keep ``entry`` (replacing one with the same id) and drop the oldest if over the limit."""
        with self._lock:
            self._entries[entry.dataset_id] = entry
            self._entries.move_to_end(entry.dataset_id)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def entries(self) -> tuple[LibraryEntry, ...]:
        """Every entry, most recently used first."""
        with self._lock:
            return tuple(reversed(self._entries.values()))

    def total_bytes(self) -> int:
        """Memory taken by every entry: its DataFrame, plus each file kept counted once.

        A file is kept by every dataset read from it (``LibraryEntry.file``), as one object:
        it is freed when the last of them is dropped.
        """
        with self._lock:
            frames = sum(entry.memory_bytes for entry in self._entries.values())
            # By identity: the same upload is one object; another upload is another copy.
            files = {id(e.file[0]): len(e.file[0].data) for e in self._entries.values() if e.file}
            return frames + sum(files.values())

    def remove(self, dataset_id: str) -> bool:
        """Forget ``dataset_id``. Return whether it was kept."""
        with self._lock:
            return self._entries.pop(dataset_id, None) is not None

    def clear(self) -> None:
        """Forget every dataset."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
