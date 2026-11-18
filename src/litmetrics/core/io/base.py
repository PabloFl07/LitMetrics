"""Types shared by every loader, independent of the file format and of Streamlit.

Flow of a file through the app:

    Source (raw bytes) --Loader.load(options)--> LoadedDataset (DataFrame + how it was read)

Which loader reads a file, and how its options are found, is in ``formats.py``.
Anything that goes wrong on the way is raised as ``ReadError``, whose message is shown as is.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class Source:
    """Raw content of a file chosen by the user, before parsing.

    Attributes:
        data: Full content of the file.
        name: File name as uploaded, used in messages to the user.
        source_id: Id of this particular upload (Streamlit's ``file_id``). It changes when the
            user uploads again, even the same file, so it only tells whether to look at the
            file again, not whether the content is new. For that, see ``compute_dataset_id``.
    """

    data: bytes
    name: str
    source_id: str


# What a loader did to the values of a column (``LoadedDataset.changed``), shared by formats.
BINARY_AS_SIZE = "binary values replaced by their size"
TIME_AS_TEXT = "times of day kept as text"


# eq=False: comparing two datasets would compare their DataFrames, which is ambiguous in pandas.
# Two datasets are the same when their ``dataset_id`` is the same.
@dataclass(frozen=True, eq=False)
class LoadedDataset:
    """A parsed file: the DataFrame plus how it was read, to show it to the user.

    Attributes:
        df: The parsed data.
        name: Name of the original file.
        format: Name of the file format, as shown to the user ("CSV").
        details: How the file was read, as label -> value in the order to show them. Both are
            text for the user (``{"separator": "Comma (,)"}``): each loader writes what applies
            to its format, so nothing else needs to know the format.
        unparsed: For each date or numeric column stored as text in the file, how many of its
            values could not be read as dates or numbers (they became missing when the column
            was converted).
        changed: Columns whose values the loader changed so the app can use them, with what
            changed, as text for the user (``{"photo": BINARY_AS_SIZE}``). An export of the
            dataset writes the changed values, not the original ones.
    """

    df: pd.DataFrame
    name: str
    format: str
    details: Mapping[str, str] = field(default_factory=dict)
    unparsed: Mapping[str, int] = field(default_factory=dict)
    changed: Mapping[str, str] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return self.df.shape[0]

    @property
    def n_cols(self) -> int:
        return self.df.shape[1]


class ReadError(Exception):
    """The file could not be turned into a dataset. The message is meant for the user."""


class Loader[OptionsT](Protocol):
    """Reads one file format. A new format only needs a new class with this method.

    ``OptionsT`` holds how the format is read (e.g. ``CsvOptions``). It must be a frozen
    dataclass, so it can be part of the dataset id and of Streamlit's cache key.
    """

    def load(self, source: Source, options: OptionsT) -> LoadedDataset:
        """Parse ``source`` with ``options``. Raise ``ReadError`` if it cannot be read."""
        ...


def compute_dataset_id(content: bytes, options: object) -> str:
    """Stable identity of a dataset: the same bytes read with the same options.

    Used as the cache key and to know whether a new upload really brings new data.
    ``options`` must have a deterministic ``repr`` (true for frozen dataclasses).
    """
    digest = hashlib.sha256()
    digest.update(content)
    digest.update(repr(options).encode("utf-8"))
    return digest.hexdigest()
