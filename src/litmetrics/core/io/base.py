"""Types shared by every loader, independent of the file format and of Streamlit.

Flow of a file through the app:

    Source (raw bytes) --Loader.load(options)--> LoadedDataset (DataFrame + how it was read)

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


# eq=False: comparing two datasets would compare their DataFrames, which is ambiguous in pandas.
# Two datasets are the same when their ``dataset_id`` is the same.
@dataclass(frozen=True, eq=False)
class LoadedDataset:
    """A parsed file: the DataFrame plus how it was read, to show it to the user.

    Attributes:
        df: The parsed data.
        name: Name of the original file.
        separator: Column separator used to read it.
        encoding: Text encoding used to read it.
        unparsed: For each date or numeric column stored as text in the file, how many of its
            values could not be read as dates or numbers (they became missing when the column
            was converted).
    """

    df: pd.DataFrame
    name: str
    separator: str
    encoding: str
    unparsed: Mapping[str, int] = field(default_factory=dict)

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
