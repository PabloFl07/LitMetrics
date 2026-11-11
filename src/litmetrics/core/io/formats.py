"""The file formats the app can read, and which one reads a given file.

Each format is one ``FileFormat`` in ``FORMATS``: its name, its file extensions, the class of
its read options, how those options are found and its loader. The rest of the app goes through
this module and never names a format: supporting a new one is a new loader plus a new entry
here (and its options class in ``ReadOptions``).

The format of a file is chosen once, by its extension, when its options are detected. From
then on the options carry it: each format has its own options class, so ``load`` picks the
loader by the type of the options and cannot pair a file with the options of another format.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from litmetrics.core.io import csv_loader, sniffing
from litmetrics.core.io.base import LoadedDataset, Loader, ReadError, Source
from litmetrics.core.io.sniffing import CsvOptions

# Read options of any supported format: the union of the ``options_type`` of every format.
type ReadOptions = CsvOptions


@dataclass(frozen=True)
class FileFormat[OptionsT]:
    """One file format.

    Attributes:
        name: Name shown to the user ("CSV").
        extensions: File extensions of the format, lowercase and with the dot (".csv").
        options_type: Class of its read options, a frozen dataclass (it is part of the dataset
            id and of the cache key). No other format may use the same class.
        detect_options: Finds how to read a file from its content.
        loader: Reads a file with those options.
    """

    name: str
    extensions: tuple[str, ...]
    options_type: type[OptionsT]
    detect_options: Callable[[bytes], OptionsT]
    loader: Loader[OptionsT]


# Every supported format. The first one whose extension matches reads the file.
FORMATS: tuple[FileFormat[Any], ...] = (
    FileFormat(
        csv_loader.FORMAT_NAME,
        (".csv",),
        CsvOptions,
        sniffing.detect_options,
        csv_loader.CsvLoader(),
    ),
)


def extensions() -> tuple[str, ...]:
    """Return the extensions of every supported format, with the dot."""
    return tuple(extension for fmt in FORMATS for extension in fmt.extensions)


def format_for(name: str) -> FileFormat[Any]:
    """Return the format of the file called ``name``, chosen by its extension.

    Raises:
        ReadError: No format has that extension.
    """
    lowered = name.lower()
    for fmt in FORMATS:
        if lowered.endswith(fmt.extensions):
            return fmt
    supported = ", ".join(extensions())
    raise ReadError(f"'{name}' is not a supported file. Supported extensions: {supported}.")


def detect_options(source: Source) -> ReadOptions:
    """Find how to read ``source``: the options of its format, guessed from its content.

    Raises:
        ReadError: The extension of the file is not supported.
    """
    return format_for(source.name).detect_options(source.data)


def load(source: Source, options: ReadOptions) -> LoadedDataset:
    """Parse ``source`` with the loader of the format of ``options``.

    Raises:
        ReadError: The file cannot be read.
        TypeError: ``options`` is not the options class of any format (a bug, not a bad file).
    """
    return _format_of(options).loader.load(source, options)


def _format_of(options: object) -> FileFormat[Any]:
    """Return the format whose options class is the class of ``options``.

    Raises:
        TypeError: No format uses that class.
    """
    for fmt in FORMATS:
        if type(options) is fmt.options_type:
            return fmt
    raise TypeError(f"{type(options).__name__} is not the read options of any file format.")
