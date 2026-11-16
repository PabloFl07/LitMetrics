"""Unit tests of ``core/io/formats.py``: the registry that picks a loader for each file.

The UI never names a format: it asks this registry which format a file has, which parts
(tables, sheets) it holds, and how to read it. These tests check the choice of format and
that the registry hands work to the right loader.
"""

import typing

import pytest
from fixtures.samples import SQLITE_AB, TWO_SHEETS

from litmetrics.core.io import formats
from litmetrics.core.io.base import ReadError, Source
from litmetrics.core.io.sniffing import CsvOptions
from litmetrics.core.io.sqlite_loader import SqliteOptions

# A small semicolon CSV: detecting its options is visible (separator ";").
_SEMICOLON_CSV = b"a;b\n1;2\n3;4\n"


# Choosing the format


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("data.csv", "CSV"),
        ("DATA.CSV", "CSV"),
        ("a.b.csv", "CSV"),
        ("data.db", "SQLite"),
        ("data.sqlite", "SQLite"),
        ("DATA.SQLITE3", "SQLite"),
        ("data.parquet", "Parquet"),
        ("DATA.PQ", "Parquet"),
        ("data.feather", "Feather"),
        ("DATA.ARROW", "Feather"),
        ("data.xlsx", "Excel"),
        ("DATA.XLS", "Excel"),
        ("b.xlsm", "Excel"),
        ("b.xlsb", "Excel"),
        ("b.ods", "Excel"),
    ],
)
def test_format_for_picks_by_extension(name, expected):
    """The format is chosen by the last extension, whatever its case.

    Case: ``DATA.CSV`` (Windows exports) or ``a.b.csv`` is rejected or given the wrong loader.
    Affects: core/io/formats.py (format_for, FORMATS)
    How: asks the format of a file name per extension the app accepts.
    """
    assert formats.format_for(name).name == expected


@pytest.mark.parametrize("name", ["data.json", "csv", "data"])
def test_format_for_rejects_unknown_extensions(name):
    """A file of no known format is a ``ReadError`` naming the file.

    Case: an unknown file falls through to the CSV loader and shows garbage as data.
    Affects: core/io/formats.py (format_for)
    How: asks the format of a JSON file, a name that is only "csv", and a name without dot.
    """
    with pytest.raises(ReadError) as error:
        formats.format_for(name)
    assert str(error.value).startswith(f"'{name}' is not a supported file.")


def test_extensions_have_a_dot_and_are_lowercase():
    """The extensions given to the uploader start with a dot and are lowercase.

    Case: the uploader is given ``CSV`` or ``csv`` and filters out valid files.
    Affects: core/io/formats.py (extensions), used by ui/pages/load.py
    How: checks every extension the registry lists.
    """
    assert ".csv" in formats.extensions()
    assert all(e.startswith(".") and e == e.lower() for e in formats.extensions())


# Consistency of the registry


def test_each_format_has_its_own_options_class():
    """No two formats share an options class.

    Case: ``load`` picks the loader by the type of the options; a shared class would send a
    file to the wrong loader.
    Affects: core/io/formats.py (FORMATS)
    How: compares the options classes of every format.
    """
    types = [fmt.options_type for fmt in formats.FORMATS]
    assert len(set(types)) == len(types)


def test_read_options_covers_every_format():
    """The ``ReadOptions`` type lists the options of every registered format.

    Case: a new format is registered but left out of the type alias, so type checking and
    cache keys silently miss it.
    Affects: core/io/formats.py (ReadOptions, FORMATS)
    How: compares the members of the alias with the options classes of ``FORMATS``.
    """
    alias = formats.ReadOptions.__value__
    covered = set(typing.get_args(alias)) or {alias}
    assert covered == {fmt.options_type for fmt in formats.FORMATS}


# Handing work to the loaders


def test_detect_options_and_load_use_the_format_of_the_file():
    """Options are detected by the loader of the file's format, and ``load`` uses them.

    Case: the registry detects with one format and reads with another.
    Affects: core/io/formats.py (detect_options, load), core/io/sniffing.py, csv_loader.py
    How: detects and loads a semicolon CSV, checks the separator found and the columns read.
    """
    source = Source(_SEMICOLON_CSV, "f.csv", "id")
    options = formats.detect_options(source)
    assert options == CsvOptions(separator=";")
    dataset = formats.load(source, options)
    assert dataset.format == "CSV"
    assert list(dataset.df.columns) == ["a", "b"]


def test_load_picks_the_loader_by_the_options_not_the_name():
    """Once options are chosen, the file name is not looked at again.

    Case: a file renamed between detection and loading (or a ``.txt`` read as CSV) fails.
    Affects: core/io/formats.py (load)
    How: loads a ``.txt`` file with CSV options.
    """
    dataset = formats.load(Source(b"a,b\n1,2\n", "f.txt", "id"), CsvOptions())
    assert dataset.format == "CSV"


def test_load_rejects_options_of_no_format():
    """Options of no format are a programming error (``TypeError``), not a ``ReadError``.

    Case: a bug passing the wrong object is shown to the user as a problem with their file.
    Affects: core/io/formats.py (load)
    How: loads with a plain ``object()`` as options.
    """
    with pytest.raises(TypeError):
        formats.load(Source(b"a,b\n1,2\n", "f.csv", "id"), object())  # type: ignore[arg-type]


# Files with several datasets


def test_list_parts_of_a_database_are_its_tables_described():
    """A database lists its tables, labelled "Table", sorted by name and described.

    Case: tables are listed in creation order, or without their size.
    Affects: core/io/formats.py (list_parts), core/io/sqlite_loader.py
    How: lists the parts of ``SQLITE_AB`` (``b`` created before ``a``).
    """
    parts = formats.list_parts(Source(SQLITE_AB, "f.db", "id"))
    assert parts is not None
    assert (parts.label, parts.names) == ("Table", ("a", "b"))
    assert dict(parts.descriptions) == {"a": "1 row · 1 column", "b": "1 row · 1 column"}


def test_list_parts_of_a_workbook_are_its_sheets_described():
    """A workbook lists its sheets, labelled "Sheet", in workbook order and described.

    Case: the Excel loader is not wired into the registry, so sheets cannot be chosen.
    Affects: core/io/formats.py (list_parts), core/io/excel_loader.py
    How: lists the parts of ``TWO_SHEETS``.
    """
    parts = formats.list_parts(Source(TWO_SHEETS, "book.xlsx", "id"))
    assert parts is not None
    assert (parts.label, parts.names) == ("Sheet", ("Sales", "Notes"))
    assert dict(parts.descriptions) == {"Sales": "2 rows · 2 columns", "Notes": "1 row · 1 column"}


def test_list_parts_of_a_single_dataset_format_is_none():
    """A CSV has no parts: the Load page reads it directly.

    Case: a CSV is shown with a list of one "table", or the page waits for a choice.
    Affects: core/io/formats.py (list_parts)
    How: lists the parts of a CSV.
    """
    assert formats.list_parts(Source(b"a,b\n1,2\n", "f.csv", "id")) is None


def test_detect_options_reads_the_chosen_part():
    """The part named in ``detect_options`` is the one ``load`` reads.

    Case: the chosen table is lost and the first one is read.
    Affects: core/io/formats.py (detect_options, load), core/io/sqlite_loader.py
    How: detects options for table ``b`` of ``SQLITE_AB`` and loads its only value.
    """
    source = Source(SQLITE_AB, "f.db", "id")
    assert formats.detect_options(source, "b") == SqliteOptions(table="b")
    assert formats.load(source, formats.detect_options(source, "b")).df.loc[0, "x"] == 2


def test_detect_options_ignores_the_part_for_single_dataset_formats():
    """A part given for a CSV is ignored, not an error.

    Case: a stale table choice from a previous database breaks reading the next CSV.
    Affects: core/io/formats.py (detect_options)
    How: detects the options of a CSV with a part name.
    """
    source = Source(_SEMICOLON_CSV, "f.csv", "id")
    assert formats.detect_options(source, "anything") == CsvOptions(separator=";")
