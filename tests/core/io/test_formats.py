import sqlite3
import typing

import pytest

from litmetrics.core.io import formats
from litmetrics.core.io.base import ReadError, Source
from litmetrics.core.io.sniffing import CsvOptions
from litmetrics.core.io.sqlite_loader import SqliteOptions


def _database() -> bytes:
    """A database with two tables, ``a`` and ``b``, of one row each."""
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(
            "CREATE TABLE b (x INTEGER); INSERT INTO b VALUES (2);"
            "CREATE TABLE a (y INTEGER); INSERT INTO a VALUES (1);"
        )
        return connection.serialize()
    finally:
        connection.close()


def test_extensions_have_a_dot_and_are_lowercase():
    assert ".csv" in formats.extensions()
    assert all(e.startswith(".") and e == e.lower() for e in formats.extensions())


@pytest.mark.parametrize("name", ["data.csv", "DATA.CSV", "a.b.csv"])
def test_format_for_picks_by_extension(name):
    assert formats.format_for(name).name == "CSV"


@pytest.mark.parametrize("name", ["data.db", "data.sqlite", "DATA.SQLITE3"])
def test_format_for_picks_sqlite_by_extension(name):
    assert formats.format_for(name).name == "SQLite"


@pytest.mark.parametrize("name", ["data.parquet", "DATA.PQ"])
def test_format_for_picks_parquet_by_extension(name):
    assert formats.format_for(name).name == "Parquet"


@pytest.mark.parametrize("name", ["data.feather", "DATA.ARROW"])
def test_format_for_picks_feather_by_extension(name):
    assert formats.format_for(name).name == "Feather"


@pytest.mark.parametrize("name", ["data.xlsx", "DATA.XLS", "b.xlsm", "b.xlsb", "b.ods"])
def test_format_for_picks_excel_by_extension(name):
    assert formats.format_for(name).name == "Excel"


def test_list_parts_of_a_workbook_are_its_sheets():
    from xlsx_builder import xlsx

    data = xlsx({"Sales": [["a"], [1], [2]], "Notes": [["b"], ["x"]]})
    parts = formats.list_parts(Source(data, "book.xlsx", "id"))
    assert parts is not None
    assert (parts.label, parts.names) == ("Sheet", ("Sales", "Notes"))
    assert dict(parts.descriptions) == {"Sales": "2 rows · 1 column", "Notes": "1 row · 1 column"}


@pytest.mark.parametrize("name", ["data.json", "csv", "data"])
def test_format_for_rejects_unknown_extensions(name):
    with pytest.raises(ReadError) as error:
        formats.format_for(name)
    assert str(error.value).startswith(f"'{name}' is not a supported file.")


def test_detect_options_uses_the_format_of_the_file():
    source = Source(b"a;b\n1;2\n3;4\n", "f.csv", "id")
    assert formats.detect_options(source) == CsvOptions(separator=";")


def test_load_reads_with_the_loader_of_the_format():
    source = Source(b"a;b\n1;2\n3;4\n", "f.csv", "id")
    dataset = formats.load(source, formats.detect_options(source))
    assert dataset.format == "CSV"
    assert list(dataset.df.columns) == ["a", "b"]


def test_load_picks_the_loader_by_the_options_not_the_name():
    # The format was chosen when the options were detected; the name is not looked at again.
    dataset = formats.load(Source(b"a,b\n1,2\n", "f.txt", "id"), CsvOptions())
    assert dataset.format == "CSV"


def test_load_rejects_options_of_no_format():
    with pytest.raises(TypeError):
        formats.load(Source(b"a,b\n1,2\n", "f.csv", "id"), object())  # type: ignore[arg-type]


def test_each_format_has_its_own_options_class():
    types = [fmt.options_type for fmt in formats.FORMATS]
    assert len(set(types)) == len(types)


def test_read_options_covers_every_format():
    alias = formats.ReadOptions.__value__
    covered = set(typing.get_args(alias)) or {alias}
    assert covered == {fmt.options_type for fmt in formats.FORMATS}


def test_list_parts_of_a_database_are_its_tables():
    parts = formats.list_parts(Source(_database(), "f.db", "id"))
    assert parts is not None
    assert (parts.label, parts.names) == ("Table", ("a", "b"))


def test_list_parts_describes_each_part():
    parts = formats.list_parts(Source(_database(), "f.db", "id"))
    assert parts is not None
    assert dict(parts.descriptions) == {"a": "1 row · 1 column", "b": "1 row · 1 column"}


def test_list_parts_of_a_single_dataset_format_is_none():
    assert formats.list_parts(Source(b"a,b\n1,2\n", "f.csv", "id")) is None


def test_detect_options_reads_the_chosen_part():
    source = Source(_database(), "f.db", "id")
    assert formats.detect_options(source, "b") == SqliteOptions(table="b")
    assert formats.load(source, formats.detect_options(source, "b")).df.loc[0, "x"] == 2


def test_detect_options_ignores_the_part_for_single_dataset_formats():
    source = Source(b"a;b\n1;2\n3;4\n", "f.csv", "id")
    assert formats.detect_options(source, "anything") == CsvOptions(separator=";")
