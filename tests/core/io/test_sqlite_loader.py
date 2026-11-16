import sqlite3

import pandas as pd
import pytest

from litmetrics.core.io.base import ReadError, Source
from litmetrics.core.io.sqlite_loader import (
    SqliteLoader,
    SqliteOptions,
    describe_tables,
    detect_options,
    table_names,
    with_table,
)


def _database(*statements: str) -> bytes:
    """Bytes of an in-memory database after running ``statements``."""
    connection = sqlite3.connect(":memory:")
    try:
        for statement in statements:
            connection.execute(statement)
        connection.commit()
        return connection.serialize()
    finally:
        connection.close()


PEOPLE = _database(
    "CREATE TABLE people (id INTEGER, name TEXT, score REAL, born TEXT)",
    "INSERT INTO people VALUES (1, 'Ana', 1.5, '2000-01-02'), (2, 'Luis', NULL, '1999-03-04')",
    'CREATE TABLE "odd ""name""" (x INTEGER)',
    'INSERT INTO "odd ""name""" VALUES (7)',
    "CREATE VIEW adults AS SELECT name FROM people",
    "CREATE TABLE empty (x INTEGER)",
)


def _source(data: bytes, name: str = "f.db") -> Source:
    return Source(data, name, "id")


def _load(data: bytes, options: SqliteOptions | None = None):
    return SqliteLoader().load(_source(data), options or SqliteOptions())


def test_table_names_lists_tables_and_views_sorted():
    assert table_names(_source(PEOPLE)) == ("adults", "empty", 'odd "name"', "people")


def test_table_names_leaves_out_sqlite_tables():
    data = _database("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT)")
    assert table_names(_source(data)) == ("t",)


def test_describe_tables_gives_rows_and_columns_sorted_by_name():
    assert describe_tables(_source(PEOPLE)) == {
        "adults": "view · 1 column",
        "empty": "0 rows · 1 column",
        'odd "name"': "1 row · 1 column",
        "people": "2 rows · 4 columns",
    }


def test_describe_tables_marks_a_broken_view_instead_of_failing():
    data = _database(
        "CREATE TABLE t (x INTEGER)",
        "CREATE VIEW v AS SELECT x FROM t",
        "DROP TABLE t",
    )
    assert describe_tables(_source(data)) == {"v": "cannot be read"}


def test_describe_tables_of_another_format_is_a_read_error():
    with pytest.raises(ReadError, match="not a SQLite database"):
        describe_tables(_source(b"PK\x03\x04 a zip file"))


def test_detect_options_reads_the_first_table():
    assert detect_options(PEOPLE) == SqliteOptions()


def test_with_table_chooses_the_table():
    assert with_table(SqliteOptions(), "people") == SqliteOptions(table="people")


def test_reads_types():
    dataset = _load(PEOPLE, SqliteOptions(table="people"))
    df = dataset.df
    assert list(df.columns) == ["id", "name", "score", "born"]
    assert pd.api.types.is_integer_dtype(df["id"])
    assert pd.api.types.is_float_dtype(df["score"])
    assert df["score"].isna().sum() == 1
    assert dataset.format == "SQLite"
    assert dataset.details == {"table": "people"}


def test_reads_a_view():
    assert list(_load(PEOPLE, SqliteOptions(table="adults")).df["name"]) == ["Ana", "Luis"]


def test_quotes_table_names():
    assert _load(PEOPLE, SqliteOptions(table='odd "name"')).df.loc[0, "x"] == 7


def test_without_table_reads_the_first_one():
    data = _database(
        "CREATE TABLE b (x INTEGER)", "CREATE TABLE a (y INTEGER)", "INSERT INTO a VALUES (1)"
    )
    assert _load(data).details == {"table": "a"}


def test_blobs_become_their_size():
    data = _database(
        "CREATE TABLE t (id INTEGER, image BLOB)",
        "INSERT INTO t VALUES (1, x'000102'), (2, NULL), (3, 'text')",
    )
    dataset = _load(data)
    assert list(dataset.df["image"].fillna("-")) == ["<3 bytes>", "-", "text"]
    assert dataset.details["binary columns"] == "image (shown as their size)"


@pytest.mark.parametrize(
    ("data", "options", "message"),
    [
        (b"", SqliteOptions(), "'f.db' is empty."),
        (b"a,b\n1,2\n", SqliteOptions(), "'f.db' is not a SQLite database."),
        (_database(), SqliteOptions(), "'f.db' has no tables."),
        (PEOPLE, SqliteOptions(table="nope"), "'f.db' has no table called 'nope'."),
        (PEOPLE, SqliteOptions(table="empty"), "Table 'empty' of 'f.db' has no data rows."),
    ],
)
def test_read_errors(data, options, message):
    with pytest.raises(ReadError) as error:
        _load(data, options)
    assert str(error.value) == message


def test_corrupted_database_is_a_read_error():
    # A valid header followed by garbage.
    data = PEOPLE[:100] + b"\xff" * (len(PEOPLE) - 100)
    with pytest.raises(ReadError, match=r"'f\.db' is damaged"):
        _load(data)


def test_table_names_of_another_format_is_a_read_error():
    with pytest.raises(ReadError, match="is not a SQLite database"):
        table_names(_source(b"PK\x03\x04 a zip file"))
