"""Unit tests of ``core/io/sqlite_loader.py``: listing and reading the tables of a database.

``samples.SQLITE_SHOWCASE`` holds every kind of object a database may have: a table, a table
whose name needs quoting, a view and an empty table.
"""

import pandas as pd
import pytest
from fixtures import files
from fixtures.samples import SQLITE_AB, SQLITE_SHOWCASE

from litmetrics.core.io.base import BINARY_AS_SIZE, ReadError, Source
from litmetrics.core.io.sqlite_loader import (
    SqliteLoader,
    SqliteOptions,
    describe_tables,
    table_names,
)


def _source(data: bytes, name: str = "f.db") -> Source:
    return Source(data, name, "id")


def _load(data: bytes, options: SqliteOptions | None = None):
    return SqliteLoader().load(_source(data), options or SqliteOptions())


# Listing tables


def test_table_names_lists_tables_and_views_sorted():
    """Tables and views are listed by name; names with quotes come back as written.

    Case: views are left out, or a name with ``"`` comes back escaped (``odd ""name""``).
    Affects: core/io/sqlite_loader.py (table_names)
    How: lists the objects of ``SQLITE_SHOWCASE``.
    """
    assert table_names(_source(SQLITE_SHOWCASE)) == ("adults", "empty", 'odd "name"', "people")


def test_table_names_leaves_out_sqlite_tables():
    """SQLite's internal tables (``sqlite_sequence``) are not offered.

    Case: an AUTOINCREMENT key adds ``sqlite_sequence``, which the user would see as data.
    Affects: core/io/sqlite_loader.py (table_names)
    How: lists the tables of a database with one AUTOINCREMENT table.
    """
    data = files.sqlite("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT)")
    assert table_names(_source(data)) == ("t",)


def test_describe_tables_gives_rows_and_columns_sorted_by_name():
    """Each table is described by rows and columns; a view only by columns.

    Case: counting the rows of a view runs its whole query (slow), or singular/plural is wrong.
    Affects: core/io/sqlite_loader.py (describe_tables), shown in the Load page and sidebar
    How: describes every object of ``SQLITE_SHOWCASE``.
    """
    assert describe_tables(_source(SQLITE_SHOWCASE)) == {
        "adults": "view · 1 column",
        "empty": "0 rows · 1 column",
        'odd "name"': "1 row · 1 column",
        "people": "2 rows · 4 columns",
    }


def test_describe_tables_marks_a_broken_view_instead_of_failing():
    """A view over a dropped table is described as unreadable, not an error for the file.

    Case: one broken view makes the whole database impossible to open.
    Affects: core/io/sqlite_loader.py (describe_tables)
    How: creates a view, drops its table, and describes the database.
    """
    data = files.sqlite(
        "CREATE TABLE t (x INTEGER)", "CREATE VIEW v AS SELECT x FROM t", "DROP TABLE t"
    )
    assert describe_tables(_source(data)) == {"v": "cannot be read"}


# Reading a table


def test_reads_types():
    """Integer, real and text columns keep their type; NULL is a missing value.

    Case: SQLite's dynamic typing makes every column ``object``, so nothing can be profiled.
    Affects: core/io/sqlite_loader.py (SqliteLoader.load)
    How: reads table ``people`` and checks columns, dtypes, missing count, format and details.
    """
    dataset = _load(SQLITE_SHOWCASE, SqliteOptions(table="people"))
    df = dataset.df
    assert list(df.columns) == ["id", "name", "score", "born"]
    assert pd.api.types.is_integer_dtype(df["id"])
    assert pd.api.types.is_float_dtype(df["score"])
    assert df["score"].isna().sum() == 1
    assert dataset.format == "SQLite"
    assert dataset.details == {"table": "people"}


def test_reads_a_view():
    """A view is read like a table.

    Case: only ``type = 'table'`` objects can be read, so views listed cannot be opened.
    Affects: core/io/sqlite_loader.py
    How: reads view ``adults`` and checks its values.
    """
    assert list(_load(SQLITE_SHOWCASE, SqliteOptions(table="adults")).df["name"]) == ["Ana", "Luis"]


def test_quotes_table_names():
    """A table whose name holds quotes and spaces is read.

    Case: the name is pasted into SQL unquoted: a syntax error, or SQL injection from the file.
    Affects: core/io/sqlite_loader.py
    How: reads table ``odd "name"`` and checks its value.
    """
    assert _load(SQLITE_SHOWCASE, SqliteOptions(table='odd "name"')).df.loc[0, "x"] == 7


def test_without_table_reads_the_first_one_by_name():
    """Without a table chosen, the first one by name is read.

    Case: the first created table is read, so the default differs from the first one listed.
    Affects: core/io/sqlite_loader.py
    How: loads ``SQLITE_AB`` (``b`` created first) without options.
    """
    assert _load(SQLITE_AB).details == {"table": "a"}


def test_blobs_become_their_size():
    """BLOB values are shown as their size; text stored in a BLOB column stays text.

    Case: raw bytes reach the preview, or a column mixing blobs and text crashes.
    Affects: core/io/sqlite_loader.py
    How: reads a BLOB column with bytes, NULL and text, and checks values and details.
    """
    data = files.sqlite(
        "CREATE TABLE t (id INTEGER, image BLOB)",
        "INSERT INTO t VALUES (1, x'000102'), (2, NULL), (3, 'text')",
    )
    dataset = _load(data)
    assert list(dataset.df["image"].fillna("-")) == ["<3 bytes>", "-", "text"]
    assert dataset.details["binary columns"] == "image (shown as their size)"
    assert dataset.changed == {"image": BINARY_AS_SIZE}


# Errors


@pytest.mark.parametrize(
    ("data", "options", "message"),
    [
        (b"", SqliteOptions(), "'f.db' is empty."),
        (b"a,b\n1,2\n", SqliteOptions(), "'f.db' is not a SQLite database."),
        (files.sqlite(), SqliteOptions(), "'f.db' has no tables."),
        (SQLITE_SHOWCASE, SqliteOptions(table="nope"), "'f.db' has no table called 'nope'."),
        (
            SQLITE_SHOWCASE,
            SqliteOptions(table="empty"),
            "Table 'empty' of 'f.db' has no data rows.",
        ),
    ],
    ids=["empty-file", "not-a-database", "no-tables", "unknown-table", "empty-table"],
)
def test_read_errors(data, options, message):
    """Files or tables with nothing to profile give a message the user can act on.

    Case: ``sqlite3.DatabaseError`` reaches the user, or an empty table reaches the Profile.
    Affects: core/io/sqlite_loader.py, core/io/base.py (ReadError)
    How: reads each bad file or table and compares the full message.
    """
    with pytest.raises(ReadError) as error:
        _load(data, options)
    assert str(error.value) == message


def test_corrupted_database_is_a_read_error():
    """A valid header followed by garbage is reported as damaged.

    Case: the header check passes and SQLite's "malformed" error crashes the page.
    Affects: core/io/sqlite_loader.py
    How: keeps the first 100 bytes of ``SQLITE_SHOWCASE`` and fills the rest with 0xFF.
    """
    data = SQLITE_SHOWCASE[:100] + b"\xff" * (len(SQLITE_SHOWCASE) - 100)
    with pytest.raises(ReadError, match=r"'f\.db' is damaged"):
        _load(data)


@pytest.mark.parametrize("listing", [table_names, describe_tables], ids=["names", "describe"])
def test_listing_tables_of_another_format_is_a_read_error(listing):
    """Listing the tables of a file that is not a database says so.

    Case: a zip renamed to ``.db`` raises ``sqlite3.DatabaseError`` while listing tables.
    Affects: core/io/sqlite_loader.py (table_names, describe_tables)
    How: lists and describes the tables of bytes that start like a zip.
    """
    with pytest.raises(ReadError, match="not a SQLite database"):
        listing(_source(b"PK\x03\x04 a zip file"))
