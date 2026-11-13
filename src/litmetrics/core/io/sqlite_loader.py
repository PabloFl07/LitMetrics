"""Read one table (or view) of a SQLite database into a LoadedDataset.

A database holds several tables, so reading one needs a choice the bytes cannot make: which
table. ``table_names`` lists them for the user, ``SqliteOptions.table`` carries the choice and
``SqliteLoader`` reads it.

The app only has the bytes of the upload, never a path. ``sqlite3`` opens them without a
temporary file through ``Connection.deserialize``, which copies the bytes into an in-memory
database: while reading, the file is in memory twice. The connection is read-only
(``PRAGMA query_only``) and closed after each read.

BLOB values (images, serialized objects...) cannot be profiled or shown: each one becomes a
text with its size (``"<1,024 bytes>"``), and the columns that had them are listed in
``LoadedDataset.details``.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace

import pandas as pd

from litmetrics.core.io.base import LoadedDataset, ReadError, Source

# Name of the format, shown to the user.
FORMAT_NAME = "SQLite"

# What one of the datasets of a database is called, shown to the user.
PART_LABEL = "Table"

# Every SQLite 3 database starts with these 16 bytes (the "magic header string").
_MAGIC = b"SQLite format 3\x00"


@dataclass(frozen=True)
class SqliteOptions:
    """How a SQLite database is read.

    Attributes:
        table: Table or view to read. ``None`` reads the first one, by name.
    """

    table: str | None = None


def detect_options(data: bytes) -> SqliteOptions:
    """Return the default options: the first table. Nothing about a database needs guessing.

    Never raises: problems with the file are reported by ``table_names`` and the loader.
    """
    return SqliteOptions()


def with_table(options: SqliteOptions, table: str) -> SqliteOptions:
    """Return ``options`` reading ``table`` instead."""
    return replace(options, table=table)


def table_names(source: Source) -> tuple[str, ...]:
    """Return the tables and views of the database in ``source``, sorted by name.

    SQLite's own tables (``sqlite_*``) are left out.

    Raises:
        ReadError: The file is empty, is not a SQLite database, is damaged or has no tables.
    """
    with _open(source) as connection:
        return _tables(connection, source.name)


class SqliteLoader:
    """``Loader`` for SQLite databases: reads one table or view."""

    def load(self, source: Source, options: SqliteOptions) -> LoadedDataset:
        """Read the table ``options.table`` of the database in ``source``.

        Raises:
            ReadError: The file is empty, is not a SQLite database or is damaged; it has no
                tables; the table does not exist; or the table has no rows.
        """
        name = source.name
        with _open(source) as connection:
            tables = _tables(connection, name)
            table = options.table if options.table is not None else tables[0]
            if table not in tables:
                raise ReadError(f"'{name}' has no table called '{table}'.")
            try:
                df = pd.read_sql_query(f"SELECT * FROM {_quote(table)}", connection)
            # pandas wraps sqlite3's errors in its own DatabaseError.
            except (sqlite3.DatabaseError, pd.errors.DatabaseError) as e:
                raise ReadError(f"Table '{table}' of '{name}' cannot be read: {e}") from e

        if df.empty:
            raise ReadError(f"Table '{table}' of '{name}' has no data rows.")
        df.columns = [str(column) for column in df.columns]
        df, binary = _describe_blobs(df)
        details = {PART_LABEL.lower(): table}
        if binary:
            details["binary columns"] = ", ".join(binary) + " (shown as their size)"
        return LoadedDataset(df, name, FORMAT_NAME, details)


@contextmanager
def _open(source: Source) -> Iterator[sqlite3.Connection]:
    """Open the bytes of ``source`` as a read-only in-memory database, and close it after.

    Raises:
        ReadError: The file is empty or is not a SQLite database.
    """
    name = source.name
    if not source.data:
        raise ReadError(f"'{name}' is empty.")
    # Checked first: sqlite3 only notices a file of another format on the first query, with
    # a less clear message.
    if not source.data.startswith(_MAGIC):
        raise ReadError(f"'{name}' is not a SQLite database.")
    with closing(sqlite3.connect(":memory:")) as connection:
        try:
            connection.deserialize(source.data)
            connection.execute("PRAGMA query_only = ON")
        except sqlite3.DatabaseError as e:
            raise ReadError(f"'{name}' is damaged and cannot be opened: {e}") from e
        yield connection


def _tables(connection: sqlite3.Connection, name: str) -> tuple[str, ...]:
    """Tables and views of an open database, sorted by name, without SQLite's own.

    Raises:
        ReadError: The database is damaged or has no tables.
    """
    try:
        rows = connection.execute(
            "SELECT name FROM sqlite_schema WHERE type IN ('table', 'view')"
            " AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' ORDER BY name"
        ).fetchall()
    except sqlite3.DatabaseError as e:
        raise ReadError(f"'{name}' is damaged and cannot be read: {e}") from e
    if not rows:
        raise ReadError(f"'{name}' has no tables.")
    return tuple(row[0] for row in rows)


def _describe_blobs(df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Replace each BLOB value with a text giving its size.

    Returns:
        The new DataFrame, and the names of the columns that had BLOB values.
    """
    # BLOBs come as ``bytes`` in object columns. SQLite does not type its columns, so a
    # column can mix them with other values: only the ``bytes`` are replaced.
    binary = tuple(
        column
        for column in df.columns
        # Text columns are ``str``: only ``object`` ones can hold bytes.
        if df[column].dtype == object
        and df[column].map(lambda value: isinstance(value, bytes)).any()
    )
    if not binary:
        return df, ()
    described = {
        column: df[column].map(
            lambda value: f"<{len(value):,} bytes>" if isinstance(value, bytes) else value
        )
        for column in binary
    }
    return df.assign(**described), binary


def _quote(identifier: str) -> str:
    """Quote a table name for SQL. The name comes from the file, so it is never trusted."""
    return '"' + identifier.replace('"', '""') + '"'
