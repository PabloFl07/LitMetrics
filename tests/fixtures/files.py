"""Builders of file bytes in each format the app reads, written by the real libraries.

The loaders are tested on bytes like the ones a user uploads, never on hand-made objects:
each builder writes a real file in memory. Excel workbooks are built by ``xlsx_builder``,
since no Excel writer is a dependency.
"""

import io
import sqlite3
import warnings
from collections.abc import Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

from fixtures.xlsx_builder import xlsx

__all__ = ["csv", "feather_file", "parquet", "sqlite", "sqlite_tables", "xlsx"]


def csv(frame: pd.DataFrame, sep: str = ",", decimal: str = ".", encoding: str = "utf-8") -> bytes:
    """Bytes of ``frame`` as a CSV with a header and without the index.

    Missing values are written as empty fields, as most exports do.
    """
    text = frame.to_csv(index=False, sep=sep, decimal=decimal)
    return text.encode(encoding)


def sqlite(*statements: str) -> bytes:
    """Bytes of a SQLite database after running ``statements`` on an empty one."""
    connection = sqlite3.connect(":memory:")
    try:
        for statement in statements:
            connection.execute(statement)
        connection.commit()
        return connection.serialize()
    finally:
        connection.close()


def sqlite_tables(tables: Mapping[str, pd.DataFrame]) -> bytes:
    """Bytes of a SQLite database with one table per item of ``tables``, in that order."""
    connection = sqlite3.connect(":memory:")
    try:
        for name, frame in tables.items():
            frame.to_sql(name, connection, index=False)
        connection.commit()
        return connection.serialize()
    finally:
        connection.close()


def parquet(data: pd.DataFrame | pa.Table, **kwargs) -> bytes:
    """Bytes of ``data`` as a Parquet file. ``kwargs`` go to ``pyarrow.parquet.write_table``."""
    table = data if isinstance(data, pa.Table) else pa.Table.from_pandas(data)
    buffer = io.BytesIO()
    pq.write_table(table, buffer, **kwargs)
    return buffer.getvalue()


def feather_file(data: pd.DataFrame | pa.Table, **kwargs) -> bytes:
    """Bytes of ``data`` as a Feather (Arrow IPC) file. ``kwargs`` go to ``write_feather``.

    ``version=1`` writes the old Feather V1 format, deprecated but still found in the wild.
    """
    table = data if isinstance(data, pa.Table) else pa.Table.from_pandas(data)
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        feather.write_feather(table, buffer, **kwargs)
    return buffer.getvalue()
