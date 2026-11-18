"""Unit tests of ``core/io/parquet_loader.py``: Parquet files.

Type conversions shared with Feather are tested in ``test_arrow_table.py``; here, what is
specific to Parquet: row groups and the errors of a bad file.
"""

import pandas as pd
import pyarrow as pa
import pytest
from fixtures import files
from fixtures.samples import PEOPLE_PARQUET

from litmetrics.core.io.base import BINARY_AS_SIZE, ReadError, Source
from litmetrics.core.io.parquet_loader import ParquetLoader, ParquetOptions


def _load(data: bytes):
    return ParquetLoader().load(Source(data, "f.parquet", "id"), ParquetOptions())


def test_reads_types():
    """Ints, floats, dates and missing text keep their type, with nothing to report.

    Case: the Arrow table is read but types are lost on the way to pandas.
    Affects: core/io/parquet_loader.py, core/io/arrow_table.py
    How: reads ``PEOPLE_PARQUET`` and checks columns, dtypes, missing count, format, details.
    """
    dataset = _load(PEOPLE_PARQUET)
    df = dataset.df
    assert list(df.columns) == ["id", "name", "score", "born"]
    assert pd.api.types.is_integer_dtype(df["id"])
    assert pd.api.types.is_float_dtype(df["score"])
    assert pd.api.types.is_datetime64_any_dtype(df["born"])
    assert df["name"].isna().sum() == 1
    assert dataset.format == "Parquet"
    assert dataset.details == {}


def test_details_list_converted_columns():
    """Columns converted on the way (here a binary column) are listed in ``details`` and
    ``changed``.

    Case: the Parquet loader drops the details of ``to_frame`` or the changes of
    ``arrow_table.changes``, so the user is not told on Load or before an export.
    Affects: core/io/parquet_loader.py, core/io/arrow_table.py
    How: reads a file with a binary column and compares the details and ``changed``.
    """
    data = files.parquet(pa.table({"x": [1], "raw": pa.array([b"abc"])}))
    dataset = _load(data)
    assert dataset.details == {"binary columns": "raw (shown as their size)"}
    assert dataset.changed == {"raw": BINARY_AS_SIZE}


def test_reads_every_row_group():
    """A file split in row groups is read whole.

    Case: only the first row group is read (an easy mistake with ``ParquetFile.read_row_group``),
    so the dataset silently loses rows.
    Affects: core/io/parquet_loader.py
    How: writes 1,000 rows in groups of 100 and checks the row count.
    """
    data = files.parquet(pa.table({"x": list(range(1000))}), row_group_size=100)
    assert _load(data).n_rows == 1000


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "'f.parquet' is empty."),
        (b"a,b\n1,2\n", "'f.parquet' is not a Parquet file."),
        (b"ARROW1\x00\x00 rest of the file", "'f.parquet' is not a Parquet file."),
        (files.parquet(pa.table({"x": pa.array([], pa.int64())})), "'f.parquet' has no data rows."),
        (files.parquet(pa.table({})), "'f.parquet' has no columns."),
    ],
    ids=["empty", "csv-renamed", "feather-renamed", "no-rows", "no-columns"],
)
def test_read_errors(data, message):
    """Files that hold no table give a message the user can act on, shown as is.

    Case: pyarrow's ``ArrowInvalid`` reaches the user, or an empty table reaches the Profile.
    Affects: core/io/parquet_loader.py, core/io/base.py (ReadError)
    How: reads each bad file and compares the full message.
    """
    with pytest.raises(ReadError) as error:
        _load(data)
    assert str(error.value) == message


@pytest.mark.parametrize(
    "data",
    [
        # Valid magic bytes at both ends, garbage in between.
        PEOPLE_PARQUET[:4] + b"\xff" * (len(PEOPLE_PARQUET) - 8) + PEOPLE_PARQUET[-4:],
        # The end of the footer cut and the magic bytes put back.
        PEOPLE_PARQUET[: len(PEOPLE_PARQUET) // 2] + b"PAR1",
    ],
    ids=["garbage-inside", "truncated"],
)
def test_damaged_file_is_a_read_error(data):
    """A file that starts and ends like Parquet but is broken is reported as damaged.

    Case: a damaged file passes the magic-bytes check and pyarrow's error (or a crash) follows.
    Affects: core/io/parquet_loader.py
    How: reads a file with garbage between valid ends and a truncated one.
    """
    with pytest.raises(ReadError, match=r"'f\.parquet' is damaged"):
        _load(data)
