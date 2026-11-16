"""Unit tests of ``core/io/feather_loader.py``: Feather (Arrow IPC) files.

Type conversions shared with Parquet are tested in ``test_arrow_table.py``; here, what is
specific to Feather: its versions, compressions and the errors of a bad file.
"""

import pandas as pd
import pyarrow as pa
import pytest
from fixtures import files
from fixtures.samples import PEOPLE_FEATHER, people

from litmetrics.core.io.base import ReadError, Source
from litmetrics.core.io.feather_loader import FeatherLoader, FeatherOptions


def _load(data: bytes):
    return FeatherLoader().load(Source(data, "f.feather", "id"), FeatherOptions())


def test_reads_types():
    """Ints, floats, dates and missing text keep their type, with nothing to report.

    Case: the Arrow table is read but types are lost on the way to pandas.
    Affects: core/io/feather_loader.py, core/io/arrow_table.py
    How: reads ``PEOPLE_FEATHER`` and checks columns, dtypes, missing count, format, details.
    """
    dataset = _load(PEOPLE_FEATHER)
    df = dataset.df
    assert list(df.columns) == ["id", "name", "score", "born"]
    assert pd.api.types.is_integer_dtype(df["id"])
    assert pd.api.types.is_float_dtype(df["score"])
    assert pd.api.types.is_datetime64_any_dtype(df["born"])
    assert df["name"].isna().sum() == 1
    assert dataset.format == "Feather"
    assert dataset.details == {}


@pytest.mark.parametrize("compression", ["lz4", "zstd", "uncompressed"])
def test_reads_compressed_files(compression):
    """Every compression Feather V2 allows is read.

    Case: a file compressed with zstd (the pandas default differs by version) fails to open.
    Affects: core/io/feather_loader.py
    How: writes ``people()`` with each compression and checks the row count.
    """
    assert _load(files.feather_file(people(), compression=compression)).n_rows == 3


# pyarrow warns that V1 is deprecated, but reads it.
@pytest.mark.filterwarnings("ignore:Feather V1:DeprecationWarning")
def test_reads_feather_v1():
    """Old Feather V1 files are still read.

    Case: only the V2 (Arrow IPC file) reader is used and V1 files are reported as damaged.
    Affects: core/io/feather_loader.py
    How: writes ``people()`` as V1 and checks the columns.
    """
    dataset = _load(files.feather_file(people(), version=1))
    assert list(dataset.df.columns) == ["id", "name", "score", "born"]


def test_details_list_converted_columns():
    """Columns converted on the way (here a list column) are listed in ``details``.

    Case: the Feather loader drops the details of ``to_frame``, so the user is not told.
    Affects: core/io/feather_loader.py, core/io/arrow_table.py
    How: reads a file with a nested column and compares the details.
    """
    data = files.feather_file(pa.table({"x": [1], "tags": pa.array([["a", "b"]])}))
    assert _load(data).details == {"nested columns": "tags (shown as JSON text)"}


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "'f.feather' is empty."),
        (b"a,b\n1,2\n", "'f.feather' is not a Feather (Arrow IPC) file."),
        (b"PAR1 rest of the file PAR1", "'f.feather' is not a Feather (Arrow IPC) file."),
        (
            files.feather_file(pa.table({"x": pa.array([], pa.int64())})),
            "'f.feather' has no data rows.",
        ),
        (files.feather_file(pa.table({})), "'f.feather' has no columns."),
    ],
    ids=["empty", "csv-renamed", "parquet-renamed", "no-rows", "no-columns"],
)
def test_read_errors(data, message):
    """Files that hold no table give a message the user can act on, shown as is.

    Case: pyarrow's ``ArrowInvalid`` reaches the user, or an empty table reaches the Profile.
    Affects: core/io/feather_loader.py, core/io/base.py (ReadError)
    How: reads each bad file and compares the full message.
    """
    with pytest.raises(ReadError) as error:
        _load(data)
    assert str(error.value) == message


@pytest.mark.parametrize(
    "data",
    [
        PEOPLE_FEATHER[: len(PEOPLE_FEATHER) // 2],
        # Valid magic bytes at both ends, garbage in between.
        PEOPLE_FEATHER[:8] + b"\xff" * (len(PEOPLE_FEATHER) - 16) + PEOPLE_FEATHER[-8:],
    ],
    ids=["truncated", "garbage-inside"],
)
def test_damaged_file_is_a_read_error(data):
    """A file that starts like Feather but is broken is reported as damaged.

    Case: a damaged file passes the magic-bytes check and pyarrow's error (or a crash) follows.
    Affects: core/io/feather_loader.py
    How: reads a truncated file and one with garbage between valid ends.
    """
    with pytest.raises(ReadError, match=r"'f\.feather' is damaged"):
        _load(data)
