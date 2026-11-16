import io
import warnings

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from litmetrics.core.io.base import ReadError, Source
from litmetrics.core.io.feather_loader import FeatherLoader, FeatherOptions, detect_options


def _feather(frame: pd.DataFrame | pa.Table, **kwargs) -> bytes:
    table = frame if isinstance(frame, pa.Table) else pa.Table.from_pandas(frame)
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        # Writing Feather V1 is deprecated; some tests still need such a file.
        warnings.simplefilter("ignore", DeprecationWarning)
        feather.write_feather(table, buffer, **kwargs)
    return buffer.getvalue()


PEOPLE_FRAME = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "name": ["Ana", "Luis", None],
        "score": [1.5, None, 3.0],
        "born": pd.to_datetime(["2000-01-02", "1999-03-04", None]),
    }
)
PEOPLE = _feather(PEOPLE_FRAME)


def _load(data: bytes):
    return FeatherLoader().load(Source(data, "f.feather", "id"), FeatherOptions())


def test_detect_options_has_nothing_to_detect():
    assert detect_options(PEOPLE) == FeatherOptions()


def test_reads_types():
    dataset = _load(PEOPLE)
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
    assert _load(_feather(PEOPLE_FRAME, compression=compression)).n_rows == 3


def test_reads_feather_v1():
    with warnings.catch_warnings():
        # pyarrow warns that V1 is deprecated, but reads it.
        warnings.simplefilter("ignore", DeprecationWarning)
        dataset = _load(_feather(PEOPLE_FRAME, version=1))
    assert list(dataset.df.columns) == ["id", "name", "score", "born"]


def test_details_list_converted_columns():
    data = _feather(pa.table({"x": [1], "tags": pa.array([["a", "b"]])}))
    assert _load(data).details == {"nested columns": "tags (shown as JSON text)"}


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "'f.feather' is empty."),
        (b"a,b\n1,2\n", "'f.feather' is not a Feather (Arrow IPC) file."),
        # A Parquet file renamed to .feather.
        (b"PAR1 rest of the file PAR1", "'f.feather' is not a Feather (Arrow IPC) file."),
        (_feather(pa.table({"x": pa.array([], pa.int64())})), "'f.feather' has no data rows."),
        (_feather(pa.table({})), "'f.feather' has no columns."),
    ],
)
def test_read_errors(data, message):
    with pytest.raises(ReadError) as error:
        _load(data)
    assert str(error.value) == message


def test_truncated_file_is_a_read_error():
    with pytest.raises(ReadError, match=r"'f\.feather' is damaged"):
        _load(PEOPLE[: len(PEOPLE) // 2])


def test_damaged_file_is_a_read_error():
    # Valid magic bytes at both ends, garbage in between.
    data = PEOPLE[:8] + b"\xff" * (len(PEOPLE) - 16) + PEOPLE[-8:]
    with pytest.raises(ReadError, match=r"'f\.feather' is damaged"):
        _load(data)
