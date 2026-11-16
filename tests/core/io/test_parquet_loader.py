import io

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from litmetrics.core.io.base import ReadError, Source
from litmetrics.core.io.parquet_loader import ParquetLoader, ParquetOptions, detect_options


def _parquet(frame: pd.DataFrame | pa.Table) -> bytes:
    table = frame if isinstance(frame, pa.Table) else pa.Table.from_pandas(frame)
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


PEOPLE = _parquet(
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Ana", "Luis", None],
            "score": [1.5, None, 3.0],
            "born": pd.to_datetime(["2000-01-02", "1999-03-04", None]),
        }
    )
)


def _load(data: bytes):
    return ParquetLoader().load(Source(data, "f.parquet", "id"), ParquetOptions())


def test_detect_options_has_nothing_to_detect():
    assert detect_options(PEOPLE) == ParquetOptions()


def test_reads_types():
    dataset = _load(PEOPLE)
    df = dataset.df
    assert list(df.columns) == ["id", "name", "score", "born"]
    assert pd.api.types.is_integer_dtype(df["id"])
    assert pd.api.types.is_float_dtype(df["score"])
    assert pd.api.types.is_datetime64_any_dtype(df["born"])
    assert df["name"].isna().sum() == 1
    assert dataset.format == "Parquet"
    assert dataset.details == {}


def test_details_list_converted_columns():
    data = _parquet(pa.table({"x": [1], "raw": pa.array([b"abc"])}))
    assert _load(data).details == {"binary columns": "raw (shown as their size)"}


def test_reads_every_row_group():
    data = io.BytesIO()
    pq.write_table(pa.table({"x": list(range(1000))}), data, row_group_size=100)
    assert _load(data.getvalue()).n_rows == 1000


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "'f.parquet' is empty."),
        (b"a,b\n1,2\n", "'f.parquet' is not a Parquet file."),
        # A Feather (Arrow IPC) file renamed to .parquet.
        (b"ARROW1\x00\x00 rest of the file", "'f.parquet' is not a Parquet file."),
        (_parquet(pa.table({"x": pa.array([], pa.int64())})), "'f.parquet' has no data rows."),
        (_parquet(pa.table({})), "'f.parquet' has no columns."),
    ],
)
def test_read_errors(data, message):
    with pytest.raises(ReadError) as error:
        _load(data)
    assert str(error.value) == message


def test_damaged_file_is_a_read_error():
    # Valid magic bytes at both ends, garbage in between.
    data = PEOPLE[:4] + b"\xff" * (len(PEOPLE) - 8) + PEOPLE[-4:]
    with pytest.raises(ReadError, match=r"'f\.parquet' is damaged"):
        _load(data)


def test_truncated_file_is_a_read_error():
    # The end of the footer was cut and the magic bytes put back.
    data = PEOPLE[: len(PEOPLE) // 2] + b"PAR1"
    with pytest.raises(ReadError, match=r"'f\.parquet' is damaged"):
        _load(data)
