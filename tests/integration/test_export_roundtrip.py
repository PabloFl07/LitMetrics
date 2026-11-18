"""Integration tests of exporting: a file read by the app, exported, and read again.

Runs the real data path twice: bytes -> ``formats.detect_options`` -> loader ->
``schema.prepare_dataset`` -> ``export.export`` -> the same steps on the exported bytes. The
unit tests of ``core/io/export.py`` look inside the written file; these check that the app
reads it back as the same dataset, which is what the user expects from an export.
"""

import pandas as pd
import pytest
from fixtures import files, samples

from litmetrics.core.io import export, formats
from litmetrics.core.io.base import LoadedDataset, Source
from litmetrics.core.schema import Schema, prepare_dataset


def _read(name: str, data: bytes, part: str | None = None) -> tuple[LoadedDataset, Schema]:
    """What the Load page does with an upload, reading ``part`` of a file with several."""
    fmt = formats.format_for(name)
    options = fmt.detect_options(data)
    if part is not None:
        options = fmt.parts.select(options, part)
    return prepare_dataset(fmt.loader.load(Source(data, name, "id"), options))


def _comparable(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` with its dates in one resolution: CSV gives seconds, Arrow microseconds, and
    the resolution is not something the user sees."""
    dates = df.select_dtypes("datetime").columns
    return df.astype(dict.fromkeys(dates, "datetime64[us]")).reset_index(drop=True)


# One file of each readable format, with dates, missing values, text and numbers.
_SOURCES = [
    pytest.param("sales.csv", samples.SALES_EXPORT_CSV, None, id="csv-cp1252-semicolon"),
    pytest.param("quality.csv", samples.QUALITY_CSV, None, id="csv-quality"),
    pytest.param("people.parquet", samples.PEOPLE_PARQUET, None, id="parquet"),
    pytest.param("people.feather", samples.PEOPLE_FEATHER, None, id="feather"),
    pytest.param("book.xlsx", samples.WORKBOOK, "Sales", id="excel-sheet"),
    pytest.param("show.db", samples.SQLITE_SHOWCASE, "people", id="sqlite-table"),
]


@pytest.mark.parametrize("target", [fmt.name for fmt in export.EXPORT_FORMATS])
@pytest.mark.parametrize(("name", "data", "part"), _SOURCES)
def test_exported_file_reads_back_as_the_same_dataset(name, data, part, target):
    """An exported dataset opens in the app with the same columns, values and types.

    Case: the export drops the index into a column, loses missing values, turns dates or
    numbers into text, or writes a CSV the sniffer misreads (separator, decimal, encoding).
    Affects: core/io/export.py, core/io/formats.py, every loader, core/schema.py
    How: reads each sample file, exports it to ``target``, reads the export with the app's
    loader and compares the frames and the semantic type of each column.
    """
    dataset, schema = _read(name, data, part)
    exported = export.export(dataset.df, target)
    back, back_schema = _read(export.export_file_name(name, part, target), exported)

    assert [(c.name, c.semantic_type) for c in back_schema.columns] == [
        (c.name, c.semantic_type) for c in schema.columns
    ]
    pd.testing.assert_frame_equal(
        _comparable(back.df), _comparable(dataset.df), check_dtype=False, check_categorical=False
    )


@pytest.mark.parametrize(
    ("name", "data", "part", "column"),
    [
        pytest.param(
            "people.parquet",
            samples.people_with_photo_parquet(),
            None,
            "photo",
            id="parquet-binary",
        ),
        pytest.param(
            "pics.db",
            files.sqlite("CREATE TABLE t (id INT, img BLOB)", "INSERT INTO t VALUES (1, x'00ff')"),
            "t",
            "img",
            id="sqlite-blob",
        ),
        pytest.param(
            "prices.csv",
            b"price\n" + b"".join(b"%d\n" % i for i in range(40)) + b"12 kg\n",
            None,
            "price",
            id="csv-unparsed-number",
        ),
    ],
)
def test_export_warns_about_what_the_loader_changed(name, data, part, column):
    """Before an export, the user is told which column will not be as in the original file.

    Case: the loader records the change but ``prepare_dataset`` drops it on the way, so the
    export of binary data (written as "<2 bytes>") or of unreadable numbers (written empty)
    gives no warning.
    Affects: core/io/sqlite_loader.py, core/io/parquet_loader.py, core/io/arrow_table.py,
    core/schema.py (prepare_dataset), core/io/export.py (export_warnings)
    How: reads each file as the Load page does and checks a warning quotes the column.
    """
    dataset, _ = _read(name, data, part)
    messages = export.export_warnings(dataset, "out.parquet", "Parquet", slow_rows=10**9)
    assert any(f"'{column}'" in message for message in messages), messages
