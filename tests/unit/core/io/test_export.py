"""Unit tests of ``core/io/export.py``: writing a DataFrame to CSV, Parquet and Feather.

Each file is read back here with pandas / pyarrow directly, to check what the writer put in
it. That the app's own loaders read it back as the same dataset is tested in
``tests/integration/test_export_roundtrip.py``.
"""

import io

import pandas as pd
import pyarrow.feather as pa_feather
import pyarrow.parquet as pq
import pytest

from litmetrics.core.io import export
from litmetrics.core.io.base import BINARY_AS_SIZE, LoadedDataset

# Readers of each export format, to look inside the bytes written.
_READ = {
    "CSV": lambda data: pd.read_csv(io.BytesIO(data)),
    "Parquet": lambda data: pq.read_table(io.BytesIO(data)).to_pandas(),
    "Feather": lambda data: pa_feather.read_table(io.BytesIO(data)).to_pandas(),
}

_NAMES = [fmt.name for fmt in export.EXPORT_FORMATS]


def test_every_export_format_has_a_reader_here():
    """The tests below cover every export format offered to the user.

    Case: a format is added to ``EXPORT_FORMATS`` and silently left out of these tests.
    Affects: core/io/export.py (EXPORT_FORMATS)
    How: compares the names of ``EXPORT_FORMATS`` with the readers of this module.
    """
    assert set(_NAMES) == set(_READ)


@pytest.mark.parametrize("name", _NAMES)
def test_index_is_not_written(name):
    """Only the columns are written, not the row index.

    Case: the index (left over from filtering rows) comes back as an extra unnamed column.
    Affects: core/io/export.py (export)
    How: exports a 2-row frame with index 10, 20 and checks the columns read back.
    """
    df = pd.DataFrame({"a": [1, 2]}, index=[10, 20])
    assert list(_READ[name](export.export(df, name)).columns) == ["a"]


@pytest.mark.parametrize("name", _NAMES)
def test_empty_dataset_keeps_its_columns(name):
    """A dataset without rows is written with its column names.

    Case: an empty table (a SQLite table without rows) raises or loses its header.
    Affects: core/io/export.py (export)
    How: exports a 0-row frame with two columns and reads back its shape and names.
    """
    df = pd.DataFrame({"a": pd.Series([], dtype="float64"), "b": pd.Series([], dtype="str")})
    back = _READ[name](export.export(df, name))
    assert list(back.columns) == ["a", "b"]
    assert len(back) == 0


@pytest.mark.parametrize("name", ["Parquet", "Feather"])
def test_arrow_formats_keep_the_dtypes_of_the_schema(name):
    """Parquet and Feather keep categories, booleans, UTC dates and missing values.

    Case: the dtypes given at load time (``schema.apply_schema``) are lost, so the exported
    file has to be typed again, or dates lose their time zone.
    Affects: core/io/export.py (_write_parquet, _write_feather)
    How: exports a frame with one column of each dtype and compares the dtypes read back.
    """
    df = pd.DataFrame(
        {
            "flag": [True, False],
            "kind": pd.Categorical(["x", "y"]),
            "when": pd.to_datetime(["2020-01-01", "2021-06-30"], utc=True),
            "n": pd.array([1, None], dtype="Int64"),
        }
    )
    back = _READ[name](export.export(df, name))
    assert back["flag"].dtype == bool
    assert isinstance(back["kind"].dtype, pd.CategoricalDtype)
    assert str(back["when"].dt.tz) == "UTC"
    assert back["n"].isna().tolist() == [False, True]
    pd.testing.assert_frame_equal(back, df, check_dtype=False, check_categorical=False)


def test_csv_is_utf8_with_comma_and_point():
    """CSV is written in UTF-8, with ``,`` between fields and ``.`` in decimals.

    Case: accents written in another encoding, or the locale's decimal comma, so the file is
    misread by other tools.
    Affects: core/io/export.py (_write_csv)
    How: exports a row with an accented word and a float, and checks the exact bytes.
    """
    df = pd.DataFrame({"city": ["A Coruña"], "price": [2.5]})
    assert export.export(df, "CSV") == "city,price\nA Coruña,2.5\n".encode()


def test_export_does_not_change_the_dataset():
    """The DataFrame exported is left as it was.

    Case: the shared, cached dataset is modified while exporting (index reset, dtypes cast),
    and every page shows the change.
    Affects: core/io/export.py (export)
    How: exports a copy of a typed frame to every format and compares it with the original.
    """
    df = pd.DataFrame({"kind": pd.Categorical(["x", "y"]), "v": [1.0, 2.0]}, index=[5, 6])
    before = df.copy()
    for name in _NAMES:
        export.export(df, name)
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize(
    ("source", "part", "name", "expected"),
    [
        ("sales.csv", None, "Parquet", "sales.parquet"),
        ("shop.db", "orders", "CSV", "shop_orders.csv"),
        ("report.v2.xlsx", "Sheet 1", "Feather", "report.v2_Sheet 1.feather"),
        ("data", None, "CSV", "data.csv"),
        ("", None, "CSV", "dataset.csv"),
    ],
    ids=["plain-file", "table-of-a-database", "dots-in-name", "no-extension", "no-name"],
)
def test_export_file_name(source, part, name, expected):
    """The download is named after the source, its table and the new format.

    Case: two tables of one database download with the same name and overwrite each other, a
    name with dots loses part of it, or an empty name gives a file called ``.csv``.
    Affects: core/io/export.py (export_file_name)
    How: builds the name for each case and compares it.
    """
    assert export.export_file_name(source, part, name) == expected


def test_unknown_format_raises():
    """Asking for a format that cannot be written is a ``ValueError`` naming the known ones.

    Case: a typo in a format name writes nothing and the user downloads an empty file.
    Affects: core/io/export.py (export_format, export)
    How: exports to "XML" and checks the error and its message.
    """
    with pytest.raises(ValueError, match="Known: CSV, Parquet, Feather"):
        export.export(pd.DataFrame({"a": [1]}), "XML")


def _dataset(df: pd.DataFrame, **fields) -> LoadedDataset:
    """A loaded dataset of ``df`` read from ``data.csv``, with ``unparsed`` / ``changed``."""
    return LoadedDataset(df, "data.csv", "CSV", **fields)


def test_clean_dataset_has_no_warnings():
    """A small dataset read without losses exports without any warning.

    Case: a warning shown on every export teaches the user to ignore it.
    Affects: core/io/export.py (export_warnings)
    How: asks the warnings of a 2-row dataset with nothing unparsed or changed, for each format.
    """
    dataset = _dataset(pd.DataFrame({"a": [1, 2]}))
    for name in _NAMES:
        assert export.export_warnings(dataset, "data.csv", name, slow_rows=10) == ()


@pytest.mark.parametrize(
    ("values", "unparsed", "expected"),
    [
        (
            [1.0, None],
            {"price": 1},
            "'price': 1 value that could not be read as numbers will be empty.",
        ),
        (
            pd.to_datetime(["2020-01-01", None]),
            {"price": 1_500},
            "'price': 1,500 values that could not be read as dates will be empty.",
        ),
    ],
    ids=["one-number", "many-dates"],
)
def test_unparsed_values_are_warned(values, unparsed, expected):
    """Values that became missing when the column was typed are announced, per column.

    Case: an export silently writes "12 kg" or "N/A" as empty cells, and the user thinks the
    file holds their data.
    Affects: core/io/export.py (export_warnings)
    How: gives a numeric or a date column with ``unparsed`` counts and reads the message.
    """
    dataset = _dataset(pd.DataFrame({"price": values}), unparsed=unparsed)
    assert export.export_warnings(dataset, "data.csv", "Parquet", slow_rows=10) == (expected,)


def test_columns_with_the_same_change_are_warned_together():
    """Columns changed the same way by the loader share one message; zero counts say nothing.

    Case: one line per column floods the narrow sidebar, or a column with 0 unparsed values is
    reported.
    Affects: core/io/export.py (export_warnings)
    How: gives two binary columns, another change and a 0 unparsed count, and reads messages.
    """
    df = pd.DataFrame({"img": ["<2 bytes>"], "doc": ["<9 bytes>"], "t": ["09:30:00"], "n": [1]})
    dataset = _dataset(
        df,
        unparsed={"n": 0},
        changed={"img": BINARY_AS_SIZE, "t": "times of day kept as text", "doc": BINARY_AS_SIZE},
    )
    assert export.export_warnings(dataset, "data.csv", "CSV", slow_rows=10) == (
        f"'img', 'doc': {BINARY_AS_SIZE}, not as in the original file.",
        "'t': times of day kept as text, not as in the original file.",
    )


@pytest.mark.parametrize(
    ("file_name", "warned"),
    [
        ("shop_orders.csv", False),
        ("shop_Sheet 1.csv", False),
        ("shop_a:b.csv", True),
        ("shop_x/y.csv", True),
        ('shop_odd "name".csv', True),
    ],
    ids=["plain", "space", "colon", "slash", "quotes"],
)
def test_unsafe_file_name_is_warned(file_name, warned):
    """A file name with characters some systems refuse is announced; spaces are fine.

    Case: a table called ``a:b`` downloads under a name the browser changes or Windows
    refuses, and the user cannot find the file.
    Affects: core/io/export.py (export_warnings)
    How: asks the warnings for each file name and checks whether the file name is quoted.
    """
    dataset = _dataset(pd.DataFrame({"a": [1]}))
    messages = export.export_warnings(dataset, file_name, "CSV", slow_rows=10)
    assert any(file_name in message for message in messages) is warned


@pytest.mark.parametrize(
    ("name", "rows", "warned"),
    [("CSV", 9, False), ("CSV", 10, True), ("Parquet", 10, False), ("Feather", 10, False)],
    ids=["csv-below", "csv-at-limit", "parquet", "feather"],
)
def test_slow_format_is_warned_from_the_limit(name, rows, warned):
    """Only a slow format (CSV), and only from ``slow_rows`` rows, gets the size warning.

    Case: a 3 M-row CSV takes 20 s with no sign of progress and the user clicks again; or the
    warning shows for fast formats or small datasets.
    Affects: core/io/export.py (export_warnings, ExportFormat.slow)
    How: asks the warnings with ``slow_rows=10`` around the limit and checks the message.
    """
    dataset = _dataset(pd.DataFrame({"a": range(rows)}))
    messages = export.export_warnings(dataset, "data.csv", name, slow_rows=10)
    expected = (f"Writing {rows:,} rows as CSV takes a while: Parquet or Feather are much faster.",)
    assert messages == (expected if warned else ())
