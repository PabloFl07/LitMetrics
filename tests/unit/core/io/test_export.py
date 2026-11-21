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
    How: asks the warnings and the time notice of a 2-row dataset with nothing unparsed or
    changed, for each format.
    """
    dataset = _dataset(pd.DataFrame({"a": [1, 2]}))
    assert export.export_warnings(dataset, "data.csv") == ()
    for name in _NAMES:
        assert export.slow_export(dataset, name, slow_seconds=10, timeout_seconds=180) is None


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
    assert export.export_warnings(dataset, "data.csv") == (expected,)


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
    assert export.export_warnings(dataset, "data.csv") == (
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
    messages = export.export_warnings(dataset, file_name)
    assert any(file_name in message for message in messages) is warned


def test_csv_estimate_counts_floats_as_the_slow_columns():
    """A float column is estimated slower to write as CSV than an integer or a text one.

    Case: an estimate from the row count alone says a wide table of floats is fast, and the
    user waits minutes with no warning (floats are written ~5 times slower).
    Affects: core/io/export.py (csv_seconds)
    How: estimates frames of the same length with a float, an integer and a text column, and
    one with two float columns.
    """
    rows = 1_000

    def seconds(values) -> float:
        return export.csv_seconds(pd.DataFrame({"a": values}))

    floats, ints, text = seconds([0.5] * rows), seconds([1] * rows), seconds(["x"] * rows)
    assert floats > ints * 3 and floats > text * 3
    two = export.csv_seconds(pd.DataFrame({"a": [0.5] * rows, "b": [0.5] * rows}))
    assert two == pytest.approx(2 * floats)


@pytest.mark.parametrize(
    ("name", "slow_factor", "expected"),
    [
        ("CSV", 1.01, None),
        ("CSV", 1.0, "slow"),
        ("Parquet", 0.0, None),
        ("Feather", 0.0, None),
    ],
    ids=["csv-below", "csv-at-limit", "parquet", "feather"],
)
def test_only_a_slow_format_past_the_limit_is_announced(name, slow_factor, expected):
    """Only a slow format (CSV), and only from ``slow_seconds``, gets the time notice.

    Case: the notice shows for fast formats or quick exports (and is learnt to be ignored),
    or a long CSV export gets none and the user clicks Download again and again.
    Affects: core/io/export.py (slow_export, csv_seconds, ExportFormat.slow)
    How: puts ``slow_seconds`` just above and at the estimate of a small frame.
    """
    dataset = _dataset(pd.DataFrame({"a": [0.5] * 100}))
    estimate = export.csv_seconds(dataset.df)
    slow = export.slow_export(
        dataset, name, slow_seconds=estimate * slow_factor, timeout_seconds=estimate * 10
    )
    assert (slow is not None) == (expected is not None)


@pytest.mark.parametrize(
    ("timeout_factor", "may_fail", "says"),
    [
        (3.0, False, "the button spins until the file is ready"),
        (1.5, True, "the download may fail"),
        (0.9, True, "the download will probably fail"),
    ],
    ids=["slow", "near-timeout", "past-timeout"],
)
def test_slow_export_says_when_the_browser_may_give_up(timeout_factor, may_fail, says):
    """Close to the time the browser waits for the file, the notice says the download may fail.

    Case: a CSV export longer than Streamlit's 180 s download timeout is announced as merely
    slow; the user waits minutes and gets "Failed to generate file" with no explanation.
    Affects: core/io/export.py (slow_export)
    How: sets the timeout around the estimate of a small frame and reads the notice.
    """
    dataset = _dataset(pd.DataFrame({"a": [0.5] * 100}))
    estimate = export.csv_seconds(dataset.df)
    slow = export.slow_export(
        dataset, "CSV", slow_seconds=0, timeout_seconds=estimate * timeout_factor
    )
    assert slow is not None
    assert slow.may_fail is may_fail
    assert says in slow.message
    assert "100 rows and 1 column" in slow.message
    assert "Parquet or Feather" in slow.message


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(0.8, "5 s"), (21.0, "25 s"), (59.5, "60 s"), (95.0, "2 min"), (250.0, "4 min")],
    ids=["minimum", "rounded-up", "under-a-minute", "minutes-rounded", "minutes"],
)
def test_estimate_is_rounded_for_the_user(seconds, text):
    """Estimates are shown rounded: seconds up to the next 5, then whole minutes.

    Case: "about 0.83 s" or "about 250.0 s", or seconds rounded down so the user thinks the
    download hangs before the time announced.
    Affects: core/io/export.py (duration_text)
    How: formats estimates around each step.
    """
    assert export.duration_text(seconds) == text


def test_report_lists_each_lost_value_by_row_then_column():
    """The problem report has one line per lost value, ordered as the user reads the file.

    Case: rows are numbered from 0 (the user looks one row too high), or lines of one row
    are scattered by column, or a date column is reported as expecting a number.
    Affects: core/io/export.py (unparsed_report, affected_rows)
    How: gives lost values at positions 4 and 0 of ``price`` and 4 of ``date`` (declared after
    ``price`` in the frame) and reads the whole report.
    """
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"] * 5), "price": [1.0] * 5})
    dataset = _dataset(
        df,
        unparsed_values={
            "price": pd.Series(["abc", "12 kg"], index=[4, 0]),
            "date": pd.Series(["someday"], index=[4]),
        },
    )
    report = export.unparsed_report(dataset)
    assert list(report.columns) == list(export.REPORT_COLUMNS)
    assert report.to_dict("records") == [
        {"row": 1, "column": "price", "value": "12 kg", "expected": "number"},
        {"row": 5, "column": "date", "value": "someday", "expected": "date"},
        {"row": 5, "column": "price", "value": "abc", "expected": "number"},
    ]
    assert export.affected_rows(report) == 2


def test_report_of_a_clean_dataset_is_empty_with_its_columns():
    """A dataset read without losses has an empty report that can still be written.

    Case: ``pd.concat`` of no parts raises, or a column read without losses (an empty Series)
    adds lines, so the dialog shows a report with nothing in it.
    Affects: core/io/export.py (unparsed_report, write_report)
    How: gives a frame with one column whose lost values are empty and reads the report and
    its CSV.
    """
    dataset = _dataset(
        pd.DataFrame({"a": [1.0]}), unparsed_values={"a": pd.Series([], dtype="str")}
    )
    report = export.unparsed_report(dataset)
    assert report.empty
    assert export.write_report(report) == b"row,column,value,expected\n"


def test_report_file_name_follows_the_export():
    """The report is named after the exported file, always as CSV.

    Case: the report of ``shop_orders.parquet`` downloads as ``shop_orders.parquet`` and
    replaces the export, or gets a name that does not say which export it belongs to.
    Affects: core/io/export.py (report_file_name)
    How: names the reports of a Parquet and a CSV export.
    """
    assert export.report_file_name("shop_orders.parquet") == "shop_orders_problems.csv"
    assert export.report_file_name("data.csv") == "data_problems.csv"
