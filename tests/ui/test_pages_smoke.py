"""Smoke tests: the app starts and every page renders without exceptions.

Pages are functions registered with st.navigation, which ``AppTest.switch_page`` cannot open,
so each page function runs on its own through ``AppTest.from_function``. The scripts passed to
it must be self-contained: they import what they need inside the function. Never import
``litmetrics.ui.pages`` at module level here: ``streamlit_echarts`` fails outside a running app.
"""

import sqlite3
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "src" / "litmetrics" / "app.py"
TIMEOUT = 30

# (module in litmetrics.ui.pages, entry point). Add every new page here.
PAGES = [
    ("home", "home_page"),
    ("load", "load_page"),
    ("profile", "profile_page"),
    ("visualize", "visualize_page"),
]

# 60 rows: an int column, a categorical one, and a float column with missing values.
CSV = b"id,age,city,score\n" + b"".join(
    b"%d,%d,%s,%s\n"
    % (i, 20 + i % 30, (b"Lugo", b"Vigo", b"Ourense")[i % 3], b"" if i % 7 == 0 else b"%.1f" % i)
    for i in range(60)
)


def _run_page(module: str, func: str, with_dataset: bool) -> None:
    """Script for AppTest: set up the state, optionally load a small dataset, draw one page."""
    import importlib

    from litmetrics.core.io.base import Source, compute_dataset_id
    from litmetrics.core.io.csv_loader import CsvLoader
    from litmetrics.core.io.sniffing import detect_options
    from litmetrics.core.schema import prepare_dataset
    from litmetrics.ui import state

    state.init_state()
    if with_dataset and state.get_dataset() is None:
        # Missing values, a duplicate row, a constant column and two correlated columns.
        data = (
            b"id,age,city,score,const\n1,20,Lugo,1.5,7\n2,21,Vigo,,7\n3,22,Ourense,4.5,7\n"
            b"4,23,Lugo,6.0,7\n5,24,Vigo,7.5,7\n5,24,Vigo,7.5,7\n"
        )
        options = detect_options(data)
        dataset, schema = prepare_dataset(
            CsvLoader().load(Source(data, "sample.csv", "s"), options)
        )
        state.set_dataset(dataset, compute_dataset_id(data, options), "sample")
        state.set_schema(schema)
    page = importlib.import_module("litmetrics.ui.pages." + module)
    getattr(page, func)()


def _errors(at: AppTest) -> list[str]:
    return [str(e.value) for e in at.exception]


def test_app_starts():
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    assert not at.exception, _errors(at)


@pytest.mark.parametrize(("module", "func"), PAGES)
@pytest.mark.parametrize("with_dataset", [False, True], ids=["no-data", "data"])
def test_page_renders(module, func, with_dataset):
    at = AppTest.from_function(
        _run_page, args=(module, func, with_dataset), default_timeout=TIMEOUT
    ).run()
    assert not at.exception, _errors(at)


def test_load_page_reads_uploaded_csv():
    at = AppTest.from_function(
        _run_page, args=("load", "load_page", False), default_timeout=TIMEOUT
    ).run()
    at.file_uploader[0].upload("people.csv", CSV, "text/csv").run()
    assert not at.exception, _errors(at)
    assert [m.value for m in at.metric][:3] == ["60", "4", "CSV"]
    assert at.session_state["dataset"].name == "people.csv"
    assert at.session_state["schema"] is not None


def test_load_page_shows_read_error():
    at = AppTest.from_function(
        _run_page, args=("load", "load_page", False), default_timeout=TIMEOUT
    ).run()
    at.file_uploader[0].upload("empty.csv", b"   \n", "text/csv").run()
    assert not at.exception, _errors(at)
    assert at.error[0].value == "'empty.csv' is empty."
    assert at.session_state["dataset"] is None


def _run_load_counting_reads() -> None:
    """Load page with ``cache.load_dataset`` counting its calls in ``session_state["reads"]``."""
    import streamlit as st

    from litmetrics.ui import cache, state
    from litmetrics.ui.pages.load import load_page

    state.init_state()
    st.session_state.setdefault("reads", 0)
    read = cache.load_dataset

    def counting(*args, **kwargs):
        st.session_state["reads"] += 1
        return read(*args, **kwargs)

    cache.load_dataset = counting
    try:
        load_page()
    finally:
        cache.load_dataset = read


def test_load_page_does_not_read_a_failed_upload_again():
    at = AppTest.from_function(_run_load_counting_reads, default_timeout=TIMEOUT).run()
    at.file_uploader[0].upload("empty.csv", b"   \n", "text/csv").run()
    at.run()
    at.run()
    assert not at.exception, _errors(at)
    assert at.error[0].value == "'empty.csv' is empty."
    assert at.session_state["reads"] == 1


def test_load_page_shows_how_the_file_was_read():
    at = AppTest.from_function(
        _run_page, args=("load", "load_page", False), default_timeout=TIMEOUT
    ).run()
    data = b"precio;ciudad\n" + b"".join(b"%d,5;Lugo\n" % i for i in range(30))
    at.file_uploader[0].upload("eu.csv", data, "text/csv").run()
    assert not at.exception, _errors(at)
    caption = " ".join(c.value for c in at.caption)
    assert "Semicolon (;)" in caption and "Comma (,)" in caption


def _parquet() -> bytes:
    """The rows of ``CSV`` as a Parquet file, plus a binary column."""
    import io

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(pd.read_csv(io.BytesIO(CSV)), preserve_index=False)
    table = table.append_column("photo", pa.array([b"\x00" * 10] * table.num_rows))
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def test_load_page_reads_uploaded_parquet():
    at = AppTest.from_function(
        _run_page, args=("load", "load_page", False), default_timeout=TIMEOUT
    ).run()
    at.file_uploader[0].upload("people.parquet", _parquet(), "application/octet-stream").run()
    assert not at.exception, _errors(at)
    assert [m.value for m in at.metric][:3] == ["60", "5", "Parquet"]
    assert "photo (shown as their size)" in " ".join(c.value for c in at.caption)


def test_load_page_reads_uploaded_feather():
    import io

    import pandas as pd

    buffer = io.BytesIO()
    pd.read_csv(io.BytesIO(CSV)).to_feather(buffer)
    at = AppTest.from_function(
        _run_page, args=("load", "load_page", False), default_timeout=TIMEOUT
    ).run()
    at.file_uploader[0].upload("people.feather", buffer.getvalue(), "application/octet-stream")
    at.run()
    assert not at.exception, _errors(at)
    assert [m.value for m in at.metric][:3] == ["60", "4", "Feather"]


# Profile: tabs, views of the Columns tab and the column inspector.

PROFILE_TABS = [
    ":material/warning: Issues",
    ":material/view_column: Columns",
    ":material/block: Missing values",
    ":material/link: Correlation",
]
COLUMN_VIEWS = ["Overview", "Numeric", "Categorical", "Dates", "Text & ids"]


def _run_profile(
    tab: str,
    view: str | None,
    inspect: str | None,
    chart: str | None = None,
    skipped: tuple[str, ...] = (),
    extra_numeric: int = 0,
) -> None:
    """Script for AppTest: load a dataset with every semantic type and draw part of Profile.

    ``tab`` and ``view`` are set through the widget keys before the page draws them. With
    ``inspect`` (a column name), only the inspector of that column is drawn, since AppTest
    cannot select a row; ``chart`` picks the chart of a numeric column. ``skipped`` columns
    are left out of the profile. ``extra_numeric`` adds that many random numeric columns.
    """
    import importlib

    import numpy as np
    import pandas as pd

    from litmetrics.core.io.base import LoadedDataset
    from litmetrics.core.schema import prepare_dataset
    from litmetrics.ui import state

    state.init_state()
    if state.get_dataset() is None:
        n = 60
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "id": range(n),
                "age": rng.integers(18, 80, n),
                "score": np.where(np.arange(n) % 7 == 0, np.nan, rng.normal(50, 10, n)),
                "income": rng.normal(3000, 500, n).round(),
                "city": rng.choice(["Lugo", "lugo ", "Vigo"], n),
                "active": rng.choice(["yes", "no"], n),
                "joined": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
                # Placeholder dates of database exports, outside the datetime64[ns] range.
                "valid_until": [
                    "9999-12-31",
                    "0001-01-01",
                    *pd.date_range("2025-01-01", periods=n - 2, freq="D").astype(str),
                ],
                "comment": [f"comment number {i}" for i in range(n)],
                "empty": [None] * n,
                # Two values: skewness is undefined.
                "pair": [1.0, 5.0] + [np.nan] * (n - 2),
            }
        )
        df = df.assign(**{f"x{i}": rng.normal(0, 1, n) for i in range(extra_numeric)})
        dataset, schema = prepare_dataset(LoadedDataset(df, "people.csv", "CSV"))
        state.set_dataset(dataset, "people", "people")
        state.set_schema(schema)
        state.set_skipped_columns(skipped)
    page = importlib.import_module("litmetrics.ui.pages.profile")
    import streamlit as st

    if inspect:
        if chart is not None:
            st.session_state[state.widget_key("inspect_numeric_chart")] = chart
        page._inspector(page._active(), inspect)
        return

    st.session_state[state.widget_key("profile_tab")] = tab
    if view is not None:
        st.session_state[state.widget_key("columns_view")] = view
    page.profile_page()


@pytest.mark.parametrize("tab", PROFILE_TABS)
def test_profile_tab_renders(tab):
    at = AppTest.from_function(_run_profile, args=(tab, None, None), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, _errors(at)


@pytest.mark.parametrize("view", COLUMN_VIEWS)
def test_profile_columns_view_renders(view):
    at = AppTest.from_function(
        _run_profile, args=(PROFILE_TABS[1], view, None), default_timeout=TIMEOUT
    ).run()
    assert not at.exception, _errors(at)
    assert at.dataframe, "the view shows a table"


# One column of each semantic type of the dataset of ``_run_profile``.
INSPECTED_COLUMNS = [
    "id", "age", "score", "city", "active", "joined", "valid_until", "comment", "empty", "pair",
]  # fmt: skip


@pytest.mark.parametrize("column", INSPECTED_COLUMNS)
def test_profile_inspector_renders_every_type(column):
    at = AppTest.from_function(
        _run_profile, args=(PROFILE_TABS[0], None, column), default_timeout=TIMEOUT
    ).run()
    assert not at.exception, _errors(at)


@pytest.mark.parametrize("chart", ["Histogram", "Boxplot"])
def test_profile_inspector_switches_numeric_chart(chart):
    at = AppTest.from_function(
        _run_profile, args=(PROFILE_TABS[0], None, "income", chart), default_timeout=TIMEOUT
    ).run()
    assert not at.exception, _errors(at)
    assert at.button_group[0].value == chart


def test_profile_correlation_with_many_columns_renders():
    # Enough numeric columns for the full-width heatmap.
    at = AppTest.from_function(
        _run_profile, args=(PROFILE_TABS[3], None, None, None, (), 14), default_timeout=TIMEOUT
    ).run()
    assert not at.exception, _errors(at)
    assert at.dataframe, "the most correlated pairs are shown below the heatmap"


def test_profile_only_runs_the_open_tab():
    at = AppTest.from_function(
        _run_profile, args=(PROFILE_TABS[2], None, None), default_timeout=TIMEOUT
    ).run()
    assert not at.exception, _errors(at)
    # The Correlation tab draws a radio; closed, it must not run.
    assert not at.radio


def test_profile_inspector_never_shows_nan():
    at = AppTest.from_function(
        _run_profile, args=(PROFILE_TABS[0], None, "pair"), default_timeout=TIMEOUT
    ).run()
    assert not at.exception, _errors(at)
    assert not any("nan" in m.value.lower() for m in at.metric)


@pytest.mark.parametrize("tab", PROFILE_TABS)
def test_profile_tab_renders_with_skipped_columns(tab):
    at = AppTest.from_function(
        _run_profile,
        args=(tab, None, None, None, ("income", "city")),
        default_timeout=TIMEOUT,
    ).run()
    assert not at.exception, _errors(at)
    assert any("(2 skipped)" in c.value for c in at.caption)


def _database() -> bytes:
    """A database with tables ``cities`` (3 rows) and ``people`` (60 rows, 4 columns)."""
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE TABLE cities (name TEXT)")
        connection.executemany(
            "INSERT INTO cities VALUES (?)", [("Lugo",), ("Vigo",), ("Ourense",)]
        )
        connection.execute("CREATE TABLE people (id INTEGER, age INTEGER, city TEXT, score REAL)")
        connection.executemany(
            "INSERT INTO people VALUES (?, ?, ?, ?)",
            [(i, 20 + i % 30, ("Lugo", "Vigo")[i % 2], i * 1.5) for i in range(60)],
        )
        connection.commit()
        return connection.serialize()
    finally:
        connection.close()


def _tables(at: AppTest):
    """Expander with the tables of the database (the datasets in memory may be listed too)."""
    return next(e for e in at.expander if e.label.startswith("Table: "))


def _open_table(at: AppTest, name: str) -> AppTest:
    """Click the Open button of table ``name`` in the list of tables of the Load page."""
    rows = [b for b in at.button if b.key and b.key.startswith("open_part_")]
    names = [m.value.split("**")[1] for m in _tables(at).markdown if m.value.startswith("**")]
    return rows[names.index(name)].click().run()


def test_load_page_reads_a_table_of_a_database_and_switches_table():
    at = AppTest.from_function(_run_load_counting_reads, default_timeout=TIMEOUT).run()
    at.file_uploader[0].upload("town.db", _database(), "application/octet-stream").run()
    assert not at.exception, _errors(at)
    tables = _tables(at)
    assert tables.label == "Table: cities"
    # Each table shows its size, written by the loader.
    captions = [c.value for c in tables.caption]
    assert "3 rows · 1 column" in captions and "60 rows · 4 columns" in captions
    assert [m.value for m in at.metric][:3] == ["3", "1", "SQLite"]
    first_id = at.session_state["dataset_id"]

    _open_table(at, "people")
    at.run()
    assert not at.exception, _errors(at)
    assert _tables(at).label == "Table: people"
    assert [m.value for m in at.metric][:2] == ["60", "4"]
    assert at.session_state["dataset_id"] != first_id
    assert "people" in " ".join(c.value for c in at.caption)
    # One read per table: the rerun after the switch reads nothing.
    assert at.session_state["reads"] == 2


def test_load_page_marks_the_open_table_as_active():
    at = AppTest.from_function(
        _run_page, args=("load", "load_page", False), default_timeout=TIMEOUT
    ).run()
    at.file_uploader[0].upload("town.db", _database(), "application/octet-stream").run()
    assert not at.exception, _errors(at)
    buttons = [b for b in at.button if b.key and b.key.startswith("open_part_")]
    assert [b.disabled for b in buttons] == [True, False]
    assert any("Active" in m.value for m in _tables(at).markdown)


def test_load_page_reopens_the_picked_table_after_opening_another_from_memory():
    at = AppTest.from_function(
        _run_page, args=("load", "load_page", False), default_timeout=TIMEOUT
    ).run()
    at.file_uploader[0].upload("town.db", _database(), "application/octet-stream").run()
    cities_id = at.session_state["dataset_id"]
    _open_table(at, "people")
    # Back to ``cities`` from the datasets in memory: the list of tables follows.
    next(b for b in at.button if b.key == f"open_{cities_id}").click().run()
    assert not at.exception, _errors(at)
    assert _tables(at).label == "Table: cities"
    # ``people`` was the last table read, and can still be opened again.
    _open_table(at, "people")
    assert not at.exception, _errors(at)
    assert [m.value for m in at.metric][:2] == ["60", "4"]


def test_load_page_shows_error_of_a_file_that_is_not_a_database():
    at = AppTest.from_function(
        _run_page, args=("load", "load_page", False), default_timeout=TIMEOUT
    ).run()
    at.file_uploader[0].upload("fake.db", b"a,b\n1,2\n", "application/octet-stream").run()
    assert not at.exception, _errors(at)
    assert at.error[0].value == "'fake.db' is not a SQLite database."
    assert not any(e.label.startswith("Table") for e in at.expander)


def _run_load_or_other_page() -> None:
    """Load page, or nothing when ``session_state["away"]`` is set, as if on another page.

    Widgets of the Load page are not drawn while away, so they lose their state, as they do
    when ``st.navigation`` shows another page.
    """
    import streamlit as st

    from litmetrics.ui import state
    from litmetrics.ui.pages.load import load_page

    state.init_state()
    if not st.session_state.get("away", False):
        load_page()


def test_load_page_keeps_the_table_selector_after_visiting_another_page():
    at = AppTest.from_function(_run_load_or_other_page, default_timeout=TIMEOUT).run()
    at.file_uploader[0].upload("town.db", _database(), "application/octet-stream").run()
    _open_table(at, "people")
    dataset_id = at.session_state["dataset_id"]

    at.session_state["away"] = True
    at.run()
    at.session_state["away"] = False
    at.run()
    assert not at.exception, _errors(at)
    assert _tables(at).label == "Table: people"
    assert at.session_state["dataset_id"] == dataset_id

    # The file is gone from the uploader, but another table can still be read.
    _open_table(at, "cities")
    assert not at.exception, _errors(at)
    assert [m.value for m in at.metric][:2] == ["3", "1"]


def test_load_page_drops_the_table_selector_when_a_csv_is_loaded():
    at = AppTest.from_function(
        _run_page, args=("load", "load_page", False), default_timeout=TIMEOUT
    ).run()
    at.file_uploader[0].upload("town.db", _database(), "application/octet-stream").run()
    at.file_uploader[0].upload("people.csv", CSV, "text/csv").run()
    assert not at.exception, _errors(at)
    assert not any(e.label.startswith("Table") for e in at.expander)
    assert [m.value for m in at.metric][:3] == ["60", "4", "CSV"]
