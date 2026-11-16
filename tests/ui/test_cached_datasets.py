"""Datasets kept in memory: the list of the Load page and the selector of the sidebar."""

from streamlit.testing.v1 import AppTest

TIMEOUT = 30

FIRST = b"a,b\n" + b"".join(b"%d,%d\n" % (i, i * 2) for i in range(10))
SECOND = b"x,y,z\n" + b"".join(b"%d,%d,%d\n" % (i, i, i) for i in range(20))


def _run_load_with_sidebar() -> None:
    """Load page plus the sidebar, as app.py draws them, starting from an empty cache."""
    import streamlit as st

    from litmetrics.ui import cache, sidebar, state
    from litmetrics.ui.pages.load import load_page

    state.init_state()
    # The library is shared by every test of the process: start each test from scratch.
    if "started" not in st.session_state:
        st.session_state["started"] = True
        cache.clear_all()
    load_page()
    sidebar.draw()


def _errors(at: AppTest) -> list[str]:
    return [str(e.value) for e in at.exception]


def _with_two_datasets() -> AppTest:
    at = AppTest.from_function(_run_load_with_sidebar, default_timeout=TIMEOUT).run()
    at.file_uploader[0].upload("first.csv", FIRST, "text/csv").run()
    at.file_uploader[0].upload("second.csv", SECOND, "text/csv").run()
    assert not at.exception, _errors(at)
    return at


def _button(at: AppTest, prefix: str, dataset_id: str):
    return next(b for b in at.button if b.key == f"{prefix}_{dataset_id}")


def test_no_sidebar_before_any_dataset():
    at = AppTest.from_function(_run_load_with_sidebar, default_timeout=TIMEOUT).run()
    assert not at.exception, _errors(at)
    assert not at.sidebar.selectbox
    assert not at.expander


def test_load_lists_every_dataset_read():
    at = _with_two_datasets()
    open_buttons = [b for b in at.button if b.key and b.key.startswith("open_")]
    assert len(open_buttons) == 2
    # The active one (the last read) cannot be opened again.
    assert [b.disabled for b in open_buttons] == [True, False]


def test_sidebar_lists_datasets_and_shows_active():
    at = _with_two_datasets()
    switcher = at.sidebar.selectbox[0]
    assert switcher.options == ["second.csv", "first.csv"]
    assert switcher.value == at.session_state["dataset_id"]
    assert at.session_state["dataset"].name == "second.csv"


def test_open_makes_a_cached_dataset_active_without_reading():
    at = _with_two_datasets()
    second_id = at.session_state["dataset_id"]
    first_button = next(
        b for b in at.button if b.key and b.key.startswith("open_") and not b.disabled
    )
    first_button.click().run()
    assert not at.exception, _errors(at)
    assert at.session_state["dataset"].name == "first.csv"
    assert [m.value for m in at.metric][:2] == ["10", "2"]
    # The file still in the uploader is not read again and does not come back.
    at.run()
    assert at.session_state["dataset"].name == "first.csv"
    assert at.sidebar.selectbox[0].value != second_id


def test_sidebar_switches_the_active_dataset():
    at = _with_two_datasets()
    second_id = at.session_state["dataset_id"]
    # AppTest lists the options as shown (``format_func``): the file names.
    at.sidebar.selectbox[0].select("first.csv").run()
    assert not at.exception, _errors(at)
    assert at.session_state["dataset_id"] != second_id
    assert at.session_state["dataset"].name == "first.csv"


def test_remove_active_dataset_leaves_none():
    at = _with_two_datasets()
    active_id = at.session_state["dataset_id"]
    _button(at, "remove", active_id).click().run()
    assert not at.exception, _errors(at)
    assert at.session_state["dataset"] is None
    assert len([b for b in at.button if b.key and b.key.startswith("open_")]) == 1


def test_remove_other_dataset_keeps_active():
    at = _with_two_datasets()
    active_id = at.session_state["dataset_id"]
    other = next(
        b for b in at.button if b.key and b.key.startswith("remove_") and active_id not in b.key
    )
    other.click().run()
    assert not at.exception, _errors(at)
    assert at.session_state["dataset_id"] == active_id
    assert at.sidebar.selectbox[0].options == ["second.csv"]


def test_clear_cache_after_confirming_removes_everything():
    at = _with_two_datasets()
    next(b for b in at.button if b.key == "clear_cache").click().run()
    # The click inside the dialog reruns the whole script in AppTest (only the dialog in a
    # browser); the clearing is a callback, so it runs either way.
    next(b for b in at.button if b.key == "clear_cache_confirm").click().run()
    assert not at.exception, _errors(at)
    assert at.session_state["dataset"] is None
    assert not at.expander
    assert not at.sidebar.selectbox


# Tables of a database in the sidebar


def _database() -> bytes:
    """A database with tables ``cities`` (3 rows) and ``people`` (60 rows, 2 columns)."""
    import sqlite3

    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE TABLE cities (name TEXT)")
        connection.executemany(
            "INSERT INTO cities VALUES (?)", [("Lugo",), ("Vigo",), ("Ourense",)]
        )
        connection.execute("CREATE TABLE people (id INTEGER, age INTEGER)")
        connection.executemany("INSERT INTO people VALUES (?, ?)", [(i, 20 + i) for i in range(60)])
        connection.commit()
        return connection.serialize()
    finally:
        connection.close()


def _run_app_like() -> None:
    """What app.py does: read the table picked, draw the Load page (or none when ``away``),
    then the sidebar."""
    import streamlit as st

    from litmetrics.ui import cache, reading, sidebar, state
    from litmetrics.ui.pages.load import load_page

    state.init_state()
    if "started" not in st.session_state:
        st.session_state["started"] = True
        cache.clear_all()
    reading.read_chosen_part()
    if not st.session_state.get("away", False):
        load_page()
    sidebar.draw()


def _with_database() -> AppTest:
    at = AppTest.from_function(_run_app_like, default_timeout=TIMEOUT).run()
    at.file_uploader[0].upload("town.db", _database(), "application/octet-stream").run()
    assert not at.exception, _errors(at)
    return at


def test_sidebar_lists_tables_of_the_active_database():
    at = _with_database()
    tables = at.sidebar.selectbox[1]
    assert tables.label == "Table"
    assert tables.options == ["cities — 3 rows · 1 column", "people — 60 rows · 2 columns"]
    assert tables.value == "cities"


def test_sidebar_has_no_table_selector_for_a_csv():
    at = AppTest.from_function(_run_app_like, default_timeout=TIMEOUT).run()
    at.file_uploader[0].upload("first.csv", FIRST, "text/csv").run()
    assert not at.exception, _errors(at)
    assert len(at.sidebar.selectbox) == 1


def test_sidebar_table_switch_reads_it_from_another_page():
    at = _with_database()
    at.session_state["away"] = True
    at.run()
    at.sidebar.selectbox[1].select("people").run()
    assert not at.exception, _errors(at)
    assert at.session_state["dataset"].n_rows == 60
    assert at.sidebar.selectbox[1].value == "people"
    # The Load page shows it too when the user comes back.
    at.session_state["away"] = False
    at.run()
    assert next(e for e in at.expander if e.label.startswith("Table: ")).label == "Table: people"


def test_sidebar_hides_tables_when_a_dataset_of_another_file_is_active():
    at = AppTest.from_function(_run_app_like, default_timeout=TIMEOUT).run()
    at.file_uploader[0].upload("first.csv", FIRST, "text/csv").run()
    at.file_uploader[0].upload("town.db", _database(), "application/octet-stream").run()
    # Back to the CSV from the datasets in memory: the database stays open in the Load page.
    at.sidebar.selectbox[0].select("first.csv").run()
    assert not at.exception, _errors(at)
    assert len(at.sidebar.selectbox) == 1


def test_excel_sheets_are_listed_and_switched_like_tables():
    from xlsx_builder import xlsx

    book = xlsx({"Sales": [["a", "b"], [1, 2], [3, 4]], "Notes": [["n"], ["x"]]})
    at = AppTest.from_function(_run_app_like, default_timeout=TIMEOUT).run()
    at.file_uploader[0].upload("book.xlsx", book, "application/octet-stream").run()
    assert not at.exception, _errors(at)
    assert [m.value for m in at.metric][:3] == ["2", "2", "Excel"]
    sheets = at.sidebar.selectbox[1]
    assert sheets.label == "Sheet"
    assert sheets.options == ["Sales — 2 rows · 2 columns", "Notes — 1 row · 1 column"]
    assert any(e.label == "Sheet: Sales" for e in at.expander)

    sheets.select("Notes").run()
    assert not at.exception, _errors(at)
    assert at.session_state["dataset"].df.columns.tolist() == ["n"]
    assert at.sidebar.selectbox[0].value is not None


# Regressions


def _run_app_counting_reads() -> None:
    """``_run_app_like`` with ``cache.load_dataset`` counting its calls in ``session_state``."""
    import streamlit as st

    from litmetrics.ui import cache, reading, sidebar, state
    from litmetrics.ui.pages.load import load_page

    state.init_state()
    if "started" not in st.session_state:
        st.session_state["started"] = True
        st.session_state["reads"] = 0
        cache.clear_all()
    read = cache.load_dataset

    def counting(*args, **kwargs):
        st.session_state["reads"] += 1
        return read(*args, **kwargs)

    cache.load_dataset = counting
    try:
        reading.read_chosen_part()
        load_page()
        sidebar.draw()
    finally:
        cache.load_dataset = read


def _two_tables(rows: int) -> bytes:
    """A database with tables ``a`` and ``b`` of ``rows`` rows: another ``rows``, another file."""
    import sqlite3

    connection = sqlite3.connect(":memory:")
    try:
        for table in "ab":
            connection.execute(f"CREATE TABLE {table} (x INTEGER)")
            connection.executemany(f"INSERT INTO {table} VALUES (?)", [(i,) for i in range(rows)])
        connection.commit()
        return connection.serialize()
    finally:
        connection.close()


def test_older_version_of_a_file_is_not_taken_for_a_table_of_the_new_one():
    at = AppTest.from_function(_run_app_like, default_timeout=TIMEOUT).run()
    at.file_uploader[0].upload("data.db", _two_tables(3), "application/octet-stream").run()
    at.file_uploader[0].upload("data.db", _two_tables(5), "application/octet-stream").run()
    # Back to the older version, same file name, from the datasets in memory.
    older = next(
        b
        for b in at.button
        if b.key and b.key.startswith("open_") and not b.key.startswith("open_part")
        if not b.disabled
    )
    older.click().run()
    assert not at.exception, _errors(at)
    assert at.session_state["dataset"].n_rows == 3
    # No table of the new file is the active one: all of them can be opened.
    assert [b.disabled for b in at.button if b.key and b.key.startswith("open_part")] == [
        False,
        False,
    ]
    assert len(at.sidebar.selectbox) == 1


def test_two_failed_reads_are_not_tried_again():
    from xlsx_builder import xlsx

    at = AppTest.from_function(_run_app_counting_reads, default_timeout=TIMEOUT).run()
    book = xlsx({"Sales": [["a"], [1]], "Blank": []})
    at.file_uploader[0].upload("book.xlsx", book, "application/octet-stream").run()
    next(b for b in at.button if b.key == "open_part_1").click().run()  # empty sheet: fails
    at.file_uploader[0].upload("bad.csv", b"   \n", "text/csv").run()  # fails too
    reads = at.session_state["reads"]
    at.run()
    at.run()
    assert not at.exception, _errors(at)
    assert at.session_state["reads"] == reads
    errors = [e.value for e in at.error]
    assert "'bad.csv' is empty." in errors
    assert "Sheet 'Blank' of 'book.xlsx' has no data rows." in errors


# Memory and closing the file


def test_memory_list_shows_the_size_of_each_dataset_and_in_all():
    at = _with_two_datasets()
    captions = " ".join(c.value for c in at.caption)
    assert " in all, most recent first." in captions
    assert captions.count(" B · read at ") == 2


def test_close_file_frees_it_and_keeps_the_dataset():
    at = _with_database()
    dataset_id = at.session_state["dataset_id"]
    next(b for b in at.button if b.key == "close_file").click().run()
    assert not at.exception, _errors(at)
    assert at.session_state["multipart"] is None
    assert at.session_state["dataset_id"] == dataset_id
    # The file still in the uploader is not opened again.
    at.run()
    assert at.session_state["multipart"] is None
    assert not any(e.label.startswith("Table: ") for e in at.expander)
    assert len(at.sidebar.selectbox) == 1


def test_clear_cache_closes_the_open_file():
    at = _with_database()
    next(b for b in at.button if b.key == "clear_cache").click().run()
    next(b for b in at.button if b.key == "clear_cache_confirm").click().run()
    at.run()
    assert not at.exception, _errors(at)
    assert at.session_state["multipart"] is None
    assert at.session_state["dataset"] is None
