"""Integration tests of the datasets kept in memory: the list of the Load page and the sidebar.

Every file read is kept in a library shared by all sessions (``ui/cache.py`` over
``core/io/library.py``). The Load page lists them (open, remove, clear all); the sidebar
switches between them, and between the tables or sheets of the open file. ``run_app`` draws
both, as app.py does, and starts each test with an empty library.
"""

from fixtures import samples
from fixtures.app_scripts import (
    button,
    buttons,
    captions,
    errors,
    memory_open_buttons,
    metrics,
    parts_expander,
    start_app,
    upload,
)


def _with_two_datasets():
    """``CSV_A`` then ``CSV_B`` uploaded: two datasets in memory, B active."""
    at = upload(start_app(), "first.csv", samples.CSV_A)
    upload(at, "second.csv", samples.CSV_B)
    assert not at.exception, errors(at)
    return at


def _clear_cache(at):
    """Click "Clear cache" and confirm it in the dialog."""
    button(at, "clear_cache").click().run()
    # The click inside the dialog reruns the whole script in AppTest (only the dialog in a
    # browser); the clearing is a callback, so it runs either way.
    button(at, "clear_cache_confirm").click().run()


# Before anything is read


def test_no_sidebar_and_no_list_before_any_dataset():
    """With nothing read, there is no sidebar selector and no list of datasets.

    Case: an empty selector or an empty "Datasets in memory" box greets a new user.
    Affects: ui/sidebar.py (draw), ui/pages/load.py (_cached_datasets)
    How: starts the app and checks there is no selectbox and no expander.
    """
    at = start_app()
    assert not at.exception, errors(at)
    assert not at.sidebar.selectbox
    assert not at.expander


# The list in the Load page


def test_load_lists_every_dataset_read_with_its_size():
    """Every file read is listed, most recent first, with its size and the total.

    Case: the second upload replaces the first in the list, or sizes are missing.
    Affects: ui/pages/load.py (_cached_datasets, _cached_row), ui/cache.py, core/io/library.py
    How: uploads two files and checks the Open buttons (the active one disabled) and captions.
    """
    at = _with_two_datasets()
    assert [b.disabled for b in memory_open_buttons(at)] == [True, False]
    assert " in all, most recent first." in captions(at)
    assert captions(at).count(" B · read at ") == 2


def test_open_makes_a_cached_dataset_active_without_reading():
    """Opening a dataset from memory makes it active; the file in the uploader does not return.

    Case: the next rerun sees the file still in the uploader and makes it active again.
    Affects: ui/pages/load.py (_open_cached, _load), ui/state.py (switch_dataset)
    How: with B active, opens A, checks its metrics, reruns, and checks A is still active.
    """
    at = _with_two_datasets()
    second_id = at.session_state["dataset_id"]
    memory_open_buttons(at)[1].click().run()
    assert not at.exception, errors(at)
    assert at.session_state["dataset"].name == "first.csv"
    assert metrics(at)[:2] == ["10", "2"]
    at.run()
    assert at.session_state["dataset"].name == "first.csv"
    assert at.sidebar.selectbox[0].value != second_id


def test_remove_active_dataset_leaves_none():
    """Removing the active dataset leaves the session without one.

    Case: the removed dataset stays active (memory not freed) or another one is opened silently.
    Affects: ui/pages/load.py (_remove_cached), ui/cache.py (forget_dataset), ui/state.py
    How: removes B (active) and checks no dataset and one entry left.
    """
    at = _with_two_datasets()
    button(at, f"remove_{at.session_state['dataset_id']}").click().run()
    assert not at.exception, errors(at)
    assert at.session_state["dataset"] is None
    assert len(memory_open_buttons(at)) == 1


def test_remove_other_dataset_keeps_active():
    """Removing another dataset keeps the active one and updates the sidebar.

    Case: any removal clears the active dataset, or the sidebar still offers the removed one.
    Affects: ui/pages/load.py (_remove_cached), ui/sidebar.py
    How: removes A while B is active.
    """
    at = _with_two_datasets()
    active_id = at.session_state["dataset_id"]
    other = next(b for b in buttons(at, "remove_") if active_id not in b.key)
    other.click().run()
    assert not at.exception, errors(at)
    assert at.session_state["dataset_id"] == active_id
    assert at.sidebar.selectbox[0].options == ["second.csv"]


def test_clear_cache_after_confirming_removes_everything():
    """Confirming "Clear cache" removes every dataset, the list and the sidebar.

    Case: the active dataset survives the clearing, so memory is not freed.
    Affects: ui/pages/load.py (_confirm_clear, _clear_cache), ui/cache.py (clear_all)
    How: clears with two datasets and checks state, expanders and sidebar.
    """
    at = _with_two_datasets()
    _clear_cache(at)
    assert not at.exception, errors(at)
    assert at.session_state["dataset"] is None
    assert not at.expander
    assert not at.sidebar.selectbox


def test_clear_cache_closes_the_open_file():
    """Clearing the cache also closes the open database.

    Case: the bytes of a large database stay in the session after "Clear cache".
    Affects: ui/pages/load.py (_clear_cache), ui/state.py (close_multipart)
    How: uploads ``TOWN_DB``, clears, reruns, and checks the file and dataset are gone.
    """
    at = upload(start_app(), "town.db", samples.TOWN_DB)
    _clear_cache(at)
    at.run()
    assert not at.exception, errors(at)
    assert at.session_state["multipart"] is None
    assert at.session_state["dataset"] is None


# The dataset selector of the sidebar


def test_sidebar_lists_datasets_and_shows_active():
    """The sidebar lists the datasets in memory by name, most recent first, on the active one.

    Case: the selector shows ids instead of names, or not the active dataset.
    Affects: ui/sidebar.py (_dataset_switcher), ui/state.py (sync_switcher)
    How: uploads two files and reads the selector.
    """
    at = _with_two_datasets()
    switcher = at.sidebar.selectbox[0]
    assert switcher.options == ["second.csv", "first.csv"]
    assert switcher.value == at.session_state["dataset_id"]
    assert at.session_state["dataset"].name == "second.csv"


def test_sidebar_switches_the_active_dataset():
    """Picking a dataset in the sidebar makes it active.

    Case: the selector changes but the page keeps showing the previous dataset.
    Affects: ui/sidebar.py (_on_switch), ui/state.py (switch_dataset)
    How: selects ``first.csv`` and checks the active dataset.
    """
    at = _with_two_datasets()
    second_id = at.session_state["dataset_id"]
    at.sidebar.selectbox[0].select("first.csv").run()
    assert not at.exception, errors(at)
    assert at.session_state["dataset_id"] != second_id
    assert at.session_state["dataset"].name == "first.csv"


# The table / sheet selector of the sidebar


def test_sidebar_lists_tables_of_the_active_database():
    """With a database open, a second selector lists its tables with their sizes.

    Case: the table selector is missing, or does not say which table is open.
    Affects: ui/sidebar.py (_part_switcher, _part_label), core/io/sqlite_loader.py
    How: uploads ``TOWN_DB`` and reads the second selector.
    """
    at = upload(start_app(), "town.db", samples.TOWN_DB)
    tables = at.sidebar.selectbox[1]
    assert tables.label == "Table"
    assert tables.options == ["cities — 3 rows · 1 column", "people — 60 rows · 4 columns"]
    assert tables.value == "cities"


def test_sidebar_has_no_table_selector_for_a_csv():
    """A CSV has no table selector.

    Case: an empty or one-item "Table" selector is shown for a CSV.
    Affects: ui/sidebar.py (_part_switcher)
    How: uploads ``CSV_A`` and counts the selectors.
    """
    at = upload(start_app(), "first.csv", samples.CSV_A)
    assert not at.exception, errors(at)
    assert len(at.sidebar.selectbox) == 1


def test_sidebar_table_switch_reads_it_from_another_page():
    """A table picked in the sidebar on another page is read, and Load shows it on return.

    Case: the sidebar only records the choice and the table is read when the user goes back to
    Load, so Profile keeps showing the previous table.
    Affects: ui/sidebar.py (_on_part_switch), ui/reading.py (read_chosen_part), app.py order
    How: goes ``away``, selects ``people``, checks the dataset, returns to Load.
    """
    at = upload(start_app(), "town.db", samples.TOWN_DB)
    at.session_state["away"] = True
    at.run()
    at.sidebar.selectbox[1].select("people").run()
    assert not at.exception, errors(at)
    assert at.session_state["dataset"].n_rows == 60
    assert at.sidebar.selectbox[1].value == "people"
    at.session_state["away"] = False
    at.run()
    assert parts_expander(at).label == "Table: people"


def test_sidebar_hides_tables_when_a_dataset_of_another_file_is_active():
    """The table selector disappears while a dataset of another file is active.

    Case: the tables of the database are offered while a CSV is shown, as if they were its.
    Affects: ui/sidebar.py (_part_switcher), ui/reading.py (active_part)
    How: uploads a CSV and ``TOWN_DB``, goes back to the CSV in the sidebar.
    """
    at = upload(start_app(), "first.csv", samples.CSV_A)
    upload(at, "town.db", samples.TOWN_DB)
    at.sidebar.selectbox[0].select("first.csv").run()
    assert not at.exception, errors(at)
    assert len(at.sidebar.selectbox) == 1


def test_excel_sheets_are_listed_and_switched_like_tables():
    """The sheets of a workbook behave like the tables of a database.

    Case: the Excel loader is not wired to the shared part machinery (no selector, no list).
    Affects: ui/sidebar.py, ui/pages/load.py (_parts), core/io/excel_loader.py, formats.py
    How: uploads ``TWO_SHEETS``, checks metrics, selector and expander, then switches sheet.
    """
    at = upload(start_app(), "book.xlsx", samples.TWO_SHEETS)
    assert not at.exception, errors(at)
    assert metrics(at)[:3] == ["2", "2", "Excel"]
    sheets = at.sidebar.selectbox[1]
    assert sheets.label == "Sheet"
    assert sheets.options == ["Sales — 2 rows · 2 columns", "Notes — 1 row · 1 column"]
    assert parts_expander(at).label == "Sheet: Sales"
    sheets.select("Notes").run()
    assert not at.exception, errors(at)
    assert at.session_state["dataset"].df.columns.tolist() == ["n"]
    assert at.sidebar.selectbox[0].value is not None
