"""Integration tests of the Load page: upload -> format registry -> loader -> schema -> state.

Each test uploads real file bytes through the uploader widget of ``run_app`` (app.py with the
Load page) and checks what the user sees (metrics, captions, errors) and what is kept in the
session. Nothing is mocked; ``session_state["reads"]`` counts the real reads.
"""

import pytest
from fixtures import samples
from fixtures.app_scripts import (
    button,
    buttons,
    captions,
    errors,
    has_parts_expander,
    memory_open_buttons,
    metrics,
    open_part,
    parts_expander,
    start_app,
    upload,
)

# Reading single-dataset files


@pytest.mark.parametrize(
    ("name", "data", "shown"),
    [
        ("people.csv", samples.PEOPLE_CSV, ["60", "4", "CSV"]),
        ("people.parquet", samples.people_with_photo_parquet(), ["60", "5", "Parquet"]),
        ("people.feather", samples.people_feather_60(), ["60", "4", "Feather"]),
    ],
    ids=["csv", "parquet", "feather"],
)
def test_upload_reads_the_file_and_makes_it_active(name, data, shown):
    """An upload of each single-dataset format is read, summarised and becomes active.

    Case: a format is registered but not wired to the page (nothing shown), or the active
    dataset in the session is not the file just uploaded.
    Affects: ui/pages/load.py, ui/reading.py, core/io/formats.py, the loader of each format,
    core/schema.py, ui/state.py
    How: uploads 60 rows in each format; checks rows, columns and format metrics, and the
    dataset and schema kept in the session.
    """
    at = upload(start_app(), name, data)
    assert not at.exception, errors(at)
    assert metrics(at)[:3] == shown
    assert at.session_state["dataset"].name == name
    assert at.session_state["schema"] is not None


def test_upload_shows_how_the_file_was_read():
    """The caption says how the file was read, from the loader's ``details``.

    Case: the page drops the details, so the user cannot tell a European CSV was understood.
    Affects: ui/pages/load.py (_summary), core/io/sniffing.py, core/io/csv_loader.py
    How: uploads ``EUROPEAN_CSV`` and reads the separator and decimal mark in the caption.
    """
    at = upload(start_app(), "eu.csv", samples.EUROPEAN_CSV)
    assert not at.exception, errors(at)
    assert "Semicolon (;)" in captions(at) and "Comma (,)" in captions(at)


def test_parquet_conversion_is_reported():
    """Columns converted on the way (binary to size) are named in the caption.

    Case: the user sees "<10 bytes>" values without knowing why.
    Affects: ui/pages/load.py, core/io/arrow_table.py
    How: uploads the Parquet file with a ``photo`` column and reads the caption.
    """
    at = upload(start_app(), "people.parquet", samples.people_with_photo_parquet())
    assert "photo (shown as their size)" in captions(at)


# Errors


def test_read_error_is_shown_and_no_dataset_is_set():
    """A file that cannot be read shows its message, and there is no active dataset.

    Case: the exception reaches the page, or a half-read dataset becomes active.
    Affects: ui/pages/load.py (_load), ui/reading.py, core/io/csv_loader.py (ReadError)
    How: uploads ``EMPTY_CSV``.
    """
    at = upload(start_app(), "empty.csv", samples.EMPTY_CSV)
    assert not at.exception, errors(at)
    assert at.error[0].value == "'empty.csv' is empty."
    assert at.session_state["dataset"] is None


def test_read_error_shows_the_file_name_as_written():
    """The error of a file whose name Markdown would read quotes that name as written.

    Case: ``ReadError`` messages quote the file name and are drawn as Markdown, so
    ``*draft*_v1.csv`` shows in italics.
    Affects: ui/pages/load.py (_load), ui/text.py (md)
    How: uploads ``EMPTY_CSV`` as ``*draft*_v1.csv`` and reads the Markdown of the error.
    """
    at = upload(start_app(), "*draft*_v1.csv", samples.EMPTY_CSV)
    assert not at.exception, errors(at)
    assert at.error[0].value == r"'\*draft\*\_v1.csv' is empty."


def test_failed_upload_is_not_read_again_on_reruns():
    """After a failed read, reruns show the error without reading the file again.

    Case: every click on the page re-reads (and re-fails on) a large broken file.
    Affects: ui/pages/load.py (_load), ui/state.py (read errors), ui/reading.py
    How: uploads ``EMPTY_CSV``, reruns twice, and checks one read and the error still shown.
    """
    at = upload(start_app(), "empty.csv", samples.EMPTY_CSV)
    at.run()
    at.run()
    assert not at.exception, errors(at)
    assert at.error[0].value == "'empty.csv' is empty."
    assert at.session_state["reads"] == 1


def test_two_failed_reads_are_not_tried_again():
    """Two different failures (a sheet and a file) are both remembered and not re-read.

    Case: remembering only the last error makes the first one be read again on each rerun.
    Affects: ui/pages/load.py, ui/reading.py (part errors), ui/state.py (read errors)
    How: opens the empty sheet of ``SHEET_AND_BLANK``, uploads ``EMPTY_CSV``, reruns twice and
    checks the read count and both messages.
    """
    at = upload(start_app(), "book.xlsx", samples.SHEET_AND_BLANK)
    button(at, "open_part_1").click().run()
    upload(at, "bad.csv", samples.EMPTY_CSV)
    reads = at.session_state["reads"]
    at.run()
    at.run()
    assert not at.exception, errors(at)
    assert at.session_state["reads"] == reads
    shown = [e.value for e in at.error]
    assert "'bad.csv' is empty." in shown
    assert "Sheet 'Blank' of 'book.xlsx' has no data rows." in shown


def test_file_that_is_not_a_database_shows_its_error():
    """A ``.db`` file that is not a database shows an error and no list of tables.

    Case: listing the tables raises outside ``ReadError`` and the page crashes, or an empty
    table list is drawn.
    Affects: ui/pages/load.py (_load), core/io/formats.py (list_parts), sqlite_loader.py
    How: uploads CSV bytes named ``fake.db``.
    """
    at = upload(start_app(), "fake.db", b"a,b\n1,2\n")
    assert not at.exception, errors(at)
    assert at.error[0].value == "'fake.db' is not a SQLite database."
    assert not has_parts_expander(at)


# Files with several datasets (tables, sheets)


def test_database_opens_its_first_table_and_lists_all():
    """A database opens its first table, lists every table with its size, marks the active.

    Case: the user has to pick a table before seeing anything, sizes are missing, or the open
    table can be "opened" again.
    Affects: ui/pages/load.py (_parts, _row), ui/reading.py, core/io/sqlite_loader.py
    How: uploads ``TOWN_DB`` and checks the expander label, captions, metrics and buttons.
    """
    at = upload(start_app(), "town.db", samples.TOWN_DB)
    assert not at.exception, errors(at)
    tables = parts_expander(at)
    assert tables.label == "Table: cities"
    table_captions = [c.value for c in tables.caption]
    assert "3 rows · 1 column" in table_captions and "60 rows · 4 columns" in table_captions
    assert metrics(at)[:3] == ["3", "1", "SQLite"]
    assert [b.disabled for b in buttons(at, "open_part_")] == [True, False]
    assert any("Active" in m.value for m in tables.markdown)


def test_opening_another_table_reads_it_once():
    """Opening another table makes it active, with one read per table.

    Case: the switch reads the table again on the next rerun, or keeps the old dataset id.
    Affects: ui/pages/load.py (_parts), ui/state.py (choose_part), ui/reading.py
    How: uploads ``TOWN_DB``, opens ``people``, reruns and checks label, metrics, id and reads.
    """
    at = upload(start_app(), "town.db", samples.TOWN_DB)
    first_id = at.session_state["dataset_id"]
    open_part(at, "people")
    at.run()
    assert not at.exception, errors(at)
    assert parts_expander(at).label == "Table: people"
    assert metrics(at)[:2] == ["60", "4"]
    assert at.session_state["dataset_id"] != first_id
    assert "people" in captions(at)
    assert at.session_state["reads"] == 2


def test_table_list_follows_a_table_reopened_from_memory():
    """Opening a table again from the datasets in memory updates the list of tables.

    Case: the list still marks the last table read as active, and it cannot be opened again.
    Affects: ui/pages/load.py (_open_cached, _parts), ui/reading.py (active_part)
    How: opens ``people``, goes back to ``cities`` from memory, then opens ``people`` again.
    """
    at = upload(start_app(), "town.db", samples.TOWN_DB)
    cities_id = at.session_state["dataset_id"]
    open_part(at, "people")
    button(at, f"open_{cities_id}").click().run()
    assert not at.exception, errors(at)
    assert parts_expander(at).label == "Table: cities"
    open_part(at, "people")
    assert not at.exception, errors(at)
    assert metrics(at)[:2] == ["60", "4"]


def test_table_list_survives_visiting_another_page():
    """The list of tables stays after visiting another page, and other tables can be read.

    Case: the uploader forgets its file on another page, so the tables of the database are
    lost and the user must upload it again.
    Affects: ui/pages/load.py (_parts), ui/state.py (multipart), ui/reading.py
    How: opens ``people``, draws no page for one rerun (``away``), comes back, checks the list
    and opens ``cities``.
    """
    at = upload(start_app(), "town.db", samples.TOWN_DB)
    open_part(at, "people")
    dataset_id = at.session_state["dataset_id"]
    at.session_state["away"] = True
    at.run()
    at.session_state["away"] = False
    at.run()
    assert not at.exception, errors(at)
    assert parts_expander(at).label == "Table: people"
    assert at.session_state["dataset_id"] == dataset_id
    open_part(at, "cities")
    assert not at.exception, errors(at)
    assert metrics(at)[:2] == ["3", "1"]


def test_csv_upload_replaces_the_table_list():
    """Uploading a CSV after a database drops the list of tables.

    Case: the tables of the old database stay listed above an unrelated CSV.
    Affects: ui/pages/load.py (_load), ui/state.py (clear_multipart)
    How: uploads ``TOWN_DB`` then ``PEOPLE_CSV``.
    """
    at = upload(start_app(), "town.db", samples.TOWN_DB)
    upload(at, "people.csv", samples.PEOPLE_CSV)
    assert not at.exception, errors(at)
    assert not has_parts_expander(at)
    assert metrics(at)[:3] == ["60", "4", "CSV"]


def test_older_version_of_a_file_is_not_taken_for_a_table_of_the_new_one():
    """Opening a table of an older version of a file opens that version, not the new one.

    Case: matching by file and table name, the old ``a`` is shown as a table of the new file
    (whose ``a`` has other rows), and picking ``b`` then reads the new file's ``b``.
    Affects: ui/pages/load.py (_open_cached), ui/reading.py (open_kept, active_part),
    ui/state.py (reopen_multipart)
    How: uploads two versions of ``data.db`` (3 and 5 rows), opens the older one from memory,
    and checks the file open again is the older one, with ``a`` active.
    """
    at = upload(start_app(), "data.db", samples.versioned_db(3))
    upload(at, "data.db", samples.versioned_db(5))
    older = next(b for b in memory_open_buttons(at) if not b.disabled)
    older.click().run()
    assert not at.exception, errors(at)
    assert at.session_state["dataset"].n_rows == 3
    assert at.session_state["multipart"][0].data == samples.versioned_db(3)
    assert [b.disabled for b in buttons(at, "open_part")] == [True, False]
    assert at.sidebar.selectbox[1].value == "a"


def test_close_file_frees_it_and_keeps_the_dataset():
    """ "Close file" forgets the database but keeps the table already read.

    Case: closing also drops the active dataset, or the file still in the uploader is opened
    again on the next rerun.
    Affects: ui/pages/load.py (_parts), ui/state.py (close_multipart), ui/sidebar.py
    How: uploads ``TOWN_DB``, closes it, reruns, and checks state, expander and sidebar.
    """
    at = upload(start_app(), "town.db", samples.TOWN_DB)
    dataset_id = at.session_state["dataset_id"]
    button(at, "close_file").click().run()
    assert not at.exception, errors(at)
    assert at.session_state["multipart"] is None
    assert at.session_state["dataset_id"] == dataset_id
    at.run()
    assert at.session_state["multipart"] is None
    assert not has_parts_expander(at)
    assert len(at.sidebar.selectbox) == 1
