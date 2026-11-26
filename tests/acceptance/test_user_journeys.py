"""Acceptance tests: what a user does with the app, in a real browser, from start to finish.

Each test is one journey written as Given / When / Then, against the app started with the
``litmetrics`` command. They check what is on the screen, not the session state: if a test
here fails, a user would see the problem. They are slow (a server per test) and left out of a
plain ``pytest`` run: ``uv run pytest -m acceptance``.
"""

from pathlib import Path

import pyarrow.parquet as pq
from fixtures import samples
from fixtures.browser import (
    RUN_TIMEOUT,
    choose_in_sidebar,
    expect_no_exception,
    go_to,
    metric,
    open_tab,
    upload,
    wait_idle,
)
from playwright.sync_api import Page, expect


def _file(tmp_path: Path, name: str, data: bytes) -> Path:
    """Write ``data`` to ``tmp_path/name``, as the file the user picks on their disk."""
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_analyst_loads_a_spanish_export_and_sees_its_problems(app: Page, tmp_path):
    """A Spanish sales export is understood and its quality problems are reported.

    Given a CSV saved by a Spanish Excel (cp1252, ``;``, decimal commas, accents) with a
    duplicate row, a mostly empty column and a constant column,
    When the user uploads it on Load and opens Profile,
    Then Load shows its size and how it was read, with accents intact, and Profile counts
    the duplicate and the constant column and lists the issues.

    Case: any step of the main flow breaks for the files most of our users have: wrong
    separator (one column), mangled accents, numbers read as text, or issues not reported.
    Affects: app.py, ui/pages/load.py, ui/pages/profile/page.py, ui/sidebar.py, core/io/sniffing.py,
    core/io/csv_loader.py, core/schema.py, core/profile.py
    How: uploads ``SALES_EXPORT_CSV`` and reads metrics, captions, the preview and the issues.
    """
    page = app
    go_to(page, "Load")
    upload(page, _file(tmp_path, "ventas.csv", samples.SALES_EXPORT_CSV))
    expect_no_exception(page)
    expect(metric(page, "Rows")).to_have_text("41")
    expect(metric(page, "Columns")).to_have_text("6")
    expect(metric(page, "Format")).to_have_text("CSV")
    how = page.get_by_text("Read with")
    expect(how).to_contain_text("Semicolon (;)")
    expect(how).to_contain_text("Comma (,)")
    expect(how).to_contain_text("cp1252")

    go_to(page, "Profile")
    expect_no_exception(page)
    expect(page.get_by_text("ventas.csv · 41 rows · 6 columns")).to_be_visible()
    expect(metric(page, "Duplicate rows")).to_have_text("1")
    expect(metric(page, "Constant columns")).to_have_text("1")
    issues = page.get_by_role("tabpanel", name="Issues")
    expect(issues).to_contain_text("1 duplicate rows")
    expect(issues).to_contain_text("descuento")
    expect(issues).to_contain_text("país")


def test_analyst_downloads_the_issues_report(app: Page, tmp_path):
    """The issues of a dataset download as a Markdown report the user can share.

    Given the Spanish sales export, with a duplicate row and a mostly empty column,
    When the user opens the Issues tab of Profile and clicks "Download report",
    Then the browser downloads ``ventas_issues.md`` with the dataset's size, its issues by
    severity (the duplicate row as a medium one) and the advice for each.

    Case: the function given to the download button fails when clicked (it runs in another
    thread, which AppTest never runs), or the report is not the one of the dataset shown.
    Affects: ui/pages/profile/page.py (_report_button), core/report.py, core/profile.py
    How: uploads ``SALES_EXPORT_CSV``, catches the download of the Issues tab and reads it.
    """
    page = app
    go_to(page, "Load")
    upload(page, _file(tmp_path, "ventas.csv", samples.SALES_EXPORT_CSV))
    go_to(page, "Profile")
    issues = page.get_by_role("tabpanel", name="Issues")
    with page.expect_download(timeout=RUN_TIMEOUT * 1000) as info:
        issues.get_by_role("button", name="Download report").click()
    download = info.value
    expect_no_exception(page)

    assert download.suggested_filename == "ventas_issues.md"
    markdown = Path(download.path()).read_text("utf-8")
    assert markdown.startswith("# Data quality report: ventas.csv\n")
    assert "| Size | 41 rows " in markdown
    assert "| Medium | Duplicate rows | whole dataset | 1 duplicate rows |" in markdown
    assert "`descuento`" in markdown
    assert "**Suggested action:** " in markdown


def test_analyst_switches_tables_of_a_database_from_any_page(app: Page, tmp_path):
    """The tables of a database can be switched from the sidebar while on Profile.

    Given a SQLite database with a small ``cities`` table and a 60-row ``people`` table,
    When the user uploads it, goes to Profile and picks ``people`` in the sidebar,
    Then Profile shows ``people`` at once, and Load lists it as the active table.

    Case: the table chosen in the sidebar is only read on the Load page, so Profile keeps
    describing the previous table without saying so.
    Affects: app.py (read_chosen_part before the page), ui/sidebar.py, ui/reading.py,
    ui/pages/load.py, ui/pages/profile/page.py, core/io/sqlite_loader.py
    How: uploads ``TOWN_DB``, switches table in the sidebar on Profile, checks both pages.
    """
    page = app
    go_to(page, "Load")
    upload(page, _file(tmp_path, "town.db", samples.TOWN_DB))
    expect(metric(page, "Format")).to_have_text("SQLite")
    expect(metric(page, "Rows")).to_have_text("3")

    go_to(page, "Profile")
    choose_in_sidebar(page, "Table", "people")
    expect_no_exception(page)
    expect(page.get_by_text("60 rows · 4 columns", exact=False).first).to_be_visible()

    go_to(page, "Load")
    expect(metric(page, "Rows")).to_have_text("60")
    expect(page.get_by_text("Table: people")).to_be_visible()


def test_a_bad_file_does_not_lose_the_dataset_being_used(app: Page, tmp_path):
    """Uploading a broken file shows why, and the previous dataset stays in use.

    Given the user is working with a valid CSV,
    When they upload an empty file by mistake,
    Then the error says the file is empty, and the previous dataset is still shown in Load
    and in Profile.

    Case: a read error replaces the active dataset with nothing, or crashes the page, and the
    user has to find and upload the good file again.
    Affects: ui/pages/load.py (_load), ui/reading.py, ui/state.py, core/io/csv_loader.py
    How: uploads ``PEOPLE_CSV`` then ``EMPTY_CSV``, reads the error, the metrics and Profile.
    """
    page = app
    go_to(page, "Load")
    upload(page, _file(tmp_path, "people.csv", samples.PEOPLE_CSV))
    expect(metric(page, "Rows")).to_have_text("60")

    upload(page, _file(tmp_path, "empty.csv", samples.EMPTY_CSV))
    expect(page.get_by_text("'empty.csv' is empty.")).to_be_visible()
    expect(metric(page, "Rows")).to_have_text("60")

    go_to(page, "Profile")
    expect_no_exception(page)
    expect(page.get_by_text("people.csv · 60 rows")).to_be_visible()


def test_analyst_goes_back_to_an_earlier_file_without_uploading_it(app: Page, tmp_path):
    """Files read earlier are one click away in the sidebar, from any page.

    Given the user uploaded two files, one after the other,
    When they pick the first one in the sidebar while on Profile,
    Then Profile describes the first file, without uploading it again.

    Case: the list of datasets in memory is not shared with the sidebar, or picking one does
    not change what the page shows.
    Affects: ui/sidebar.py (_dataset_switcher), ui/cache.py, core/io/library.py,
    ui/pages/profile/page.py
    How: uploads ``CSV_A`` and ``CSV_B``, switches to ``first.csv`` on Profile.
    """
    page = app
    go_to(page, "Load")
    upload(page, _file(tmp_path, "first.csv", samples.CSV_A))
    expect(metric(page, "Rows")).to_have_text("10")
    upload(page, _file(tmp_path, "second.csv", samples.CSV_B))
    expect(metric(page, "Rows")).to_have_text("20")

    go_to(page, "Profile")
    expect(page.get_by_text("second.csv · 20 rows")).to_be_visible()
    choose_in_sidebar(page, "Dataset", "first.csv")
    expect_no_exception(page)
    expect(page.get_by_text("first.csv · 10 rows")).to_be_visible()


def test_correlation_heatmap_is_drawn(app: Page, tmp_path):
    """The correlation heatmap is drawn by ECharts in the browser.

    Given a loaded dataset with several numeric columns,
    When the user opens the Correlation tab of Profile,
    Then a chart is drawn on the page.

    Case: the chart option is valid Python but not valid for ECharts (NaN, numpy types), or
    the component fails to load: AppTest cannot see either, only a browser can.
    Affects: ui/pages/profile/page.py (_correlation), ui/charts.py (heatmap_options),
    ui/echarts.py, streamlit-echarts
    How: uploads ``PEOPLE_CSV``, opens Correlation and waits for the canvas inside the chart's
    container (key ``profile_correlation_heatmap``) to be drawn at the heatmap's height. The
    table of pairs is a canvas too, so any canvas would not do. ECharts draws after the script
    run ends, so the wait is longer than for widgets.
    """
    page = app
    go_to(page, "Load")
    upload(page, _file(tmp_path, "people.csv", samples.PEOPLE_CSV))
    go_to(page, "Profile")
    open_tab(page, "Correlation")
    expect_no_exception(page)
    heatmap = page.locator(".st-key-profile_correlation_heatmap canvas").first
    expect(heatmap).to_be_visible(timeout=RUN_TIMEOUT * 1000)
    box = heatmap.bounding_box()
    assert box is not None and box["height"] >= 300, box


def test_analyst_exports_a_table_of_a_database_from_profile(app: Page, tmp_path):
    """A table of a database is downloaded as Parquet from the sidebar, without leaving Profile.

    Given a SQLite database with a 3-row ``cities`` table and a 60-row ``people`` table,
    When the user picks ``people`` in the sidebar on Profile and exports it as Parquet,
    Then the browser downloads ``town_people.parquet`` holding the 60 rows of ``people``.

    Case: the file is written from the table read first instead of the active one, the
    download gets the wrong name or extension, or the function given to the download button
    fails when clicked (it runs in another thread, which AppTest never runs).
    Affects: ui/sidebar.py (_export), core/io/export.py, ui/reading.py (active_part)
    How: uploads ``TOWN_DB``, switches table on Profile, opens the Export dialog, picks
    Parquet, catches the download and reads the file with pyarrow.
    """
    page = app
    go_to(page, "Load")
    upload(page, _file(tmp_path, "town.db", samples.TOWN_DB))
    go_to(page, "Profile")
    choose_in_sidebar(page, "Table", "people")

    sidebar = page.locator('[data-testid="stSidebar"]')
    sidebar.get_by_role("button", name="Export").click()
    page.get_by_role("radio", name="Parquet").click()
    wait_idle(page)
    with page.expect_download(timeout=RUN_TIMEOUT * 1000) as info:
        page.get_by_role("dialog").get_by_role("button", name="Download", exact=True).click()
    download = info.value
    expect_no_exception(page)

    assert download.suggested_filename == "town_people.parquet"
    table = pq.read_table(download.path())
    assert table.num_rows == 60
    assert table.column_names == ["id", "age", "city", "score"]


def test_analyst_downloads_the_report_of_values_lost_in_an_export(app: Page, tmp_path):
    """Before exporting, the user gets the list of values that will be empty, by row.

    Given a CSV of 60 orders with "12 kg" and "abc" in ``price`` and "someday" in ``date``,
    When the user opens Export, unfolds the problem report and downloads it,
    Then the dialog says 2 rows are affected and the browser downloads
    ``orders_problems.csv`` with the 3 values, their rows and columns.

    Case: the dialog does not open from the sidebar, the report is not offered, or the
    function given to its download button fails when clicked (it runs in another thread,
    which AppTest never runs).
    Affects: ui/sidebar.py (_export_dialog, _problem_report), core/io/export.py
    (unparsed_report, write_report), core/schema.py (apply_schema)
    How: uploads ``ORDERS_WITH_TYPOS_CSV``, opens the dialog, catches the report's download
    and reads its lines.
    """
    page = app
    go_to(page, "Load")
    upload(page, _file(tmp_path, "orders.csv", samples.ORDERS_WITH_TYPOS_CSV))

    page.locator('[data-testid="stSidebar"]').get_by_role("button", name="Export").click()
    dialog = page.get_by_role("dialog")
    report = dialog.get_by_text("Problem report: 2 rows affected")
    expect(report).to_be_visible(timeout=RUN_TIMEOUT * 1000)
    report.click()
    with page.expect_download(timeout=RUN_TIMEOUT * 1000) as info:
        dialog.get_by_role("button", name="Download report").click()
    download = info.value
    expect_no_exception(page)

    assert download.suggested_filename == "orders_problems.csv"
    assert Path(download.path()).read_text("utf-8").splitlines() == [
        "row,column,value,expected",
        "5,price,12 kg,number",
        "30,date,someday,date",
        "30,price,abc,number",
    ]


def test_analyst_builds_a_chart_in_visualization(app: Page, tmp_path):
    """A loaded dataset can be charted in Visualization, and the chart kind changed.

    Given a loaded CSV with numbers, categories and dates,
    When the user opens Visualization and switches from the first chart to a scatter,
    Then a chart is drawn each time and the page shows no error.

    Case: the page is not registered or breaks on a real upload, or an option of the chart
    builder is valid Python but not valid ECharts (AppTest cannot see that, a browser can).
    Affects: app.py (build_pages), ui/pages/visualize.py, ui/chart_style.py, ui/cache.py,
    core/visualize/, ui/echarts.py, streamlit-echarts
    How: uploads ``chart_csv``, opens Visualization, waits for the canvas of the chart (key
    ``viz_chart``), clicks the "Scatter" pill and waits for it again.
    """
    page = app
    go_to(page, "Load")
    upload(page, _file(tmp_path, "chart.csv", samples.chart_csv()))
    go_to(page, "Visualization")
    expect_no_exception(page)
    chart = page.locator(".st-key-viz_chart canvas").first
    expect(chart).to_be_visible(timeout=RUN_TIMEOUT * 1000)
    page.get_by_text("Scatter", exact=True).click()
    wait_idle(page)
    expect_no_exception(page)
    expect(page.get_by_text("Y axis", exact=True)).to_be_visible()
    expect(chart).to_be_visible(timeout=RUN_TIMEOUT * 1000)
    box = chart.bounding_box()
    assert box is not None and box["height"] >= 300, box
