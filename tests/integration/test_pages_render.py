"""Integration tests: the app starts and every page and part of Profile draws without errors.

These are the broad safety net: every page with and without a dataset, every Profile tab and
Columns view, and the column inspector for each semantic type, on ``every_type_frame``, which
holds the hard cases (placeholder dates, an all-missing column, undefined skewness).
What each page shows is checked in ``test_load_page.py`` and the acceptance tests.
"""

from pathlib import Path

import pytest
from fixtures.app_scripts import TIMEOUT, captions, errors, run_page, run_profile
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "src" / "litmetrics" / "app.py"

# (module in litmetrics.ui.pages, entry point). Add every new page here.
PAGES = [
    ("home", "home_page"),
    ("load", "load_page"),
    ("profile", "profile_page"),
    ("visualize", "visualize_page"),
]
PROFILE_TABS = [
    ":material/warning: Issues",
    ":material/view_column: Columns",
    ":material/block: Missing values",
    ":material/link: Correlation",
]
COLUMN_VIEWS = ["Overview", "Numeric", "Categorical", "Dates", "Text & ids"]
# One column of each semantic type and hard case of ``samples.every_type_frame``.
INSPECTED_COLUMNS = [
    "id", "age", "score", "city", "active", "joined", "valid_until", "comment", "empty", "pair",
]  # fmt: skip


def _profile(*args) -> AppTest:
    """Run ``run_profile`` with ``args`` and return the result, failing on any exception."""
    at = AppTest.from_function(run_profile, args=args, default_timeout=TIMEOUT).run()
    assert not at.exception, errors(at)
    return at


def test_app_starts():
    """app.py runs: page config, styles, navigation, sidebar.

    Case: an import error, a missing theme or a broken page registration stops the whole app.
    Affects: app.py, ui/style.py, ui/sidebar.py, every page module (imported by app.py)
    How: runs app.py as a file in AppTest (it opens the Home page).
    """
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    assert not at.exception, errors(at)


@pytest.mark.parametrize(("module", "func"), PAGES)
@pytest.mark.parametrize("with_dataset", [False, True], ids=["no-data", "data"])
def test_page_renders(module, func, with_dataset):
    """Every page draws with and without an active dataset.

    Case: a page assumes a dataset exists (``None`` attribute error) or fails on real data.
    Affects: ui/pages/*, ui/state.py, ui/cache.py, core/ (through the pages)
    How: draws each page on its own, empty and with ``QUALITY_CSV`` loaded.
    """
    at = AppTest.from_function(
        run_page, args=(module, func, with_dataset), default_timeout=TIMEOUT
    ).run()
    assert not at.exception, errors(at)


@pytest.mark.parametrize("tab", PROFILE_TABS)
def test_profile_tab_renders(tab):
    """Every Profile tab draws on a dataset with every semantic type.

    Case: one tab fails on a hard column (all missing, dates out of the nanosecond range).
    Affects: ui/pages/profile/page.py, ui/cache.py, core/profile.py, core/columns.py, ui/charts.py
    How: opens each tab through its widget key on ``every_type_frame``.
    """
    _profile(tab)


@pytest.mark.parametrize("view", COLUMN_VIEWS)
def test_profile_columns_view_renders(view):
    """Every view of the Columns tab draws a table.

    Case: a view raises or shows nothing for the columns of its type.
    Affects: ui/pages/profile/columns_tab.py (tab), core/columns.py
    How: opens the Columns tab with each view and checks a dataframe is drawn.
    """
    at = _profile(PROFILE_TABS[1], view)
    assert at.dataframe, "the view shows a table"


@pytest.mark.parametrize("column", INSPECTED_COLUMNS)
def test_profile_inspector_renders_every_type(column):
    """The inspector draws for a column of each type and each hard case.

    Case: the inspector of one type raises (e.g. a histogram of an all-missing column).
    Affects: ui/pages/profile/columns_tab.py (_inspector), core/columns.py (inspect_column),
    ui/charts.py
    How: draws only the inspector of each column (AppTest cannot select a table row).
    """
    _profile(PROFILE_TABS[0], None, column)


def test_profile_inspector_never_shows_nan():
    """No metric of the inspector shows "nan", even when a statistic is undefined.

    Case: the skewness of a column with two values is NaN and is printed as "nan".
    Affects: ui/pages/profile/columns_tab.py (_inspector)
    How: inspects ``pair`` (two values) and reads every metric.
    """
    at = _profile(PROFILE_TABS[0], None, "pair")
    assert not any("nan" in m.value.lower() for m in at.metric)


@pytest.mark.parametrize("chart", ["Histogram", "Boxplot"])
def test_profile_inspector_switches_numeric_chart(chart):
    """The numeric inspector draws both of its charts.

    Case: the boxplot builder fails on data the histogram accepts, or the choice is ignored.
    Affects: ui/pages/profile/columns_tab.py (_inspector), ui/charts.py (histogram/boxplot_options)
    How: inspects ``income`` with each chart chosen and checks the selected option.
    """
    at = _profile(PROFILE_TABS[0], None, "income", chart)
    assert at.button_group[0].value == chart


def test_profile_inspector_shows_a_column_name_as_written():
    """The inspector's title is the column name as written, whatever characters it holds.

    Case: the title is drawn as Markdown, so ``*net* $ income`` shows in italics, or as a
    formula between two ``$``.
    Affects: ui/pages/profile/columns_tab.py (_inspector), ui/text.py (md)
    How: renames ``income`` to ``*net* $ income``, inspects it and reads the subheader, whose
    Markdown must escape every special character.
    """
    name = "*net* $ income"
    at = _profile(PROFILE_TABS[0], None, name, None, (), 0, (("income", name),))
    assert at.subheader[0].value == r"\*net\* \$ income"


def test_profile_correlation_with_many_columns_renders():
    """With many numeric columns the heatmap goes full width with the top pairs below.

    Case: the wide layout (another code path) raises or drops the table of pairs.
    Affects: ui/pages/profile/page.py (_correlation), core/profile.py, ui/charts.py (heatmap)
    How: adds 14 numeric columns, opens Correlation and checks a table is drawn.
    """
    at = _profile(PROFILE_TABS[3], None, None, None, (), 14)
    assert at.dataframe, "the most correlated pairs are shown below the heatmap"


def test_profile_only_runs_the_open_tab():
    """Only the open tab runs (R-E1).

    Case: every tab is computed on every rerun, so a click on Missing values also computes the
    correlation of millions of rows.
    Affects: ui/pages/profile/page.py (profile_page, ``st.tabs(on_change="rerun")``)
    How: opens Missing values and checks the radio of the Correlation tab was not drawn.
    """
    at = _profile(PROFILE_TABS[2])
    assert not at.radio


@pytest.mark.parametrize("tab", PROFILE_TABS)
def test_profile_tab_renders_with_skipped_columns(tab):
    """Every tab draws with columns skipped, and the summary says how many.

    Case: a tab uses the whole dataset with the view's schema (``KeyError`` on a skipped name).
    Affects: ui/pages/profile/page.py, ui/cache.py (view_id), ui/state.py (skipped columns)
    How: skips ``income`` and ``city`` and opens each tab.
    """
    at = _profile(tab, None, None, None, ("income", "city"))
    assert "(2 skipped)" in captions(at)
