"""Unit tests of ``core/report.py``: the data quality report rendered as Markdown.

The report is read by people in a Markdown viewer, so the tests check what breaks it there:
names that split a table, sections numbered wrong, a division by an empty dataset, and
content of columns the user left out.
"""

import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from litmetrics.core import profile, report
from litmetrics.core.io.base import LoadedDataset
from litmetrics.core.schema import infer_schema

# Settings of a dataset small enough to be used whole: no sample, every column compared.
SETTINGS = report.ReportSettings(
    high_missing_pct=50.0,
    high_correlation=0.9,
    correlation_method="pearson",
    sample_rows=None,
    seed=0,
    correlation_columns=2,
    numeric_columns=2,
)
# Separator of table cells: a ``|`` not escaped.
_CELL = re.compile(r"(?<!\\)\|")


def _render(dataset: LoadedDataset, **kwargs) -> str:
    """The report of ``dataset`` with its real base pass, schema and issues (no correlation)."""
    df = dataset.df
    base = profile.base_profile(df)
    schema = infer_schema(df)
    pairs = pd.DataFrame({"column_a": [], "column_b": [], "correlation": []})
    issues = profile.detect_issues(base, schema, pairs, high_missing_pct=50.0, high_correlation=0.9)
    return report.issues_report(
        dataset,
        base,
        schema,
        issues,
        kwargs.pop("settings", SETTINGS),
        memory="1 KB",
        generated_at=datetime(2026, 11, 25, 16, 40),
        version="0.0.0",
        **kwargs,
    )


def _table_rows(markdown: str) -> list[str]:
    return [line for line in markdown.splitlines() if line.startswith("|")]


def test_names_with_pipes_and_backticks_keep_every_table_row_whole():
    """A column named with ``|`` or a backtick stays in its cell, in every table.

    Case: ``a|b`` splits a row of the summary and columns tables into one cell more, and a
    backtick in a name closes the inline code early: the tables fall apart in the viewer.
    Affects: core/report.py (_code, _cell), templates/issues_report.md.j2
    How: renders a dataset whose constant column is ``a|b`` and whose sparse one is
    ``x`y``, then counts the cells of every table row against its header.
    """
    df = pd.DataFrame({"a|b": [1] * 6, "x`y": [1.0, *[np.nan] * 5], "v": range(6)})
    markdown = _render(LoadedDataset(df, "data.csv", "CSV"))
    rows = _table_rows(markdown)
    widths = {len(_CELL.split(row)) for row in rows if not row.startswith("| |")}
    # Tables of 2 (dataset), 4 (summary) and 5 (columns) cells, plus the edges.
    assert widths == {4, 6, 7}
    assert "`` x`y ``" in markdown
    # Outside a table, no backslash is shown in the name.
    assert "- `a|b`: Has a single value" in markdown


def test_each_kind_of_issue_is_explained_once_with_all_its_cases():
    """Why a kind of issue matters and what to do are written once, followed by every case.

    Case: three sparse columns repeat the same explanation three times, and the reader has
    to scroll through it to find which columns are affected.
    Affects: core/report.py (groups), core/profile.py (ISSUE_INFO), templates/issues_report.md.j2
    How: renders a dataset with three mostly empty columns and a constant one, and reads the
    section of each kind.
    """
    sparse = [1.0, *[np.nan] * 5]
    df = pd.DataFrame({"a": sparse, "b": sparse, "c": sparse, "k": [7] * 6, "v": range(6)})
    markdown = _render(LoadedDataset(df, "sparse.csv", "CSV"))
    issues = markdown[markdown.index(". Issues\n") : markdown.index(". Columns\n")]
    assert issues.count("**Why it matters:**") == 2
    assert issues.count("**Suggested action:**") == 2
    missing = issues[issues.index("### Many missing values") : issues.index("### Constant")]
    assert "**Found:** 3" in missing
    assert [line[:5] for line in missing.splitlines() if line.startswith("- ")] == [
        "- `a`",
        "- `b`",
        "- `c`",
    ]


def test_empty_dataset_gives_a_report_without_issues():
    """A dataset without rows is reported, with no issue and no division by zero.

    Case: the share of rows with missing values divides by ``n_rows`` (ZeroDivisionError), or
    an empty list of issues leaves an empty "Issues" section.
    Affects: core/report.py (issues_report), templates/issues_report.md.j2
    How: renders a 0-row dataset with two columns.
    """
    df = pd.DataFrame({"a": pd.Series([], dtype="float64"), "b": pd.Series([], dtype="str")})
    markdown = _render(LoadedDataset(df, "empty.csv", "CSV"))
    assert "| Rows with a missing value | 0 (0.0 %) |" in markdown
    assert "No issues found with the current thresholds (section 4)." in markdown
    assert ". Issues" not in markdown


@pytest.mark.parametrize("with_issues", [False, True], ids=["no-issues", "issues"])
@pytest.mark.parametrize("changed", [False, True], ids=["as-read", "values-changed-on-load"])
def test_sections_are_numbered_in_order(with_issues, changed):
    """Sections are numbered 1, 2, 3... whichever of the optional ones are there.

    Case: without issues (no "Issues" section) or without values changed on load, the
    sections after them keep their numbers and the report jumps from 2 to 4, or "Method" is
    cited by a number that does not exist.
    Affects: core/report.py (sections), templates/issues_report.md.j2
    How: renders data with and without a constant column and an unparsed value, and reads
    the headings and the section cited when there are no issues.
    """
    df = pd.DataFrame({"price": [1.0, 2.0, 3.0], "flag": [1, 1, 1 if with_issues else 2]})
    unparsed = {"price": 1} if changed else {}
    markdown = _render(LoadedDataset(df, "orders.csv", "CSV", unparsed=unparsed))
    numbers = re.findall(r"^## (\d+)\. ", markdown, flags=re.MULTILINE)
    assert numbers == [str(i) for i in range(1, len(numbers) + 1)]
    assert len(numbers) == 4 + with_issues + changed
    if not with_issues:
        assert f"(section {numbers[-1]})" in markdown


def test_columns_left_out_are_named_but_not_described():
    """Columns the user skipped are listed as left out, and not in "Values changed on load".

    Case: the dataset keeps ``unparsed`` of every column of the file, so a skipped column
    shows up in the report the user made without it.
    Affects: core/report.py (issues_report)
    How: renders a view without ``date``, whose ``unparsed`` still names it.
    """
    df = pd.DataFrame({"price": [1.0, 2.0, np.nan]})
    dataset = LoadedDataset(df, "orders.csv", "CSV", unparsed={"date": 4, "price": 1})
    markdown = _render(dataset, skipped=("date",))
    assert "| Columns left out | `date` |" in markdown
    assert "| `price` | 1 |" in markdown
    assert "\n| `date` |" not in markdown


def test_correlation_sample_and_columns_left_out_are_disclosed():
    """The method says when correlation used a sample or compared only some columns.

    Case: on a large dataset the report claims every pair was checked, so the reader trusts
    that no correlated pair is missing.
    Affects: core/report.py (ReportSettings), templates/issues_report.md.j2
    How: renders with settings of a sampled correlation that kept 30 of 40 columns.
    """
    settings = report.ReportSettings(50.0, 0.9, "pearson", 200_000, 0, 30, 40)
    df = pd.DataFrame({"v": [1, 2, 3]})
    markdown = _render(LoadedDataset(df, "big.csv", "CSV"), settings=settings)
    assert "a random sample of 200,000 rows (seed 0)" in markdown
    assert "Only 30 of the 40 numeric columns were compared" in markdown
    assert "computed over every row" not in markdown


@pytest.mark.parametrize(
    ("name", "part", "expected"),
    [
        pytest.param("titanic.csv", None, "titanic_issues.md", id="file"),
        pytest.param("shop.db", "orders", "shop_orders_issues.md", id="table-of-a-database"),
    ],
)
def test_report_file_name(name, part, expected):
    """The report is named after its dataset, and two tables of one file get two names.

    Case: the reports of two tables of ``shop.db`` download as the same file.
    Affects: core/report.py (report_file_name)
    How: names the report of a file and of a table of a database.
    """
    assert report.report_file_name(name, part) == expected
