"""Unit tests of ``core/profile.py``: the base pass, missing values, correlation and issues.

Several functions here replace a pandas call with faster numpy code (performance rules R-P*).
For those, the tests compare against pandas (or an exact reference) on data built to trip the
fast path: missing values, ``-0.0``, nullable dtypes, large values with a small spread.
"""

import numpy as np
import pandas as pd
import pytest
from fixtures.samples import frame_with_missing

from litmetrics.core import profile
from litmetrics.core.profile import IssueKind
from litmetrics.core.schema import IDENTIFIER_MIN_ROWS, infer_schema


def _no_pairs() -> pd.DataFrame:
    """An empty table of correlated pairs: issues are found without correlation."""
    return pd.DataFrame({"column_a": [], "column_b": [], "correlation": []})


def _issues(df: pd.DataFrame) -> list[tuple[IssueKind, tuple[str, ...]]]:
    """Kind and columns of every issue of ``df``, without correlation, high-missing at 50%."""
    found = profile.detect_issues(
        profile.base_profile(df),
        infer_schema(df),
        _no_pairs(),
        high_missing_pct=50.0,
        high_correlation=0.9,
    )
    return [(i.kind, i.columns) for i in found]


# Base pass


def test_by_column_reads_the_counts_of_one_column():
    """``by_column`` gives the counts of a column by its name.

    Case: the index is lost (positions instead of names), so a page reads the wrong column.
    Affects: core/profile.py (BaseProfile.by_column)
    How: reads missing and unique counts of two columns by name.
    """
    base = profile.base_profile(pd.DataFrame({"a": [1, 1, None], "b": ["x", "y", "z"]}))
    assert base.by_column.loc["a", "missing"] == 1
    assert base.by_column.loc["b", "unique"] == 3
    assert list(base.by_column.index) == ["a", "b"]


def test_base_profile_columns(quality_df):
    """Missing, distinct and repeated counts per column, and their shares.

    Case: shares over the wrong total (distinct % over all rows instead of the values present).
    Affects: core/profile.py (base_profile)
    How: computes ``quality_df`` and compares counts worked out by hand.
    """
    table = profile.base_profile(quality_df).columns.set_index("column")
    assert table.loc["x", "missing"] == 1
    assert table.loc["x", "missing_pct"] == pytest.approx(20.0)
    assert table.loc["city", "unique"] == 2
    assert table.loc["city", "unique_pct"] == pytest.approx(50.0)
    assert table.loc["city", "repeated"] == 3
    assert table.loc["y", "repeated"] == 0
    assert table.loc["const", "unique"] == 1


def test_base_profile_summary(quality_df):
    """The headline numbers of Profile: rows, missing cells, rows with missing, constants.

    Case: missing cells counted per row instead of per cell, or memory measured as 0.
    Affects: core/profile.py (base_profile), shown in the headline of ui/pages/profile/page.py
    How: computes ``quality_df`` (2 missing cells of 20, in 2 rows; 1 constant column).
    """
    base = profile.base_profile(quality_df)
    assert base.n_rows == 5
    assert base.missing_cells_pct == pytest.approx(2 / 20 * 100)
    assert base.rows_with_missing == 2
    assert base.duplicate_rows == 0
    assert base.constant_columns == 1
    assert base.memory_bytes > 0


@pytest.mark.parametrize(
    ("df", "duplicates"),
    [
        (pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]}), 1),
        (pd.DataFrame({"id": [1, 2, 3], "b": ["x", "x", "x"]}), 0),
    ],
    ids=["one-repeat", "distinct-column"],
)
def test_base_profile_counts_duplicate_rows(df, duplicates):
    """A row is a duplicate only when every column repeats.

    Case: the shortcut that skips hashing when a column is all distinct counts wrong.
    Affects: core/profile.py (base_profile)
    How: counts one repeated row, and none when one column has no repeats.
    """
    assert profile.base_profile(df).duplicate_rows == duplicates


def test_base_profile_missing_per_row(quality_df):
    """Rows are grouped by how many values they miss.

    Case: rows without missing values are left out of the chart.
    Affects: core/profile.py (base_profile.missing_per_row)
    How: checks ``quality_df`` has 3 complete rows and 2 rows missing one value.
    """
    per_row = profile.base_profile(quality_df).missing_per_row
    assert dict(zip(per_row["missing_in_row"], per_row["rows"], strict=True)) == {0: 3, 1: 2}


def test_base_profile_all_missing_column():
    """A column with no values has 0 distinct values and 0% distinct, not NaN.

    Case: dividing by 0 values present shows "nan%" on screen.
    Affects: core/profile.py (base_profile)
    How: computes a column of two missing values.
    """
    table = profile.base_profile(pd.DataFrame({"a": [None, None]})).columns
    assert table.loc[0, "unique"] == 0
    assert table.loc[0, "unique_pct"] == 0.0


def test_base_profile_empty():
    """An empty DataFrame gives 0% missing and 0 duplicates.

    Case: 0 cells makes the missing share 0 / 0 = NaN.
    Affects: core/profile.py (base_profile)
    How: computes an empty DataFrame.
    """
    base = profile.base_profile(pd.DataFrame())
    assert base.missing_cells_pct == 0.0
    assert base.duplicate_rows == 0


def test_base_profile_in_threads_matches_one_thread():
    """Computing columns in threads gives the same tables as in one thread.

    Case: threads write to a shared result or return columns out of order.
    Affects: core/profile.py (base_profile), core/parallel.py (map_columns)
    How: compares 6 columns of 2,000 rows (one with missing values) with 1 and 4 workers.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"c{i}": rng.normal(size=2_000).round(i) for i in range(6)})
    df.iloc[::7, 2] = np.nan
    one, many = profile.base_profile(df), profile.base_profile(df, workers=4)
    pd.testing.assert_frame_equal(one.columns, many.columns)
    pd.testing.assert_frame_equal(one.missing_per_row, many.missing_per_row)


def _hash_counts(series: pd.Series) -> tuple[int, int]:
    """Distinct values and values repeated, by ``value_counts`` (the slow reference)."""
    counts = series.value_counts()
    counts = counts[counts > 0]
    return int(counts.size), int(counts[counts > 1].sum())


@pytest.mark.parametrize(
    "series",
    [
        pd.Series([1.5, np.nan, 1.5, 2.0, -0.0, 0.0, np.nan]),
        pd.Series([3, 1, 2, 3, 3, 9]),
        pd.Series([np.nan, np.nan]),
        pd.Series([], dtype="float64"),
        pd.Series([1, None, 1, 2**60, 2**60 + 1], dtype="Int64"),
        pd.Series([1.0, None, 1.0], dtype="Float64"),
        pd.Series(np.random.default_rng(0).integers(0, 50, 1_000).astype(float)),
        pd.Series(["a", "a", "b"], dtype=pd.CategoricalDtype(["a", "b", "c"])),
    ],
    ids=[
        "float-nan-zeros",
        "int",
        "all-missing",
        "empty",
        "Int64-big",
        "Float64",
        "few",
        "unused-category",
    ],
)
def test_distinct_counts_match_value_counts(series):
    """The fast distinct / repeated counts equal those of ``value_counts``.

    Case: sorting-based counting treats ``-0.0`` and ``0.0`` apart, counts NaN as a value,
    loses precision above 2**53 in ``Int64``, or counts unused categories.
    Affects: core/profile.py (distinct_and_repeated)
    How: compares both counts on each tricky series.
    """
    assert profile.distinct_and_repeated(series) == _hash_counts(series)


# Missing values


def test_missing_by_column_only_lists_incomplete_sorted(quality_df):
    """Only columns with missing values are listed, most missing first.

    Case: complete columns fill the chart with zero bars.
    Affects: core/profile.py (missing_by_column)
    How: makes ``city`` miss 3 values and checks the order city, x.
    """
    df = quality_df.assign(city=[None, None, None, "a", "a"])
    table = profile.missing_by_column(profile.base_profile(df))
    assert list(table["column"]) == ["city", "x"]
    assert list(table["missing"]) == [3, 1]


def test_rows_with_missing(quality_df):
    """The rows with missing values are returned in order, up to the limit.

    Case: the limit is ignored and a huge table is sent to the browser.
    Affects: core/profile.py (rows_with_missing)
    How: asks for 1 and 10 rows of ``quality_df`` (rows 2 and 4 miss a value).
    """
    assert list(profile.rows_with_missing(quality_df, 1).index) == [2]
    assert list(profile.rows_with_missing(quality_df, 10).index) == [2, 4]


# Correlation


def test_correlation_known_pair(quality_df):
    """A perfectly correlated pair gives 1; the constant column is left out.

    Case: the constant column is kept and fills the matrix with NaN.
    Affects: core/profile.py (correlation_matrix, top_correlated_pairs)
    How: correlates ``quality_df`` (y = 2x) and checks matrix, rows used and the top pair.
    """
    result = profile.correlation_matrix(quality_df, "pearson", max_columns=10, sample_rows=100)
    assert list(result.matrix.columns) == ["x", "y"]
    assert result.matrix.loc["x", "y"] == pytest.approx(1.0)
    assert result.rows_used == 5
    pairs = profile.top_correlated_pairs(result.matrix, 5)
    assert len(pairs) == 1
    assert pairs.loc[0, "correlation"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("df", "columns"),
    [
        (pd.DataFrame({"a": [1, 2, 3], "b": list("xyz")}), []),
        (
            pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 1.0, 2.0], "flag": [True, False, True]}),
            ["a", "b"],
        ),
    ],
    ids=["one-numeric", "bool-left-out"],
)
def test_correlation_uses_only_numeric_columns(df, columns):
    """Only numeric columns are correlated; booleans are not numbers here.

    Case: a matrix of one column is drawn, or booleans (0/1) correlate with everything.
    Affects: core/profile.py (correlation_matrix, top_correlated_pairs)
    How: correlates a frame with one numeric column, and one with a boolean.
    """
    result = profile.correlation_matrix(df, "pearson", 10, 100)
    assert list(result.matrix.columns) == columns
    if not columns:
        assert profile.top_correlated_pairs(result.matrix, 5).empty


def test_correlation_limits_columns_keeping_most_complete():
    """Over the column limit, the most complete columns are kept.

    Case: the first columns are kept, so a nearly empty one takes the place of a full one.
    Affects: core/profile.py (correlation_matrix)
    How: correlates 3 columns with a limit of 2; ``a`` has a missing value.
    """
    df = pd.DataFrame(
        {"a": [1.0, 2.0, np.nan, 4.0], "b": [1.0, 3.0, 2.0, 5.0], "c": [4.0, 1.0, 2.0, 3.0]}
    )
    result = profile.correlation_matrix(df, "spearman", max_columns=2, sample_rows=100)
    assert list(result.matrix.columns) == ["b", "c"]
    assert result.numeric_columns == 3


def test_spearman_uses_the_sample_and_pearson_every_row():
    """Spearman (ranks, slow) uses the sample; Pearson (fast) every row.

    Case: Pearson is sampled (needlessly inexact) or Spearman ranks millions of rows.
    Affects: core/profile.py (correlation_matrix)
    How: correlates 500 rows with a 100-row sample and checks ``rows_used`` of each method.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    assert profile.correlation_matrix(df, "spearman", 10, sample_rows=100).rows_used == 100
    assert profile.correlation_matrix(df, "pearson", 10, sample_rows=100).rows_used == 500


def test_top_pairs_sorted_by_absolute_value():
    """Strong negative correlations rank as high as positive ones.

    Case: pairs are sorted by signed value, so -0.9 comes last.
    Affects: core/profile.py (top_correlated_pairs)
    How: ranks a 3x3 matrix with -0.9, 0.5 and 0.3.
    """
    matrix = pd.DataFrame(
        [[1.0, -0.9, 0.3], [-0.9, 1.0, 0.5], [0.3, 0.5, 1.0]],
        index=list("abc"),
        columns=list("abc"),
    )
    pairs = profile.top_correlated_pairs(matrix, 2)
    assert list(zip(pairs["column_a"], pairs["column_b"], strict=True)) == [("a", "b"), ("b", "c")]


def _exact_pearson(df: pd.DataFrame) -> np.ndarray:
    """Pearson of each pair over its shared rows, centred on the mean of those rows."""
    names = list(df.columns)
    out = np.eye(len(names))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i != j:
                pair = df[[a, b]].dropna().to_numpy()
                x, y = pair[:, 0] - pair[:, 0].mean(), pair[:, 1] - pair[:, 1].mean()
                out[i, j] = (x @ y) / np.sqrt((x @ x) * (y @ y))
    return out


def test_pearson_matches_an_exact_reference_with_missing_values():
    """Pearson with missing values is exact, even on large values with a small spread.

    Case: the fast matrix product uses every row for the means (not each pair's shared rows),
    or sums 1.7e12-sized values without centring and loses precision.
    Affects: core/profile.py (correlation_matrix)
    How: compares ``frame_with_missing`` with a pair-by-pair exact computation (1e-12), and
    with pandas, which itself is only 1e-6 close on the ``big`` column.
    """
    df = frame_with_missing()
    result = profile.correlation_matrix(df, "pearson", 10, 100).matrix.to_numpy()
    np.testing.assert_allclose(result, _exact_pearson(df), atol=1e-12)
    np.testing.assert_allclose(result, df.corr(method="pearson").to_numpy(), atol=1e-6)


def test_spearman_matches_pandas():
    """Spearman equals pandas without missing values, and is close with them.

    Case: ties ranked differently; with missing values, ranking once per column (not per pair)
    drifts too far from the real value.
    Affects: core/profile.py (correlation_matrix)
    How: compares with pandas on ``frame_with_missing`` without and with its missing rows.
    """
    full = frame_with_missing().dropna()
    result = profile.correlation_matrix(full, "spearman", 10, 10_000).matrix.to_numpy()
    np.testing.assert_allclose(result, full.corr(method="spearman").to_numpy(), atol=1e-12)
    df = frame_with_missing()
    result = profile.correlation_matrix(df, "spearman", 10, 10_000).matrix.to_numpy()
    np.testing.assert_allclose(result, df.corr(method="spearman").to_numpy(), atol=0.01)


def test_correlation_of_a_pair_without_shared_rows_is_undefined():
    """Two columns that are never present together have no correlation (NaN), not 0.

    Case: a pair without shared rows is reported as uncorrelated, or as an issue.
    Affects: core/profile.py (correlation_matrix, top_correlated_pairs)
    How: correlates ``a`` and ``b`` filled in disjoint rows; ``c`` overlaps both.
    """
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, np.nan, np.nan, np.nan],
            "b": [np.nan, np.nan, np.nan, 1.0, 5.0, 2.0],
            "c": [1.0, 3.0, 2.0, 4.0, 6.0, 5.0],
        }
    )
    matrix = profile.correlation_matrix(df, "pearson", 10, 100).matrix
    assert np.isnan(matrix.loc["a", "b"])
    assert matrix.loc["a", "a"] == 1.0
    assert profile.top_correlated_pairs(matrix).shape[0] == 2


def test_correlation_ignores_infinite_values():
    """Infinite values are left out of the correlation.

    Case: one ``inf`` turns the whole row and column of the matrix into NaN.
    Affects: core/profile.py (correlation_matrix)
    How: correlates y = 2x with an ``inf`` in the last row of x.
    """
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.inf], "b": [2.0, 4.0, 6.0, 1.0]})
    matrix = profile.correlation_matrix(df, "pearson", 10, 100).matrix
    assert matrix.loc["a", "b"] == pytest.approx(1.0)


# Issues


def test_detect_issues(quality_df):
    """Every kind of issue in ``quality_df`` plus a duplicate row is found.

    Case: one detector stops running (e.g. after a refactor of the base pass) silently.
    Affects: core/profile.py (detect_issues), core/schema.py (infer_schema)
    How: adds a duplicate row and checks high-missing, constant, duplicates and correlation.
    """
    df = pd.concat([quality_df, quality_df.iloc[[1]]], ignore_index=True)
    corr = profile.correlation_matrix(df, "pearson", 10, 100)
    issues = profile.detect_issues(
        profile.base_profile(df),
        infer_schema(df),
        profile.top_correlated_pairs(corr.matrix),
        high_missing_pct=10.0,
        high_correlation=0.9,
    )
    found = {(i.kind, i.columns) for i in issues}
    assert (IssueKind.HIGH_MISSING, ("x",)) in found
    assert (IssueKind.CONSTANT, ("const",)) in found
    assert (IssueKind.DUPLICATE_ROWS, ()) in found
    assert (IssueKind.HIGH_CORRELATION, ("x", "y")) in found


@pytest.mark.parametrize(
    ("df", "column"),
    [
        (
            pd.DataFrame(
                {
                    "code": [f"A{i}" for i in range(IDENTIFIER_MIN_ROWS)],
                    "value": np.linspace(0, 1, IDENTIFIER_MIN_ROWS),
                }
            ),
            "code",
        ),
        (
            pd.DataFrame(
                {"n": range(IDENTIFIER_MIN_ROWS), "f": np.linspace(0, 1, IDENTIFIER_MIN_ROWS)}
            ),
            "n",
        ),
    ],
    ids=["text-code", "integer"],
)
def test_detect_possible_id(df, column):
    """Distinct text or integers are a possible id; distinct floats are not.

    Case: every continuous measurement (all distinct floats) is flagged as an id.
    Affects: core/profile.py (detect_issues), core/schema.py (IDENTIFIER_MIN_ROWS)
    How: pairs a distinct text / integer column with distinct floats; only the first is flagged.
    """
    assert _issues(df) == [(IssueKind.POSSIBLE_ID, (column,))]


def test_every_issue_kind_is_described():
    """Every kind of issue has its title, severity and advice for the report.

    Case: a new ``IssueKind`` is detected but missing from ``ISSUE_INFO``, and the report of
    any dataset with that issue fails with ``KeyError`` when downloaded.
    Affects: core/profile.py (ISSUE_INFO, Issue.info)
    How: compares the keys of ``ISSUE_INFO`` with ``IssueKind`` and checks each severity.
    """
    assert set(profile.ISSUE_INFO) == set(IssueKind)
    assert all(info.severity in profile.SEVERITIES for info in profile.ISSUE_INFO.values())
