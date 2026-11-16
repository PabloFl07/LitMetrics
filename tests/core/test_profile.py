import numpy as np
import pandas as pd
import pytest

from litmetrics.core import profile
from litmetrics.core.profile import IssueKind
from litmetrics.core.schema import IDENTIFIER_MIN_ROWS, infer_schema


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, np.nan],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0],
            "const": [7, 7, 7, 7, 7],
            "city": ["a", "b", None, "a", "a"],
        }
    )


def _no_pairs() -> pd.DataFrame:
    return pd.DataFrame({"column_a": [], "column_b": [], "correlation": []})


# Base pass


def test_by_column_reads_the_counts_of_one_column():
    base = profile.base_profile(pd.DataFrame({"a": [1, 1, None], "b": ["x", "y", "z"]}))
    assert base.by_column.loc["a", "missing"] == 1
    assert base.by_column.loc["b", "unique"] == 3
    assert list(base.by_column.index) == ["a", "b"]


def test_base_profile_columns(df):
    table = profile.base_profile(df).columns.set_index("column")
    assert table.loc["x", "missing"] == 1
    assert table.loc["x", "missing_pct"] == pytest.approx(20.0)
    assert table.loc["city", "unique"] == 2
    assert table.loc["city", "unique_pct"] == pytest.approx(50.0)
    assert table.loc["city", "repeated"] == 3
    assert table.loc["y", "repeated"] == 0
    assert table.loc["const", "unique"] == 1


def test_base_profile_summary(df):
    base = profile.base_profile(df)
    assert base.n_rows == 5
    assert base.missing_cells_pct == pytest.approx(2 / 20 * 100)
    assert base.rows_with_missing == 2
    assert base.duplicate_rows == 0
    assert base.constant_columns == 1
    assert base.memory_bytes > 0


def test_base_profile_counts_duplicates():
    base = profile.base_profile(pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]}))
    assert base.duplicate_rows == 1


def test_base_profile_no_duplicates_when_a_column_is_all_distinct():
    base = profile.base_profile(pd.DataFrame({"id": [1, 2, 3], "b": ["x", "x", "x"]}))
    assert base.duplicate_rows == 0


def test_base_profile_missing_per_row(df):
    per_row = profile.base_profile(df).missing_per_row
    assert dict(zip(per_row["missing_in_row"], per_row["rows"], strict=True)) == {0: 3, 1: 2}


def test_base_profile_all_missing_column():
    table = profile.base_profile(pd.DataFrame({"a": [None, None]})).columns
    assert table.loc[0, "unique"] == 0
    assert table.loc[0, "unique_pct"] == 0.0


def test_base_profile_empty():
    base = profile.base_profile(pd.DataFrame())
    assert base.missing_cells_pct == 0.0
    assert base.duplicate_rows == 0


def test_distinct_and_repeated_ignores_unused_categories():
    series = pd.Series(["a", "a", "b"], dtype=pd.CategoricalDtype(["a", "b", "c"]))
    assert profile.distinct_and_repeated(series) == (2, 2)


def test_distinct_and_repeated_ignores_missing():
    assert profile.distinct_and_repeated(pd.Series([1.0, np.nan, np.nan, 2.0])) == (2, 0)


# Missing values


def test_missing_by_column_only_lists_incomplete_sorted(df):
    df = df.assign(city=[None, None, None, "a", "a"])
    table = profile.missing_by_column(profile.base_profile(df))
    assert list(table["column"]) == ["city", "x"]
    assert list(table["missing"]) == [3, 1]


def test_rows_with_missing(df):
    assert list(profile.rows_with_missing(df, 1).index) == [2]
    assert list(profile.rows_with_missing(df, 10).index) == [2, 4]


# Correlation


def test_correlation_known_pair(df):
    result = profile.correlation_matrix(df, "pearson", max_columns=10, sample_rows=100)
    # The constant column is left out.
    assert list(result.matrix.columns) == ["x", "y"]
    assert result.matrix.loc["x", "y"] == pytest.approx(1.0)
    assert result.rows_used == 5
    pairs = profile.top_correlated_pairs(result.matrix, 5)
    assert len(pairs) == 1
    assert pairs.loc[0, "correlation"] == pytest.approx(1.0)


def test_correlation_needs_two_numeric_columns():
    result = profile.correlation_matrix(
        pd.DataFrame({"a": [1, 2, 3], "b": list("xyz")}), "pearson", 10, 100
    )
    assert result.matrix.empty
    assert profile.top_correlated_pairs(result.matrix, 5).empty


def test_correlation_ignores_bool_columns():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 1.0, 2.0], "flag": [True, False, True]})
    result = profile.correlation_matrix(df, "pearson", 10, 100)
    assert list(result.matrix.columns) == ["a", "b"]


def test_correlation_limits_columns_keeping_most_complete():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0],
            "b": [1.0, 3.0, 2.0, 5.0],
            "c": [4.0, 1.0, 2.0, 3.0],
        }
    )
    result = profile.correlation_matrix(df, "spearman", max_columns=2, sample_rows=100)
    assert list(result.matrix.columns) == ["b", "c"]
    assert result.numeric_columns == 3


def test_spearman_uses_the_sample_and_pearson_every_row():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    assert profile.correlation_matrix(df, "spearman", 10, sample_rows=100).rows_used == 100
    assert profile.correlation_matrix(df, "pearson", 10, sample_rows=100).rows_used == 500


def test_top_pairs_sorted_by_absolute_value():
    matrix = pd.DataFrame(
        [[1.0, -0.9, 0.3], [-0.9, 1.0, 0.5], [0.3, 0.5, 1.0]],
        index=list("abc"),
        columns=list("abc"),
    )
    pairs = profile.top_correlated_pairs(matrix, 2)
    assert list(zip(pairs["column_a"], pairs["column_b"], strict=True)) == [("a", "b"), ("b", "c")]


# Issues


def test_detect_issues(df):
    df = pd.concat([df, df.iloc[[1]]], ignore_index=True)
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


def test_detect_possible_id_skips_floats():
    n = IDENTIFIER_MIN_ROWS
    df = pd.DataFrame({"code": [f"A{i}" for i in range(n)], "value": np.linspace(0, 1, n)})
    issues = profile.detect_issues(
        profile.base_profile(df),
        infer_schema(df),
        _no_pairs(),
        high_missing_pct=50.0,
        high_correlation=0.9,
    )
    assert [(i.kind, i.columns) for i in issues] == [(IssueKind.POSSIBLE_ID, ("code",))]


# Equivalence with pandas of the faster methods


def _hash_counts(series: pd.Series) -> tuple[int, int]:
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
    ],
    ids=["float-nan-zeros", "int", "all-missing", "empty", "Int64-big", "Float64", "few"],
)
def test_distinct_counts_match_value_counts(series):
    assert profile.distinct_and_repeated(series) == _hash_counts(series)


def test_base_profile_in_threads_matches_one_thread():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"c{i}": rng.normal(size=2_000).round(i) for i in range(6)})
    df.iloc[::7, 2] = np.nan
    one, many = profile.base_profile(df), profile.base_profile(df, workers=4)
    pd.testing.assert_frame_equal(one.columns, many.columns)
    pd.testing.assert_frame_equal(one.missing_per_row, many.missing_per_row)


def _frame_with_missing(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=3_000)
    df = pd.DataFrame(
        {
            "x": x,
            "y": 2 * x + rng.normal(size=3_000),
            "z": rng.normal(size=3_000),
            # Large values with a small spread: the sums cancel out unless centred.
            "big": 1.7e12 + x * 1e3 + rng.normal(size=3_000),
        }
    )
    for name, share in (("x", 0.1), ("y", 0.3), ("big", 0.2)):
        df.loc[rng.random(3_000) < share, name] = np.nan
    return df


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


def test_pearson_matches_pandas_with_missing_values():
    df = _frame_with_missing()
    result = profile.correlation_matrix(df, "pearson", 10, 100).matrix.to_numpy()
    # pandas loses precision on the large column (2.5e-8 off); the exact value is the reference.
    np.testing.assert_allclose(result, _exact_pearson(df), atol=1e-12)
    np.testing.assert_allclose(result, df.corr(method="pearson").to_numpy(), atol=1e-6)


def test_spearman_matches_pandas_without_missing_values():
    df = _frame_with_missing().dropna()
    result = profile.correlation_matrix(df, "spearman", 10, 10_000).matrix
    np.testing.assert_allclose(result.to_numpy(), df.corr(method="spearman").to_numpy(), atol=1e-12)


def test_spearman_with_missing_values_is_close_to_pandas():
    # Ranked once per column, not per pair: a small difference when values are missing.
    df = _frame_with_missing()
    result = profile.correlation_matrix(df, "spearman", 10, 10_000).matrix
    np.testing.assert_allclose(result.to_numpy(), df.corr(method="spearman").to_numpy(), atol=0.01)


def test_correlation_of_a_pair_without_shared_rows_is_undefined():
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
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.inf], "b": [2.0, 4.0, 6.0, 1.0]})
    matrix = profile.correlation_matrix(df, "pearson", 10, 100).matrix
    assert matrix.loc["a", "b"] == pytest.approx(1.0)


def test_detect_possible_id_of_integer_column():
    n = IDENTIFIER_MIN_ROWS
    df = pd.DataFrame({"n": range(n), "f": np.linspace(0, 1, n)})
    issues = profile.detect_issues(
        profile.base_profile(df),
        infer_schema(df),
        _no_pairs(),
        high_missing_pct=50.0,
        high_correlation=0.9,
    )
    assert [(i.kind, i.columns) for i in issues] == [(IssueKind.POSSIBLE_ID, ("n",))]
