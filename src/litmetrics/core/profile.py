"""Profile of a dataset: missing values, distinct values, duplicates, correlation and issues.

Pure functions over a DataFrame, independent of Streamlit. The Profile page shows their results
and ``ui/cache.py`` caches them per dataset.

Built for large datasets (rules in docs/performance.md):
- ``base_profile`` is the only pass over the whole data for the cheap, exact counts (R-P1).
  Every section starts from its result instead of calling ``isna`` or ``nunique`` again.
- Expensive measures run on ``sample(df, n)``, a sample with a fixed seed (R-P2). Spearman is
  always computed on it (R-P3).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from litmetrics.core.schema import IDENTIFIER_MIN_ROWS, Schema, SemanticType, mostly_distinct

type CorrelationMethod = Literal["pearson", "spearman"]

# Seed of every sample, so the same dataset always gives the same results.
SAMPLE_SEED = 0


@dataclass(frozen=True, eq=False)
class BaseProfile:
    """Exact counts over the whole dataset, computed in a single pass.

    Attributes:
        n_rows: Rows of the dataset.
        columns: One row per column of the dataset: ``column``, ``pandas_dtype``, ``missing``,
            ``missing_pct``, ``unique`` (distinct non-missing values), ``unique_pct`` (their
            share of the non-missing values) and ``repeated`` (non-missing values that appear
            more than once, counting every copy).
        missing_per_row: How many rows have 0, 1, 2... missing values, as ``missing_in_row`` and
            ``rows``. Only the counts that occur are listed.
        duplicate_rows: Rows that are an exact copy of an earlier one.
        memory_bytes: Memory used by the DataFrame, text included.
    """

    n_rows: int
    columns: pd.DataFrame
    missing_per_row: pd.DataFrame
    duplicate_rows: int
    memory_bytes: int

    @property
    def missing_cells_pct(self) -> float:
        cells = self.n_rows * len(self.columns)
        return float(self.columns["missing"].sum() / cells * 100) if cells else 0.0

    @property
    def rows_with_missing(self) -> int:
        counts = self.missing_per_row
        complete = counts.loc[counts["missing_in_row"] == 0, "rows"].sum()
        return self.n_rows - int(complete)

    @property
    def constant_columns(self) -> int:
        return int((self.columns["unique"] == 1).sum())


class IssueKind(StrEnum):
    HIGH_MISSING = "high_missing"
    CONSTANT = "constant"
    POSSIBLE_ID = "possible_id"
    DUPLICATE_ROWS = "duplicate_rows"
    HIGH_CORRELATION = "high_correlation"


@dataclass(frozen=True)
class Issue:
    """A problem found in the dataset. ``message`` is meant for the user."""

    kind: IssueKind
    columns: tuple[str, ...]
    message: str


@dataclass(frozen=True, eq=False)
class CorrelationResult:
    """Correlation between numeric columns.

    Attributes:
        matrix: Square matrix of the columns used. Empty if fewer than two could be used.
        numeric_columns: Numeric non-constant columns in the dataset, used or not. More than
            ``len(matrix)`` means some were left out because of ``max_columns``.
        rows_used: Rows the matrix was computed on. Fewer than the dataset has when it was
            computed on a sample.
    """

    matrix: pd.DataFrame
    numeric_columns: int
    rows_used: int


# Sampling


def sample[T: (pd.DataFrame, pd.Series)](data: T, n: int) -> T:
    """``n`` random rows of ``data`` with a fixed seed, or all of it if it has ``n`` or fewer.

    The seed is the same for every call, so a DataFrame and any of its columns are sampled on
    the same rows.
    """
    if len(data) <= n:
        return data
    return data.sample(n=n, random_state=SAMPLE_SEED)


# Base pass


def base_profile(df: pd.DataFrame) -> BaseProfile:
    """Missing values, distinct values, duplicates and memory of ``df``, exactly."""
    n_rows = len(df)
    missing = df.isna()
    missing_by_column = missing.sum().to_numpy()
    per_row = missing.sum(axis=1).value_counts().sort_index()
    non_missing = n_rows - missing_by_column
    counted = [distinct_and_repeated(series) for _, series in df.items()]
    unique = np.array([u for u, _ in counted], dtype=np.int64)
    repeated = np.array([r for _, r in counted], dtype=np.int64)
    columns = pd.DataFrame(
        {
            "column": df.columns,
            "pandas_dtype": [str(dtype) for dtype in df.dtypes],
            "missing": missing_by_column,
            "missing_pct": _pct(missing_by_column, np.full(len(df.columns), n_rows)),
            "unique": unique,
            "unique_pct": _pct(unique, non_missing),
            "repeated": repeated,
        }
    )
    # A column with a different value in every row (an id) makes every row different: no
    # need to compare whole rows, the slowest count of all (R-P4).
    all_distinct = n_rows > 0 and bool((unique == n_rows).any())
    return BaseProfile(
        n_rows=n_rows,
        columns=columns,
        missing_per_row=pd.DataFrame(
            {"missing_in_row": per_row.index.to_numpy(), "rows": per_row.to_numpy()}
        ),
        duplicate_rows=0 if all_distinct else _count_duplicate_rows(df),
        memory_bytes=int(df.memory_usage(deep=True).sum()),
    )


def distinct_and_repeated(series: pd.Series) -> tuple[int, int]:
    """Distinct non-missing values of a column, and how many values appear more than once.

    Both come from one ``value_counts`` (a hash table), instead of ``nunique`` here and
    ``duplicated`` in the text table.

    Deviation from R-P4, measured (Principle 4): with pandas 3.1 and numpy 2.5, on 7.3 M rows,
    ``np.unique`` took 5.4 s on an id column and 6.0 s on a date column, against 0.2 s and 1.0 s
    for pandas' hash-based counting; ``np.unique`` sorts, and only wins on columns with very few
    distinct values (50 ms against 70 ms).
    """
    # sort=False: ordering millions of distinct values by count is slower than counting them
    # (5.8 s against 1.6 s on 7.3 M phone numbers).
    # Filtered as a numpy array: boolean indexing of a Series with millions of text labels
    # spends more time checking the index than counting.
    counts = series.value_counts(sort=False).to_numpy()
    # A category column also lists the categories that never appear.
    counts = counts[counts > 0]
    return int(counts.size), int(counts[counts > 1].sum())


def _count_duplicate_rows(df: pd.DataFrame) -> int:
    """Rows equal to an earlier one, through a 64-bit hash of each row (R-P4).

    Faster than ``df.duplicated()``. Two different rows with the same hash would count as a
    duplicate; with 64 bits that is negligible for any dataset that fits in memory.
    """
    if df.shape[1] == 0:
        return 0
    return int(pd.util.hash_pandas_object(df, index=False).duplicated().sum())


def _pct(part: np.ndarray, whole: np.ndarray) -> np.ndarray:
    """``part / whole * 100`` element by element, 0 where ``whole`` is 0."""
    return np.divide(part * 100.0, whole, out=np.zeros(len(part), dtype=float), where=whole > 0)


# Missing values


def missing_by_column(base: BaseProfile) -> pd.DataFrame:
    """Columns with at least one missing value, the most incomplete first."""
    table = base.columns[["column", "missing", "missing_pct"]]
    table = table[table["missing"] > 0]
    return table.sort_values("missing", ascending=False, kind="stable", ignore_index=True)


def rows_with_missing(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """First ``n`` rows that have at least one missing value (R-P7)."""
    positions = np.flatnonzero(df.isna().any(axis=1).to_numpy())[:n]
    return df.iloc[positions]


# Correlation


def correlation_matrix(
    df: pd.DataFrame,
    method: CorrelationMethod,
    max_columns: int,
    sample_rows: int,
    columns: Sequence[str] | None = None,
) -> CorrelationResult:
    """Correlation between the numeric columns of ``df``.

    ``columns`` are the columns to compare, usually the ``NUMERIC`` ones of the schema, so that
    integer ids (which grow with the row order) do not show up as correlated. ``None`` takes
    every numeric column. Infinite values are ignored like missing ones.

    Constant columns are left out (their correlation is undefined). If more than
    ``max_columns`` remain, the ones with the fewest missing values are kept, so the matrix
    stays readable and is computed over as many rows as possible.

    Pearson uses every row. Spearman ranks the values, which is slow on large data, so it uses
    ``sample(df, sample_rows)`` (R-P3).
    """
    numeric = df.select_dtypes("number") if columns is None else df[list(columns)]
    # As missing: a single inf makes std() NaN, and the column would be left out.
    numeric = numeric.astype("float64")
    numeric = numeric.where(np.isfinite(numeric.to_numpy()))
    # std > 0 leaves out constant and empty columns; much cheaper than ``nunique() > 1`` (R-P1).
    numeric = numeric.loc[:, (numeric.std() > 0).to_numpy()]
    n_numeric = numeric.shape[1]
    if n_numeric < 2:
        return CorrelationResult(pd.DataFrame(), n_numeric, len(df))
    if n_numeric > max_columns:
        # sort_values is stable, so ties keep the original column order.
        keep = numeric.isna().sum().sort_values(kind="stable").index[:max_columns]
        numeric = numeric[[c for c in numeric.columns if c in keep]]
    if method == "spearman":
        numeric = sample(numeric, sample_rows)
    return CorrelationResult(numeric.corr(method=method), n_numeric, len(numeric))


def top_correlated_pairs(matrix: pd.DataFrame, n: int | None = None) -> pd.DataFrame:
    """The ``n`` pairs of distinct columns with the strongest correlation, by absolute value.

    ``None`` returns every pair.
    """
    if matrix.empty:
        return pd.DataFrame({"column_a": [], "column_b": [], "correlation": []})
    columns = list(matrix.columns)
    values = matrix.to_numpy()
    rows = [
        (a, b, values[i, j])
        for i, a in enumerate(columns)
        for j, b in enumerate(columns)
        if j > i and not np.isnan(values[i, j])
    ]
    pairs = pd.DataFrame(rows, columns=["column_a", "column_b", "correlation"])
    order = pairs["correlation"].abs().sort_values(ascending=False, kind="stable").index
    pairs = pairs.loc[order]
    return (pairs if n is None else pairs.head(n)).reset_index(drop=True)


# Issues


def detect_issues(
    df: pd.DataFrame,
    base: BaseProfile,
    schema: Schema,
    pairs: pd.DataFrame,
    *,
    high_missing_pct: float,
    high_correlation: float,
) -> list[Issue]:
    """Problems worth pointing out to the user, from results already computed (R-P6).

    Args:
        df: The dataset, only used to look at the dtype of the columns.
        base: Output of ``base_profile(df)``.
        schema: Semantic types of ``df``. Identifiers are flagged as possible ids.
        pairs: Output of ``top_correlated_pairs`` with every pair (``n=None``), so no pair
            above ``high_correlation`` is missed.
    """
    issues: list[Issue] = []
    n_rows = base.n_rows

    for row in base.columns.itertuples(index=False):
        name = str(row.column)
        n_values = n_rows - int(row.missing)
        if row.missing_pct > high_missing_pct:
            issues.append(
                Issue(IssueKind.HIGH_MISSING, (name,), f"{row.missing_pct:.1f}% of values missing")
            )
        if row.unique == 1:
            issues.append(Issue(IssueKind.CONSTANT, (name,), "Has a single value"))
        elif _is_possible_id(df[name], schema.type_of(name), int(row.unique), n_values):
            share = "All" if row.unique == n_values else f"{row.unique_pct:.0f}% of the"
            issues.append(
                Issue(IssueKind.POSSIBLE_ID, (name,), f"{share} values are distinct (possible id)")
            )

    if base.duplicate_rows:
        issues.append(
            Issue(IssueKind.DUPLICATE_ROWS, (), f"{base.duplicate_rows:,} duplicate rows")
        )

    for row in pairs.itertuples(index=False):
        if abs(row.correlation) > high_correlation:
            issues.append(
                Issue(
                    IssueKind.HIGH_CORRELATION,
                    (str(row.column_a), str(row.column_b)),
                    f"Correlation {row.correlation:+.2f}",
                )
            )
    return issues


def _is_possible_id(series: pd.Series, kind: SemanticType, unique: int, n_values: int) -> bool:
    """Identifiers of the schema, and integer numbers that (almost) never repeat.

    A few repeats are allowed (``schema.IDENTIFIER_MIN_DISTINCT``): in a dirty file, duplicated
    rows repeat their ids. Dates and free text are often all distinct too, but they are not
    keys. Floats neither: measurements rarely repeat.
    """
    if kind is SemanticType.IDENTIFIER:
        return True
    return (
        kind is SemanticType.NUMERIC
        and ptypes.is_integer_dtype(series)
        and n_values >= IDENTIFIER_MIN_ROWS
        and mostly_distinct(unique, n_values)
    )
