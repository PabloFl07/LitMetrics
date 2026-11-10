"""Profile of a dataset: missing values, distinct values, duplicates, correlation and issues.

Pure functions over a DataFrame, independent of Streamlit. The Profile page shows their results
and ``ui/cache.py`` caches them per dataset.

Built for large datasets (rules in docs/performance.md):
- ``base_profile`` is the only pass over the whole data for the cheap, exact counts (R-P1).
  Every section starts from its result instead of calling ``isna`` or ``nunique`` again.
- Expensive measures run on ``sampling.sample(df, n)``, a sample with a fixed seed (R-P2).
  Spearman is always computed on it (R-P3).
- Correlation is computed with matrix products over the rows where both columns have a value,
  not pair by pair: same result as ``DataFrame.corr``, 3.6x faster for Pearson (R-P3).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from litmetrics.core.parallel import map_columns
from litmetrics.core.sampling import positions
from litmetrics.core.schema import IDENTIFIER_MIN_ROWS, Schema, SemanticType, mostly_distinct

type CorrelationMethod = Literal["pearson", "spearman"]


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


# Base pass


def base_profile(df: pd.DataFrame, workers: int = 1) -> BaseProfile:
    """Missing values, distinct values, duplicates and memory of ``df``, exactly.

    Args:
        workers: Threads for the distinct counts (``parallel.map_columns``). Only the numeric
            columns, counted by sorting, run in parallel; it gives back the same result.
    """
    n_rows = len(df)
    missing = df.isna()
    missing_by_column = missing.sum().to_numpy()
    per_row = np.bincount(missing.to_numpy().sum(axis=1), minlength=1) if n_rows else []
    non_missing = n_rows - missing_by_column
    counted = map_columns(distinct_and_repeated, (series for _, series in df.items()), workers)
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
    # Only the counts that occur: how many rows have 0, 1, 2... missing values.
    occurs = np.flatnonzero(per_row)
    # A column with a different value in every row (an id) makes every row different: no
    # need to compare whole rows, the slowest count of all (R-P4).
    all_distinct = n_rows > 0 and bool((unique == n_rows).any())
    return BaseProfile(
        n_rows=n_rows,
        columns=columns,
        missing_per_row=pd.DataFrame(
            {"missing_in_row": occurs, "rows": np.asarray(per_row, dtype=np.int64)[occurs]}
        ),
        duplicate_rows=0 if all_distinct else _count_duplicate_rows(df),
        memory_bytes=int(df.memory_usage(deep=True).sum()),
    )


def distinct_and_repeated(series: pd.Series) -> tuple[int, int]:
    """Distinct non-missing values of a column, and how many values appear more than once.

    Numbers are counted by sorting (``_sorted_counts``) and everything else with pandas' hash
    table (``value_counts``). Measured on 3 M rows (docs/performance.md, R-P4):

    | column | hash | sort |
    |---|---:|---:|
    | float, all distinct | 449 ms | 74 ms |
    | int, random ids | 211 ms | 97 ms |
    | int, 10,000 values | 22 ms | 43 ms |
    | date, 1,500 values | 23 ms | 240 ms |

    Sorting wins by far on numbers with many distinct values and loses a little on numbers
    with few. It also releases the GIL, so ``base_profile`` can run it in threads; the hash
    table cannot. Dates, categories and text keep the hash table.
    """
    if ptypes.is_float_dtype(series.dtype) or ptypes.is_integer_dtype(series.dtype):
        if isinstance(series.dtype, np.dtype):
            return _sorted_counts(series.to_numpy())
        # Nullable dtypes (Int64, Float64): numpy cannot hold their missing values. Kept in
        # their own numpy type: float64 would merge integers above 2**53.
        return _sorted_counts(series.dropna().to_numpy(dtype=series.dtype.numpy_dtype))
    # sort=False: ordering millions of distinct values by count is slower than counting them
    # (5.8 s against 1.6 s on 7.3 M phone numbers).
    # Filtered as a numpy array: boolean indexing of a Series with millions of text labels
    # spends more time checking the index than counting.
    counts = series.value_counts(sort=False).to_numpy()
    # A category column also lists the categories that never appear.
    counts = counts[counts > 0]
    return int(counts.size), int(counts[counts > 1].sum())


def _sorted_counts(values: np.ndarray) -> tuple[int, int]:
    """``distinct_and_repeated`` of a numpy array of numbers, by sorting it.

    NaN is missing and left out. Equal neighbours in the sorted array are copies of one value.
    """
    if values.dtype.kind == "f":
        values = values[~np.isnan(values)]
    if values.size == 0:
        return 0, 0
    ordered = np.sort(values)
    starts_value = np.empty(ordered.size, dtype=bool)
    starts_value[0] = True
    np.not_equal(ordered[1:], ordered[:-1], out=starts_value[1:])
    starts = np.flatnonzero(starts_value)
    counts = np.diff(starts, append=ordered.size)
    return int(starts.size), int(counts[counts > 1].sum())


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
    stays readable and is computed over as many rows as possible. Only the kept columns are
    converted to a matrix.

    Each pair is computed over the rows where both columns have a value, as
    ``DataFrame.corr`` does. Pearson uses every row. Spearman ranks the values, which is slow
    on large data, so it uses ``sample(df, sample_rows)`` (R-P3). Each column is ranked once,
    over all its values, not again for each pair over the rows the pair shares: when values
    are missing this differs slightly from ``DataFrame.corr(method="spearman")`` (6e-6 on
    3 M rows with 10% missing at random), and it is 9x faster.
    """
    names = list(df.select_dtypes("number").columns if columns is None else columns)
    # Usually every candidate fits in the matrix: it is filled while checking the columns, so
    # each one is converted once. With more candidates than ``max_columns`` only their counts
    # are kept, and the chosen ones are converted again: one column at a time in memory
    # instead of all of them.
    fill = len(names) <= max_columns
    full = np.empty((len(df), len(names)), dtype=np.float64, order="F") if fill else None
    kept: list[int] = []
    missing: list[int] = []
    for i, name in enumerate(names):
        values = _finite(df[name])
        present = ~np.isnan(values)
        count = int(np.count_nonzero(present))
        # A spread (min < max) leaves out constant and empty columns: their correlation is
        # undefined. Cheaper than ``std() > 0`` (13 ms against 30 ms on 3 M values) and than
        # ``nunique() > 1``.
        if count > 1 and (finite := values[present]).min() < finite.max():
            kept.append(i)
            missing.append(len(values) - count)
            if full is not None:
                full[:, i] = values
    n_numeric = len(kept)
    if n_numeric < 2:
        return CorrelationResult(pd.DataFrame(), n_numeric, len(df))
    if n_numeric > max_columns:
        # Stable sort, so ties keep the original column order.
        best = set(np.argsort(missing, kind="stable")[:max_columns].tolist())
        kept = [i for j, i in enumerate(kept) if j in best]
    labels = [names[i] for i in kept]
    rows = None if method == "pearson" else _sample_rows(len(df), sample_rows)
    if full is not None:
        # Selecting columns copies the matrix: skip it when every column is kept.
        matrix = full if len(kept) == len(names) else full[:, kept]
        if rows is not None:
            matrix = matrix[rows]
    else:
        data = df[labels] if rows is None else df[labels].iloc[rows]
        matrix = np.column_stack([_finite(data[name]) for name in labels])
    if method == "spearman":
        # Ranks of each column; missing values stay NaN.
        # A writable copy: ``_pairwise_pearson`` works in place, and pandas hands out
        # read-only views under copy-on-write.
        matrix = pd.DataFrame(matrix).rank().to_numpy(copy=True)
    result = pd.DataFrame(_pairwise_pearson(matrix), index=labels, columns=labels)
    return CorrelationResult(result, n_numeric, len(matrix))


def _sample_rows(length: int, n: int) -> np.ndarray | slice:
    """Positions of the rows of ``sampling.sample`` with ``n`` rows, or all of them."""
    return slice(None) if length <= n else positions(length, n)


def _finite(series: pd.Series) -> np.ndarray:
    """Values of a numeric column as ``float64``, with missing and infinite values as NaN."""
    values = series.to_numpy(dtype="float64", na_value=np.nan, copy=True)
    # As missing: a single inf makes std() NaN, and the column would be left out.
    values[np.isinf(values)] = np.nan
    return values


def _pairwise_pearson(x: np.ndarray) -> np.ndarray:
    """Pearson correlation of every pair of columns of ``x``, over the rows where both have a
    value (not NaN). NaN where a pair has fewer than two such rows or no spread. ``x`` is
    overwritten.

    Four matrix products give, for every pair at once, the count, sums and sums of squares of
    the rows they share. Columns are centred first: the sums of large values (dates as numbers,
    amounts in millions) would otherwise cancel out and lose precision (pandas loses it:
    2.5e-8 off on values around 1e12). Without missing values one product is enough.
    """
    present = ~np.isnan(x)
    x -= np.nanmean(x, axis=0)
    if present.all():
        products = x.T @ x
        count = np.full(products.shape, float(len(x)))
        var = np.diag(products)[:, None] * np.ones((1, x.shape[1]))
        cov = products
    else:
        x[~present] = 0.0
        weights = present.astype(np.float64)
        count = weights.T @ weights
        sums = x.T @ weights  # [i, j]: sum of column i over the rows where j has a value
        products = x.T @ x
        np.square(x, out=x)
        squares = x.T @ weights
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = products - sums * sums.T / count
            var = squares - sums * sums / count
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.sqrt(var * var.T)
    corr[(count < 2) | ~(var > 0) | ~(var.T > 0)] = np.nan
    # Rounding can leave values just past 1; the diagonal is exactly 1 where defined.
    corr = np.clip(corr, -1.0, 1.0)
    diagonal = np.diag(corr).copy()
    np.fill_diagonal(corr, np.where(np.isnan(diagonal), np.nan, 1.0))
    return corr


def top_correlated_pairs(matrix: pd.DataFrame, n: int | None = None) -> pd.DataFrame:
    """The ``n`` pairs of distinct columns with the strongest correlation, by absolute value.

    ``None`` returns every pair.
    """
    if matrix.empty:
        return pd.DataFrame({"column_a": [], "column_b": [], "correlation": []})
    columns = np.asarray(matrix.columns, dtype=object)
    rows, cols = np.triu_indices(len(columns), k=1)
    values = matrix.to_numpy()[rows, cols]
    defined = ~np.isnan(values)
    rows, cols, values = rows[defined], cols[defined], values[defined]
    # Stable: equal values keep the order of the matrix.
    order = np.argsort(-np.abs(values), kind="stable")
    if n is not None:
        order = order[:n]
    return pd.DataFrame(
        {
            "column_a": columns[rows[order]],
            "column_b": columns[cols[order]],
            "correlation": values[order],
        }
    )


# Issues


def detect_issues(
    base: BaseProfile,
    schema: Schema,
    pairs: pd.DataFrame,
    *,
    high_missing_pct: float,
    high_correlation: float,
) -> list[Issue]:
    """Problems worth pointing out to the user, from results already computed (R-P6).

    Args:
        base: Output of ``base_profile``. Its ``pandas_dtype`` tells integer columns apart.
        schema: Semantic types of the dataset. Identifiers are flagged as possible ids.
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
        elif _is_possible_id(
            str(row.pandas_dtype), schema.type_of(name), int(row.unique), n_values
        ):
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


def _is_possible_id(dtype: str, kind: SemanticType, unique: int, n_values: int) -> bool:
    """Identifiers of the schema, and integer numbers that (almost) never repeat.

    A few repeats are allowed (``schema.IDENTIFIER_MIN_DISTINCT``): in a dirty file, duplicated
    rows repeat their ids. Dates and free text are often all distinct too, but they are not
    keys. Floats neither: measurements rarely repeat.
    """
    if kind is SemanticType.IDENTIFIER:
        return True
    return (
        kind is SemanticType.NUMERIC
        and ptypes.is_integer_dtype(pd.api.types.pandas_dtype(dtype))
        and n_values >= IDENTIFIER_MIN_ROWS
        and mostly_distinct(unique, n_values)
    )
