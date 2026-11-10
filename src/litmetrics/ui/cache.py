"""Streamlit cache wrappers around pure core functions.

The ``dataset_id`` is the cache key. Arguments starting with ``_`` are not hashed by Streamlit,
which avoids hashing the whole file or DataFrame on every call (R-C2 in
docs/performance.md). This is only correct because ``dataset_id`` already identifies those
arguments: never call these functions with a ``dataset_id`` that does not match the data passed
with it.

The cache is shared by all sessions, so two users uploading the same file read it once.

- The dataset itself is kept with ``st.cache_resource``: every caller gets the same object, with
  no copy (R-C1, R-C4). That is safe because nothing modifies it: ``core/`` is pure.
- Everything else uses ``st.cache_data`` and returns aggregates, whose size grows with the
  number of columns, never with the number of rows (R-C1). One wrapper per section, not per
  measure (R-C6), each with a limit of entries (R-C3).

Which id to pass. Measures of one column (the per-type tables, the overview, the inspector,
the correlation of some columns) do not depend on the other columns, so they take the id of the
whole dataset with the whole DataFrame, and the columns to describe as ``names``. Leaving
columns out of the profile then reuses them instead of computing them again. Only what depends
on the set of columns (``base_profile``: duplicate rows; ``rows_with_missing``) takes the id of
the view (``view_id``) with the DataFrame without those columns.
"""

import hashlib
from collections.abc import Iterable, Mapping

import pandas as pd
import streamlit as st

from litmetrics.config import LoadConfig, ProfileConfig
from litmetrics.core import columns, profile
from litmetrics.core.io.base import LoadedDataset, Source
from litmetrics.core.io.csv_loader import CsvLoader
from litmetrics.core.io.sniffing import CsvOptions
from litmetrics.core.schema import ColumnSchema, Schema, SemanticType, prepare_dataset

_PROFILE_MAX = ProfileConfig.cache_max_datasets
_WORKERS = ProfileConfig.workers


def view_id(dataset_id: str, skipped: Iterable[str]) -> str:
    """Cache key of a dataset with the columns ``skipped`` left out.

    A dataset without some columns is other data, so it needs its own ``dataset_id``: passing
    the same id with fewer columns would return results cached for all of them.
    """
    names = sorted(skipped)
    if not names:
        return dataset_id
    digest = hashlib.sha256("\0".join(names).encode()).hexdigest()[:16]
    return f"{dataset_id}-skip-{digest}"


# Loading


# Exceptions (ReadError) are not cached, so a failed read is tried again on the next call.
@st.cache_resource(
    max_entries=LoadConfig.cache_max_datasets, show_spinner="Reading CSV…", show_time=True
)
def load_dataset(
    dataset_id: str, _source: Source, options: CsvOptions
) -> tuple[LoadedDataset, Schema]:
    """Parse ``_source`` with ``options`` and type its columns, once per ``dataset_id``.

    Returns the dataset, with every column converted to the dtype of its semantic type, and its
    schema (see ``schema.prepare_dataset``). Callers must not modify them: they are shared.
    """
    return prepare_dataset(CsvLoader().load(_source, options))


# Profile: page header, Issues and Missing values


@st.cache_data(max_entries=_PROFILE_MAX, show_spinner="Profiling the dataset…", show_time=True)
def base_profile(dataset_id: str, _df: pd.DataFrame) -> profile.BaseProfile:
    """Exact counts of ``_df`` that every section starts from (R-P1)."""
    return profile.base_profile(_df, _WORKERS)


@st.cache_data(max_entries=_PROFILE_MAX, show_spinner=False)
def rows_with_missing(dataset_id: str, _df: pd.DataFrame, n: int) -> pd.DataFrame:
    """First ``n`` incomplete rows. Cached: finding them looks at every cell."""
    return profile.rows_with_missing(_df, n)


# Each method is a separate entry, so switching back and forth does not recompute.
@st.cache_data(max_entries=_PROFILE_MAX * 2, show_spinner="Computing correlation…", show_time=True)
def correlation(
    dataset_id: str,
    _df: pd.DataFrame,
    names: tuple[str, ...],
    method: profile.CorrelationMethod,
    max_columns: int,
    sample_rows: int,
) -> profile.CorrelationResult:
    """Correlation between the columns ``names`` (the numeric ones of the schema)."""
    return profile.correlation_matrix(_df, method, max_columns, sample_rows, columns=names)


# Profile: Columns tab


@st.cache_data(max_entries=_PROFILE_MAX, show_spinner="Describing numeric columns…", show_time=True)
def numeric_stats(
    dataset_id: str, _df: pd.DataFrame, names: tuple[str, ...], sample_rows: int
) -> pd.DataFrame:
    return columns.numeric_stats(_df, names, sample_rows, _WORKERS)


@st.cache_data(
    max_entries=_PROFILE_MAX, show_spinner="Describing categorical columns…", show_time=True
)
def categorical_stats(dataset_id: str, _df: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
    return columns.categorical_stats(_df, names)


@st.cache_data(max_entries=_PROFILE_MAX, show_spinner="Describing date columns…", show_time=True)
def datetime_stats(
    dataset_id: str,
    _df: pd.DataFrame,
    names: tuple[str, ...],
    _unparsed: Mapping[str, int],
    now: pd.Timestamp,
) -> pd.DataFrame:
    """Date table. ``now`` is part of the key, so the "future" count follows the clock."""
    return columns.datetime_stats(_df, names, _unparsed, now)


@st.cache_data(max_entries=_PROFILE_MAX, show_spinner="Describing text columns…", show_time=True)
def text_stats(
    dataset_id: str,
    _df: pd.DataFrame,
    names: tuple[str, ...],
    schema: Schema,
    sample_rows: int,
) -> pd.DataFrame:
    base = base_profile(dataset_id, _df)
    return columns.text_stats(_df, names, schema, base, sample_rows)


@st.cache_data(max_entries=_PROFILE_MAX, show_spinner="Describing columns…", show_time=True)
def column_overview(
    dataset_id: str, _df: pd.DataFrame, schema: Schema, sample_rows: int
) -> pd.DataFrame:
    """One row per column of ``schema``. Reuses the cached base pass and numeric table."""
    base = base_profile(dataset_id, _df)
    numeric = numeric_stats(dataset_id, _df, schema.names_of(columns.NUMERIC_TYPES), sample_rows)
    return columns.column_overview(_df, schema, base, numeric, sample_rows, _WORKERS)


@st.cache_data(max_entries=ProfileConfig.cache_max_columns, show_spinner=False)
def inspect_column(
    dataset_id: str,
    _df: pd.DataFrame,
    name: str,
    kind: SemanticType,
    sample_rows: int,
    _unparsed: Mapping[str, int],
    now: pd.Timestamp | None,
) -> tuple[pd.DataFrame, columns.ColumnDetail]:
    """Measures and chart data of one column, for the inspector.

    Only that column is computed: the row is the same as in its per-type table, without
    computing the table of every column of its type. Small, so many columns fit in the cache.

    Returns:
        The one-row per-type table of the column (with ``unparsed`` for numbers and dates),
        and its chart data.

    Args:
        now: Date columns only (``datetime_stats``); pass None for the others, so their entry
            does not expire with the clock.
    """
    names = (name,)
    unique = int(base_profile(dataset_id, _df).columns.set_index("column").loc[name, "unique"])
    stats: pd.Series | None = None
    if kind in columns.NUMERIC_TYPES:
        table = columns.with_unparsed(columns.numeric_stats(_df, names, sample_rows), _unparsed)
        stats = table.iloc[0]
    elif kind in columns.CATEGORICAL_TYPES:
        table = columns.categorical_stats(_df, names)
    elif kind in columns.DATE_TYPES:
        table = columns.datetime_stats(_df, names, _unparsed, now)
    else:
        schema = Schema((ColumnSchema(name, str(_df[name].dtype), kind),))
        base = base_profile(dataset_id, _df)
        table = columns.text_stats(_df, names, schema, base, sample_rows)
    detail = columns.column_detail(
        _df,
        name,
        kind,
        unique=unique,
        stats=stats,
        sample_rows=sample_rows,
        outlier_limit=ProfileConfig.max_outlier_dots,
        n_examples=ProfileConfig.text_examples,
    )
    return table, detail
