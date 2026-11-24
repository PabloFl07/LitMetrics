"""Streamlit cache wrappers around pure core functions.

The ``dataset_id`` is the cache key. Arguments starting with ``_`` are not hashed by Streamlit,
which avoids hashing the whole file or DataFrame on every call (R-C2 in
docs/performance.md). This is only correct because ``dataset_id`` already identifies those
arguments: never call these functions with a ``dataset_id`` that does not match the data passed
with it.

The cache is shared by all sessions, so two users uploading the same file read it once.

- The datasets read are kept in a ``DatasetLibrary`` held by ``st.cache_resource``: every caller
  gets the same object, with no copy (R-C1, R-C4). That is safe because nothing modifies it:
  ``core/`` is pure. Unlike a function cache, the library can be listed and emptied, so the user
  can go back to a dataset or forget it (``cached_datasets``, ``forget_dataset``, ``clear_all``).
- Everything else uses ``st.cache_data`` and returns aggregates, whose size grows with the
  number of columns, never with the number of rows (R-C1). One wrapper per section, not per
  measure (R-C6), each with a limit of entries (R-C3).

Which id to pass. Measures of one column (the per-type tables, the overview, the inspector,
the correlation of some columns) do not depend on the other columns, so they take the id of the
whole dataset with the whole DataFrame, and the columns to describe as ``names``. Leaving
columns out of the profile then reuses them instead of computing them again. Only what depends
on the set of columns (``base_profile``: duplicate rows; ``rows_with_missing``) takes the id of
the view (``view_id``) with the DataFrame without those columns. The same holds for
Visualization: ``column_hints`` (columns offered) takes the view, ``chart_data`` the whole
dataset.
"""

import hashlib
from collections.abc import Iterable, Mapping
from datetime import datetime

import pandas as pd
import streamlit as st

from litmetrics.config import LoadConfig, ProfileConfig, VisualizeConfig
from litmetrics.core import columns, profile, visualize
from litmetrics.core.io import formats
from litmetrics.core.io.base import LoadedDataset, Source
from litmetrics.core.io.library import DatasetLibrary, LibraryEntry
from litmetrics.core.schema import Schema, SemanticType, prepare_dataset

_PROFILE_MAX = ProfileConfig.cache_max_datasets
_WORKERS = ProfileConfig.workers
# Largest file (bytes) kept with its datasets in the library (``load_dataset``).
_KEEP_FILE_BYTES = LoadConfig.keep_file_max_mb * 1024 * 1024


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


@st.cache_resource
def _library() -> DatasetLibrary:
    """The datasets read, shared by every session (R-C1, R-C4): one object, never copied."""
    return DatasetLibrary(LoadConfig.cache_max_datasets)


def load_dataset(
    dataset_id: str,
    _source: Source,
    options: formats.ReadOptions,
    part: str | None = None,
    _parts: formats.Parts | None = None,
) -> tuple[LoadedDataset, Schema]:
    """Parse ``_source`` with ``options`` and type its columns, once per ``dataset_id``.

    ``options`` are the read options of the file (``formats.detect_options``), and choose the
    loader. ``part`` is the dataset of a file with several, to name it in the library, and
    ``_parts`` the datasets of that file: with them the file is kept with the dataset (up to
    ``LoadConfig.keep_file_max_mb``), so opening it from memory offers the other datasets.

    The result is kept in the library (``cached_datasets``), not in a Streamlit function
    cache, so the user can list it and forget it. A failed read is not kept: a ``ReadError``
    is raised again on the next call.

    Returns the dataset, with every column converted to the dtype of its semantic type, and its
    schema (see ``schema.prepare_dataset``). Callers must not modify them: they are shared.

    Raises:
        ReadError: The file cannot be read.
    """
    library = _library()
    if (entry := library.get(dataset_id)) is not None:
        return entry.dataset, entry.schema
    with st.spinner("Reading the file…", show_time=True):
        dataset, schema = prepare_dataset(formats.load(_source, options))
    keep = _parts is not None and len(_source.data) <= _KEEP_FILE_BYTES
    file = (_source, _parts) if keep else None
    library.put(LibraryEntry.create(dataset_id, dataset, schema, part, datetime.now(), file))
    return dataset, schema


def cached_datasets() -> tuple[LibraryEntry, ...]:
    """Datasets kept in memory, most recently used first."""
    return _library().entries()


def cached_bytes() -> int:
    """Memory taken by the datasets kept in memory, together."""
    return _library().total_bytes()


def cached_dataset(dataset_id: str) -> LibraryEntry | None:
    """Dataset ``dataset_id`` if it is still kept (it may have been dropped as the oldest)."""
    return _library().get(dataset_id)


def forget_dataset(dataset_id: str) -> None:
    """Drop ``dataset_id`` from memory, for every session.

    Its Profile results stay until they are pushed out by newer ones: ``st.cache_data`` can only
    forget one entry given all its arguments.
    """
    _library().remove(dataset_id)


def clear_all() -> None:
    """Drop every dataset and every computed result (Profile), for every session."""
    _library().clear()
    st.cache_data.clear()


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
    dataset_id: str,
    _df: pd.DataFrame,
    names: tuple[str, ...],
    sample_rows: int,
    _unparsed: Mapping[str, int],
) -> pd.DataFrame:
    """Numeric table. ``_unparsed`` is ``LoadedDataset.unparsed`` of the same dataset."""
    return columns.numeric_stats(_df, names, sample_rows, _WORKERS, unparsed=_unparsed)


@st.cache_data(
    max_entries=_PROFILE_MAX, show_spinner="Describing categorical columns…", show_time=True
)
def categorical_stats(dataset_id: str, _df: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
    """Categorical and boolean table."""
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
    """Text and identifier table. Reuses the cached base pass."""
    base = base_profile(dataset_id, _df)
    return columns.text_stats(_df, names, schema, base, sample_rows)


@st.cache_data(max_entries=_PROFILE_MAX, show_spinner="Describing columns…", show_time=True)
def column_overview(
    dataset_id: str,
    _df: pd.DataFrame,
    schema: Schema,
    sample_rows: int,
    _unparsed: Mapping[str, int],
) -> pd.DataFrame:
    """One row per column of ``schema``. Reuses the cached base pass and numeric table."""
    base = base_profile(dataset_id, _df)
    names = schema.names_of(columns.NUMERIC_TYPES)
    numeric = numeric_stats(dataset_id, _df, names, sample_rows, _unparsed)
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
    """Measures and chart data of one column, for the inspector (``columns.inspect_column``).

    Small, so many columns fit in the cache.

    Args:
        now: Date columns only (``datetime_stats``); pass None for the others, so their entry
            does not expire with the clock.
    """
    return columns.inspect_column(
        _df,
        name,
        kind,
        base_profile(dataset_id, _df),
        sample_rows=sample_rows,
        unparsed=_unparsed,
        now=now,
        outlier_limit=ProfileConfig.max_outlier_dots,
        n_examples=ProfileConfig.text_examples,
    )


# Visualization


@st.cache_data(max_entries=VisualizeConfig.cache_max_hints, show_spinner=False)
def column_hints(dataset_id: str, _df: pd.DataFrame, schema: Schema) -> visualize.ColumnHints:
    """Columns of ``_df`` that can go in a chart, and what the filters need, once per dataset.

    Depends on the set of columns offered: pass ``view_id(dataset_id, skipped)`` with the
    DataFrame and schema without the skipped columns.
    """
    return visualize.column_hints(_df, schema)


@st.cache_data(max_entries=VisualizeConfig.cache_max_charts, show_spinner="Preparing the chart…")
def chart_data(dataset_id: str, _df: pd.DataFrame, spec: visualize.DataSpec) -> visualize.ChartData:
    """Aggregated data of one chart, keyed by what changes the data only, filters included
    (R-V1, R-V3, R-V8). A chart only reads its own columns, so it takes the id of the whole
    dataset: skipping columns does not throw its entries away.

    Raises:
        ValueError: ``spec`` does not fit the dataset (``visualize.chart_data``). Not cached.
    """
    return visualize.chart_data(_df, spec)
