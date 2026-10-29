"""Streamlit cache wrappers around pure core functions.

The ``dataset_id`` is the cache key. Arguments starting with ``_`` are not hashed by Streamlit,
which avoids hashing the whole file or DataFrame on every call. This is only correct because
``dataset_id`` already identifies those arguments: never call these functions with a
``dataset_id`` that does not match the data passed with it.

The cache is shared by all sessions, so two users uploading the same file read it once.
"""

import pandas as pd
import streamlit as st

from litmetrics.config import LoadConfig
from litmetrics.core.io.base import LoadedDataset, Source
from litmetrics.core.io.csv_loader import CsvLoader
from litmetrics.core.io.sniffing import CsvOptions
from litmetrics.core.schema import profile_columns


# Exceptions (ReadError) are not cached, so a failed read is tried again on the next call.
@st.cache_data(max_entries=LoadConfig.cache_max_datasets, show_spinner="Reading CSV…")
def load_csv(dataset_id: str, _source: Source, options: CsvOptions) -> LoadedDataset:
    """Parse ``_source`` with ``options``, once per ``dataset_id``."""
    return CsvLoader().load(_source, options)


@st.cache_data(max_entries=LoadConfig.cache_max_datasets, show_spinner=False)
def column_profile(dataset_id: str, _df: pd.DataFrame) -> pd.DataFrame:
    """Per-column profile of ``_df`` (see ``profile_columns``), once per ``dataset_id``."""
    return profile_columns(_df)
