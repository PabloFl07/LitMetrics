"""App-wide settings. Read as class attributes (``PageConfig.layout``), never instantiated."""

import os
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PageConfig:
    """Arguments for ``st.set_page_config``."""

    page_title: str = "LitMetrics"
    layout: Literal["centered", "wide"] = "wide"
    page_icon: str = ":material/insights:"


@dataclass(frozen=True)
class StyleConfig:
    """Style variants: a theme ``theme_<n>.toml`` and a stylesheet ``styles_<n>.css``.

    How they work and how to add one: docs/styling.md.

    Variant 0 is the default look: ``theme_0.toml`` and ``styles.css``.
    """

    # Environment variable with the number of the active variant. ``litmetrics --style <n>``
    # sets it, and the app reads it to pick the stylesheet that goes with the theme.
    env_var: str = "LITMETRICS_STYLE"
    # Variant used when the variable is not set.
    default_variant: int = 0
    # Folder of the theme files, relative to the project root.
    themes_dir: str = ".streamlit/themes"
    # Variants whose stylesheet stands alone, without the shared rules of ``styles_base.css``.
    standalone_variants: tuple[int, ...] = (4,)


@dataclass(frozen=True)
class LoadConfig:
    """Settings of the Load page and of the dataset cache."""

    # Rows shown in the preview table.
    preview_rows: int = 100
    # Height (px) of the preview table.
    preview_height: int = 500
    # Datasets kept in memory by each cached function (shared by all sessions).
    cache_max_datasets: int = 5


@dataclass(frozen=True)
class ProfileConfig:
    """Settings of the Profile page and of its cache."""

    # Rows of the sample used for the expensive measures (quantiles, histograms, Spearman,
    # text lengths, outliers drawn). A dataset with fewer rows is used whole.
    sample_rows: int = 200_000
    # Threads that profile columns at the same time (R-P8 in docs/performance.md). Up to 8:
    # 8 numeric columns took 182 ms in 8 threads against 590 ms in one, and more threads only
    # add memory (one column being sorted per thread).
    workers: int = max(1, min(8, os.cpu_count() or 1))
    # Datasets kept in memory by each cached Profile function: the last ones opened.
    cache_max_datasets: int = 3
    # Columns inspected kept in memory by the cache of the column inspector.
    cache_max_columns: int = 32
    # A column with more missing values than this (%) is reported as an issue.
    high_missing_pct: float = 50.0
    # A pair of columns with a larger absolute correlation is reported as an issue.
    high_correlation: float = 0.9
    # Numeric columns shown at most in the correlation heatmap.
    max_heatmap_columns: int = 30
    # Columns drawn at most in the "missing values by column" chart (the most incomplete).
    missing_chart_columns: int = 30
    # Pairs listed in the "most correlated" table.
    top_pairs: int = 10
    # Rows with missing values shown in the sample table.
    missing_rows_preview: int = 100
    # Outliers drawn as dots in the boxplot of a column (the most extreme ones).
    max_outlier_dots: int = 300
    # The boxplot opens zoomed on the box when the data spans this many times the whiskers.
    box_zoom_ratio: float = 3.0
    # Example values listed in the inspector of text and identifier columns.
    text_examples: int = 8
    # Height (px) at which the column tables stop growing and scroll.
    table_max_height: int = 560
    # Height (px) of a row of the column tables. Set on the table, so it does not depend on the
    # font size of the style variant.
    table_row_height: int = 33
