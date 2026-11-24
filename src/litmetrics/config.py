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
    # Folder of the theme files, relative to the package (``src/litmetrics/``). Inside the
    # package, so an installed ``litmetrics`` finds them too.
    themes_dir: str = "ui/themes"
    # Variants whose stylesheet stands alone, without the shared rules of ``styles_base.css``.
    standalone_variants: tuple[int, ...] = (4,)


@dataclass(frozen=True)
class LoadConfig:
    """Settings of the Load page and of the dataset cache."""

    # Largest file (MB) the uploader accepts. ``litmetrics`` passes it as
    # ``--server.maxUploadSize``; ``.streamlit/config.toml`` repeats it for a plain
    # ``streamlit run`` (tests/unit/test_cli.py checks both agree).
    max_upload_mb: int = 700
    # Rows shown in the preview table.
    preview_rows: int = 100
    # Height (px) of the preview table.
    preview_height: int = 500
    # Datasets kept in memory by each cached function (shared by all sessions).
    cache_max_datasets: int = 5
    # Largest file with several datasets (MB) whose bytes are kept with its datasets in
    # memory, so opening one of them again offers the others (the sidebar table selector).
    # A larger file is only kept while it is open: its other tables need the upload again.
    keep_file_max_mb: int = 200


@dataclass(frozen=True)
class ExportConfig:
    """Settings of the Export dialog of the sidebar."""

    # Estimated seconds (export.csv_seconds) from which exporting to a slow format (CSV) shows
    # how long it takes. CSV took 19.8 s for 3.1 M rows, against 0.6 s for Parquet.
    slow_seconds: float = 10.0
    # Seconds the browser waits for the file of a download button before failing. Fixed by
    # Streamlit's frontend (180 s in 1.66, not configurable): from half of it, the warning says
    # the download may fail.
    download_timeout_seconds: float = 180.0
    # Lines of the problem report shown in the dialog; the downloaded report has them all.
    report_preview_rows: int = 200


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
    # From this many columns the heatmap takes the whole width (the pairs go below it), so
    # its cells stay wide enough for their numbers.
    heatmap_full_width_columns: int = 13
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


@dataclass(frozen=True)
class VisualizeConfig:
    """Settings of the Visualization page: limits of what reaches the browser (R-V2) and of
    the heuristics that pick the first chart of each kind (``core/visualize.py``).
    """

    # Points of a scatter chart: a random sample with a fixed seed above this.
    max_points: int = 10_000
    # Points of each line of a time series. A finer time unit is coarsened to fit.
    max_line_points: int = 2_000
    # Buckets the automatic time unit aims for at most, so the line stays readable.
    auto_line_points: int = 400
    # Categories of a bar axis; the rest are grouped in "Other".
    max_categories: int = 50
    # Categories the person can ask to show on an axis (bars, boxes, heatmap rows and columns).
    max_shown_categories: int = 30
    # Slices of a pie.
    max_slices: int = 12
    # Series drawn when a chart is split by a column (one color each).
    max_series: int = 8
    # Outliers drawn in a boxplot, over all its boxes.
    max_outliers: int = 300
    # Values sampled for the quantiles of a boxplot and for the trimming of extreme values.
    quantile_sample: int = 200_000
    # Rows sampled to rank numeric pairs by correlation (default axes of a scatter).
    correlation_sample: int = 50_000
    # Numeric columns compared to find that pair: the first ones, the matrix grows as n².
    pair_columns: int = 20
    # Intervals a numeric x axis of a bar or line chart is cut into at first.
    x_bins: int = 10
    # Bars of a histogram at first.
    histogram_bins: int = 30
    # Extreme values are trimmed when the whole range is this many times the central one...
    trim_ratio: float = 10.0
    # ...and the central range goes from this quantile to 1 minus it.
    trim_quantile: float = 0.005
    # Chart data kept by the cache: many combinations of options, each one small (R-V3).
    cache_max_charts: int = 256
    # Datasets whose usable columns (``column_hints``) the cache keeps.
    cache_max_hints: int = 8
    # Timings of the chart data and option under its table, for development only: set the
    # environment variable ``LITMETRICS_DEV=1`` before starting the app.
    show_timings: bool = os.environ.get("LITMETRICS_DEV") == "1"
