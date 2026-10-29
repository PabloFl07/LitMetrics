"""App-wide settings. Read as class attributes (``PageConfig.layout``), never instantiated."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PageConfig:
    """Arguments for ``st.set_page_config``."""

    page_title: str = "LitMetrics"
    layout: str = "wide"
    page_icon: str = ":fire:"
    initial_sidebar_state: str = "auto"


@dataclass(frozen=True)
class LoadConfig:
    """Settings of the Load page and of the dataset cache."""

    # Rows shown in the preview table.
    preview_rows: int = 100
    # Datasets kept in memory by each cached function (shared by all sessions).
    cache_max_datasets: int = 5
