from dataclasses import dataclass


@dataclass(frozen=True)
class PageConfig:
    page_title: str = "LitMetrics"
    layout: str = "wide"
    page_icon: str = ":fire:"
    initial_sidebar_state: str = "auto"
