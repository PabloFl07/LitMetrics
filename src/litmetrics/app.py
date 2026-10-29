"""Streamlit entry point: page setup, session state and top navigation."""

import streamlit as st

from litmetrics.config import PageConfig
from litmetrics.ui import state
from litmetrics.ui.pages.home import home
from litmetrics.ui.pages.load import load_page
from litmetrics.ui.pages.quality import quality
from litmetrics.ui.pages.settings import settings
from litmetrics.ui.pages.statistics import statistics
from litmetrics.ui.pages.visualize import visualize


def build_pages() -> tuple[st.Page, ...]:
    """Pages shown in the top navigation, in order. ``url_path`` is the URL of each page."""
    return (
        st.Page(home, title="Home", icon=":material/home:", default=True, url_path="home"),
        st.Page(load_page, title="Load", icon=":material/upload_file:", url_path="load"),
        st.Page(quality, title="Quality", icon=":material/fact_check:", url_path="quality"),
        st.Page(
            visualize, title="Visualization", icon=":material/bar_chart:", url_path="visualize"
        ),
        st.Page(statistics, title="Statistics", icon=":material/functions:", url_path="statistics"),
        st.Page(settings, title="Settings", icon=":material/settings:", url_path="settings"),
    )


def main() -> None:
    """Run on every rerun: set up the page, make sure the state exists and draw the page."""
    st.set_page_config(
        page_title=PageConfig.page_title, page_icon=PageConfig.page_icon, layout=PageConfig.layout
    )
    state.init_state()

    nav = st.navigation(pages=build_pages(), position="top")

    nav.run()


if __name__ == "__main__":
    main()
