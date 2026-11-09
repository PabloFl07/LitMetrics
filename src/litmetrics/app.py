"""Streamlit entry point: page setup, session state and top navigation."""

import streamlit as st

from litmetrics.config import PageConfig
from litmetrics.ui import state, style
from litmetrics.ui.pages.home import home
from litmetrics.ui.pages.load import load_page
from litmetrics.ui.pages.profile import profile_page
from litmetrics.ui.pages.visualize import visualize


def build_pages() -> tuple[st.Page, ...]:
    """Pages shown in the top navigation, in order. ``url_path`` is the URL of each page."""
    return (
        st.Page(home, title="Home", icon=":material/home:", default=True, url_path="home"),
        st.Page(load_page, title="Load", icon=":material/upload_file:", url_path="load"),
        st.Page(profile_page, title="Profile", icon=":material/query_stats:", url_path="profile"),
        st.Page(
            visualize, title="Visualization", icon=":material/bar_chart:", url_path="visualize"
        ),
    )


def main() -> None:
    """Run on every rerun: set up the page, make sure the state exists and draw the page."""
    st.set_page_config(
        page_title=PageConfig.page_title, page_icon=PageConfig.page_icon, layout=PageConfig.layout
    )
    style.apply()
    state.init_state()

    nav = st.navigation(pages=build_pages(), position="top")

    nav.run()


if __name__ == "__main__":
    main()
