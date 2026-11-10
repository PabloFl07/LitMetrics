"""Home page: what the app does and where to start."""

import streamlit as st

from litmetrics.ui import style


def home_page() -> None:
    """Entry point registered in app.py."""
    style.page_header("Home", "Explore a dataset: load it, profile it, plot it.")
    st.info("Start in the Load page.", icon=":material/upload_file:")
