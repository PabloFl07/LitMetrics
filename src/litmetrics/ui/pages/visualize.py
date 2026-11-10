"""Visualization page: charts of the active dataset. Not built yet."""

import streamlit as st

from litmetrics.ui import style


def visualize_page() -> None:
    """Entry point registered in app.py."""
    style.page_header("Visualization", "Charts of the active dataset.")
    st.info("Coming soon.", icon=":material/construction:")
