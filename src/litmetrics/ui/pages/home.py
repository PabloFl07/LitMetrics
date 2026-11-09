import streamlit as st

from litmetrics.ui import style


def home():
    style.page_header("Home", "Explore a dataset: load it, profile it, plot it.")
    st.info("Start in the Load page.", icon=":material/upload_file:")
