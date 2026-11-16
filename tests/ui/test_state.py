"""ui/state.py: what is kept and what is cleared when the dataset changes.

Session state only exists inside a script run, so each check runs as an AppTest script.
"""

from streamlit.testing.v1 import AppTest

TIMEOUT = 30


def _check_set_dataset() -> None:
    import pandas as pd
    import streamlit as st

    from litmetrics.core.io.base import LoadedDataset
    from litmetrics.ui import state

    state.init_state()
    dataset = LoadedDataset(pd.DataFrame({"a": [1]}), "f.csv", "CSV")
    assert state.set_dataset(dataset, "one", "upload-1")
    st.session_state[state.widget_key("x_axis")] = "a"
    state.set_skipped_columns(["a"])
    # Same content uploaded again: nothing is reset.
    assert not state.set_dataset(dataset, "one", "upload-2")
    assert st.session_state[state.widget_key("x_axis")] == "a"
    assert state.get_skipped_columns() == ("a",)
    assert not state.source_changed("upload-2")
    # New content: the widgets and choices that depend on the columns are cleared.
    assert state.set_dataset(dataset, "two", "upload-3")
    assert state.widget_key("x_axis") not in st.session_state
    assert state.get_skipped_columns() == ()
    st.session_state["checked"] = True


def _check_read_error() -> None:
    import streamlit as st

    from litmetrics.ui import state

    state.init_state()
    assert state.get_read_error("upload-1") is None
    state.set_read_error("upload-1", "'f.csv' is empty.")
    assert state.get_read_error("upload-1") == "'f.csv' is empty."
    # Another upload is read again.
    assert state.get_read_error("upload-2") is None
    # A second failure does not make the first one be tried again.
    state.set_read_error("upload-2", "'g.csv' is empty.")
    assert state.get_read_error("upload-1") == "'f.csv' is empty."
    assert state.get_read_error("upload-2") == "'g.csv' is empty."
    st.session_state["checked"] = True


def test_set_dataset_keeps_choices_for_the_same_content():
    at = AppTest.from_function(_check_set_dataset, default_timeout=TIMEOUT).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["checked"]


def test_read_error_is_remembered_per_upload():
    at = AppTest.from_function(_check_read_error, default_timeout=TIMEOUT).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["checked"]
