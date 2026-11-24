"""The "Skip columns" selector, shared by Profile and Visualization.

One widget key and one piece of state (``state.get_skipped_columns``): a column skipped in one
page is skipped in the other. Profile leaves it out of every tab; Visualization does not offer
it in any selector or filter.
"""

import streamlit as st

from litmetrics.core.schema import Schema
from litmetrics.ui import state


def skip_selector(schema: Schema) -> None:
    """Columns to leave out of Profile and Visualization, such as codes stored as numbers."""
    key = state.widget_key("skipped_columns")
    st.multiselect(
        "Skip columns",
        [c.name for c in schema.columns],
        key=key,
        on_change=lambda: state.set_skipped_columns(state.widget_value(key)),
        # Kept when the user visits another page and comes back.
        persist_state="session",
        placeholder="None: every column is used",
        help=(
            "Columns left out of Profile (every tab and the numbers above) and of the charts "
            "of Visualization. Useful for codes stored as numbers (phone numbers, postal "
            "codes): statistics and charts of them mean nothing."
        ),
    )
