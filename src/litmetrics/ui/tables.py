"""Rows of the dataset prepared to be shown with ``st.dataframe``."""

import pandas as pd


def for_display(rows: pd.DataFrame) -> pd.DataFrame:
    """``rows`` with ``category`` columns as text, for ``st.dataframe``.

    Streamlit 1.66 draws a category value that ends in a space ("madrid ") as an empty cell,
    which hides exactly the badly written values the Profile page points out. Only meant for
    the few rows sent to the browser (previews), never for a whole column.
    """
    categories = [
        name for name, dtype in rows.dtypes.items() if isinstance(dtype, pd.CategoricalDtype)
    ]
    if not categories:
        return rows
    return rows.astype(dict.fromkeys(categories, "str"))
