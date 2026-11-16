"""Unit tests of ``ui/tables.py``: rows prepared for ``st.dataframe``."""

import pandas as pd

from litmetrics.ui.tables import for_display


def test_for_display_turns_categories_into_text():
    """Categorical columns are shown as text; other columns and the input are unchanged.

    Case: ``st.dataframe`` shows categories as codes or fails on mixed categories; or the
    cached dataset is changed in place.
    Affects: ui/tables.py (for_display)
    How: prepares a categorical with a missing value next to an int column.
    """
    rows = pd.DataFrame({"city": pd.Categorical(["Lugo", "lugo ", None]), "n": [1, 2, 3]})
    shown = for_display(rows)
    assert shown["city"].tolist()[:2] == ["Lugo", "lugo "]
    assert pd.isna(shown["city"].iloc[2])
    assert shown["n"].dtype == rows["n"].dtype
    assert isinstance(rows["city"].dtype, pd.CategoricalDtype)


def test_for_display_without_categories_returns_the_rows():
    """Rows without categories are returned as they are, not copied.

    Case: every preview copies its rows on every rerun.
    Affects: ui/tables.py (for_display)
    How: prepares a frame with one int column and checks it is the same object.
    """
    rows = pd.DataFrame({"n": [1]})
    assert for_display(rows) is rows
