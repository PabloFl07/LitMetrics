import pandas as pd

from litmetrics.ui.tables import for_display


def test_for_display_turns_categories_into_text():
    rows = pd.DataFrame({"city": pd.Categorical(["Lugo", "lugo ", None]), "n": [1, 2, 3]})
    shown = for_display(rows)
    assert shown["city"].tolist()[:2] == ["Lugo", "lugo "]
    assert pd.isna(shown["city"].iloc[2])
    assert shown["n"].dtype == rows["n"].dtype
    # The argument is not changed.
    assert isinstance(rows["city"].dtype, pd.CategoricalDtype)


def test_for_display_without_categories_returns_the_rows():
    rows = pd.DataFrame({"n": [1]})
    assert for_display(rows) is rows
