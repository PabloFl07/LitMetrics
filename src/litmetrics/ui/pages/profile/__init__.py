"""Profile page: how the active dataset looks, and what to check before analysing it.

Headline quality numbers on top, then four tabs:
- Issues: problems found, with a filter by kind.
- Columns: an overview with one row per column and one table per semantic type, with only the
  measures that make sense for it. Selecting a row of any of them inspects that column.
- Missing values: by column, by row and a sample of incomplete rows.
- Correlation: heatmap of the numeric columns and the strongest pairs.

Modules: ``page`` (entry point, headline, Issues, Missing values, Correlation),
``columns_tab`` (Columns tab and the inspector of one column) and ``shared`` (the active
dataset and the pieces every tab uses).

How it stays fast on large datasets (docs/performance.md):
- Only the open tab runs (R-E1), and every tab with widgets is an ``st.fragment``, so a widget
  reruns its tab and nothing else (R-E2). Fragments read the dataset from ``state``.
- Everything is computed through ``ui/cache.py``, once per dataset. Expensive measures use a
  sample, and the page says so where it applies (R-P2).
- Measures of one column are cached under the id of the whole dataset (``shared.Active``), so
  leaving columns out reuses them; only the counts over whole rows are recomputed.
- Tables and charts receive aggregates or capped rows, never the whole dataset (R-E3, R-V2).
"""

from litmetrics.ui.pages.profile.page import profile_page

__all__ = ["profile_page"]
