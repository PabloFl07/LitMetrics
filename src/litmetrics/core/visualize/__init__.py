"""Data side of the Visualization page: what a chart shows, aggregated from the dataset.

Pure: no Streamlit. ``chart_data(df, spec)`` turns the dataset into the small, aggregated
``ChartData`` that a chart draws (R-V1, R-V2 in docs/performance.md). Everything that changes
the numbers is in ``DataSpec``, a frozen dataclass that is the cache key (R-V3), row filters
included (R-V8); nothing about the look (colors, title, legend) is there, so changing the look
never touches pandas.

The page is a chart builder: the person picks the kind, the columns and the look. This package
only gives it what to offer and where to start. One module per concern, all re-exported here,
so callers use ``visualize.<name>``:

- ``spec``: chart kinds, ``ROLES`` (for each kind, which columns it takes and of which semantic
  types; the editor is generated from this table), row filters, ``DataSpec``, ``ChartData``
  and ``describe`` (default title).
- ``hints``: ``column_hints``, the columns that can go in a chart, in dataset order, without
  the ones that never make a useful chart (constant, empty, row numbers), plus what the
  filters need; ``available_kinds``; ``default_spec``, a reasonable first chart of a kind,
  never a recommendation.
- ``saved``: ``spec_to_dict`` / ``spec_from_dict`` / ``fit_to_dataset``: a ``DataSpec`` as
  JSON, and how one saved from another dataset is adapted to the current one.
- ``data``: ``chart_data`` and ``to_frame``, the only code that reads rows.
"""

from litmetrics.core.visualize.data import chart_data, to_frame
from litmetrics.core.visualize.hints import (
    ColumnHints,
    available_kinds,
    column_hints,
    default_spec,
)
from litmetrics.core.visualize.saved import (
    config_file_name,
    fit_to_dataset,
    spec_from_dict,
    spec_to_dict,
)
from litmetrics.core.visualize.spec import (
    AGGREGATIONS,
    DEFAULT_AGGREGATIONS,
    ROLES,
    Accepts,
    Agg,
    ChartData,
    ChartKind,
    DataSpec,
    DateFilter,
    RangeFilter,
    Role,
    RowFilter,
    Series,
    TimeUnit,
    ValuesFilter,
    describe,
)

__all__ = [
    "AGGREGATIONS",
    "DEFAULT_AGGREGATIONS",
    "ROLES",
    "Accepts",
    "Agg",
    "ChartData",
    "ChartKind",
    "ColumnHints",
    "DataSpec",
    "DateFilter",
    "RangeFilter",
    "Role",
    "RowFilter",
    "Series",
    "TimeUnit",
    "ValuesFilter",
    "available_kinds",
    "chart_data",
    "column_hints",
    "config_file_name",
    "default_spec",
    "describe",
    "fit_to_dataset",
    "spec_from_dict",
    "spec_to_dict",
    "to_frame",
]
