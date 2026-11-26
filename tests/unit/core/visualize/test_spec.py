"""Unit tests of ``core/visualize/spec.py``: ``DataSpec`` as the cache key of a chart's data."""

from dataclasses import replace

from litmetrics.core import visualize as viz

K = viz.ChartKind


def test_filters_are_part_of_the_cache_key():
    """Two specs that differ only in their filters are two cache entries (R-V8).

    Case: a filter is left out of the hash and the cache returns the unfiltered chart.
    Affects: core/visualize/spec.py (DataSpec, ValuesFilter)
    How: compares equality and hash of specs with and without the same filter.
    """
    plain = viz.DataSpec(K.BAR, x="plan")
    filtered = replace(plain, filters=(viz.ValuesFilter("churn", ("yes",)),))
    assert plain != filtered
    assert hash(replace(plain, filters=(viz.ValuesFilter("churn", ("yes",)),))) == hash(filtered)
