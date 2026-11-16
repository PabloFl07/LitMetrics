"""Unit tests of the pure helpers of ``ui/cache.py``.

The cached functions themselves run in the integration tests of the pages; here, the key of a
view without some columns, which decides which cached results are shared.
"""

from litmetrics.ui import cache


def test_view_id_without_skipped_columns_is_the_dataset_id():
    """With no column skipped, the view shares the cache of the whole dataset.

    Case: an empty skip list gives a new key, so every result is computed twice.
    Affects: ui/cache.py (view_id)
    How: asks the id of a view that skips nothing.
    """
    assert cache.view_id("abc", []) == "abc"


def test_view_id_depends_on_the_set_of_skipped_columns_only():
    """The same columns skipped in any order give the same id; other columns another one.

    Case: the order of the multiselect changes the key (cache misses), or two different views
    share a key (results of one shown for the other).
    Affects: ui/cache.py (view_id)
    How: compares ids of ``[b, a]`` and ``[a, b]``, and of ``[a]`` and ``[b]``.
    """
    assert cache.view_id("abc", ["b", "a"]) == cache.view_id("abc", ["a", "b"])
    assert cache.view_id("abc", ["a"]) != cache.view_id("abc", ["b"])
    assert cache.view_id("abc", ["a"]).startswith("abc-skip-")
