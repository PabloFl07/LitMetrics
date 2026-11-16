"""Pure helpers of ui/cache.py. The cached functions are exercised by the page smoke tests."""

from litmetrics.ui import cache


def test_view_id_without_skipped_columns_is_the_dataset_id():
    assert cache.view_id("abc", []) == "abc"


def test_view_id_ignores_the_order_of_the_skipped_columns():
    assert cache.view_id("abc", ["b", "a"]) == cache.view_id("abc", ["a", "b"])
    assert cache.view_id("abc", ["a"]) != cache.view_id("abc", ["b"])
    assert cache.view_id("abc", ["a"]).startswith("abc-skip-")
