"""Run a function over the columns of a dataset in several threads (R-P8 in docs/performance.md).

Threads only help where the work releases the GIL: numpy sorting and arithmetic, pyarrow
kernels. pandas' hash tables (``value_counts``, ``nunique``) hold it, so running them in
threads gains nothing (measured: 3.1 s against 2.9 s for 8 columns). Use ``map_columns`` for
numpy-heavy work per column, never to speed up pandas counting.
"""

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor


def map_columns[T, R](func: Callable[[T], R], items: Iterable[T], workers: int) -> list[R]:
    """``[func(item) for item in items]``, in ``workers`` threads when that is more than one.

    The order of the results is the order of ``items``. An exception in ``func`` is raised here.
    """
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [func(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        return list(pool.map(func, items))
