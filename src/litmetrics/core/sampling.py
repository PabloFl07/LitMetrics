"""Random samples with a fixed seed, for the measures that are too slow on every row (R-P2).

Positions are drawn with numpy (``Generator.choice``) instead of ``DataFrame.sample``: on 3 M
rows numpy takes 8 ms, pandas 90 ms, and the sample is taken again for every column and every
chart. The positions depend only on the length and the size of the sample, so a DataFrame and
any of its columns are sampled on the same rows, and the same column always gives the same
sample.
"""

import numpy as np
import pandas as pd

# Seed of every sample, so the same dataset always gives the same results.
SAMPLE_SEED = 0


def positions(length: int, n: int) -> np.ndarray:
    """``n`` distinct positions in ``range(length)``, in increasing order (fixed seed).

    Sorted, so the sample keeps the order of the data, and reading it touches memory in order.
    """
    rng = np.random.default_rng(SAMPLE_SEED)
    picked = rng.choice(length, size=n, replace=False, shuffle=False)
    picked.sort()
    return picked


def sample[T: (pd.DataFrame, pd.Series)](data: T, n: int) -> T:
    """``n`` random rows of ``data`` with a fixed seed, or all of it if it has ``n`` or fewer."""
    if len(data) <= n:
        return data
    return data.iloc[positions(len(data), n)]


def sample_array(values: np.ndarray, n: int) -> np.ndarray:
    """``n`` random elements of ``values`` with a fixed seed, or all of them if there are fewer."""
    if values.size <= n:
        return values
    return values[positions(values.size, n)]
