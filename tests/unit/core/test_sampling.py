"""Unit tests of ``core/sampling.py``: the fixed-seed random samples shared by every module.

Every statistic estimated on a sample must be the same on every rerun and must use the same
rows in every module, or numbers would change with each click and disagree between tables.
"""

import numpy as np
import pandas as pd

from litmetrics.core import sampling


def test_sample_returns_everything_when_small():
    """A frame smaller than the sample is returned as is, not copied.

    Case: small datasets are copied on every call (memory) or shuffled.
    Affects: core/sampling.py (sample)
    How: samples 10 rows of a 5-row frame and checks it is the same object.
    """
    df = pd.DataFrame({"a": range(5)})
    assert sampling.sample(df, 10) is df


def test_sample_is_reproducible_and_aligned():
    """The same rows are picked every time, for a frame and for one of its columns.

    Case: a new random seed per call, or a column sampled alone gets other rows than the frame.
    Affects: core/sampling.py (sample)
    How: samples 50 of 1,000 rows twice, and column ``a`` alone, and compares the indexes.
    """
    df = pd.DataFrame({"a": range(1000), "b": range(1000)})
    first = sampling.sample(df, 50)
    assert len(first) == 50
    assert first.index.equals(sampling.sample(df, 50).index)
    assert sampling.sample(df["a"], 50).index.equals(first.index)


def test_positions_keep_the_order_and_have_no_repeats():
    """Picked positions are distinct and in increasing order.

    Case: sampling with replacement counts a row twice; unordered positions scramble dates.
    Affects: core/sampling.py (positions)
    How: picks 200 of 1,000 positions and checks uniqueness and order.
    """
    picked = sampling.positions(1000, 200)
    assert len(set(picked.tolist())) == 200
    assert (np.diff(picked) > 0).all()


def test_sample_array():
    """Arrays are sampled like frames: as is when small, reproducible when not.

    Case: the numpy path (used by threads, R-P8) forgets the fixed seed.
    Affects: core/sampling.py (sample_array)
    How: samples 2,000 of 1,000 values (same object) and 10 values twice.
    """
    values = np.arange(1000.0)
    assert sampling.sample_array(values, 2000) is values
    picked = sampling.sample_array(values, 10)
    assert picked.size == 10
    np.testing.assert_array_equal(picked, sampling.sample_array(values, 10))
