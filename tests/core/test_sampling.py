import numpy as np
import pandas as pd

from litmetrics.core import sampling


def test_sample_returns_everything_when_small():
    df = pd.DataFrame({"a": range(5)})
    assert sampling.sample(df, 10) is df


def test_sample_is_reproducible_and_aligned():
    df = pd.DataFrame({"a": range(1000), "b": range(1000)})
    first = sampling.sample(df, 50)
    assert len(first) == 50
    assert first.index.equals(sampling.sample(df, 50).index)
    # A column sampled alone gets the same rows as the whole frame.
    assert sampling.sample(df["a"], 50).index.equals(first.index)


def test_sample_keeps_the_order_and_has_no_repeats():
    picked = sampling.positions(1000, 200)
    assert len(set(picked.tolist())) == 200
    assert (np.diff(picked) > 0).all()


def test_sample_array():
    values = np.arange(1000.0)
    assert sampling.sample_array(values, 2000) is values
    picked = sampling.sample_array(values, 10)
    assert picked.size == 10
    np.testing.assert_array_equal(picked, sampling.sample_array(values, 10))
