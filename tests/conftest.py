"""Shared pytest fixtures and the level marker of each test.

Every test under ``tests/integration/`` is marked ``integration`` and every test under
``tests/acceptance/`` ``acceptance``, from its folder: ``pytest -m integration`` runs one level.
Acceptance tests are left out of a plain ``pytest`` run (``addopts`` in pyproject.toml).

The data behind each fixture is described in ``fixtures/samples.py``. Fixtures return a new
object on each call, so a test may change it freely.
"""

from pathlib import Path

import pandas as pd
import pytest
from fixtures import samples

from litmetrics.core.schema import Schema

_LEVELS = {"integration": pytest.mark.integration, "acceptance": pytest.mark.acceptance}
_TESTS = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark each test with the level of the folder it lives in."""
    for item in items:
        level = item.path.relative_to(_TESTS).parts[0]
        if level in _LEVELS:
            item.add_marker(_LEVELS[level])


@pytest.fixture
def quality_df() -> pd.DataFrame:
    """``samples.quality_frame``: missing values, a constant column and a correlated pair."""
    return samples.quality_frame()


@pytest.fixture
def typed_df() -> pd.DataFrame:
    """``samples.typed_frame``: one column of each semantic type, with an outlier."""
    return samples.typed_frame()


@pytest.fixture
def typed_schema() -> Schema:
    """``samples.typed_schema``: the semantic types of ``typed_df``."""
    return samples.typed_schema()
