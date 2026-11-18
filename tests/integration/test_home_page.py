"""Integration tests of the Home page: which steps it lets the user open, through ``AppTest``.

The Home page links to every step of the flow. Steps that need a dataset are locked (link
disabled) until one is active; the top navigation is not affected and is not tested here.
"""

import pytest
from fixtures.app_scripts import TIMEOUT, errors, run_page
from streamlit.testing.v1 import AppTest


def _links(with_dataset: bool) -> dict[str, bool]:
    """Draw the Home page and return ``url_path`` -> disabled for each "Open ..." step link."""
    at = AppTest.from_function(
        run_page, args=("home", "home_page", with_dataset), default_timeout=TIMEOUT
    ).run()
    assert not at.exception, errors(at)
    return {
        link.proto.page: link.proto.disabled
        for link in at.get("page_link")
        if link.proto.label.startswith("Open ")
    }


@pytest.mark.parametrize(
    ("with_dataset", "expected"),
    [
        (False, {"load": False, "profile": True, "visualize": True}),
        (True, {"load": False, "profile": False, "visualize": False}),
    ],
    ids=["no-data-locks-steps", "data-unlocks-steps"],
)
def test_step_links_need_a_dataset(with_dataset, expected):
    """Steps that need a dataset can only be opened from Home once one is active.

    Case: the Profile link stays enabled with no dataset (the user lands on an empty page), or
    stays disabled after a file is loaded (the user cannot move on from Home).
    Affects: ui/pages/home.py (_steps), ui/state.py (get_dataset)
    How: draws Home without and with ``QUALITY_CSV`` loaded and reads ``disabled`` of each
    step link.
    """
    assert _links(with_dataset) == expected
