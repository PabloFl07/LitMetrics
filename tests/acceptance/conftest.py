"""Fixtures of the acceptance tests: a running app per test and a browser page on it.

Each test gets its own server, so the datasets kept in memory (shared by every session of a
server) never leak from one test to the next. ``page`` comes from pytest-playwright; run with
``--headed`` to watch, or ``--tracing retain-on-failure`` to keep a trace of failed tests.
"""

from collections.abc import Iterator

import pytest
from fixtures.browser import free_port, start_app, wait_idle
from playwright.sync_api import Page


@pytest.fixture
def browser_context_args(browser_context_args: dict) -> dict:
    """A desktop-sized window: the sidebar and the five headline metrics fit side by side."""
    return {**browser_context_args, "viewport": {"width": 1440, "height": 1000}}


@pytest.fixture
def app(page: Page, tmp_path) -> Iterator[Page]:
    """The app started with ``litmetrics``, open in ``page`` on its Home page.

    The server log is written to ``tmp_path/streamlit.log`` and printed if the test fails.
    """
    port = free_port()
    log = tmp_path / "streamlit.log"
    server = start_app(port, log)
    try:
        page.goto(f"http://127.0.0.1:{port}/")
        wait_idle(page)
        yield page
    finally:
        server.terminate()
        server.wait(timeout=10)
        print(log.read_text())
