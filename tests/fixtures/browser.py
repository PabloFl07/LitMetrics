"""Helpers of the acceptance tests: start the app and drive it in Playwright as a user would.

Selectors use what the user sees (link and option names, labels) or Streamlit's stable
``data-testid`` attributes, never generated CSS classes.
"""

import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Locator, Page, expect

# Seconds to wait for the server to answer, and for one script run to finish.
STARTUP_TIMEOUT = 60
RUN_TIMEOUT = 30


def free_port() -> int:
    """A TCP port nothing listens on now."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_app(port: int, log: Path) -> subprocess.Popen:
    """Start the app with the ``litmetrics`` command (theme, upload limit), headless on ``port``.

    Raises:
        RuntimeError: The server did not answer its health check in time; the log is included.
    """
    command = [
        sys.executable,
        "-c",
        "from litmetrics import main; main()",
        "--server.headless=true",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
    ]
    with log.open("w") as out:
        process = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline and process.poll() is None:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health") as response:
                if response.status == 200:
                    return process
        except OSError:
            time.sleep(0.3)
    process.terminate()
    raise RuntimeError("The app did not start:\n" + log.read_text())


def wait_idle(page: Page) -> None:
    """Wait until Streamlit has finished running the script.

    The short pause lets the run triggered by the last action start; then the app must report
    ``notRunning`` and its status widget must be gone. Acting while a run is going on can
    interrupt it (a click on another page stops the read of an upload), so call this after
    every action, and still check the visible outcome before the next one.
    """
    page.wait_for_timeout(300)
    timeout = RUN_TIMEOUT * 1000
    page.wait_for_selector(
        '[data-testid="stApp"][data-test-script-state="notRunning"]', timeout=timeout
    )
    page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=timeout)


def go_to(page: Page, title: str) -> None:
    """Open a page with the top navigation and wait for it. A new URL would start a new
    session and lose the dataset."""
    link = page.get_by_role("link", name=title, exact=True).first
    href = link.get_attribute("href") or ""
    link.click()
    expect(page).to_have_url(href)
    wait_idle(page)


def upload(page: Page, path: Path) -> None:
    """Upload ``path`` with the file uploader of the Load page."""
    page.locator('[data-testid="stFileUploaderDropzoneInput"]').set_input_files(str(path))
    # The rerun starts once the upload ends: wait for the file to be listed, then for the run.
    expect(page.locator('[data-testid="stFileUploader"]')).to_contain_text(path.name)
    wait_idle(page)


def metric(page: Page, label: str) -> Locator:
    """The value of the metric labelled ``label``."""
    card = page.locator('[data-testid="stMetric"]').filter(has_text=label)
    return card.locator('[data-testid="stMetricValue"]')


def choose_in_sidebar(page: Page, label: str, option: str) -> None:
    """Pick ``option`` in the sidebar selectbox labelled ``label``."""
    page.locator('[data-testid="stSidebar"]').get_by_role("combobox", name=label).click()
    page.get_by_role("option", name=option).first.click()
    wait_idle(page)


def open_tab(page: Page, name: str) -> None:
    """Open the tab whose label contains ``name``."""
    page.get_by_role("tab", name=name).click()
    wait_idle(page)


def expect_no_exception(page: Page) -> None:
    """Fail if the page shows a Streamlit exception box."""
    expect(page.locator('[data-testid="stException"]')).to_have_count(0)
