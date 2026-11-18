"""Home page: what the app does, where the user is, and where to go next.

Sections from top to bottom:
- The active dataset: its name, size and format, with links to Profile and Load. With no
  dataset, a prompt to load one and the formats that can be read.
- The three steps of the flow (Load -> Profile -> Visualization), each in a card with what it
  does, whether it is ready and a link to its page.

Nothing here touches the rows: every number comes from the shape of the DataFrame and the
dataset library, so the page is instant on any dataset.
"""

from dataclasses import dataclass
from typing import Literal

import streamlit as st

from litmetrics.core.io import formats
from litmetrics.core.io.base import LoadedDataset
from litmetrics.ui import cache, state, style, text

# Badge text of a step: whether its page works yet.
type _Status = Literal["Ready", "Coming soon"]


@dataclass(frozen=True)
class _Step:
    """One step of the flow, as shown in its card.

    Attributes:
        title: Name of the step, the same as its page in the navigation.
        icon: Material icon name, the same as its page.
        summary: What the user does there, in one sentence.
        status: Badge text: "Ready" or "Coming soon".
        url_path: ``url_path`` of its page in ``app.build_pages``.
        needs_dataset: Whether its page only works on an active dataset. Without one, the card
            shows the step as locked and its link is disabled (the top navigation still opens
            the page, which says what is missing).
    """

    title: str
    icon: str
    summary: str
    status: _Status
    url_path: str
    needs_dataset: bool


_STEPS = (
    _Step(
        "Load",
        "upload_file",
        "Upload a file. Its separator, encoding, header and types are detected for you.",
        "Ready",
        "load",
        needs_dataset=False,
    ),
    _Step(
        "Profile",
        "query_stats",
        "Check quality before analysing: missing values, duplicates, outliers, correlation.",
        "Ready",
        "profile",
        needs_dataset=True,
    ),
    _Step(
        "Visualization",
        "bar_chart",
        "Plot the columns with interactive charts and pick points to inspect rows.",
        "Coming soon",
        "visualize",
        needs_dataset=True,
    ),
)

# Badge color of each step status (Streamlit badge colors).
_STATUS_COLOR: dict[_Status, str] = {"Ready": "green", "Coming soon": "orange"}
# Badge shown instead of "Ready" on a step that needs a dataset while none is active.
_LOCKED_STATUS = "Needs a dataset"


def home_page() -> None:
    """Entry point registered in app.py."""
    style.page_header("LitMetrics", "Explore a dataset in three steps: load it, check it, plot it.")
    pages = _pages()
    dataset = state.get_dataset()
    if dataset is None:
        _start(pages)
    else:
        _active(dataset, pages)
    _steps(pages, has_dataset=dataset is not None)


def _pages() -> dict[str, st.Page]:
    """Pages of the navigation by ``url_path``, for ``st.page_link``.

    Imported here, not at module level: app.py imports this module to register it.
    """
    from litmetrics.app import build_pages

    return {page.url_path: page for page in build_pages()}


# COMPONENTS


def _start(pages: dict[str, st.Page]) -> None:
    """No dataset yet: what can be loaded and a link to the Load page."""
    with style.card("home_start"):
        st.markdown("#### :material/rocket_launch: Start with a file")
        names = ", ".join(fmt.name for fmt in formats.FORMATS)
        st.markdown(f"LitMetrics reads {names} files. Upload one in the Load page.")
        st.page_link(pages["load"], label="Load a file", icon=":material/upload_file:")


def _active(dataset: LoadedDataset, pages: dict[str, st.Page]) -> None:
    """The active dataset: name, size, format, and the next step."""
    with style.card("home_active"):
        st.markdown(f"#### :material/dataset: {text.md(dataset.name)}")
        in_memory = cache.cached_datasets()
        st.caption(
            f"Active dataset · {len(in_memory)} in memory "
            f"({text.format_bytes(cache.cached_bytes())}), switch in the sidebar"
        )
        # Keyed container: the stylesheets turn these metrics into cards.
        with st.container(key="headline"):
            rows, cols, fmt = st.columns(3)
        for where, label, value, icon in (
            (rows, "Rows", f"{dataset.n_rows:,}", "table_rows"),
            (cols, "Columns", f"{dataset.n_cols:,}", "view_column"),
            (fmt, "Format", dataset.format, "description"),
        ):
            where.metric(label, value, icon=f":material/{icon}:", border=True)
        profile, load = st.columns(2)
        profile.page_link(pages["profile"], label="Profile it", icon=":material/query_stats:")
        load.page_link(pages["load"], label="Load another file", icon=":material/upload_file:")


def _steps(pages: dict[str, st.Page], *, has_dataset: bool) -> None:
    """The steps of the flow, one card each, in order.

    Without ``has_dataset``, steps that need one are locked: their link is disabled and a
    "Ready" badge becomes "Needs a dataset".
    """
    st.subheader("How it works", anchor=False)
    for number, (where, step) in enumerate(zip(st.columns(len(_STEPS)), _STEPS, strict=True), 1):
        with where, style.card(f"home_step_{step.url_path}"):
            st.markdown(f"#### :material/{step.icon}: {number}. {step.title}")
            locked = step.needs_dataset and not has_dataset
            if locked and step.status == "Ready":
                st.badge(_LOCKED_STATUS, color="gray", icon=":material/lock:")
            else:
                st.badge(step.status, color=_STATUS_COLOR[step.status])
            st.markdown(step.summary)
            st.page_link(
                pages[step.url_path],
                label=f"Open {step.title}",
                icon=":material/lock:" if locked else None,
                disabled=locked,
                help="Load a file first." if locked else None,
            )
