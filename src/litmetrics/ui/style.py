"""Look of the app beyond the theme: the stylesheet of the style variant, the CSS variables
derived from the theme, and the markup shared by every page (header, cards).

Colors, fonts and radii live in the theme file of the variant (``.streamlit/themes/``). The
stylesheet only adds what the theme cannot set, using the ``--lm-*`` variables written by
``ui/theme.py``. How variants work: docs/styling.md.
"""

import os
from functools import cache
from pathlib import Path

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from litmetrics.config import StyleConfig
from litmetrics.ui import theme

_HERE = Path(__file__).parent


def variant() -> int:
    """Number of the active style variant: ``LITMETRICS_STYLE``, else the default."""
    value = os.environ.get(StyleConfig.env_var, "")
    return int(value) if value.isdigit() else StyleConfig.default_variant


def stylesheet(number: int) -> Path:
    """Stylesheet of variant ``number``: ``styles.css`` for 0, ``styles_<n>.css`` otherwise.

    Falls back to ``styles.css`` when the variant has no stylesheet of its own.
    """
    path = _HERE / (f"styles_{number}.css" if number else "styles.css")
    return path if path.is_file() else _HERE / "styles.css"


def stylesheets(number: int) -> tuple[Path, ...]:
    """Stylesheets of variant ``number``, in the order they are loaded: the shared rules
    (``styles_base.css``) and then the variant's own, which win on equal selectors.
    Variants in ``StyleConfig.standalone_variants`` load only their own.
    """
    own = stylesheet(number)
    if number in StyleConfig.standalone_variants and own.name != "styles.css":
        return (own,)
    return (_HERE / "styles_base.css", own)


@cache
def _css(path: Path) -> str:
    """Contents of a stylesheet, read once per process."""
    return path.read_text(encoding="utf-8")


def apply() -> None:
    """Inject the theme variables and the stylesheet. Call it on every rerun, before the page
    draws anything.
    """
    css = "\n".join(_css(path) for path in stylesheets(variant()))
    # A <style>-only st.html takes no space on the page.
    st.html(f"<style>{theme.css_variables()}\n{css}</style>")


def page_header(title: str, subtitle: str) -> None:
    """Title and one-line subtitle at the top of a page, styled by ``.st-key-page_header``."""
    with st.container(key="page_header", gap="xsmall"):
        st.title(title, anchor=False)
        st.caption(subtitle)


def card(name: str) -> DeltaGenerator:
    """Bordered container drawn as a card, styled by ``[class*="st-key-card_"]``.

    Use it as ``with style.card("missing_by_column"):``. ``name`` must be unique on the page.
    """
    return st.container(border=True, key=f"card_{name}")
