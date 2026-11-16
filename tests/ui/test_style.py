"""Stylesheets of the style variants (ui/style.py)."""

import re

import pytest

from litmetrics.config import StyleConfig
from litmetrics.ui import style

SHARED = [0, 1, 2, 3]


def _rules(text: str) -> set[str]:
    """Top-level rules of a stylesheet, without comments and with normalized spaces."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    rules, depth, start = set(), 0, 0
    for i, char in enumerate(text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                rules.add(re.sub(r"\s+", " ", text[start : i + 1]).strip())
                start = i + 1
    return rules


@pytest.mark.parametrize("number", SHARED)
def test_shared_variants_load_the_base_first(number):
    paths = style.stylesheets(number)
    assert paths[0].name == "styles_base.css"
    assert all(p.is_file() for p in paths)


@pytest.mark.parametrize("number", StyleConfig.standalone_variants)
def test_standalone_variants_load_only_their_own(number):
    assert [p.name for p in style.stylesheets(number)] == [f"styles_{number}.css"]


@pytest.mark.parametrize("number", SHARED)
def test_variants_do_not_repeat_a_shared_rule(number):
    # A rule every variant writes the same way belongs in styles_base.css only.
    base = _rules(style.stylesheets(number)[0].read_text(encoding="utf-8"))
    own = _rules(style.stylesheet(number).read_text(encoding="utf-8"))
    assert not base & own


def test_no_variant_uses_the_header_blur():
    # The header background is opaque in every theme, so a blur behind it does nothing.
    for path in style.stylesheet(0).parent.glob("styles*.css"):
        assert "backdrop-filter" not in path.read_text(encoding="utf-8"), path.name
