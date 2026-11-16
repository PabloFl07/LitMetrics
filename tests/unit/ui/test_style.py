"""Unit tests of ``ui/style.py`` and of the stylesheets of the style variants."""

import re

import pytest

from litmetrics.config import StyleConfig
from litmetrics.ui import style

# Variants that load styles_base.css before their own stylesheet.
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
    """Shared variants load ``styles_base.css`` first, and every file exists.

    Case: the base loads after the variant and overrides it, or a file is missing.
    Affects: ui/style.py (stylesheets), ui/styles/
    How: lists the stylesheets of each shared variant.
    """
    paths = style.stylesheets(number)
    assert paths[0].name == "styles_base.css"
    assert all(p.is_file() for p in paths)


@pytest.mark.parametrize("number", StyleConfig.standalone_variants)
def test_standalone_variants_load_only_their_own(number):
    """Standalone variants load only their own stylesheet.

    Case: the base rules leak into a variant designed without them.
    Affects: ui/style.py (stylesheets), config.py (StyleConfig.standalone_variants)
    How: lists the stylesheets of each standalone variant.
    """
    assert [p.name for p in style.stylesheets(number)] == [f"styles_{number}.css"]


@pytest.mark.parametrize("number", SHARED)
def test_variants_do_not_repeat_a_shared_rule(number):
    """No variant repeats a rule that is in ``styles_base.css``.

    Case: a rule copied into each variant is later fixed in one place only.
    Affects: ui/styles/styles_base.css, ui/styles/styles_<n>.css
    How: compares the top-level rules of the base and of each variant.
    """
    base = _rules(style.stylesheets(number)[0].read_text(encoding="utf-8"))
    own = _rules(style.stylesheet(number).read_text(encoding="utf-8"))
    assert not base & own


def test_no_variant_uses_the_header_blur():
    """No stylesheet uses ``backdrop-filter``.

    Case: the header is opaque in every theme, so the blur costs paint time for nothing.
    Affects: ui/styles/styles*.css
    How: searches every stylesheet for the property.
    """
    for path in style.stylesheet(0).parent.glob("styles*.css"):
        assert "backdrop-filter" not in path.read_text(encoding="utf-8"), path.name
