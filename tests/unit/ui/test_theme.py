"""Unit tests of ``ui/theme.py`` and of the theme files of the style variants.

Every variant must define the colors ``ui/theme.py`` reads. Every variant but the original one
(variant 4, kept as it was) must also meet WCAG AA contrast in light and dark mode.
"""

import tomllib
from pathlib import Path
from typing import Any

import pytest

from litmetrics.config import StyleConfig
from litmetrics.ui import theme

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src" / "litmetrics"
THEMES = sorted((PACKAGE / StyleConfig.themes_dir).glob("theme_*.toml"))
# The original theme of the app, kept as it was: it is not held to WCAG AA.
ORIGINAL = "theme_4"
NEW_THEMES = [path for path in THEMES if path.stem != ORIGINAL]
# WCAG AA contrast ratios: body text, and large text / graphical objects such as a focus ring.
AA_TEXT = 4.5
AA_GRAPHIC = 3.0
SEMANTIC = ["blue", "green", "yellow", "red", "orange", "violet", "gray"]


def _lookup(path: Path):
    """``st.get_option``-like reader of a theme file: ``"theme.light.primaryColor"``."""
    doc = tomllib.loads(path.read_text(encoding="utf-8"))

    def lookup(name: str) -> Any:
        node: Any = doc
        for part in name.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    return lookup


def _luminance(color: str) -> float:
    """Relative luminance of ``#rrggbb`` as WCAG 2 defines it."""
    channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a: str, b: str) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_there_are_variants():
    """Variants 0 to 3 exist as theme files.

    Case: a theme file is renamed or deleted and ``--style`` fails for it.
    Affects: ui/themes/theme_<n>.toml, config.py (StyleConfig.themes_dir)
    How: lists the theme files in the package.
    """
    assert [p.stem for p in THEMES][:4] == ["theme_0", "theme_1", "theme_2", "theme_3"]


@pytest.mark.parametrize("path", THEMES, ids=lambda p: p.stem)
@pytest.mark.parametrize("mode", theme.MODES)
def test_variant_defines_every_color(path, mode):
    """Every variant defines, in both modes, the colors the charts need, as ``#rrggbb``.

    Case: a variant misses a color and charts fail (or go black) only with that variant.
    Affects: ui/theme.py (palette), ui/themes/theme_<n>.toml
    How: reads the palette of each variant and mode and checks the categorical colors.
    """
    palette = theme.palette(mode, _lookup(path))
    assert all(len(c) == 7 and c.startswith("#") for c in palette.chart_categorical)


@pytest.mark.parametrize("path", NEW_THEMES, ids=lambda p: p.stem)
@pytest.mark.parametrize("mode", theme.MODES)
def test_variant_meets_wcag_aa(path, mode):
    """Text, links, buttons, focus ring and badges meet WCAG AA on their backgrounds.

    Case: a color edit makes muted text or a badge unreadable in one mode.
    Affects: ui/theme.py (palette, tokens), ui/themes/theme_<n>.toml
    How: computes the contrast of each pair against its threshold (4.5 text, 3.0 graphics).
    """
    lookup = _lookup(path)
    p = theme.palette(mode, lookup)
    tokens = theme.tokens(p, mode)
    for surface in (p.background, p.secondary_background):
        assert _contrast(p.text, surface) >= AA_TEXT
        assert _contrast(tokens["text-muted"], surface) >= AA_TEXT
        assert _contrast(p.link, surface) >= AA_TEXT
    # Streamlit writes primary buttons in white.
    assert _contrast("#FFFFFF", p.primary) >= AA_TEXT
    assert _contrast(p.primary, p.background) >= AA_GRAPHIC
    assert _contrast(tokens["focus"], p.background) >= AA_GRAPHIC
    for color in SEMANTIC:
        text = lookup(f"theme.{mode}.{color}TextColor")
        badge = lookup(f"theme.{mode}.{color}BackgroundColor")
        assert _contrast(text, badge) >= AA_TEXT, color
        assert _contrast(text, p.background) >= AA_TEXT, color


@pytest.mark.parametrize("path", THEMES, ids=lambda p: p.stem)
def test_css_variables_follow_the_mode(path):
    """CSS variables switch with the browser mode through ``light-dark()``.

    Case: variables hold only the light colors, so the stylesheets break in dark mode.
    Affects: ui/theme.py (css_variables)
    How: builds the CSS of each variant and checks the ``:root`` block and two variables.
    """
    css = theme.css_variables(_lookup(path))
    assert css.startswith(":root {")
    assert "--lm-accent: light-dark(" in css
    assert "--lm-font-mono:" in css


def test_config_toml_only_points_at_a_variant():
    """``.streamlit/config.toml`` sets only ``theme.base``.

    Case: a color left in config.toml overrides that color in every variant.
    Affects: .streamlit/config.toml
    How: reads the theme section of config.toml.
    """
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert set(config["theme"]) == {"base"}


def test_active_theme_is_read_through_streamlit():
    """Without a lookup, the palette is read from Streamlit's options (variant 0 by default).

    Case: the default path reads a hard-coded file instead of the theme Streamlit applies.
    Affects: ui/theme.py (palette)
    How: compares the default palette with the one read from ``theme_0.toml``.
    """
    assert theme.palette("light") == theme.palette("light", _lookup(THEMES[0]))


def test_chart_colors_come_from_the_theme():
    """Chart colors and the chart theme are taken from the theme file.

    Case: the accent or the diverging scale of the heatmap ignores the variant.
    Affects: ui/theme.py (chart_colors, chart_theme)
    How: compares chart colors in dark mode and the chart theme in light mode with the file.
    """
    lookup = _lookup(NEW_THEMES[0])
    colors = theme.chart_colors("dark", lookup)
    assert colors.accent == lookup("theme.dark.primaryColor")
    assert colors.diverging[0] == lookup("theme.dark.blueColor")
    assert colors.diverging[-1] == lookup("theme.dark.redColor")
    chart = theme.chart_theme("light", lookup)
    assert chart.palette[0] == lookup("theme.light.primaryColor")
    assert chart.font.startswith("'IBM Plex Sans'")


@pytest.mark.parametrize("color", ["#abc", "rgb(1, 2, 3)", "red"])
def test_palette_rejects_colors_that_are_not_rrggbb(color):
    """A color that is not ``#rrggbb`` is a ``ValueError`` when the theme is read.

    Case: a short or named color passes and breaks color mixing (``mix``, ``with_alpha``) later.
    Affects: ui/theme.py (palette)
    How: replaces the light primary color with each bad format.
    """
    lookup = _lookup(THEMES[0])

    def with_bad_primary(name: str) -> Any:
        return color if name == "theme.light.primaryColor" else lookup(name)

    with pytest.raises(ValueError, match="#rrggbb"):
        theme.palette("light", with_bad_primary)
