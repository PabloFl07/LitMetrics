"""Colors and fonts of the active theme, for the parts of the look Streamlit does not draw.

The theme file of the style variant (``.streamlit/themes/theme_<n>.toml``) is the only place
colors are written. This module reads both modes from it and turns them into:

- CSS variables for the stylesheets (``css_variables``). Each one is
  ``light-dark(<light>, <dark>)``: Streamlit sets ``color-scheme`` on the app from the active
  mode, so the browser picks the right value as soon as the user switches, without a rerun.
- Colors and fonts of the ECharts charts (``chart_colors``, ``chart_theme``).

Tones the theme has no option for (hover, pressed, focus, muted text, soft tints, shadows) are
derived here from theme colors, never written as literals. Changing ``primaryColor`` in the
theme file changes all of them.
"""

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any, Literal

import streamlit as st

from litmetrics.ui.charts import ChartColors, ChartTheme, mix, with_alpha

type Mode = Literal["light", "dark"]
# Reads a theme option by its full name ("theme.light.primaryColor"); None when it is not set.
type Lookup = Callable[[str], Any]

MODES: tuple[Mode, ...] = ("light", "dark")

# Ends of the mixes that darken (light mode) or lighten (dark mode) a theme color. They are not
# theme colors: they only set the direction of the mix.
_WHITE = "#FFFFFF"
_BLACK = "#000000"
# Share of the background mixed into the text color for secondary text. Keeps it above 4.5:1
# in every variant (tests/ui/test_theme.py).
_MUTED_TEXT = 0.35
# Share of black (light mode) or white (dark mode) mixed into the accent on hover and when
# pressed.
_HOVER = 0.12
_PRESSED = 0.24
# Opacity of the accent in soft tints: active navigation link, hover backgrounds. Dark mode
# needs more to be seen.
_SOFT_ALPHA: dict[Mode, float] = {"light": 0.10, "dark": 0.20}
# Opacity of semantic colors (info, success...) in their soft tints.
_SEMANTIC_SOFT_ALPHA: dict[Mode, float] = {"light": 0.10, "dark": 0.16}
# Shadows: the text color in light mode (a tinted gray), black in dark mode, where a gray
# shadow would glow. Opacity of the subtle and of the raised shadow.
_SHADOW_ALPHA: dict[Mode, tuple[float, float]] = {"light": (0.06, 0.12), "dark": (0.35, 0.55)}
# Share of white mixed into the accent for the light end of bar gradients and the box fill.
_CHART_LIGHT = 0.45
# Share of the page background mixed into the inner stops of the diverging scale, and into its
# neutral middle (gray). Opaque mixes, not transparency: the heatmap draws striped bands behind
# its cells, and they would show through.
_DIVERGING_INNER = 0.5
_DIVERGING_MIDDLE = 0.85


@dataclass(frozen=True)
class Palette:
    """Theme colors of one mode, as ``#rrggbb``. Each attribute is the theme option of the
    same name in camelCase (``secondary_background`` is ``secondaryBackgroundColor``).
    """

    primary: str
    background: str
    secondary_background: str
    text: str
    border: str
    link: str
    red: str
    orange: str
    yellow: str
    green: str
    blue: str
    violet: str
    gray: str
    chart_categorical: tuple[str, ...]


def _option_name(field: str) -> str:
    """Theme option of a ``Palette`` attribute: ``secondary_background`` ->
    ``secondaryBackgroundColor``.
    """
    if field == "chart_categorical":
        return "chartCategoricalColors"
    head, *rest = field.split("_")
    return head + "".join(word.capitalize() for word in rest) + "Color"


def palette(mode: Mode, lookup: Lookup = st.get_option) -> Palette:
    """Colors of ``mode`` in the active theme: ``[theme.<mode>]``, else ``[theme]``.

    Args:
        lookup: Where to read options from. ``st.get_option`` (the default) already includes
            the theme file set as ``theme.base``; tests pass a reader of a theme file.

    Raises:
        ValueError: The theme does not set one of the colors, in that mode or in ``[theme]``.
    """
    values: dict[str, Any] = {}
    for field in fields(Palette):
        name = _option_name(field.name)
        value = lookup(f"theme.{mode}.{name}")
        if value is None:
            value = lookup(f"theme.{name}")
        if value is None:
            raise ValueError(
                f"The theme does not set {name} for {mode} mode. Add it to the theme file "
                "(docs/styling.md lists the required options)."
            )
        values[field.name] = tuple(value) if isinstance(value, list) else value
    return Palette(**values)


def active_mode() -> Mode:
    """Mode of the browser that runs this session. Light until the browser has reported it.

    It can lag one rerun behind right after the user switches mode in the settings menu.
    """
    return "dark" if st.context.theme.type == "dark" else "light"


def tokens(p: Palette, mode: Mode) -> dict[str, str]:
    """Design tokens of one mode, by CSS variable name without the ``--lm-`` prefix."""
    # Away from the background: darker in light mode, lighter in dark mode.
    away = _BLACK if mode == "light" else _WHITE
    shadow_base = p.text if mode == "light" else _BLACK
    subtle, raised = _SHADOW_ALPHA[mode]
    soft = _SEMANTIC_SOFT_ALPHA[mode]
    return {
        "accent": p.primary,
        "accent-hover": mix(p.primary, away, _HOVER),
        "accent-pressed": mix(p.primary, away, _PRESSED),
        # Accent used as text (active link, selected tab): linkColor meets 4.5:1 in both modes,
        # primaryColor only in light mode.
        "accent-text": p.link,
        "accent-soft": with_alpha(p.primary, _SOFT_ALPHA[mode]),
        # Keyboard focus ring: at least 3:1 against the background.
        "focus": p.primary if mode == "light" else p.link,
        "canvas": p.background,
        "surface": p.secondary_background,
        "text": p.text,
        "text-muted": mix(p.text, p.background, _MUTED_TEXT),
        "border": p.border,
        "border-strong": mix(p.border, p.text, 0.25),
        "shadow": with_alpha(shadow_base, subtle),
        "shadow-raised": with_alpha(shadow_base, raised),
        "info": p.blue,
        "info-soft": with_alpha(p.blue, soft),
        "success": p.green,
        "success-soft": with_alpha(p.green, soft),
        "warning": p.yellow,
        "warning-soft": with_alpha(p.yellow, soft),
        "error": p.red,
        "error-soft": with_alpha(p.red, soft),
    }


def font_family(option: str | None, fallback: str) -> str:
    """CSS font stack of a theme font option, ending in ``fallback``.

    ``"'IBM Plex Mono':https://..."`` gives ``"'IBM Plex Mono', <fallback>"``.
    """
    if not option:
        return fallback
    return f"{option.split(':http', 1)[0]}, {fallback}"


def css_variables(lookup: Lookup = st.get_option) -> str:
    """``:root`` rule with every token as ``--lm-<name>: light-dark(<light>, <dark>)``, plus
    the fonts (``--lm-font``, ``--lm-font-mono``).
    """
    light, dark = (tokens(palette(mode, lookup), mode) for mode in MODES)
    lines = [
        f"  --lm-{name}: light-dark({value}, {dark[name]});"
        if value != dark[name]
        else f"  --lm-{name}: {value};"
        for name, value in light.items()
    ]
    lines.append(f"  --lm-font: {font_family(lookup('theme.font'), 'sans-serif')};")
    lines.append(f"  --lm-font-mono: {font_family(lookup('theme.codeFont'), 'monospace')};")
    return ":root {\n" + "\n".join(lines) + "\n}"


def chart_colors(mode: Mode | None = None, lookup: Lookup = st.get_option) -> ChartColors:
    """Series colors of the chart builders for ``mode`` (default: the active one)."""
    p = palette(mode or active_mode(), lookup)
    return ChartColors(
        accent=p.primary,
        accent_light=mix(p.primary, _WHITE, _CHART_LIGHT),
        highlight=p.orange,
        outlier=p.red,
        median=p.orange,
        mean=p.green,
        background=p.background,
        emphasis=p.text,
        # Blue for negative, red for positive, same steps towards the page background on both
        # sides, so 0 is a quiet gray close to the page in either mode.
        diverging=(
            p.blue,
            mix(p.blue, p.background, _DIVERGING_INNER),
            mix(p.gray, p.background, _DIVERGING_MIDDLE),
            mix(p.red, p.background, _DIVERGING_INNER),
            p.red,
        ),
    )


def chart_theme(mode: Mode | None = None, lookup: Lookup = st.get_option) -> ChartTheme:
    """Palette and font shared by every chart. The first series color is the accent."""
    p = palette(mode or active_mode(), lookup)
    return ChartTheme(
        palette=(p.primary, *p.chart_categorical[1:]),
        font=font_family(lookup("theme.font"), "sans-serif"),
    )
