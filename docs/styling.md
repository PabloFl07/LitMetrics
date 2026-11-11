# Styling

How the app gets its look: theme files, style variants, the CSS variables derived from the
theme, the stylesheets and the selectors they may use. It covers `ui/themes/`, `ui/styles/`,
`ui/theme.py`, `ui/style.py` and `.streamlit/config.toml`.

## Overview

```
uv run litmetrics --style <n>
      │  __init__.streamlit_args
      ├──► --theme.base=<package>/ui/themes/theme_<n>.toml   (Streamlit reads the theme)
      ├──► --server.maxUploadSize=<LoadConfig.max_upload_mb>
      └──► LITMETRICS_STYLE=<n>                               (the app picks the stylesheet)
              │
              ▼
app.py ──► style.apply()   (every rerun, before the page draws anything)
              │
              ├── theme.css_variables()  ──► :root { --lm-*: light-dark(<light>, <dark>) }
              └── style.stylesheets(n)   ──► ui/styles/styles_base.css + the variant's own
              │
              ▼
        one <style> block through st.html
```

Two layers, with a strict split:

| Layer | Where | Sets |
| --- | --- | --- |
| Theme | `src/litmetrics/ui/themes/theme_<n>.toml` | Colors (both modes), fonts, font sizes, radii, padding, widget borders |
| Stylesheet | `src/litmetrics/ui/styles/` | Only what the theme cannot set: shadows, cards, tab weight, the active nav link, focus rings... |

Charts follow the theme too: builders in `ui/charts.py` take `theme.chart_colors()` and
`echarts.show` applies `theme.chart_theme()` (palette and font).

## Style variants

A variant is a theme file plus a stylesheet with the same number.

| Variant | Name | Theme | Stylesheet | Loads `styles_base.css` |
| --- | --- | --- | --- | --- |
| 0 (default) | Slate Blue: mix of the others, IBM Plex, compact | `theme_0.toml` | `styles.css` | yes |
| 1 | Indigo Slate: clean dashboard, soft shadows | `theme_1.toml` | `styles_1.css` | yes |
| 2 | Graphite Blue: dense and flat, hairlines | `theme_2.toml` | `styles_2.css` | yes |
| 3 | Canvas Navy: BI canvas, white cards on a gray page | `theme_3.toml` | `styles_3.css` | yes |
| 4 | The original LitMetrics look (Inter), kept as it was | `theme_4.toml` | `styles_4.css` | no |

Pick one:

```bash
uv run litmetrics                 # variant 0, or LITMETRICS_STYLE if set
uv run litmetrics --style 2       # variant 2
LITMETRICS_STYLE=3 uv run litmetrics
```

- `--style` wins over `LITMETRICS_STYLE`, which wins over `StyleConfig.default_variant`.
- A variant with no theme file stops the command with `Unknown style ...`.
- A variant with a theme but no stylesheet of its own falls back to `styles.css`
  (`style.stylesheet`).
- `litmetrics` always passes the theme as an absolute `--theme.base` (and the upload limit as
  `--server.maxUploadSize`): Streamlit only reads `.streamlit/config.toml` from the directory
  it starts in, and without a theme every chart fails. The themes live inside the package, so
  this also works when LitMetrics is installed instead of run from a clone. The relative
  `base` in `config.toml` is only for a plain `streamlit run` from the project root.

Settings live in `StyleConfig` (`config.py`): `env_var`, `default_variant`, `themes_dir`
(relative to the package), `standalone_variants`.

## Theme files

`.streamlit/config.toml` only sets `[theme] base`. Any other theme key there would override the
same key in every variant (`test_config_toml_only_points_at_a_variant`).

Each theme file has a `[theme]` table (fonts, sizes, radii, spacing) and a `[theme.light]` and
`[theme.dark]` table with the colors of each mode. A color missing in a mode is read from
`[theme]`.

### Required options

`theme.palette` raises `ValueError` when one of these is missing in a mode (and in `[theme]`):

| Option | Used for |
| --- | --- |
| `primaryColor` | Accent, its hover / pressed tones, soft tints, first chart series |
| `backgroundColor` | Page (`--lm-canvas`), chart background |
| `secondaryBackgroundColor` | Surfaces (`--lm-surface`) |
| `textColor` | Text, muted text, light-mode shadows |
| `borderColor` | Borders (`--lm-border`, `--lm-border-strong`) |
| `linkColor` | Accent used as text (`--lm-accent-text`), dark-mode focus ring |
| `redColor`, `orangeColor`, `yellowColor`, `greenColor`, `blueColor`, `violetColor`, `grayColor` | Semantic tokens, chart highlights, the diverging scale |
| `chartCategoricalColors` | Chart series after the first |

Write every color as `#rrggbb`: the derived tones (`charts.mix`, `charts.with_alpha`) cannot read
`#abc`, `rgb()` or names, and `theme.palette` rejects them.

The semantic text / background pairs (`blueTextColor`, `blueBackgroundColor`...) are not read by
`ui/theme.py`, but Streamlit uses them for badges and alerts, and the WCAG test checks them.

## CSS variables

`theme.css_variables()` writes one `:root` rule. Each token is
`light-dark(<light>, <dark>)`: Streamlit sets `color-scheme` from the active mode, so the browser
switches colors as soon as the user changes mode, without a rerun. Tokens with the same value in
both modes are written once.

| Variable | From |
| --- | --- |
| `--lm-accent` | `primaryColor` |
| `--lm-accent-hover`, `--lm-accent-pressed` | Accent mixed 12 % / 24 % towards black (light) or white (dark) |
| `--lm-accent-text` | `linkColor` (meets 4.5:1 in both modes; `primaryColor` only in light) |
| `--lm-accent-soft` | Accent at 10 % (light) / 20 % (dark) opacity |
| `--lm-focus` | `primaryColor` (light), `linkColor` (dark) |
| `--lm-canvas` | `backgroundColor` |
| `--lm-surface` | `secondaryBackgroundColor` |
| `--lm-text` | `textColor` |
| `--lm-text-muted` | Text mixed 35 % towards the background |
| `--lm-border` | `borderColor` |
| `--lm-border-strong` | Border mixed 25 % towards the text |
| `--lm-shadow`, `--lm-shadow-raised` | Text color (light) or black (dark) at low opacity |
| `--lm-info`, `--lm-success`, `--lm-warning`, `--lm-error` | `blueColor`, `greenColor`, `yellowColor`, `redColor` |
| `--lm-info-soft`, `--lm-success-soft`, ... | The same at 10 % (light) / 16 % (dark) opacity |
| `--lm-font`, `--lm-font-mono` | `font`, `codeFont` (family only, URL removed) |

Every derived tone comes from a theme color, with its ratios as constants in `ui/theme.py`.
Changing `primaryColor` changes all accent tones; nothing else needs editing.

## Stylesheets

All in `src/litmetrics/ui/styles/`. `style.apply()` reads each file once per process
(`style._css`, cached) and injects them after the variables, in this order:

1. `styles_base.css`: rules every one of variants 0 to 3 writes the same way (top navigation,
   page header, focus rings, KPI cards, cards, tabs, buttons, file uploader, expanders, numbers
   in widgets).
2. The variant's own stylesheet: only what it does differently. With the same selector, it wins
   over the base.

Variants in `StyleConfig.standalone_variants` (4) load only their own stylesheet.

### Rules

- Only what the theme cannot set. If a theme option exists, use it instead.
- Colors only through `var(--lm-*)`. No hex values, no `rgb()` literals: they would not follow
  the theme or switch with the mode. (Variant 4 predates this and uses neutral translucent grays.)
- Our elements by key: `st.container(key="x")` gets the class `.st-key-x`.
- Streamlit elements by `data-testid` or ARIA attributes. Never `.st-emotion-cache-*` or `css-*`
  classes (they change on every release), never DOM depth.
- No `!important`.
- Spacing in multiples of 4 px.
- A rule that is the same in variants 0 to 3 goes in `styles_base.css` only
  (`test_variants_do_not_repeat_a_shared_rule`).
- No `backdrop-filter` on the header: it is opaque in every theme, so a blur does nothing
  (`test_no_variant_uses_the_header_blur`).

### Keys we style

| Selector | Element | Drawn by |
| --- | --- | --- |
| `.st-key-page_header` | Title and subtitle of every page | `style.page_header` |
| `[class*="st-key-card_"]` | Bordered card containers | `style.card(name)` |
| `.st-key-headline` | Row of metrics shown as KPI cards | Load and Profile pages |
| `.st-key-inspector` | Column inspector of Profile | Profile page |
| `[class*="st-key-measure_card_"]` | Measure groups of the inspector | Profile page |

Renaming one of these keys in a page breaks its style silently: search the stylesheets first.

### Streamlit elements we style

By `data-testid`: `stHeader`, `stTopNavLink`, `stMetric`, `stMetricLabel`, `stMetricValue`,
`stTabs`, `stDataFrame`, `stExpander`, `stForm`, `stFileUploaderDropzone`,
`stCaptionContainer`, `stMarkdownBadge`, `stNumberInputField`, `stSliderThumbValue`.

These are not a public API. After upgrading Streamlit, run the browser check of the `verify`
skill and look at every variant.

## Accessibility

`tests/ui/test_theme.py` checks WCAG AA in both modes for every variant except 4 (the original,
kept as it was):

- Text, muted text and links: at least 4.5:1 on the page and on surfaces.
- White text on `primaryColor` (Streamlit's primary buttons): at least 4.5:1.
- `primaryColor` and the focus ring against the page: at least 3:1.
- Each semantic text color on its badge background and on the page: at least 4.5:1.

Contrast against `backgroundColor`, as of the current theme files (recompute after changing a
theme; the test is the source of truth):

| Variant | Mode | Text | Muted text | Link | White on primary | Primary | Focus |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | light | 17.0 | 5.4 | 6.8 | 5.7 | 5.5 | 5.5 |
| 0 | dark | 14.5 | 6.7 | 9.5 | 5.0 | 3.7 | 9.5 |
| 1 | light | 17.7 | 5.5 | 7.9 | 6.3 | 6.3 | 6.3 |
| 1 | dark | 15.3 | 6.9 | 9.5 | 4.9 | 3.8 | 9.5 |
| 2 | light | 17.7 | 5.5 | 6.7 | 5.2 | 5.2 | 5.2 |
| 2 | dark | 15.1 | 6.7 | 10.6 | 4.6 | 4.2 | 10.6 |
| 3 | light | 15.9 | 5.2 | 7.4 | 6.3 | 5.8 | 5.8 |
| 3 | dark | 15.4 | 6.9 | 9.2 | 4.7 | 4.0 | 9.2 |
| 4 | light | 17.9 | 5.6 | 6.7 | 5.2 | 5.2 | 5.2 |
| 4 | dark | 15.2 | 6.9 | 7.4 | 3.7 | 5.1 | 7.4 |
