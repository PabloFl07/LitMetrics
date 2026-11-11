<div align="center">

# LitMetrics

**Explore a dataset in the browser: load a CSV, analyze its data, plot it and run statistics.**

[![Python](https://img.shields.io/badge/python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.66-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Version](https://img.shields.io/badge/version-0.3.0-blue)](pyproject.toml)

</div>

---

LitMetrics is a Streamlit app for the first look at a new dataset. Upload a CSV and it works
out how the file is written, reads it, figures out what each column means, and shows what
needs attention before any analysis: missing values, duplicates, constant columns, likely
identifiers and strongly correlated pairs.

It is built for real-world files: European Excel exports with `;` and cp1252, files without a
header, dates and numbers stored as text, and datasets with millions of rows.

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Development](#development)
- [Documentation](#documentation)

## Features

### Load

- **Automatic format detection**: encoding (BOM, UTF‑8, cp1252, latin‑1), separator, decimal
  mark (`3.5` or `3,5`) and whether the first row is a header.
- **Fast reading** with the pyarrow CSV engine, falling back to pandas' C engine for files
  pyarrow cannot parse.
- **Clear error messages** for empty, binary, malformed or mixed-encoding files. A failed
  upload never replaces the dataset already loaded.
- **Typed columns from the start**: each column gets a semantic type (numeric, categorical,
  datetime, boolean, text, identifier) and is converted once at load time. Dates and numbers
  stored as text become real dates and numbers; values that cannot be converted are counted.
- **Shared cache**: a file is read once per content and read options, even across sessions.

### Profile

- **Headline numbers**: missing cells, rows with missing values, duplicate rows, constant
  columns and memory use.
- **Issues**: columns with many missing values, constant columns, possible identifiers,
  duplicate rows and highly correlated pairs, with a filter by kind.
- **Columns**: an overview with one row per column, one table per semantic type with the
  measures that make sense for it, and an inspector for any column (histogram and boxplot,
  top values, date ranges, text lengths and examples).
- **Missing values**: by column, by row, and a sample of incomplete rows.
- **Correlation**: Pearson or Spearman heatmap of the numeric columns and the strongest pairs.
- **Skip columns**: leave out data that is hard to deal with, such as codes stored as numbers
  (phone numbers, postal codes), so it does not distort the statistics.

### Look

- **Five style variants**, each a Streamlit theme plus a stylesheet, in light and dark mode.
  Charts follow the theme. Variants 0 to 3 meet WCAG AA contrast in both modes.

### Built for large datasets

Only the open tab is computed, widgets rerun only their own section, results are cached per
dataset, expensive measures run on a fixed-seed sample (and the page says so), and the browser
only receives aggregates or capped rows. The rules are documented in
[`docs/performance.md`](docs/performance.md).

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/getting-started/installation/). uv installs
Python 3.14 if it is not on the system.

```bash
git clone <repository-url> litmetrics
cd litmetrics
uv sync
uv run litmetrics
```

The app opens at <http://localhost:8501>. Arguments after `litmetrics` are passed on to
`streamlit run`:

```bash
uv run litmetrics --server.port 8502
```

Pick a style variant (0 to 4, default 0) with `--style`, or with the `LITMETRICS_STYLE`
environment variable:

```bash
uv run litmetrics --style 2
```

## Project structure

```
src/litmetrics/
├── app.py              Page setup and top navigation
├── config.py           Thresholds and limits (frozen dataclasses)
├── core/               Pure logic, no Streamlit
│   ├── io/             Source, loaders, format detection
│   ├── schema.py       Semantic types and column conversion
│   ├── profile.py      Missing values, duplicates, correlation, issues
│   ├── columns.py      Per-column tables and detail
│   ├── sampling.py     Random samples with a fixed seed
│   └── parallel.py     One column per thread for numpy work
└── ui/                 Streamlit layer
    ├── state.py        Session state
    ├── cache.py        Cache wrappers
    ├── charts.py       ECharts option builders
    ├── echarts.py      Draws charts with the theme of the page
    ├── theme.py        Colors and fonts read from the theme
    ├── style.py        Stylesheets, page header and cards
    ├── tables.py       Rows prepared for st.dataframe
    ├── themes/         Streamlit theme of each style variant
    ├── styles/         Stylesheet of each style variant
    └── pages/          One module per page
tests/                  pytest and AppTest smoke tests
scripts/                bench_profile.py: benchmark of Load and Profile
docs/                   Reference documentation
```

## Development

```bash
uv sync                          # install, including dev tools
uv run pytest -q                 # tests (about 5 s)
uv run ruff format src tests     # format
uv run ruff check src tests      # lint
uv run python scripts/bench_profile.py   # time Load + Profile on 3 M rows
```

- Dependencies are pinned in `uv.lock`. Add them with `uv add <package>`
  (`uv add --dev <package>` for tools); never edit the lock file by hand.
- Code style: Ruff with a line length of 100 and the rules in `pyproject.toml`, including
  pandas-vet. Code, comments and UI text are in English.
- AI-agent conventions (architecture rules, gotchas) are in
  [`AGENTS.md`](AGENTS.md).

## Documentation

- [`docs/data-loading.md`](docs/data-loading.md): how a file becomes the active dataset.
- [`docs/performance.md`](docs/performance.md): rules that keep the app fast on large data.
- [`docs/styling.md`](docs/styling.md): themes, style variants, CSS variables and selectors.
- [`CHANGELOG.md`](CHANGELOG.md): what changed in each version.
- [`AGENTS.md`](AGENTS.md): architecture rules, conventions and commands for LLMs.
