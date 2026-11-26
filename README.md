<div align="center">

# LitMetrics

**Explore a dataset in the browser: load a file, analyze its data and plot it.**

[![Python](https://img.shields.io/badge/python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.66-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Version](https://img.shields.io/badge/version-0.6.0-blue)](pyproject.toml)

</div>

---

LitMetrics is a Streamlit app for the first look at a new dataset. Upload a CSV, Excel,
Parquet, Feather or SQLite file and it works out how the file is written, reads it, figures out what each column means, and shows what
needs attention before any analysis: missing values, duplicates, constant columns, likely
identifiers and strongly correlated pairs. Then it charts the data with a chart builder.

It is built for real-world files: European Excel exports with `;` and cp1252, files without a
header, dates and numbers stored as text, and datasets with millions of rows.

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Development](#development)
- [Documentation](#documentation)

## Features

### Home

- **Where you are**: the active dataset (rows, columns, format, datasets in memory) with links
  to Profile, Visualization and Load, or the formats that can be read when nothing is loaded
  yet.
- **One card per step** of the flow; steps that need a dataset stay locked until one is loaded.

### Load

- **Many formats**: CSV, Excel and OpenDocument workbooks (`.xlsx`, `.xlsm`, `.xlsb`, `.xls`,
  `.ods`), Parquet, Feather and SQLite databases. For workbooks and databases, pick the sheet
  or table to read, also from the sidebar of any page.
- **Automatic CSV format detection**: encoding (BOM, UTF‑8, cp1252, latin‑1), separator, decimal
  mark (`3.5` or `3,5`) and whether the first row is a header.
- **Fast reading** with the pyarrow CSV engine, falling back to pandas' C engine for files
  pyarrow cannot parse.
- **Clear error messages** for empty, binary, malformed or mixed-encoding files. A failed
  upload never replaces the dataset already loaded.
- **Typed columns from the start**: each column gets a semantic type (numeric, categorical,
  datetime, boolean, text, identifier) and is converted once at load time. Dates and numbers
  stored as text become real dates and numbers; values that cannot be converted are counted.
- **Shared cache**: a file is read once per content and read options, even across sessions.
- **Datasets in memory**: the last datasets read are kept; switch between them from the
  sidebar, or open and remove them from the Load page.

### Profile

- **Headline numbers**: missing cells, rows with missing values, duplicate rows, constant
  columns and memory use.
- **Issues**: columns with many missing values, constant columns, possible identifiers,
  duplicate rows and highly correlated pairs, with a filter by kind.
- **Data quality report**: the issues as a Markdown file (`<file>_issues.md`), with why each
  kind matters and what to do about it, a summary of the dataset and its columns, and the
  exact criterion of each issue.
- **Columns**: an overview with one row per column, one table per semantic type with the
  measures that make sense for it, and an inspector for any column (histogram and boxplot,
  top values, date ranges, text lengths and examples).
- **Missing values**: by column, by row, and a sample of incomplete rows.
- **Correlation**: Pearson or Spearman heatmap of the numeric columns and the strongest pairs.
- **Skip columns**: leave out data that is hard to deal with, such as codes stored as numbers
  (phone numbers, postal codes), so it does not distort the statistics.

### Visualization

- **Chart builder**: bar, line, histogram, scatter, boxplot, pie and heatmap, only the kinds
  the dataset has columns for. Pick the columns of each role, the aggregation, a split by
  category, bins, time unit and categories shown.
- **Sensible columns only**: constant and empty columns, free text, identifiers and row
  numbers are never offered for a chart (they can still filter rows).
- **Row filters** on any column: dates, values or a range, as many as needed.
- **Style**: title, palette from the theme, colors, orientation, stacking, order, labels and
  their decimals, log scale, compact axis numbers, legend and size.
- **Honest charts**: extreme values can be trimmed, intervals with too few rows get no mean,
  and a note under the chart says what was cut or sampled.
- **Save and load** a chart as JSON, also on another dataset; download the data behind the
  chart as CSV.

### Export

- **Download the active dataset** as CSV, Parquet or Feather from the sidebar of any page,
  named after the file and its table (`shop_orders.parquet`). The file is only written when
  you click Download.
- **No silent losses**: before the download, a warning lists what the file will not hold as
  in the original (values that could not be read, written empty; values the loader changed,
  such as binary data or nested values; risky file names; slow CSV exports of large datasets).
- **Problem report**: every value written empty, with its row, column, text in the file and
  expected type, downloadable as `<file>_problems.csv`.

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
│   ├── io/             Loaders, format registry, CSV detection, datasets in memory, export
│   ├── schema.py       Semantic types and column conversion
│   ├── profile.py      Missing values, duplicates, correlation, issues
│   ├── columns.py      Per-column tables and detail
│   ├── report.py       Data quality report (Markdown, from a Jinja2 template)
│   ├── visualize/      Chart data: chart kinds, columns offered, filters, saved charts
│   ├── sampling.py     Random samples with a fixed seed
│   └── parallel.py     One column per thread for numpy work
└── ui/                 Streamlit layer
    ├── state.py        Session state
    ├── cache.py        Cache wrappers
    ├── charts.py       ECharts option builders
    ├── chart_style.py  Look of the Visualization charts
    ├── echarts.py      Draws charts with the theme of the page
    ├── theme.py        Colors and fonts read from the theme
    ├── style.py        Stylesheets, page header and cards
    ├── tables.py       Rows prepared for st.dataframe
    ├── reading.py      Reads a dataset into the session
    ├── text.py         Text for the user: escaped names, byte sizes
    ├── skip.py         "Skip columns" selector shared by Profile and Visualization
    ├── sidebar.py      Dataset and table switcher and Export, shown on every page
    ├── themes/         Streamlit theme of each style variant
    ├── styles/         Stylesheet of each style variant
    └── pages/          One module (or package) per page
tests/
├── fixtures/           Test subjects: sample data, file builders, AppTest scripts
├── unit/               One module at a time
├── integration/        Modules together, pages through AppTest
└── acceptance/         User journeys in a real browser (Playwright)
scripts/                bench_profile.py: benchmark of Load and Profile
docs/                   Reference documentation
```

## Development

```bash
uv sync                          # install, including dev tools
uv run pytest -q                 # unit + integration tests (about 15 s)
uv run playwright install chromium   # once, for the acceptance tests
uv run pytest -m acceptance -q   # user journeys in a headless browser (about 30 s)
uv run ruff format src tests     # format
uv run ruff check src tests      # lint
uv run python scripts/bench_profile.py   # time Load + Profile on 3 M rows
```

- Continuous integration (GitHub Actions, `.github/workflows/ci.yaml`) runs lint, unit and
  integration tests, then the acceptance tests, on every push to `main` and `dev`.

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
- [`docs/state-and-cache.md`](docs/state-and-cache.md): cache, session state and what survives
  a reload.
- [`CHANGELOG.md`](CHANGELOG.md): what changed in each version.
- [`AGENTS.md`](AGENTS.md): architecture rules, conventions and commands for LLMs.
