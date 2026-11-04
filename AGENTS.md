# litmetrics

Streamlit app to explore a dataset: load a CSV, check its quality, plot it with ECharts and run
statistics. Flow of the user: Load → Profile → Visualization → Statistics.

Status: Load and Profile work. Home, Visualization, Statistics and Settings are stubs.

## Commands

```bash
uv sync                                       # install (Python 3.14, deps pinned in uv.lock)
uv run litmetrics                             # run the app (args go to streamlit run)
uv run pytest -q                              # all tests (~2 s)
uv run ruff format src tests                  # format
uv run ruff check src tests                   # lint
```

- `uv run litmetrics` = `streamlit run src/litmetrics/app.py` (entry point in `__init__.py`).
- Lint only `src` and `tests`. `drafts/` is not part of the package and fails ruff on purpose.
- Add dependencies with `uv add <pkg>` (`uv add --dev <pkg>` for tools). Never edit `uv.lock`.
- Before saying a change is done, run the `verify` skill (format, lint, tests, page smoke tests).

## Layout

```
src/litmetrics/
  app.py          st.set_page_config + st.navigation. Every page is registered in build_pages()
  config.py       Frozen dataclasses with settings, read as class attributes (LoadConfig.preview_rows)
  core/           Pure logic. NEVER imports streamlit
    io/base.py      Source, LoadedDataset, ReadError, Loader protocol, compute_dataset_id
    io/sniffing.py  Detects separator, encoding and header -> CsvOptions
    io/csv_loader.py
    schema.py       Semantic type of each column (numeric, categorical, datetime, ...)
    profile.py      Quality summary, column profile, missing values, correlation, issues
  ui/             The only package that imports streamlit
    state.py        The ONLY module that touches st.session_state
    cache.py        @st.cache_data wrappers around core functions
    charts.py       ECharts option builders: plain dicts, no streamlit, no streamlit_echarts
    echarts.py      (create when needed) the only module importing streamlit_echarts
    pages/          One module per page
tests/core/       pytest, mirrors src/litmetrics/core
tests/ui/         AppTest smoke tests of the pages
drafts/           Design notes and plans (Spanish). Intent, not truth: the code wins
```

## Architecture rules

- `core/` is pure and testable without Streamlit. Functions take a DataFrame and return new
  objects. Never mutate the argument, never use `inplace=True`.
- Pages only decide what to show. Computation goes in `core/`, its caching in `ui/cache.py`,
  chart configuration in `ui/charts.py`.
- Session state goes through `ui/state.py` functions. Never write `st.session_state[...]`
  anywhere else. New keys get a private constant, a default in `_defaults()` and typed accessors.
- A widget whose options depend on the dataset columns uses `key=state.widget_key("name")`, so
  it is cleared when a new dataset is loaded.
- Cache wrappers in `ui/cache.py` take `dataset_id` first and the data as `_df` / `_source`
  (underscore = not hashed). Only call them with a `dataset_id` that matches the data.
  Other arguments must be hashable: frozen dataclasses with tuples, not lists.
- Anything that becomes part of a cache key or of `compute_dataset_id` (options, schema, rules)
  is a `@dataclass(frozen=True)`.
- User-facing errors from reading a file are `ReadError` with a message shown as is.
- A new file format = a new class with `load(source, options) -> LoadedDataset` in `core/io/`.
- Thresholds and limits go in `config.py`, not as literals in pages.
- Charts use [streamlit-echarts](https://github.com/andfanilo/streamlit-echarts) 0.7: any
  ECharts option, JavaScript callbacks, events, click/box/lasso selection returned to Python,
  themes and maps. Use the `add-chart` skill; it has the API checked against the installed
  version.
- New UI (widgets, layout, interactions): consider
  [streamlit-extras](https://github.com/arnaudmiribel/streamlit-extras) through the
  `streamlit-extras` skill, which compares each extra with the native option first. Ask the user
  before replacing working UI with an extra, and let them choose when several extras fit.
  Deprecated extras are never used.

## Style

- Code, comments, docstrings and UI text in English.
- ruff: line length 100, rules in `pyproject.toml` (includes pandas-vet `PD`). A `# noqa` always
  says why: `# noqa: PD101 (all columns at once)`.
- Python 3.14: `type X = ...` aliases, `class Loader[T](Protocol)`, `X | None`. No
  `from __future__ import annotations`.
- Module docstring explains what the module is for. Function docstrings say what it returns and
  when it raises (`Args:`, `Raises:`, `Attributes:` sections). Inline code in docstrings uses
  double backticks.
- Comments explain why, not what. Module-level constants get a comment with their meaning and
  unit.
- Pages: one public entry point (`<name>_page()`, docstring "Entry point registered in app.py.")
  and private `_section()` functions in the order they are drawn.

## Gotchas

- Streamlit reruns the whole script on every interaction. Anything slow in the main path runs on
  every click: cache it.
- Pinned to Streamlit 1.66 and pandas 3. Newer APIs are used (`st.pills`, `st.badge`,
  `st.iframe`, `st.navigation(position="top")`); training data is often older. Check an API
  against the installed version with the `streamlit-api` skill before using it.
- `st.components.v1.html` is deprecated: use `st.iframe`.
- `streamlit_echarts` (0.7) cannot even be imported outside a running Streamlit app (not even
  `JsCode`), so `litmetrics.ui.pages.profile` cannot be imported from plain pytest. Test pages
  through `AppTest` (see `tests/ui/`), and test `ui/charts.py` builders directly. JavaScript
  callbacks in builders use the `Js` marker, converted to `JsCode` in `ui/echarts.py`.
- AppTest does not run the chart's JavaScript: selection and chart events are only testable in
  the browser (Playwright).
- `AppTest.switch_page()` does not work with pages defined as functions. Smoke tests run each
  page function on its own with `AppTest.from_function`.
- Every `st_echarts(...)` call needs an explicit, unique `key`: without it the chart remounts and
  replays its animation on every rerun.
- `st.stop()` and `st.rerun()` raise `BaseException` subclasses: never catch `BaseException`
  or use a bare `except:`.
- Writing `st.session_state[key]` for a widget after the widget was drawn in the same rerun
  raises. Change widget values in callbacks (`on_change` / `on_click`).
