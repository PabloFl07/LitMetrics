# litmetrics

Streamlit app to explore a dataset: load a CSV, check its quality, plot it with ECharts and run
statistics. Flow of the user: Load → Profile → Visualization → Statistics.

Status: Load and Profile work. Home and Visualization are stubs. Statistics has no page yet.

## Commands

```bash
uv sync                                       # install (Python 3.14, deps pinned in uv.lock)
uv run litmetrics                             # run the app (args go to streamlit run)
uv run litmetrics --style 2                   # run it with style variant 2 (docs/styling.md)
uv run pytest -q                              # all tests (~5 s)
uv run ruff format src tests                  # format
uv run ruff check src tests                   # lint
uv run python scripts/bench_profile.py        # time Load + Profile on 3 M rows (docs/performance.md)
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
    io/sniffing.py  Detects separator, encoding, header and decimal mark -> CsvOptions
    io/csv_loader.py
    schema.py       Semantic type of each column, and conversion to its dtype at load time
    profile.py      Base pass (missing, distinct, duplicates), correlation, issues
    columns.py      Column table of Profile: overview, per-type tables, one column (inspector)
    sampling.py     Random samples with a fixed seed (numpy positions), shared by every module
    parallel.py     map_columns: one column per thread, for numpy work only (R-P8)
  ui/             The only package that imports streamlit
    state.py        The ONLY module that touches st.session_state
    cache.py        Cache wrappers around core functions (load_dataset, Profile sections)
    charts.py       ECharts option builders: plain dicts, no streamlit, no streamlit_echarts
    echarts.py      The only module importing streamlit_echarts: ``echarts.show`` draws themed charts
    theme.py        Reads the theme (both modes): CSS variables --lm-* and chart colors
    style.py        Injects the variant stylesheet, page header and cards (``style.card``)
    tables.py       Rows prepared for ``st.dataframe`` (``tables.for_display`` for previews)
    themes/         theme_<n>.toml: Streamlit theme of each style variant (docs/styling.md)
    styles/         Stylesheets of the style variants (docs/styling.md):
      styles_base.css Rules shared by style variants 0-3, loaded before the variant's own
      styles.css      CSS of style variant 0; styles_<n>.css of variant n
    pages/          One module per page
tests/core/       pytest, mirrors src/litmetrics/core
tests/ui/         AppTest smoke tests of the pages and tests of ui/charts.py
scripts/          bench_profile.py: benchmark of Load and Profile
docs/             Reference docs: data-loading.md (Load flow), performance.md (rules R-*),
                  styling.md (themes, variants, CSS selectors)
drafts/           Design notes and plans (Spanish). Intent, not truth: the code wins
```

## Architecture rules

- `core/` is pure and testable without Streamlit. Functions take a DataFrame and return new
  objects. Never mutate the argument, never use `inplace=True`.
- Pages only decide what to show. Computation goes in `core/`, its caching in `ui/cache.py`,
  chart configuration in `ui/charts.py`. A cache wrapper only calls one core function with
  cached inputs: choosing what to compute is core logic (`columns.inspect_column`).
- The counts of one column come from `BaseProfile.by_column.loc[name]`, not from indexing
  `BaseProfile.columns` again.
- Session state goes through `ui/state.py` functions. Never write `st.session_state[...]`
  anywhere else. New keys get a private constant, a default in `_defaults()` and typed accessors.
- A widget whose options depend on the dataset columns uses `key=state.widget_key("name")`, so
  it is cleared when a new dataset is loaded.
- Cache wrappers in `ui/cache.py` take `dataset_id` first and the data as `_df` / `_source`
  (underscore = not hashed). Only call them with a `dataset_id` that matches the data.
  Other arguments must be hashable: frozen dataclasses with tuples, not lists. Measures of
  single columns take the id of the whole dataset with the whole DataFrame and the columns as
  `names`; only what depends on the set of columns (duplicate rows, rows with missing values)
  takes `cache.view_id(dataset_id, skipped)` with the DataFrame without those columns.
- The loaded dataset is cached with `st.cache_resource` and shared, not copied: never modify it.
- Code that touches every row (loading, profiling, charts of a whole column) follows the rules
  in `docs/performance.md` and cites the rule it applies (`R-L2`, `R-C1`...).
- Anything that becomes part of a cache key or of `compute_dataset_id` (options, schema, rules)
  is a `@dataclass(frozen=True)`.
- User-facing errors from reading a file are `ReadError` with a message shown as is.
- A new file format = a new class with `load(source, options) -> LoadedDataset` in `core/io/`.
- Thresholds and limits go in `config.py`, not as literals in pages.
- Colors, fonts and radii go in the theme file of each style variant
  (`ui/themes/theme_<n>.toml`); `.streamlit/config.toml` only sets `theme.base`. The
  stylesheets (`ui/styles/styles.css`, `ui/styles/styles_<n>.css`) only add what the theme
  cannot set, use colors only through `var(--lm-*)` (from `ui/theme.py`), target our elements
  by `key` (`.st-key-<key>`) and Streamlit ones by `data-testid` / ARIA, never by
  `.st-emotion-cache-*` classes. Every page starts with `style.page_header(title, subtitle)`. See docs/styling.md.
- Charts are drawn with `echarts.show(option, key=..., height=...)`; builders take their colors
  as `theme.chart_colors()`, never hex literals.
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
  callbacks in builders would need a `Js` marker converted to `JsCode` in `ui/echarts.py`: no
  builder has one yet, so neither exists. The `add-chart` skill has the code to add them.
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
