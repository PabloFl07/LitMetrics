# Changelog

Notable changes of LitMetrics, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (before 1.0, a minor version may
change behavior).

## [Unreleased]

### Added

- **Acceptance tests**: user journeys run in a real browser against the app started with
  `litmetrics` (Playwright), written as Given / When / Then: loading a Spanish export (cp1252,
  `;`, decimal commas) and seeing its quality issues, switching the tables of a database from
  the sidebar on another page, keeping the active dataset after a failed upload, going back to
  an earlier file from the sidebar, and the correlation heatmap drawn by ECharts. They are
  left out of a plain `pytest` run: `uv run pytest -m acceptance`. New dev dependency:
  `pytest-playwright` (run `uv run playwright install chromium` once).
- **Integration test of the data path**: the same survey written as CSV, Parquet, Feather,
  SQLite and Excel must give the same schema, counts and statistics.
- A test that checks the suite itself: every test has a docstring with `Case`, `Affects` and
  `How`, and every test file lives in a level folder.
- **Continuous integration** with GitHub Actions (`.github/workflows/ci.yaml`), on every push
  to `main` and `dev` or run by hand from the Actions tab. Three jobs: **lint** (the same
  pre-commit hooks as locally, on all files), **unit + integration** (`pytest`) and
  **acceptance** (Playwright with Chromium, only once the other tests pass; Playwright traces
  of a failed run are kept for 7 days). Dependencies are installed from `uv.lock`
  (`uv sync --locked`), and a new push to the same branch cancels the run still in progress.

### Changed

- **Test suite reorganized** in three levels, each in its own folder and marked from it:
  `tests/unit/` (one module, mirrors `src/litmetrics/`), `tests/integration/` (modules
  together, pages through AppTest) and `tests/acceptance/`. `pytest -m integration` runs one
  level.
- Test subjects (sample files and DataFrames, file builders for each format, AppTest scripts,
  browser helpers) moved to `tests/fixtures/`; shared pytest fixtures in `tests/conftest.py`.
- Every test now says which bug it tries to catch, which parts of the app it touches and what
  it does. Near-duplicate tests were merged into parametrized ones, and tests that only
  restated the implementation were removed.
- `AGENTS.md` has a new **Tests** section with the rules of the suite: quality over coverage,
  integration before isolation, real objects over mocks, subjects in `tests/fixtures/` and the
  docstring every test must have.

## [0.4.0] - 2026-11-16

### Added

- **Parquet files** (`.parquet`, `.pq`) and **Feather files** (`.feather`, `.arrow`; Feather V1
  and V2, compressed or not). Their columns keep the types stored in the file. Types the app
  cannot profile are converted, and the Load page lists them: decimals and durations become
  numbers, times become text, binary values are shown as their size and nested values (lists,
  structs, maps) as JSON text.
- **SQLite databases** (`.db`, `.sqlite`, `.sqlite3`): the Load page lists the tables and
  views of the database and reads the one picked (the first one, by name, by default).
  BLOB values are shown as their size.
- **Excel workbooks** (`.xlsx`, `.xlsm`, `.xlsb`, `.xls`) and OpenDocument spreadsheets
  (`.ods`), read with calamine. Sheets are picked like the tables of a database (list in the
  Load page, selector in the sidebar), each with its size (left out above 10 MB, where
  finding it would take seconds); hidden sheets are listed last.
  Cells keep their type (numbers, dates, booleans), and columns that mix numbers and text are
  read as text so the schema can still find them numeric. New dependency: `python-calamine`.
- **Close file** in the list of tables or sheets frees the open database or workbook; the
  datasets already read from it stay in memory. **Clear cache** closes it too.
- Formats whose files hold several datasets declare them in `core/io/formats.py`
  (`FileFormat.parts`); the Load page shows the selector without knowing the format, ready
  for Excel sheets or other multi-dataset formats.
- **Datasets in memory**: the last datasets read (5) are kept, and the user can go back to
  one without uploading its file again.
  - A **sidebar**, shown on every page, has a selector of the active dataset among them.
  - Each shows the memory it takes, and the list the total.
  - The Load page lists them (name, table of a database, format, size, when it was read)
    with buttons to open or remove each one, and a **Clear cache** button that asks for
    confirmation and removes every dataset and every Profile result.
  - The cache is shared by every browser tab, so removing a dataset or clearing the cache
    affects them all.
- The tables of a SQLite database are listed above the preview, like the datasets in
  memory: one row per table with its size (rows and columns; only the columns of a view) and
  an Open button. The title of the list shows the open table, also while it is collapsed.
- While a table of a database is the active dataset, the sidebar has a **Table** selector
  below the dataset one, to switch table from any page.
- The per-type tables of the Columns tab (Numeric, Categorical, Dates, Text) open the
  inspector of the column in the selected row, like the overview.
- `docs/state-and-cache.md`: cache, session state and what survives a reload.

### Changed

- The correlation heatmap shows the numbers in its cells for any number of columns (smaller
  as the cells shrink), and from 13 columns it takes the whole width, with the strongest pairs
  below it.
- The datasets read are kept in a library of their own (`core/io/library.py`) instead of a
  `st.cache_resource` function cache, so they can be listed and removed. The least recently
  used one is dropped first. Two tabs reading the same file at the same moment may now read
  it twice.

### Fixed

- In dark mode, the numbers at the end of horizontal bars looked hollow (only their outline
  could be read); they now take the text color of the theme.
- Time-of-day columns (`09:30`, `17:05:10`) were taken for dates: Excel time cells lost
  every value, and text times in CSV, Parquet and Feather files were dated today. They are
  now text.
- After uploading a new version of a database or workbook with the same name and opening the
  older one from memory, the list of tables marked a table of the new file as active and did
  not let it be opened.
- File, table, sheet and column names with Markdown characters (`*draft*`, `$5$`,
  `:red[x]`) were drawn as italics, formulas or colors; they now show as written.
- With two failed reads at once (an empty sheet and a broken upload), both files were read
  again on every interaction. Each failed read is now remembered on its own.

## [0.3.0] - 2026-11-11

### Added

- **Style variants**: five looks (0 to 4), each a Streamlit theme plus a stylesheet, in light
  and dark mode. Pick one with `uv run litmetrics --style <n>` or the `LITMETRICS_STYLE`
  environment variable; variant 0 is the default. Charts take their colors and font from the
  theme, and variants 0 to 3 meet WCAG AA contrast in both modes. See
  [`docs/styling.md`](docs/styling.md).
- **Decimal mark detection**: files that write `3,5` (European exports separated by `;`) are
  read as numbers. The Load page says how the file was read: separator, decimal mark and
  whether the first row held the column names.
- **Numeric column inspector**: switch between a histogram and a horizontal boxplot. Both
  mark the median and the mean; the boxplot draws the outliers as dots and opens zoomed on the
  box when they are far away. Columns with few distinct values get one bar per value.
  "All measures" is shown as cards grouped by kind.
- `scripts/bench_profile.py`: times Load and Profile on a generated dataset of 3 M rows.
- `docs/styling.md`: themes, style variants, CSS variables and the selectors stylesheets may
  use.

### Changed

- The upload limit is 700 MB (was 500 MB).
- The column overview no longer has a one-line summary column, and the categorical table no
  longer shows an example of spelling variants: the inspector shows both in full.
- Faster and lighter on large files:
  - The encoding is checked without decoding the whole file at once, and Arrow's memory is
    released after each read (about 1 GB less held after reading a 569 MB file).
  - Types are guessed on a random sample of each column instead of its first rows, and
    dates and numbers stored as text are parsed once per distinct value (3 M text dates:
    3.9 s before, 0.1 s now).
  - Categorical and boolean text columns are stored as `category` (counting them is about
    6x faster).
  - Numeric measures are computed with numpy, one column per thread; distinct values of
    numeric columns are counted by sorting.
  - Correlation is computed with matrix products: 3.6x faster for Pearson, 9x for Spearman.
  - Skipping columns reuses the measures already computed for the other columns.

### Fixed

- The Columns tab of Profile failed on datasets with placeholder dates such as `9999-12-31`
  or `0001-01-01`.
- `litmetrics` installed as a package (instead of run from a clone of the repository) did not
  start: the theme files were outside the package. The theme and the upload limit are now
  passed on the command line, so the app also works when started from any other directory.
- A file that could not be read was read again on every interaction; the error is now
  remembered for that upload. A file already loaded is no longer copied on every rerun.
- Category values ending in a space (`"madrid "`) showed as empty cells in the previews.
- Type guessing could be wrong on sorted files, whose first rows are not like the rest.

## [0.2.0] - 2026-11-06

### Added

- **Load page**: upload a CSV file. Encoding (BOM, UTF-8, cp1252, latin-1), separator and
  header are detected; the file is read with pyarrow, falling back to pandas' C engine.
  Empty, binary, malformed and mixed-encoding files give a clear error, and a failed upload
  never replaces the dataset already loaded.
- **Semantic types**: every column is numeric, categorical, date, boolean, text or
  identifier, and is converted once at load time. Values that cannot be converted are
  counted.
- **Profile page**: headline numbers; issues (many missing values, constant columns,
  possible identifiers, duplicate rows, highly correlated pairs) with a filter by kind; a
  column overview, one table per semantic type and an inspector for any column; missing
  values by column and by row; Pearson or Spearman correlation heatmap. Columns can be left
  out of the profile.
- Shared cache: a file is read once per content and read options, even across sessions.
- `uv run litmetrics` starts the app.
- `docs/data-loading.md` and `docs/performance.md`.

## [0.1.0] - 2026-10-22

### Added

- Project structure: Streamlit app with top navigation, uv, ruff, pre-commit and pytest.

[Unreleased]: https://github.com/PabloFl07/LitMetrics/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/PabloFl07/LitMetrics/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/PabloFl07/LitMetrics/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/PabloFl07/LitMetrics/releases/tag/v0.2.0
[0.1.0]: https://github.com/PabloFl07/LitMetrics/commit/39f9997
