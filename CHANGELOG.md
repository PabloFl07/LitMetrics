# Changelog

Notable changes of LitMetrics, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (before 1.0, a minor version may
change behavior).

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

[0.3.0]: https://github.com/PabloFl07/LitMetrics/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/PabloFl07/LitMetrics/releases/tag/v0.2.0
[0.1.0]: https://github.com/PabloFl07/LitMetrics/commit/39f9997
