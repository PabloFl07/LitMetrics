# Performance guidelines

litmetrics is meant to work with datasets of millions of rows. Streamlit reruns the script on
every interaction, so a slow computation in the wrong place makes the whole app feel slow. This
document collects the rules we follow to avoid that, with the measurements behind them.

It is written for contributors. It focuses on the two parts of the app that compute the most:

- **Profile**, which summarizes every column of a dataset.
- **Visualization**, where charts are configured and update in real time.

Every rule has a code (`R-C2`, `R-V1`...). Cite it in pull requests and reviews instead of
explaining the reason again, for example: *"cache this per section, not per measure (R-C6)"*.

## Budgets

| Situation | Target |
|---|---|
| Interacting with a widget (rerun with cache) | < 200 ms |
| Opening a Profile tab for the first time, 3 M rows | < 5 s, with a spinner |
| Changing a style setting of a chart | < 100 ms, no data recomputed |
| Any function in `core/` on 3 M rows | > 1 s → sample or rewrite it (R-P2) |

## 0. Principles

1. **Compute once, per dataset.** Every expensive computation depends on
   `(dataset_id, parameters)` and is cached. An interaction that does not change the data
   recomputes nothing.
2. **Aggregates, not rows.** What is cached and what is sent to the browser grows with the number
   of *columns*, *groups* or *bins*, never with the number of rows.
3. **Run only what is visible.** A closed tab runs no code, and a widget only reruns its own
   section.
4. **Measure before optimizing.** A rule can be broken if a measurement justifies it. Leave a
   comment in the code saying why.

## 1. Loading (`core/io`)

- **R-L1. Read with the pyarrow engine.** `pd.read_csv(..., engine="pyarrow")` is 3.7× faster
  than the default reader (0.6 s vs 2.2 s for 1 M rows). Parquet is 6× faster (0.35 s). If the
  pyarrow engine does not support a read option, fall back to the C engine only in that case.
- **R-L2. Convert types once, when loading or when the schema is confirmed.**
  - Dates stored as text are parsed once, with a fixed format if one was detected. Functions in
    `core/` never call `pd.to_datetime` on a column again.
  - **Parse each distinct value once** (`schema._by_unique`): `pd.factorize`, convert the
    distinct values, spread them back with the codes. On pandas 3 `pd.to_datetime` on a `str`
    column ignores its own cache: 3.9 s for 3 M dates with a fixed format, against 0.1 s for
    the 1,500 distinct ones. Same result, value for value. Numbers stored as text go the same
    way. When every value is different it costs one `factorize` (75 ms) more.
  - Text with few distinct values (`categorical` and `boolean` columns in the schema) is
    converted to `category`. A `groupby` on `category` is 2.5× faster (37 ms vs 95 ms), a
    `value_counts` 6× faster (14 ms vs 87 ms), and it uses less memory.
  - Type inference looks at a random sample of each column (`sampling.sample`), never at its
    first rows: in a sorted file they are not like the rest.
- **R-L3. A stable `dataset_id`.** It is computed once at load time (a hash of the file bytes,
  or name + size + modification time) and is the key of the whole cache. If the data changes (a
  persistent filter, a cleaning step), the result is a new dataset with a new id.
- **R-L4. Check the encoding without decoding the file.** `bytes.isascii()` takes 58 ms on
  569 MB and allocates nothing; `data.decode("utf-8")` takes 146 ms and builds a 569 MB string
  (more if the text is not ASCII). Files that are not ASCII are decoded in 64 MB chunks with an
  incremental decoder: 246 ms, 64 MB at most.
- **R-L5. Give Arrow's memory back after reading.** The pyarrow engine allocates through Arrow's
  pool, which keeps freed blocks. After reading 569 MB the process held 1.9 GB more for a
  525 MB DataFrame; `pa.default_memory_pool().release_unused()` brings it down to 0.95 GB.

## 2. Cache (`ui/cache.py`)

- **R-C1. `st.cache_data` only returns aggregates.** Every cache hit deserializes and copies the
  value. Caching a 500 MB DataFrame costs 0.7 s to store it, 0.33 s *on every rerun* and one more
  copy in memory. The numeric stats of 12 columns take 1.5 KB. If a function returns something
  the size of the dataset (a mask, a sample, a filtered DataFrame), do not cache it with
  `cache_data`.
- **R-C2. The DataFrame goes in with a leading underscore and the key is the id.** Streamlit
  does not hash arguments whose name starts with `_`. Hashing a 500 MB DataFrame takes about
  100 ms per call and per function.
  ```python
  @st.cache_data(max_entries=3, show_spinner="Profiling...", show_time=True)
  def numeric_stats(dataset_id: str, _df: pd.DataFrame) -> pd.DataFrame:
      return profile.numeric_stats(_df)
  ```
  The other parameters (method, number of bins, columns) have no underscore, because they are
  part of the key.
- **R-C3. Every cached function has a limit.**
  - Profile functions: `max_entries=3`, the last datasets opened.
  - Visualization functions: `max_entries` from 32 to 64, because there are many combinations
    of parameters (R-V3).
  - Streamlit 1.66 has no limit by size (`max_size`, `server.maxCacheDataSize` do not exist):
    `max_entries` is the only bound. Keep entries small (R-C1) so that the count is enough.
- **R-C4. The dataset is stored in one place, without copies.** It goes in `st.session_state`
  (one per user) or in `st.cache_resource` (shared, no copy). Nothing modifies it in place: the
  functions in `core/` are pure and return new objects.
- **R-C5. Do not cache what is cheap.** Below about 50 ms (a `value_counts` takes 40 ms, an
  `np.histogram` 37 ms), caching brings nothing and adds entries. Cache the higher-level function
  that groups several steps, not each step.
- **R-C6. One cached function per section, not per measure.** For example `base_profile`,
  `numeric_stats` and `correlation(method)`. That means fewer entries and less hashing than
  twenty small functions.

## 3. Page execution (`ui/`)

- **R-E1. Lazy tabs.** By default, `st.tabs` runs *every* tab on every rerun. With
  `on_change="rerun"`, only the open one runs:
  ```python
  tabs = st.tabs(LABELS, on_change="rerun", key="profile_tab")
  issues, columns, ranges, missing, correlation = tabs
  if ranges.open:
      _ranges(dataset_id, dataset)
  ```
- **R-E2. Every interactive section is an `@st.fragment`.** A widget inside a fragment only
  reruns that fragment. Examples: the correlation method, the groups of measures, the issue
  filter and the editor of a chart. The fragment reads the dataset from session state; it does
  not receive results computed outside it.
- **R-E3. Never render the whole dataset.** `st.dataframe` sends everything it receives to the
  browser. Previews use `head(n)` or a sample, with `n` in the config.
- **R-E4. Show a spinner with elapsed time for anything that may take more than 1 s.** With
  `show_spinner=..., show_time=True` in `st.cache_data`, the user knows the app has not frozen.

## 4. Profile (`core/profile.py`, `core/stats`)

- **R-P1. One base pass.** The cheap counts are computed together in a single cached function:
  missing values per column, missing values per row, distinct values, duplicates, memory and
  dtypes. The sections start from its result, so no section calls `df.isna()` or `nunique()`
  over the whole dataset on its own.
- **R-P2. Sample what is expensive, count what is cheap.** Samples come from
  `core/sampling.py`: positions drawn with numpy (8 ms on 3 M rows, against 90 ms for
  `DataFrame.sample`), so the same column always gives the same sample.

  | On the full data (exact) | On a sample with a fixed seed |
  |---|---|
  | missing values, distinct values, duplicates, min/max, `value_counts`, memory | quantiles, histograms, skewness and kurtosis, Spearman, text lengths, outliers *to draw* |

  With a 200,000-row sample, the largest error was 0.02 relative in the quantiles (in the tails)
  and 0.005 in Spearman. The sample size goes in `ProfileConfig`. The UI says when a result
  comes from a sample, for example *"computed on 200,000 of 3,000,000 rows"*. If the dataset fits
  in the sample, all of it is used.
- **R-P3. Correlation with matrix products; Spearman always on the sample.** `DataFrame.corr`
  with missing values works pair by pair. Four matrix products (count, sums, sums of squares
  and cross products over the rows each pair shares) give the same matrix: Pearson 0.57 s vs
  1.9 s on 3 M rows and 12 columns, and more precise (pandas is 2.5e-8 off on values near
  1e12; centring the columns first avoids it). Spearman ranks each column once and then does
  the same: 0.48 s vs 4.4 s on a 200,000-row sample. Ranking once per column, not per pair,
  differs from pandas when values are missing (6e-6 with 10% missing at random). On every row
  Spearman still takes 9.5 s, so it stays on the sample.
- **R-P4. Duplicates and distinct values are the most expensive exact counts.**
  - `df.duplicated()`: 3.9 s. With `pd.util.hash_pandas_object(df, index=False).duplicated()`,
    2.6 s. Skipped when a column has a different value in every row.
  - Distinct values: sort numbers, hash the rest. On 3 M rows:

    | Column | `value_counts` (hash) | `np.sort` + compare neighbours |
    |---|---:|---:|
    | float, all distinct | 449 ms | 74 ms |
    | int, random ids | 211 ms | 97 ms |
    | int, 10,000 values | 22 ms | 43 ms |
    | date, 1,500 values | 23 ms | 240 ms |

    `np.unique` is not the same: it was 5-6 s on 7.3 M ids and dates.
  - Both are computed only once, in the base pass (R-P1). To skip constant columns, check
    `min < max` on the finite values or reuse the distinct counts of the base pass instead of
    calling `nunique()` again.
- **R-P5. Vectorize over rows, loop only over columns.** Going through 100,000 rows with
  `itertuples` takes 370 ms. A Python loop over *columns* (`for name in df`) is fine. A loop over
  *rows*, `apply(axis=1)` or `iterrows` in `core/` is not.
- **R-P6. Issue detection reuses results.** `detect_issues` receives the cached results of the
  other sections and does not recompute them.
- **R-P7. Row previews are capped.** Rows with missing values, text examples and the outliers
  drawn in a boxplot use `head(n)` or `sample(n)`, with `n` in `ProfileConfig`.
- **R-P8. Threads per column, only where the GIL is released.** `core/parallel.map_columns`
  runs one column per thread (`ProfileConfig.workers`, up to 8). numpy sorting and arithmetic
  release the GIL: 8 float columns counted by sorting take 182 ms in 8 threads, 590 ms in one.
  pandas' hash tables do not: 8 `value_counts` take 3.1 s in threads and 2.9 s without. Numeric
  summaries (`numeric_stats`) are computed with numpy per column for the same reason: 185 ms
  in threads against 2.8 s for the same measures on a DataFrame.

## 5. Visualization

A configurable chart has two parts with very different costs:

```
data spec (x, y, aggregation, bins, filters, group)  ──►  chart_data()    cached, expensive
style spec (colors, title, axes, theme, legend)     ──►  build_option()  pure, cheap
```

- **R-V1. Separate data from style.** `chart_data(dataset_id, _df, data_spec)` aggregates the
  data and is cached. `build_option(data, style_spec)` only builds the ECharts `dict` and is not
  cached. Changing a color, the title or the theme does not touch pandas.
- **R-V2. Only aggregated or capped data reaches the browser.** The ECharts `option` is
  serialized to JSON on every rerun. 100,000 points are 1.7 MB and 47 ms; one million are
  16.7 MB and 0.5 s, *before* the browser draws them.

  | Chart type | What is sent | Indicative cap |
  |---|---|---|
  | Bars | aggregated groups; top-k + "Other" | 50 categories |
  | Histogram | bins computed on the server (`np.histogram`) | 100 bins |
  | Line / time series | `resample` or LTTB on the server | 2,000 points per series |
  | Scatter | sample with a fixed seed; with many more points, a density map (`np.histogram2d`, 0.2 s with 3 M) | 10,000 points |
  | Boxplot | quartiles + the most extreme outliers | 300 outliers |

  If data is cut, the chart says so, for example *"10,000 of 3 M points"*.
- **R-V3. The cache key only contains what changes the data.** `data_spec` is a hashable
  `dataclass(frozen=True)`; style never goes in. Since the space of combinations is large, the
  function gets a high `max_entries` and returns small values (R-C3).
- **R-V4. The editor of each chart is an `@st.fragment`.** Changing a parameter reruns that
  chart, not the page or the other charts.
- **R-V5. A stable `key` in `st_echarts`.** The key depends on the chart, not on its
  parameters. If the key changes, the component is mounted again from scratch: it flickers and
  loses the animation.
- **R-V6. ECharts options for many elements.** From a few thousand elements on, use
  `"large": True` in scatter and bar, `"progressive"` in scatter, `"sampling": "lttb"` in line,
  and `"animation": False`. They make *drawing* faster but do not shrink the JSON, so R-V2 still
  applies.
- **R-V7. Widgets that do not fire too often.** `st.slider` already reruns on release and
  `st.text_input` on Enter. Only if a measurement justifies it, group expensive parameters in an
  `st.form` with an "Apply" button. Style parameters never go in a form.
- **R-V8. Filters inside the cached function.** A filter mask has the size of the dataset: it is
  computed inside `chart_data` and thrown away. What is cached is the aggregated result, not the
  filtered DataFrame (R-C1).
- **R-V9. Group on `category` with `observed=True`.** Once R-L2 is applied, this comes for free.
  A time aggregation of 3 M rows (`resample("D")`) takes about 1 s, so it is cached.

## Appendix: measurements

The figures in this document are indicative. They were taken on a synthetic dataset of 3 M rows
and 17 columns (about 500 MB in memory): 8 float columns with up to 10% missing values,
4 integer columns, 2 low-cardinality text columns, 1 unique text code, 1 short free-text column
and 1 datetime column. `scripts/bench_profile.py` generates a dataset with the same structure
and times every step from reading the CSV to the Profile page:

```bash
uv run python scripts/bench_profile.py [--rows 3000000] [--csv data/bench.csv]
```

Environment of the table below: Python 3.13, pandas 3.0, NumPy 2.5, pyarrow 26, Streamlit 1.67,
on a 2-core Linux machine. Absolute times will differ on other machines; the ratios between
operations are what the rules rely on.

### Load and Profile, before and after the rules R-L2 to R-P8

3 M rows, 18 columns (569 MB CSV), Python 3.14, pandas 3.1, NumPy 2.5, pyarrow 26, 20-core Linux,
`ProfileConfig.workers = 8`.

| Step | Before | After |
|---|---:|---:|
| Check the encoding | 158 ms, +568 MB | 36 ms, +0 MB |
| Read the CSV (memory kept afterwards) | 541 ms (+1,924 MB) | 349 ms (+948 MB) |
| `infer_schema` | 1,123 ms | 490 ms |
| `apply_schema` | 4,129 ms | 184 ms |
| `base_profile` | 3,694 ms | 478 ms |
| `numeric_stats` | 2,779 ms, +1,325 MB | 185 ms |
| `column_overview` | 1,254 ms | 88 ms |
| Pearson | 1,928 ms, +985 MB | 565 ms |
| Spearman (200,000-row sample) | 4,246 ms | 477 ms |
| Rerun of any Profile tab with the cache | 17-31 ms | 17-31 ms |

| Operation | Time |
|---|---|
| `read_csv` 1 M rows: default / `engine="pyarrow"` / `read_parquet` | 2,200 / 600 / 350 ms |
| `df.isna().sum()` | 43 ms |
| `df.isna().sum(axis=1).value_counts()` | 277 ms |
| `df.nunique()`, every column | 2,523 ms |
| `nunique()` numeric: pandas / `np.unique` per column | 2,135 / 350 ms |
| `df.duplicated().sum()`: pandas / `hash_pandas_object` | 3,862 / 2,557 ms |
| Quantiles ×5, numeric: full / 200 k sample | 749 / 57 ms |
| Skewness + kurtosis | 620 ms |
| Pearson correlation, 12 columns | 1,096 ms |
| Spearman correlation: full / `rank()` + Pearson / 200 k sample | 27,390 / 9,167 / 1,266 ms |
| `value_counts` of one column | 40 ms |
| `np.histogram`, one column | 37 ms |
| `pd.to_datetime` 1 M strings: `format="mixed"` / fixed format | 718 / 525 ms |
| `itertuples` over 100 k rows | 370 ms |
| Hash of a DataFrame in `cache_data` (no underscore) | 100 ms per call |
| `pickle` of a DataFrame: store / load (every hit) | 731 / 333 ms |
| JSON of a scatter: 10 k / 100 k / 1 M points | 6 ms, 0.2 MB / 47 ms, 1.7 MB / 492 ms, 16.7 MB |
| `groupby` mean: text / `category` | 95 / 37 ms |
| `resample("D").mean()` | 1,045 ms |
| `np.histogram2d` 100×100 | 202 ms |
