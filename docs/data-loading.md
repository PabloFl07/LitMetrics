# Data loading

How a CSV file becomes the active dataset of a session: format detection, parsing, error
handling, caching and session state. It covers the Load page and everything under `core/io/`,
plus the parts of `ui/state.py` and `ui/cache.py` that keep the dataset.

## Overview

```
st.file_uploader
      │  bytes, name, file_id
      ▼
Source ──► state.source_changed? ──no──► (nothing to do: draw the active dataset)
              │ yes
              ▼
        detect_options(bytes) ──► CsvOptions(separator, encoding, has_header)
              │
              ▼
        compute_dataset_id(bytes, options) ──► dataset_id (sha256)
              │
              ▼
        cache.load_dataset(dataset_id, source, options)
              │   CsvLoader.load ──► LoadedDataset ──► prepare_dataset ──► (typed dataset, schema)
              │        │
              │        └─ ReadError ──► st.error, previous dataset kept
              ▼
        state.set_dataset(dataset, dataset_id, source_id)
              │ new content?
              ▼
        state.set_schema(schema)
```

| Module | Role |
|---|---|
| `core/io/base.py` | Format-independent types: `Source`, `LoadedDataset`, `ReadError`, `Loader`, `compute_dataset_id` |
| `core/io/sniffing.py` | Guesses `CsvOptions` (encoding, separator, header) from the bytes |
| `core/io/csv_loader.py` | `CsvLoader`: turns a `Source` + `CsvOptions` into a `LoadedDataset` |
| `core/schema.py` | Semantic type of each column, and conversion of each column to the dtype of its type |
| `ui/cache.py` | Cache wrappers, shared by all sessions |
| `ui/state.py` | The only module that touches `st.session_state` |
| `ui/pages/load.py` | Orchestrates the flow above and shows a summary and a preview |

Typing the columns is part of loading (R-L2 in `docs/performance.md`).
`schema.prepare_dataset` infers the semantic type of each column and converts it once:
dates and numbers stored as text become `datetime64` and `float64`/`int64`, and categorical
text becomes `category`. Values that cannot be converted become missing, and how many there
were per column is kept in `LoadedDataset.unparsed`. Pages after Load never parse the columns
again.

`core/` never imports Streamlit, so detection and parsing are plain functions that can be
tested with pytest.

## Core types (`core/io/base.py`)

**`Source`** (frozen dataclass): the raw upload before parsing.

- `data`: the full file as `bytes`.
- `name`: the file name, used in error messages.
- `source_id`: Streamlit's `file_id`. It changes on every upload, even of the same file, so it
  only answers "is this an upload I have not processed yet?", not "is this new data?".

**`LoadedDataset`** (frozen, `eq=False`): the parsed result.

- `df`, `name`, `separator`, `encoding`, plus the properties `n_rows` and `n_cols`.
- `unparsed`: for each date or numeric column that was text in the file, how many values
  could not be converted and became missing. Empty right after `CsvLoader.load`; filled by
  `schema.prepare_dataset`.
- `eq=False` because comparing two dataclasses would compare their DataFrames, which is
  ambiguous in pandas. Two datasets are "the same" when their `dataset_id` is the same.

**`ReadError`**: the only exception a loader raises on purpose. Its message is written for the
user and the page shows it as is (`st.error(str(e))`).

**`Loader[OptionsT]`** (Protocol): one method, `load(source, options) -> LoadedDataset`.

- Being a `Protocol`, a loader does not inherit from anything: any class with that method
  qualifies (`CsvLoader` is a plain class).
- `OptionsT` is the format-specific way to read the file (`CsvOptions` for CSV). It must be a
  frozen dataclass, because it becomes part of the dataset id and of the cache key.
- Loaders do not guess their options. Detection lives apart (`sniffing.py`), so the caller can
  show the options or, later, let the user override them.

**`compute_dataset_id(content, options)`**: sha256 of the bytes plus `repr(options)`. The same
file read with different options is a different dataset (a different DataFrame). It relies on
`repr` being deterministic, which frozen dataclasses guarantee.

### Two ids, two questions

| Id | Changes when | Answers | Cost |
|---|---|---|---|
| `source_id` | Every upload, even the same file | Do I need to look at this upload? | Free (string compare) |
| `dataset_id` | Content or read options change | Is this new data? Cache key | Hash of the whole file |

Streamlit reruns the page on every click while the file stays in the uploader. `source_id`
lets those reruns skip everything. `dataset_id` is only computed for a new upload, and decides
whether the state that depends on the data must be reset.

## Format detection (`core/io/sniffing.py`)

A CSV file does not say how it is written. Reading it with the wrong settings does not always
fail: it can give one single column, a header taken as a data row, or garbled accents
(`MÃ¡laga` instead of `Málaga`). Detection avoids asking the user for settings they usually do
not know.

`detect_options(data)` returns `CsvOptions(separator, encoding, has_header)`. Every function
returns a safe default when the data is not clear (comma, header present) and never raises:
reporting problems is the loader's job.

### Encoding

Why: the bytes of a CSV are only text once decoded. Excel on Windows often saves in cp1252,
most other tools in UTF‑8. A wrong guess either fails the read or silently corrupts every
non‑ASCII character.

How (`detect_encoding`), from most to least strict:

1. **BOM** at the start of the file states the encoding explicitly:
   - UTF‑8 BOM → `utf-8-sig`, which also strips the BOM. With plain `utf-8` the first column
     would be named `﻿id` instead of `id`.
   - UTF‑16 BOM (LE or BE) → `utf-16`.
2. **`utf-8`** if the whole file decodes with it. UTF‑8 is strict: random non‑UTF‑8 bytes
   almost never form valid sequences, so a successful decode is strong evidence.
3. **`cp1252`** (Windows / Excel exports) if the whole file decodes with it.
4. **`latin-1`** as last resort. It maps every byte to a character, so it never fails.

The **whole file** is decoded, not a sample: a single invalid byte near the end is enough to
make the real read fail. Unlike the other two checks, this one has to be exhaustive.

### Separator

Why: a European Excel exports with `;` (because `,` is the decimal mark), other tools use `,`,
tab or `|`. With the wrong separator every row becomes one column.

How (`detect_separator`): `csv.Sniffer().sniff` on the first 64 KiB (`SAMPLE_BYTES`),
restricted to the candidates in `SEPARATORS` (`,` `;` tab `|`). The sniffer picks the character
that appears a consistent number of times per line. If it cannot decide (e.g. a single-column
file), it falls back to `,`.

The sample is decoded with the detected encoding (`errors="replace"`) and, when the file is
larger than the sample, cut at the last newline: the last line of the sample is probably
incomplete and would confuse the sniffer.

### Header

Why: if the first row holds column names and it is read as data, every column gets a text
value and numeric columns become text. If it is data and read as a header, the first row is
lost and the columns get meaningless names.

How (`detect_header`): `csv.Sniffer.has_header` is not used because it fails on files where
every column is text. The rule is:

> The first row is **data** only if it contains a number and every value matches the type of
> its column. Otherwise it is a **header**.

1. Read up to `SAMPLE_ROWS` (50) rows of the sample.
2. For each column, using the rows after the first, decide its type (`_column_is_number`):
   numeric if ≥ 90 % of the non-empty values are numbers, text if ≤ 10 %, otherwise mixed.
3. For each value of the first row: skip it if it is empty or its column is mixed. If its type
   (number or text) differs from its column's, the first row is a **header**.
4. If no mismatch was found, it is data only if at least one value was a number in a numeric
   column. With no number there is nothing to tell names from values, so a header is assumed.

| Sample | Result | Reason |
|---|---|---|
| `age,city` / `30,Paris` | header | `age` is text, its column is numeric |
| `30,Paris` / `41,Rome` | no header | types match and the row has a number |
| `name,city` / `Ana,Paris` | header | only text: assume a header |

Numbers are recognized with a regex: integers and decimals with `.` or `,` as decimal mark,
optional sign and exponent (`12`, `-3.5`, `3,5`, `.5`, `1e-3`). Thousands separators
(`1.000.000`) are not recognized. Fewer than two rows, or a sample that the `csv` module cannot
read, defaults to header.

When there is no header, `CsvLoader` renames pandas' default `0, 1, 2…` to
`column_1, column_2…`.

## Reading the CSV (`core/io/csv_loader.py`)

`CsvLoader.load(source, options)` checks the cheap problems first, then calls `pd.read_csv` and
translates its exceptions into `ReadError`.

`pd.read_csv` runs first with `engine="pyarrow"`, which is about 3.7x faster (R-L1 in
`docs/performance.md`). If pyarrow raises `ValueError` or `pa.ArrowException` (ragged rows, an
option it does not support…), the file is read again with the C engine. The C engine also gives
the clearer error messages, so the table below describes its errors:

| Situation | Detected by | Message to the user |
|---|---|---|
| File empty or only whitespace | `not data.strip()` before parsing | `'x.csv' is empty.` |
| Binary file (e.g. Excel renamed to `.csv`) | Null byte in the first 64 KiB, unless the encoding is UTF‑16 (which legitimately has them) | `'x.csv' is not a text file. It may be an Excel file renamed to .csv.` |
| Parts of the file in different encodings | `UnicodeDecodeError` | `'x.csv' mixes text encodings. Save it as UTF-8 and upload it again.` |
| pandas sees no columns | `pd.errors.EmptyDataError` | `'x.csv' is empty.` |
| Rows with a wrong number of fields, broken quotes… | `pd.errors.ParserError` | `'x.csv' is malformed: <pandas detail>` |
| Header only, or only blank lines | `df.empty` after parsing | `'x.csv' has no data rows.` |

Notes:

- Exceptions are chained (`raise ReadError(...) from e`), so the original pandas error is kept
  in the traceback for debugging.
- `UnicodeDecodeError` should be rare: the encoding was already checked on the whole file.
- With the C engine, `low_memory=False` makes pandas read the whole file before choosing each
  column's dtype. Otherwise it reads in chunks and a column can end up with mixed types.
  pyarrow rejects this option and does not need it.
- Anything else (e.g. `MemoryError`) is not caught and reaches Streamlit's error screen.

## Caching (`ui/cache.py`)

The cache lives in memory and is **shared by all sessions**, so two users uploading the same
file only read it once. Rules R-C1 to R-C6 in `docs/performance.md` explain the choices below.

- `load_dataset` uses `@st.cache_resource`: every caller gets the same `LoadedDataset` object,
  with no copy. That is safe because nothing modifies it (`core/` is pure). It keeps at most
  `LoadConfig.cache_max_datasets` (5) datasets.
- The Profile wrappers use `@st.cache_data` and return aggregates, whose size grows with the
  number of columns, not of rows. They keep `ProfileConfig.cache_max_datasets` (3) datasets
  each (`column_detail`: `ProfileConfig.cache_max_columns`).

Convention for the wrappers:

- The first argument is `dataset_id`, which is the real cache key.
- The data is passed as `_df` / `_source`. Streamlit does not hash arguments starting with
  `_`, which avoids hashing a whole DataFrame on every call. This is only correct because
  `dataset_id` already identifies that data: never call a wrapper with a `dataset_id` that does
  not match the data.
- Other arguments must be hashable (frozen dataclasses, tuples).
- Exceptions are not cached: a `ReadError` is tried again on the next call.

`load_dataset(dataset_id, _source, options)` reads the file and runs
`schema.prepare_dataset`, with `show_spinner="Reading CSV…"`. It returns
`(LoadedDataset, Schema)`. The Profile wrappers (`base_profile`, `rows_with_missing`,
`correlation`, `numeric_stats`, `categorical_stats`, `datetime_stats`, `text_stats`,
`column_overview`, `column_detail`) follow the same pattern with `dataset_id` and `_df`.

When the user leaves columns out of the profile, the data is different, so it needs a
different key: `cache.view_id(dataset_id, skipped)` returns one, and is the id to pass along
with the reduced DataFrame.

`dataset_id` must be passed even when it looks redundant. Without it the key of `load_dataset`
would be only `options`, and two different files with the same detected options (e.g. both
comma, UTF‑8, with header) would share one entry: the second upload would show the first
file's data.

## Session state (`ui/state.py`)

`st.session_state` is per browser session and survives reruns and page changes. Only
`ui/state.py` reads or writes it, through typed functions, so the rules about what is cleared
when the dataset changes live in one place.

| Key | Content | Reset when |
|---|---|---|
| `dataset` | Active `LoadedDataset` or `None` | `clear_dataset` |
| `dataset_id` | Content hash of the active dataset | `clear_dataset` |
| `source_id` | Last upload that was processed | Every processed upload; `clear_dataset` |
| `schema` | `Schema` of the active dataset | New content; `clear_dataset` |
| `skipped_columns` | Columns the user left out of the profile (tuple) | New content; `clear_dataset` |
| `w_*` | Widgets that depend on the columns | New content; `clear_dataset` |

Functions:

- `init_state()`: called from `app.py` on every rerun. Fills missing keys with their defaults
  and keeps existing values.
- `source_changed(source_id)`: the cheap check done on every rerun.
- `set_dataset(dataset, dataset_id, source_id)`: always marks the upload as processed. If
  `dataset_id` equals the active one it returns `False` and changes nothing else, so uploading
  the same file again keeps the schema and the user's widget choices. Otherwise it clears the
  dependent state, stores the dataset and returns `True`.
- `clear_dataset()`: forgets the dataset and everything derived from it.
- `get_dataset()`, `get_dataset_id()`, `get_schema()`, `set_schema()`.
- `require_dataset()`, `require_dataset_id()`, `require_schema()`: the same, but raise
  `RuntimeError` when nothing is loaded. For code that only runs with a dataset.
- `get_skipped_columns()`, `set_skipped_columns(names)`: columns left out of the profile.
- `widget_key(name)`: returns `"w_" + name`. Any widget whose options depend on the columns
  must use `key=state.widget_key(...)`, so a new dataset does not leave it pointing at columns
  that no longer exist.
- `widget_value(key)`: current value of a widget, for its `on_change` callback.

## The Load page (`ui/pages/load.py`)

`load_page()` runs on every rerun:

1. `_upload_source()` draws the uploader and builds a `Source` from the chosen file.
2. If there is one, `_load(source)`:
   - returns at once if `source_id` was already processed;
   - detects the options and computes `dataset_id`;
   - calls `cache.load_dataset`, which reads the file and types its columns. On `ReadError`
     it shows the message and returns **without** marking the upload as processed, so the
     error stays visible on later reruns and the previous dataset stays active;
   - calls `state.set_dataset` and, only if the content is new, stores the schema returned by
     `load_dataset`.
3. Draws the active dataset from the session state: a summary (rows, columns, encoding) and the
   first `LoadConfig.preview_rows` rows. Only the head is sent to the browser.

Because the dataset lives in the session state, it stays after the file is removed from the
uploader or the user moves to another page.

### What happens in each scenario

| Action | Detection + hash | Parse | State |
|---|---|---|---|
| Any click while the file stays in the uploader | No | No | Unchanged |
| First upload of a file | Yes | Yes | Dataset, schema, widgets reset |
| Same file uploaded again | Yes | No (cache) | Only `source_id` updated; schema and widgets kept |
| Same file uploaded by another session | Yes | No (shared cache) | That session's state |
| File that cannot be read | Yes | Fails | Error shown; previous dataset kept |
| File removed from the uploader | No | No | Dataset kept |

## Adding a new format

See the `add-loader` skill for the full checklist. In short:

1. An options dataclass, `@dataclass(frozen=True)`, with tuples instead of lists.
2. A class in `core/io/` with `load(source, options) -> LoadedDataset` that raises `ReadError`
   with a user-facing message for every expected failure.
3. Make `cache.load_dataset` (or a sibling wrapper with the same signature) pick the loader,
   and keep calling `schema.prepare_dataset` on the result.
4. Accept the extension in the uploader and choose the loader on the Load page.
5. Tests in `tests/core/io/`.

## Tests

`tests/core/io/test_csv_loader.py` covers `CsvLoader`: dtypes, files without a header, cp1252
with `;`, UTF‑8 with BOM, empty values, the fallback from pyarrow to the C engine on ragged
rows, and every `ReadError` message. `sniffing.py` has no tests of its own; it is exercised
through those cases. Column typing is tested in `tests/core/test_schema.py`, and the Load page
in `tests/ui/test_pages_smoke.py`.
