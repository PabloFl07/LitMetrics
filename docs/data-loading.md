# Data loading

How a file becomes the active dataset of a session: format detection, parsing, error
handling, caching and session state. It covers the Load page and everything under `core/io/`,
plus the parts of `ui/state.py` and `ui/cache.py` that keep the dataset.

## Overview

```
st.file_uploader
      │  UploadedFile (file_id)
      ▼
state.source_changed? ──no──► (nothing to do: draw the active dataset)
              │ yes
              ▼
state.get_read_error? ──yes──► st.error (the remembered message; nothing is read)
              │ no
              ▼
        Source(bytes, name, file_id)
              │
              ▼
  formats.list_parts(source) ──Parts──► state.set_multipart(source, parts)
              │ None (one dataset)               │  (every rerun, uploader or not)
              │                                  ▼
              │                     selectbox (restored from state) ──► part
              │                                  │
              │                     read_id = file_id + "/" + part, processed once
              ▼                                  │
              ◄──────────────────────────────────┘
        formats.detect_options(source, part) ──► options of its format (CsvOptions, SqliteOptions)
              │
              ▼
        compute_dataset_id(bytes, options) ──► dataset_id (sha256)
              │
              ▼
        cache.load_dataset(dataset_id, source, options)
              │   formats.load ──► LoadedDataset ──► prepare_dataset ──► (typed dataset, schema)
              │        │
              │        └─ ReadError ──► state.set_read_error, st.error, previous dataset kept
              ▼
        state.set_dataset(dataset, dataset_id, source_id)
              │ new content?
              ▼
        state.set_schema(schema)
```

| Module | Role |
|---|---|
| `core/io/base.py` | Format-independent types: `Source`, `LoadedDataset`, `ReadError`, `Loader`, `compute_dataset_id` |
| `core/io/formats.py` | The supported formats (`FORMATS`), and which one reads a file: `format_for`, `detect_options`, `load` |
| `core/io/sniffing.py` | Guesses `CsvOptions` (encoding, separator, header, decimal mark) from the bytes |
| `core/io/csv_loader.py` | `CsvLoader`: turns a `Source` + `CsvOptions` into a `LoadedDataset` |
| `core/io/sqlite_loader.py` | `SqliteLoader` and `table_names`: one table or view of a SQLite database |
| `core/io/parquet_loader.py` | `ParquetLoader`: a Parquet file |
| `core/io/feather_loader.py` | `FeatherLoader`: a Feather (V1, V2) or Arrow IPC file |
| `core/io/arrow_table.py` | `to_frame`: Arrow table of a columnar file → DataFrame, converting the types the app cannot use |
| `core/schema.py` | Semantic type of each column, and conversion of each column to the dtype of its type |
| `ui/cache.py` | Cache wrappers, shared by all sessions |
| `ui/state.py` | The only module that touches `st.session_state` |
| `ui/pages/load.py` | Orchestrates the flow above and shows a summary and a preview |

Typing the columns is part of loading (R-L2 in `docs/performance.md`).
`schema.prepare_dataset` infers the semantic type of each column and converts it once:
dates and numbers stored as text become `datetime64` and `float64`/`int64`, and categorical
and boolean text ("yes"/"no") becomes `category`. Each distinct value is converted once and
spread to every row (`schema._by_unique`): 40x faster on dates, which repeat. Values that
cannot be converted become missing, and how many there were per column is kept in
`LoadedDataset.unparsed`. Pages after Load never parse the columns again.

`core/` never imports Streamlit, so detection and parsing are plain functions that can be
tested with pytest.

## Core types (`core/io/base.py`)

**`Source`** (frozen dataclass): the raw upload before parsing.

- `data`: the full file as `bytes`.
- `name`: the file name, used in error messages.
- `source_id`: Streamlit's `file_id`. It changes on every upload, even of the same file, so it
  only answers "is this an upload I have not processed yet?", not "is this new data?".

**`LoadedDataset`** (frozen, `eq=False`): the parsed result.

- `df`, `name`, plus the properties `n_rows` and `n_cols`.
- `format`: name of the file format shown to the user (`"CSV"`).
- `details`: how the file was read, as label → value, both already text for the user and in
  the order to show them. `CsvLoader` writes `separator`, `encoding`, `decimal mark` and
  `column names`. The Load page shows them without knowing the format.
- `unparsed`: for each date or numeric column that was text in the file, how many values
  could not be converted and became missing. Empty right after `Loader.load`; filled by
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

## Formats (`core/io/formats.py`)

The only module that knows which formats exist. `FORMATS` holds one `FileFormat` per format:
`name` (shown to the user), `extensions` (lowercase, with the dot), `options_type` (its frozen
options class), `detect_options(bytes)` and `loader`. Today it has four entries: CSV (`.csv`,
`CsvOptions`, `sniffing.detect_options`, `CsvLoader`), SQLite (`.db`, `.sqlite`,
`.sqlite3`, `SqliteOptions`, `SqliteLoader`, with parts), Parquet (`.parquet`, `.pq`,
`ParquetOptions`, `ParquetLoader`) and Feather (`.feather`, `.arrow`, `FeatherOptions`,
`FeatherLoader`). `ReadOptions` is the union of the
options classes of every format: the type of the options outside `core/io/`.

- `extensions()`: every supported extension. The uploader of the Load page accepts these.
- `format_for(name)`: the format of a file, by its extension (case-insensitive). `ReadError`
  if none matches.
- `detect_options(source, part=None)`: the read options of the file, guessed by its format,
  for the dataset `part` when the file holds several. This is the only place where the format
  is chosen by the file name.
- `list_parts(source)`: see "Files with several datasets" below.
- `load(source, options)`: reads the file with the loader of the format of `options`. Each
  format has its own options class, so the options carry the format and a file cannot be read
  with the loader of another one. Options of no format raise `TypeError` (a bug, not a bad file).

`ui/cache.py` and the Load page only call these functions, so they do not change when a format
is added.

### Files with several datasets

A database has tables, a workbook has sheets: which one to read cannot be guessed from the
bytes. Such a format sets `FileFormat.parts` to a `PartsSpec(label, names, select)`:

- `label`: what one dataset is called ("Table").
- `names(source)`: lists them, raising `ReadError` when the file cannot be read.
- `select(options, name)`: the options that read the one called `name`.

`list_parts(source)` returns `Parts(label, names)`, or `None` for single-dataset formats, and
`detect_options(source, part)` applies `select`. The Load page draws a selectbox from `Parts`
and never names the format, so a new multi-dataset format needs no change outside `core/io/`.

## CSV detection (`core/io/sniffing.py`)

A CSV file does not say how it is written. Reading it with the wrong settings does not always
fail: it can give one single column, a header taken as a data row, or garbled accents
(`MÃ¡laga` instead of `Málaga`). Detection avoids asking the user for settings they usually do
not know.

`detect_options(data)` returns `CsvOptions(separator, encoding, has_header, decimal)`. Every function
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
2. **`utf-8`** if every byte is ASCII (`bytes.isascii()`, which decodes nothing), or if the
   whole file decodes with it. UTF‑8 is strict: random non‑UTF‑8 bytes almost never form valid
   sequences, so a successful decode is strong evidence.
3. **`cp1252`** (Windows / Excel exports) if the whole file decodes with it.
4. **`latin-1`** as last resort. It maps every byte to a character, so it never fails.

The **whole file** is checked, not a sample: a single invalid byte near the end is enough to
make the real read fail. Unlike the other checks, this one has to be exhaustive. It is decoded
in chunks of 64 MB with an incremental decoder (a character split between two chunks is not
an error), so the text of the whole file is never in memory at once (R-L4).

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

### Decimal mark

Why: a European file separated by `;` writes `3,5`. Read with `.` as the decimal mark, every
such column stays text, and a column of distinct amounts even looks like an identifier.

How (`detect_decimal`): with `,` as separator the decimal mark is `.`. Otherwise, the cells of
the sample rows are counted: `,` wins only when more cells look like `3,5` than like `3.5`.
Integers say nothing. The mark is passed to `pd.read_csv(decimal=...)`, which both engines
support. A thousands separator (`1.000,50`) is still not supported: those values stay text and
are counted in `unparsed`.

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
- After every read, failed or not, Arrow's memory pool gives back the blocks it no longer uses
  (`release_unused`, R-L5): about 1 GB on a 569 MB file.

## Reading a SQLite database (`core/io/sqlite_loader.py`)

`SqliteOptions(table)`: the table or view to read; `None` reads the first one by name.
`detect_options` has nothing to guess and returns `SqliteOptions()`.

The app only has bytes, so `sqlite3.Connection.deserialize` opens them as an in-memory
database (no temporary file; the file is in memory twice while reading). The connection is
read-only (`PRAGMA query_only`) and closed after each read. `table_names` lists tables and
views sorted by name, leaving out SQLite's own (`sqlite_*`). The table is read with
`pd.read_sql_query("SELECT * FROM <quoted name>")`; names come from the file and are quoted
with doubled `"`. BLOB values become a text with their size (`"<1,024 bytes>"`) and the columns
that had them are listed in `details` ("binary columns"), next to "table".

| Situation | Message to the user |
|---|---|
| Empty file | `'x.db' is empty.` |
| No `SQLite format 3` header (another format renamed) | `'x.db' is not a SQLite database.` |
| Damaged database | `'x.db' is damaged and cannot be read: <sqlite detail>` |
| No tables or views | `'x.db' has no tables.` |
| Table not in the file | `'x.db' has no table called 't'.` |
| Query fails | `Table 't' of 'x.db' cannot be read: <detail>` |
| Table without rows | `Table 't' of 'x.db' has no data rows.` |

## Reading Parquet and Feather files (`core/io/parquet_loader.py`, `feather_loader.py`)

Both formats store typed columns, so there is nothing to detect: `ParquetOptions` and
`FeatherOptions` are empty frozen dataclasses that only carry the format, and
`detect_options` returns them as they are. Feather V2 is the Arrow IPC file format, so `.arrow`
files are read as Feather; Feather V1 (deprecated by pyarrow) and LZ4/ZSTD compression are read
too. Arrow IPC *streams* (`.arrows`) are not.

The bytes are wrapped in `pa.BufferReader` (no copy) and read with `pq.read_table` or
`feather.read_table`. `arrow_table.to_frame` then turns the Arrow table into a DataFrame:

- integers, floats, text, booleans, timestamps (with or without timezone) and dictionary
  columns (→ `category`) keep their type;
- dates become `datetime64` (`date_as_object=False`);
- times of day become text (`"13:45:00"`), as in a CSV;
- decimals become `float64`; durations become `float64` seconds;
- binary values become a text with their size (`"<1,024 bytes>"`), as in SQLite;
- nested values (lists, structs, maps) become JSON text, so they can be hashed and counted.
  This goes through Python objects one value at a time: slow on large columns, but rare;
- a pandas index saved with the file (not a `RangeIndex`) becomes columns again;
- column names become strings and repeated names get `.1`, `.2`... as `read_csv` does.

Each converted column is listed in `details` under its kind (`"decimal columns": "price (read
as floating-point numbers)"`); a file with nothing to convert has no `details`. The conversion
uses `self_destruct` and `split_blocks`, so each Arrow column is freed as soon as it is
converted, and Arrow's pool is released after each read (R-L5).

| Situation | Message to the user |
|---|---|
| Empty file | `'x.parquet' is empty.` |
| Another format renamed (no `PAR1` at both ends; no `ARROW1`/`FEA1` at the start) | `'x.parquet' is not a Parquet file.` / `'x.feather' is not a Feather (Arrow IPC) file.` |
| Damaged or truncated file | `'x.parquet' is damaged and cannot be read: <pyarrow detail>` |
| No columns | `'x.parquet' has no columns.` |
| No rows | `'x.parquet' has no data rows.` |

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

`load_dataset(dataset_id, _source, options)` reads the file with `formats.load` and runs
`schema.prepare_dataset`, with `show_spinner="Reading the file…"`. `options` is whatever
frozen dataclass the format of the file uses. It returns
`(LoadedDataset, Schema)`. The Profile wrappers (`base_profile`, `rows_with_missing`,
`correlation`, `numeric_stats`, `categorical_stats`, `datetime_stats`, `text_stats`,
`column_overview`, `inspect_column`) follow the same pattern with `dataset_id` and `_df`.
The ones that count values that were not numbers or dates also take
`LoadedDataset.unparsed` as `_unparsed`: it belongs to the dataset, so the id covers it. Each
wrapper only calls one function of `core/` with cached inputs; `inspect_column`, for example,
passes the cached base pass to `columns.inspect_column`, which decides what to compute for
the type of the column.

When the user leaves columns out of the profile, the data is different, so what depends on the
set of columns (`base_profile`, `rows_with_missing`) needs a different key:
`cache.view_id(dataset_id, skipped)` returns one, and is the id to pass along with the reduced
DataFrame. Measures of single columns do not depend on the others: they keep the id of the whole
dataset and the whole DataFrame, and receive the columns to describe as `names`, so skipping
columns reuses them.

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
| `dataset` | Active `LoadedDataset` or `None` | New content |
| `dataset_id` | Content hash of the active dataset | New content |
| `source_id` | Last read that was processed (`read_id`: upload, plus part) | Every processed read |
| `read_error` | Last read (or listing of parts) that failed, and its message | Next failure |
| `multipart` | Active file with several datasets: its `Source` (bytes kept) and `Parts` | New upload with parts; cleared when a single-dataset file is read |
| `part` | Dataset of that file picked by the user | New upload with parts |
| `part_selector` | Selectbox of the dataset (not `w_`: switching must not reset it) | Restored from `part` after visiting another page |
| `schema` | `Schema` of the active dataset | New content |
| `skipped_columns` | Columns the user left out of the profile (tuple) | New content |
| `w_*` | Widgets that depend on the columns | New content |

Functions:

- `init_state()`: called from `app.py` on every rerun. Fills missing keys with their defaults
  and keeps existing values.
- `source_changed(source_id)`: the cheap check done on every rerun.
- `set_dataset(dataset, dataset_id, source_id)`: always marks the upload as processed. If
  `dataset_id` equals the active one it returns `False` and changes nothing else, so uploading
  the same file again keeps the schema and the user's widget choices. Otherwise it clears the
  dependent state, stores the dataset and returns `True`.
- `get_dataset()`, `get_dataset_id()`, `get_schema()`, `set_schema()`.
- `require_dataset()`, `require_dataset_id()`, `require_schema()`: the same, but raise
  `RuntimeError` when nothing is loaded. For code that only runs with a dataset.
- `get_skipped_columns()`, `set_skipped_columns(names)`: columns left out of the profile.
- `get_read_error(source_id)`, `set_read_error(source_id, message)`: the error of an upload
  that could not be read, so later reruns show it without reading the file again.
- `get_multipart()`, `set_multipart(source, parts)`, `clear_multipart()`: the active file
  with several datasets. Its bytes stay in the session (memory: one copy of the file per
  session) because the uploader forgets its file when another page is shown, and another
  table must still be readable. `set_part(name)` and `restore_part_widget()`: the chosen
  dataset, given back to the selectbox (`PART_WIDGET`) after visiting another page, since a
  widget that is not drawn loses its state.
- `widget_key(name)`: returns `"w_" + name`. Any widget whose options depend on the columns
  must use `key=state.widget_key(...)`, so a new dataset does not leave it pointing at columns
  that no longer exist.
- `widget_value(key)`: current value of a widget, for its `on_change` callback.

## The Load page (`ui/pages/load.py`)

`load_page()` runs on every rerun:

1. Draws the uploader.
2. If there is a file, `_load(uploaded)`:
   - shows the remembered error if listing its parts failed;
   - for formats with parts, lists them once per upload and keeps the file in the state
     (`state.set_multipart`); `_part()` then draws the selectbox from the state on every
     rerun, with or without the file in the uploader, and reads the chosen dataset with
     `read_id = file_id + "/" + part`, so each table is one read, done once;
   - returns at once if its `file_id` was already processed, before reading its bytes
     (`getvalue()` copies the whole file on every call);
   - if reading it already failed, shows the remembered error and returns;
   - builds the `Source`, detects the options of its format (`formats.detect_options`) and
     computes `dataset_id`;
   - calls `cache.load_dataset`, which reads the file and types its columns. On `ReadError`
     it remembers and shows the message and returns **without** marking the upload as
     processed, so the error stays visible on later reruns and the previous dataset stays
     active. Exceptions are not cached: without the remembered message, every rerun would read
     the file again;
   - calls `state.set_dataset` and, only if the content is new, stores the schema returned by
     `load_dataset`.
3. Draws the active dataset from the session state: a summary (rows, columns, format, and how
   the file was read: `LoadedDataset.details`) and the first `LoadConfig.preview_rows`
   rows. Only the head is sent to the browser.

Because the dataset lives in the session state, it stays after the file is removed from the
uploader or the user moves to another page.

### What happens in each scenario

| Action | Detection + hash | Parse | State |
|---|---|---|---|
| Any click while the file stays in the uploader | No | No | Unchanged |
| First upload of a file | Yes | Yes | Dataset, schema, widgets reset |
| Same file uploaded again | Yes | No (cache) | Only `source_id` updated; schema and widgets kept |
| Same file uploaded by another session | Yes | No (shared cache) | That session's state |
| File that cannot be read | Yes | Fails | Error shown and remembered; previous dataset kept |
| Any click after a file failed | No | No | The remembered error is shown again |
| File removed from the uploader | No | No | Dataset kept |

## Adding a new format

See the `add-loader` skill for the full checklist. In short:

1. An options dataclass, `@dataclass(frozen=True)`, with tuples instead of lists.
2. A class in `core/io/` with `load(source, options) -> LoadedDataset` that raises `ReadError`
   with a user-facing message for every expected failure, and fills `format` and `details`.
3. A `FileFormat` entry in `formats.FORMATS`, and its options class added to
   `formats.ReadOptions`. The cache and the uploader pick it up from there.
4. Tests in `tests/core/io/`.

## Tests

`tests/core/io/test_csv_loader.py` covers `CsvLoader`: dtypes, files without a header, cp1252
with `;`, UTF‑8 with BOM, empty values, decimal commas, the fallback from pyarrow to the C
engine on ragged rows, and every `ReadError` message. `tests/core/io/test_sqlite_loader.py` covers tables, views, quoted names, BLOBs and every
`ReadError` of the SQLite loader. `tests/core/io/test_parquet_loader.py` and
`test_feather_loader.py` cover types, row groups, compression, Feather V1 and every
`ReadError`; `test_arrow_table.py` covers each type conversion, the restored index and
repeated column names. `tests/core/io/test_formats.py` covers choosing the format
by extension, choosing the loader by the options, and that `ReadOptions` lists every format. `tests/core/io/test_sniffing.py` covers the detection: header, separator, encoding (including a character split between two chunks
and a cp1252 byte at the end) and decimal mark. Column typing is tested in
`tests/core/test_schema.py`, and the Load page in `tests/ui/test_pages_smoke.py` (including
that a failed upload is read only once).
