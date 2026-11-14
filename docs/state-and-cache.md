# Cache and session state

How litmetrics keeps data between reruns, pages, browser tabs and page reloads. It covers
`ui/cache.py`, `ui/state.py`, `core/io/library.py` and the code that connects them
(`ui/reading.py`, `ui/sidebar.py`, `app.py`).

`docs/data-loading.md` follows one file from the uploader to the active dataset;
`docs/performance.md` has the rules R-C1 to R-C6 behind the cache. This document is about the
model as a whole: what lives where, how long it lives, and who can see it.

## Two layers

```
                    server process (one per `uv run litmetrics`)
┌──────────────────────────────────────────────────────────────────────────┐
│  CACHE: shared by every session                         ui/cache.py      │
│                                                                          │
│   st.cache_resource ─► DatasetLibrary (core/io/library.py)               │
│                          dataset_id ─► LibraryEntry(dataset, schema,     │
│                                                     part, loaded_at)     │
│                          LRU, LoadConfig.cache_max_datasets (5)          │
│                                                                          │
│   st.cache_data ─────► Profile results (aggregates), keyed by dataset_id │
│                          or view_id, ProfileConfig.cache_max_* entries   │
└──────────────────────────────────────────────────────────────────────────┘
          ▲  same objects, no copy                ▲
          │                                       │
┌─────────┴──────────────────────┐   ┌────────────┴───────────────────────┐
│ SESSION A (browser tab)        │   │ SESSION B (another tab or user)    │
│ st.session_state  ui/state.py  │   │ st.session_state                   │
│  dataset, dataset_id ──────────┼─► │  dataset, dataset_id               │
│  schema, skipped_columns, w_*  │   │  ...                               │
│  multipart (file bytes), part  │   │                                    │
└────────────────────────────────┘   └────────────────────────────────────┘
```

| | Cache (`ui/cache.py`) | Session state (`ui/state.py`) |
|---|---|---|
| Holds | Every dataset read (library) and every Profile result | Which dataset this tab is using, and the user's choices about it |
| Scope | The server process: every tab and every user | One browser session (one tab, one connection) |
| Survives a rerun / page change | Yes | Yes |
| Survives a page reload (F5) | Yes | **No**: a new session starts empty |
| Survives a server restart | No | No |
| Bounded by | `max_entries` / library size, least recently used dropped | The life of the session |
| Key | `dataset_id` (content hash) or `view_id` | Private constants in `ui/state.py` |

The split follows one idea: **the data is shared, the choices are personal.** Reading and
profiling a file is expensive and the result is the same for anyone, so it goes in the cache.
Which dataset is open, which columns are skipped and which tab is selected belong to one user in
one tab, so they go in the session.

## The cache (`ui/cache.py`)

### Datasets: the library

`load_dataset(dataset_id, _source, options, part)` is the only way a file becomes a dataset.
It looks the id up in a `DatasetLibrary` held by `@st.cache_resource` (`_library()`), and only
on a miss reads the file (`formats.load` + `schema.prepare_dataset`) and stores a
`LibraryEntry`.

The library is used instead of `@st.cache_data` on the loader because:

- **No copies (R-C1, R-C4).** `cache_resource` returns the same object to every caller. A
  `cache_data` hit would deserialize and copy the whole DataFrame on every rerun.
- **It can be listed and edited.** A function cache is hidden: it cannot say what it holds or
  forget one entry without all its arguments. The library offers `cached_datasets()`,
  `cached_dataset(id)`, `forget_dataset(id)` and `clear_all()`, which is what the Load page
  ("Datasets in memory") and the sidebar selector are built on.
- **It remembers the schema and the part.** Opening a dataset from memory needs its `Schema`
  and, for a table of a database, the name of the table (`LibraryEntry.label` = `file / table`).

`DatasetLibrary` is pure (`core/`), thread-safe (one lock; each session runs in its own thread)
and keeps at most `LoadConfig.cache_max_datasets` entries. `get` marks an entry as just used, so
the eviction order is "least recently used", not "oldest read".

Because the objects are shared, **nothing may modify a dataset or schema** that came from the
cache. `core/` is pure, so this holds as long as nobody calls an in-place pandas method on
`dataset.df`.

### Results: `st.cache_data`

The Profile wrappers (`base_profile`, `rows_with_missing`, `correlation`, `numeric_stats`,
`categorical_stats`, `datetime_stats`, `text_stats`, `column_overview`, `inspect_column`)
use `@st.cache_data`. They return aggregates whose size grows with the number of columns, not
of rows, so copying them on each hit is cheap (R-C1). Each has a `max_entries`
(`ProfileConfig.cache_max_datasets`, or `cache_max_columns` for the inspector) (R-C3).

Conventions (details in `docs/data-loading.md` → *Caching*):

- `dataset_id` comes first and is the real key. The data goes as `_df` / `_source` / `_unparsed`
  and is not hashed (R-C2). Never pass a `dataset_id` that does not match the data.
- Other arguments are hashable: frozen dataclasses and tuples.
- Measures of single columns take the id of the whole dataset and the columns as `names`. Only
  what depends on the set of columns takes `view_id(dataset_id, skipped)`.
- Exceptions are not cached. A `ReadError` is remembered in the session instead
  (`state.set_read_error`), so a broken file is not read again on every rerun.

### Ids

| Id | Computed by | Identifies | Used as |
|---|---|---|---|
| `dataset_id` | `compute_dataset_id(bytes, options)` | Content + read options | Key of the library and of every cache wrapper |
| `view_id` | `cache.view_id(dataset_id, skipped)` | Dataset without some columns | Key of `base_profile` / `rows_with_missing` |
| `source_id` / `read_id` | Uploader `file_id` (+ `/part`) | One upload (one table of it) | Session only: "was this read already processed?" |

`dataset_id` depends only on the content, so the same file uploaded in two tabs, by two users,
or after a reload is the same entry: it is read once.

### Emptying it

| Action | Library | `cache_data` | Sessions using that dataset |
|---|---|---|---|
| A sixth dataset is read | Least recently used dropped | Untouched (own LRU) | Keep it open (see below) |
| "Remove" in Load → Datasets in memory (`forget_dataset`) | That entry dropped | Untouched until pushed out | This session: `state.clear_dataset()`. Others keep it open |
| "Clear cache" (`clear_all`) | Emptied | Emptied | This session: no dataset. Others keep it open |
| Server restart | Gone | Gone | Gone |

A session that has a dataset open holds a reference to it in `st.session_state`, so the
DataFrame stays in memory until that session switches dataset or ends, even after it left the
library. The sidebar handles that case: the active dataset is added to the selector although
it is not in the library ("dropped from memory, but still open here"). Picking a dataset that is
no longer kept shows a toast asking to upload the file again.

## The session state (`ui/state.py`)

`ui/state.py` is the only module that reads or writes `st.session_state`. Every key has a
private constant, a default in `_defaults()` and typed accessors. `init_state()` runs on every
rerun from `app.py` and fills only the missing keys.

| Key | Content | Default |
|---|---|---|
| `dataset` | Active `LoadedDataset` (a reference to the library's object) | `None` |
| `dataset_id` | Its `dataset_id` | `None` |
| `source_id` | Last read processed (`read_id`) | `None` |
| `read_error` | `(read_id, message)` of the last read that failed | `None` |
| `multipart` | File with several datasets: `(Source, Parts)`, its bytes included | `None` |
| `part` | Dataset of that file picked by the user | `None` (the first) |
| `schema` | `Schema` of the active dataset | `None` |
| `skipped_columns` | Columns left out of the profile | `()` |
| `w_*` | Widgets whose options depend on the columns (`state.widget_key`) | — |
| `dataset_switcher` | Sidebar selector of datasets | Set to `dataset_id` before each draw |
| `part_switcher` | Sidebar selector of tables | Set to the active part before each draw |

### Changing the active dataset

There are three ways in, and all of them clear the dependent state (`schema`,
`skipped_columns`, every `w_*` widget) through `_reset_dependent_state()`:

| Function | Called from | Reads a file? | `source_id` |
|---|---|---|---|
| `set_dataset(dataset, id, source_id)` | `reading.read` (upload, table picked) | Yes (or a cache hit) | Set to the read |
| `switch_dataset(dataset, id, schema)` | Sidebar selector, Load → "Open" | No: taken from the library | Kept |
| `clear_dataset()` | Load → "Remove" / "Clear cache" | No | Kept |

`set_dataset` and `switch_dataset` do nothing beyond `source_id` when the id is already the
active one, so uploading the same file again keeps the schema, the skipped columns and the
widget choices. `switch_dataset` and `clear_dataset` keep `source_id` so the file still in the
uploader is not read again and does not replace the dataset the user just picked.

### Widgets

- A widget whose options are columns uses `key=state.widget_key("name")` (prefix `w_`), so it is
  deleted when the dataset changes and never points at a column that no longer exists.
- The two sidebar selectors are not `w_` widgets: they list datasets, not columns. Their value
  is overwritten before each draw (`sync_switcher`, `sync_part_switcher`), because the active
  dataset can also change from the Load page.
- Values are changed from `on_change` / `on_click` callbacks, never after the widget was drawn
  in the same rerun.

### Order of a rerun (`app.py`)

1. `state.init_state()`: missing keys get their defaults.
2. `reading.read_chosen_part()`: if a table was picked in the sidebar, read it now, so the page
   drawn in this rerun already shows it.
3. `nav.run()`: the page. The Load page may read an upload here.
4. `sidebar.draw()`: after the page, so a dataset read in step 3 is already listed.

## Persistence across page reloads

Reloading the page (F5, opening the URL again, a new tab) starts a **new Streamlit session**.
`st.session_state` belongs to the old one, so the new session starts with the defaults of
`_defaults()`. The cache lives in the server process, so it is still there.

### What is kept and what is lost

| After a reload | Kept? | Why |
|---|---|---|
| Datasets read (library) | **Yes** | `st.cache_resource`, process-wide |
| Profile results | **Yes** | `st.cache_data`, process-wide; opening the dataset again is a cache hit |
| Current page | Yes | It is in the URL (`url_path` of `st.Page`: `/profile`, `/load`...) |
| Active dataset (`dataset`, `dataset_id`) | **No** | Session state |
| Schema | No, but it comes back | Stored in the `LibraryEntry`; restored by `switch_dataset` |
| Skipped columns, open tab, selected method, other `w_*` widgets | No | Session state; they start from their defaults |
| File in the uploader | No | The browser forgets it |
| File with several datasets (`multipart`, its bytes, `part`) | No | Session state. Tables already read are still in the library (`file / table`); the others need the file again |
| Remembered read error | No | The file is no longer in the uploader anyway |

So after a reload the user lands on the same page with no active dataset (the Profile page
shows its "Load a file" message), and the sidebar selector lists every dataset in memory with
the placeholder "Choose a dataset". One pick restores it: `switch_dataset` takes the dataset
and schema from the library without reading any file, and the Profile is served from the
cache. The same works from Load → "Datasets in memory" → "Open".

### Why the active dataset is not persisted

The active dataset is deliberately **not** saved anywhere that outlives the session (no
`st.query_params`, cookie or browser storage). It can be reached again with one click from the
dataset selector of the sidebar, which every page shows while the cache holds a dataset, so
restoring it automatically would save one click and cost:

- **Stale ids.** The id would point at a library that changes under it: the dataset may have
  been pushed out (LRU), removed, or cleared by another session, or the server may have
  restarted. Every restore would need a fallback path and a message for "that dataset is gone",
  which the selector already handles in one place.
- **Shared library, personal choice.** The library is shared by every session, so "the last
  dataset" is not a per-user fact. Picking it from the list makes the choice explicit.
- **Partial restores.** Restoring only the dataset would bring back a Profile without the
  skipped columns and widget choices the user had, which is more confusing than starting clean.
  Persisting all of that would mean a second copy of the state with its own rules.
- **Simplicity.** `ui/state.py` keeps one rule: the session state lives as long as the session.

If this changes (for example to restore the dataset from the URL), the restore must go through
`cache.cached_dataset(id)` + `state.switch_dataset`, treat a missing entry as "no dataset", and
run before the page draws (next to `reading.read_chosen_part()` in `app.py`).

### Server restart

Everything is lost: the cache is in memory only, nothing is written to disk. The user uploads
the file again; `compute_dataset_id` gives the same id as before.

## Adding state or a cache entry

- **New session key:** a private constant in `ui/state.py`, a default in `_defaults()`, typed
  accessors, and a decision about whether `_reset_dependent_state()` must clear it. If it
  depends on the columns, prefer a `w_` widget key.
- **New cached computation:** a wrapper in `ui/cache.py` that calls one core function, with
  `dataset_id` first, the data as `_df`, hashable arguments, a `max_entries` and an aggregate as
  result (R-C1 to R-C6). Pick `dataset_id` or `view_id` by whether it depends on the set of
  columns.
- **Something that must survive a reload** belongs in the cache, keyed by content, not in the
  session. Ask whether it is really shared by every user first.
