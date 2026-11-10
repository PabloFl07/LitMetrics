"""Time every step from reading a CSV to the Profile page, on a synthetic dataset.

Usage (from the project root):
    uv run python scripts/bench_profile.py [--rows 3000000] [--csv data/bench.csv]

The dataset has the structure described in docs/performance.md: 8 float columns with 10%
missing values, 4 integer columns, an integer id, two low-cardinality text columns (one of them
yes/no), a unique text code, short free text and a date stored as text. It is written once to
``--csv`` and reused by later runs.

Prints the time and the peak memory of each step (Linux only for memory: it reads
``/proc/self/status``). Run it before and after changing anything in ``core/``, and compare.
"""

import argparse
import gc
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from litmetrics.config import ProfileConfig
from litmetrics.core import columns as cols
from litmetrics.core import profile
from litmetrics.core.io.base import Source, compute_dataset_id
from litmetrics.core.io.csv_loader import CsvLoader
from litmetrics.core.io.sniffing import detect_options
from litmetrics.core.schema import apply_schema, infer_schema

_STATUS = Path("/proc/self/status")


def _status_mb(field: str) -> int | None:
    if not _STATUS.is_file():
        return None
    for line in _STATUS.read_text().splitlines():
        if line.startswith(field):
            return int(line.split()[1]) // 1024
    return None


def _reset_peak() -> None:
    # Writing 5 to clear_refs resets the peak resident memory (VmHWM) on Linux.
    clear_refs = Path("/proc/self/clear_refs")
    if clear_refs.exists():
        clear_refs.write_text("5")


def step[T](name: str, func: Callable[[], T]) -> T:
    """Run ``func`` once and print its time and how much memory it needed on top."""
    gc.collect()
    _reset_peak()
    before = _status_mb("VmRSS")
    start = time.perf_counter()
    result = func()
    elapsed = (time.perf_counter() - start) * 1000
    peak = _status_mb("VmHWM")
    memory = f"peak +{peak - before:,} MB" if before is not None and peak is not None else ""
    print(f"{name:38s} {elapsed:9,.0f} ms   {memory}", flush=True)
    return result


def write_dataset(path: Path, rows: int) -> None:
    rng = np.random.default_rng(1)
    data: dict[str, object] = {}
    for i in range(8):
        values = rng.normal(100 * i, 10 + i, rows)
        values[rng.random(rows) < 0.1] = np.nan
        data[f"f{i}"] = values
    for i in range(4):
        data[f"i{i}"] = rng.integers(0, 10 ** (i + 1), rows)
    data["user_id"] = np.arange(rows)
    data["city"] = rng.choice(["Madrid", "Barcelona", "Sevilla", "madrid ", "Valencia"], rows)
    data["status"] = rng.choice(["yes", "no"], rows)
    data["code"] = np.char.add("C-", np.arange(rows).astype(str))
    words = np.array(
        ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit"]
    )
    data["note"] = np.char.add(np.char.add(rng.choice(words, rows), " "), rng.choice(words, rows))
    days = pd.to_timedelta(rng.integers(0, 1500, rows), unit="D")
    data["date"] = (pd.Timestamp("2020-01-01") + days).strftime("%d/%m/%Y")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=3_000_000)
    parser.add_argument("--csv", type=Path, default=Path("data/bench.csv"))
    args = parser.parse_args()
    if not args.csv.is_file():
        print(f"Writing {args.rows:,} rows to {args.csv}…", flush=True)
        write_dataset(args.csv, args.rows)

    data = args.csv.read_bytes()
    sample_rows, workers = ProfileConfig.sample_rows, ProfileConfig.workers
    print(f"{args.csv}: {len(data) / 2**20:,.0f} MB, {workers} workers\n")

    options = step("detect_options", lambda: detect_options(data))
    step("compute_dataset_id", lambda: compute_dataset_id(data, options))
    raw = step("CsvLoader.load", lambda: CsvLoader().load(Source(data, "bench", "1"), options))
    schema = step("infer_schema", lambda: infer_schema(raw.df))
    typed = step("apply_schema", lambda: apply_schema(raw.df, schema))
    df, schema = typed.df, typed.schema
    print(f"{'(dataset in memory)':38s} {df.memory_usage(deep=True).sum() / 2**20:9,.0f} MB\n")

    base = step("base_profile", lambda: profile.base_profile(df, workers))
    numeric_names = schema.names_of(cols.NUMERIC_TYPES)
    numeric = step(
        "numeric_stats", lambda: cols.numeric_stats(df, numeric_names, sample_rows, workers)
    )
    step(
        "categorical_stats",
        lambda: cols.categorical_stats(df, schema.names_of(cols.CATEGORICAL_TYPES)),
    )
    step(
        "datetime_stats",
        lambda: cols.datetime_stats(df, schema.names_of(cols.DATE_TYPES), typed.unparsed),
    )
    step(
        "text_stats",
        lambda: cols.text_stats(df, schema.names_of(cols.TEXT_TYPES), schema, base, sample_rows),
    )
    step(
        "column_overview",
        lambda: cols.column_overview(df, schema, base, numeric, sample_rows, workers),
    )
    for method in ("pearson", "spearman"):
        step(
            f"correlation {method}",
            lambda m=method: profile.correlation_matrix(df, m, 30, sample_rows, numeric_names),
        )
    step("rows_with_missing", lambda: profile.rows_with_missing(df, 100))


if __name__ == "__main__":
    main()
