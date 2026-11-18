"""Turn an Arrow table (from a Parquet or Feather file) into a DataFrame the app can profile.

Columnar files store typed columns, so nothing is guessed: integers, floats, dates, booleans
and categories (dictionary columns) arrive with their dtype. A few Arrow types have no useful
pandas equivalent for the rest of the app and are converted first, in Arrow:

- dates (``date32``/``date64``) become ``datetime64`` instead of ``datetime.date`` objects;
- times of day become text (``"13:45:00"``), as they would in a CSV;
- decimals become ``float64``: exact decimals would be ``Decimal`` objects, which no
  statistic accepts;
- durations become ``float64`` seconds: the profile has no type for ``timedelta``;
- binary values become a text with their size (``"<1,024 bytes>"``), as in SQLite;
- nested values (lists, structs, maps) become JSON text: as Python lists and dicts they cannot
  be hashed, so they could not even be counted.

Every converted column is listed in ``details`` (shown to the user), grouped by conversion.

The pandas index saved by ``DataFrame.to_parquet`` (or ``to_feather``) is restored by Arrow
and then turned back into columns: it is data of the file, and the rest of the app expects a
``RangeIndex``.
"""

import json
from collections.abc import Callable

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types as pat

from litmetrics.core.io.base import BINARY_AS_SIZE, TIME_AS_TEXT

# Nanoseconds in one unit of each Arrow duration unit.
_DURATION_NS = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}


def to_frame(table: pa.Table) -> tuple[pd.DataFrame, dict[str, str]]:
    """Convert ``table`` to a DataFrame with unique string column names.

    The table is consumed (``self_destruct``): it cannot be used afterwards.

    Returns:
        The DataFrame, and the conversions made, as ``LoadedDataset.details`` entries
        (``{"decimal columns": "price (read as floating-point numbers)"}``). Empty when no
        column needed one.
    """
    converted: dict[str, list[str]] = {}
    for index, arrow_field in enumerate(table.schema):
        conversion = _conversion(arrow_field.type)
        if conversion is None:
            continue
        label, convert = conversion
        new_column = convert(table.column(index).combine_chunks())
        table = table.set_column(index, pa.field(arrow_field.name, new_column.type), new_column)
        converted.setdefault(label, []).append(str(arrow_field.name))

    # R-L5: self_destruct + split_blocks free each Arrow column as soon as it is converted,
    # so the file is not held twice (as Arrow and as pandas) at the end of the conversion.
    df = table.to_pandas(date_as_object=False, self_destruct=True, split_blocks=True)
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index(allow_duplicates=True)
    df.columns = _unique([str(column) for column in df.columns])
    details = {label: ", ".join(names) + _NOTES[label] for label, names in converted.items()}
    return df, details


# Note added after the names of the converted columns of each kind, in ``details``.
_NOTES = {
    "time columns": " (read as text)",
    "decimal columns": " (read as floating-point numbers)",
    "duration columns": " (read as seconds)",
    "binary columns": " (shown as their size)",
    "nested columns": " (shown as JSON text)",
}


# What each conversion does to the values, as ``LoadedDataset.changed`` notes.
_CHANGES = {
    "time columns": TIME_AS_TEXT,
    "decimal columns": "exact decimals kept as floating-point numbers",
    "duration columns": "durations kept as seconds",
    "binary columns": BINARY_AS_SIZE,
    "nested columns": "nested values kept as JSON text",
}


def changes(schema: pa.Schema) -> dict[str, str]:
    """Columns of ``schema`` that ``to_frame`` converts, with what the conversion does to their
    values (``LoadedDataset.changed``). Call it before ``to_frame``, which consumes the table.

    Keys are the names in the file: a duplicated name, renamed by ``to_frame``, keeps only its
    first occurrence.
    """
    found: dict[str, str] = {}
    for arrow_field in schema:
        if (conversion := _conversion(arrow_field.type)) is not None:
            found.setdefault(str(arrow_field.name), _CHANGES[conversion[0]])
    return found


def _conversion(
    arrow_type: pa.DataType,
) -> tuple[str, Callable[[pa.Array], pa.Array]] | None:
    """The ``details`` label and the conversion of a column of ``arrow_type``, or ``None``
    if pandas reads it well as it is."""
    if pat.is_dictionary(arrow_type):
        # Categories of a convertible type (rare) are decoded and converted like any column.
        inner = _conversion(arrow_type.value_type)
        if inner is None:
            return None
        label, convert = inner
        return label, lambda array: convert(array.dictionary_decode())
    if pat.is_time(arrow_type):
        return "time columns", _times_to_text
    if pat.is_decimal(arrow_type):
        return "decimal columns", lambda array: pc.cast(array, pa.float64())
    if pat.is_duration(arrow_type):
        return "duration columns", _durations_to_seconds
    if (
        pat.is_binary(arrow_type)
        or pat.is_large_binary(arrow_type)
        or pat.is_fixed_size_binary(arrow_type)
    ):
        return "binary columns", _binary_to_size
    if pat.is_nested(arrow_type):
        return "nested columns", _nested_to_json
    return None


def _times_to_text(array: pa.Array) -> pa.Array:
    """``"13:45:00"``: Arrow writes every digit of the unit (``"13:45:00.000000"``)."""
    return pc.replace_substring_regex(pc.cast(array, pa.string()), r"\.0+$", "")


def _durations_to_seconds(array: pa.Array) -> pa.Array:
    """Durations as ``float64`` seconds, whatever their unit."""
    per_unit = _DURATION_NS[array.type.unit] / 1_000_000_000
    return pc.multiply(pc.cast(pc.cast(array, pa.int64()), pa.float64()), per_unit)


def _binary_to_size(array: pa.Array) -> pa.Array:
    """``"<1,024 bytes>"`` for each value; missing values stay missing."""
    sizes = pc.binary_length(array).to_pylist()
    return pa.array([None if size is None else f"<{size:,} bytes>" for size in sizes], pa.string())


def _nested_to_json(array: pa.Array) -> pa.Array:
    """Each value as JSON text. Arrow cannot cast nested types to text, so this goes through
    Python objects, one value at a time: slow, but nested columns are rare in tabular data."""
    return pa.array(
        [
            None if value is None else json.dumps(value, default=str, ensure_ascii=False)
            for value in array.to_pylist()
        ],
        pa.string(),
    )


def _unique(names: list[str]) -> list[str]:
    """``names`` with repeats renamed ``"a.1"``, ``"a.2"``... as ``pd.read_csv`` does.

    Arrow allows repeated column names, and the restored index may repeat a column name.
    A new name never takes the name of another column of the file.
    """
    taken = set(names)
    used: set[str] = set()
    result: list[str] = []
    for name in names:
        new, suffix = name, 0
        while new in used:
            suffix += 1
            new = f"{name}.{suffix}"
            if new in taken:
                new = name
        used.add(new)
        result.append(new)
    return result
