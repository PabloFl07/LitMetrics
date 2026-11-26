"""A chart configuration as JSON: ``spec_to_dict`` / ``spec_from_dict``, the name of the file
it is saved in, and how one saved from another dataset is adapted to the current one
(``fit_to_dataset``).
"""

import datetime as dt
import re
from collections.abc import Collection, Mapping
from dataclasses import asdict, fields, replace
from enum import StrEnum
from typing import Any

from litmetrics.config import VisualizeConfig
from litmetrics.core.visualize.hints import ColumnHints
from litmetrics.core.visualize.spec import (
    FILTER_TYPES,
    ROLES,
    Agg,
    ChartKind,
    DataSpec,
    DateFilter,
    RangeFilter,
    RowFilter,
    TimeUnit,
    ValuesFilter,
)

# Configuration as JSON


# Characters Windows, macOS or Linux refuse in a file name, and runs of them, replaced by "_".
_UNSAFE_FILE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def config_file_name(name: str, kind: ChartKind) -> str:
    """File name of a saved configuration: ``name`` as typed, made safe, with ``.json``.

    An empty name gives ``chart_<kind>.json``; a name that already ends in ``.json`` does not
    get it twice; characters some systems refuse (``/``, ``:``...) become ``_``.
    """
    stem = name.strip()
    if stem.lower().endswith(".json"):
        stem = stem[: -len(".json")]
    stem = _UNSAFE_FILE_CHARS.sub("_", stem).strip(" .")
    return f"{stem or f'chart_{kind}'}.json"


def spec_to_dict(spec: DataSpec) -> dict[str, Any]:
    """``spec`` as plain JSON values; each filter carries its ``type`` (``FILTER_TYPES``)."""
    out: dict[str, Any] = {
        f.name: _plain(getattr(spec, f.name)) for f in fields(spec) if f.name != "filters"
    }
    names = {cls: name for name, cls in FILTER_TYPES.items()}
    out["filters"] = [
        {"type": names[type(f)], **{k: _plain(v) for k, v in asdict(f).items()}}
        for f in spec.filters
    ]
    return out


def _plain(value: Any) -> Any:
    """``value`` as a JSON value: enums as their text, tuples as lists."""
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def spec_from_dict(raw: Mapping[str, Any]) -> DataSpec:
    """The ``DataSpec`` that ``spec_to_dict`` wrote. Unknown keys are ignored.

    Raises:
        ValueError: A value has the wrong type or is not one of the allowed ones (unknown
            chart kind, aggregation, filter type, malformed date...).
    """
    if not isinstance(raw, Mapping):
        raise ValueError("The chart data must be an object.")
    try:
        kind = ChartKind(raw["kind"])
    except KeyError:
        raise ValueError("The configuration has no chart kind.") from None
    columns = {name: _optional(raw.get(name), str, name) for name in ("x", "y", "value", "group")}
    filters = raw.get("filters", [])
    if not isinstance(filters, list):
        raise ValueError("'filters' must be a list.")
    return DataSpec(
        kind,
        **columns,
        agg=Agg(raw.get("agg", Agg.COUNT)),
        bins=_bounded(raw.get("bins", VisualizeConfig.histogram_bins), "bins"),
        top_k=_bounded(raw.get("top_k", 20), "top_k"),
        time_unit=TimeUnit(raw.get("time_unit", TimeUnit.AUTO)),
        trim=_typed(raw.get("trim", True), bool, "trim"),
        min_rows=_bounded(raw.get("min_rows", VisualizeConfig.min_rows_per_point), "min_rows"),
        filters=tuple(_filter_from_dict(f) for f in filters),
    )


def _filter_from_dict(raw: Any) -> RowFilter:
    """One filter of ``spec_from_dict``.

    Raises:
        ValueError: Unknown type, missing column or values of the wrong type.
    """
    if not isinstance(raw, Mapping) or raw.get("type") not in FILTER_TYPES:
        raise ValueError(f"Unknown filter: {raw!r}.")
    column = _typed(raw.get("column"), str, "column")
    cls = FILTER_TYPES[raw["type"]]
    if cls is DateFilter:
        start, end = (_optional(raw.get(k), str, k) for k in ("start", "end"))
        for day in (start, end):
            if day is not None:
                dt.date.fromisoformat(day)
        return DateFilter(column, start, end)
    if cls is ValuesFilter:
        values = raw.get("values", [])
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError("The values of a filter must be a list of text.")
        return ValuesFilter(column, tuple(values))
    low, high = (_optional(raw.get(k), (int, float), k) for k in ("low", "high"))
    return RangeFilter(
        column, None if low is None else float(low), None if high is None else float(high)
    )


def _typed(value: Any, kind: type | tuple[type, ...], name: str) -> Any:
    """``value`` if it is a ``kind`` (``bool`` is never taken for a number).

    Raises:
        ValueError: It is not.
    """
    if not isinstance(value, kind) or (isinstance(value, bool) and kind is not bool):
        raise ValueError(f"{name!r} has a wrong value: {value!r}.")
    return value


def _optional(value: Any, kind: type | tuple[type, ...], name: str) -> Any:
    """``value`` if it is None or a ``kind``.

    Raises:
        ValueError: It is neither.
    """
    return None if value is None else _typed(value, kind, name)


def _bounded(value: Any, name: str) -> int:
    """``value`` as a count from 1 to 1,000 (bins, categories shown).

    Raises:
        ValueError: It is not an integer in that range.
    """
    count = _typed(value, int, name)
    if not 1 <= count <= 1_000:
        raise ValueError(f"{name!r} is out of range: {count}.")
    return count


def fit_to_dataset(
    spec: DataSpec,
    columns: Collection[str],
    hints: ColumnHints,
    skipped: Collection[str] = (),
) -> tuple[DataSpec, tuple[str, ...]]:
    """``spec`` (saved from any dataset) with only what the current one can draw.

    A column that does not exist, or cannot take its role here, is removed (a required role is
    then filled by the editor with another column), and so is a filter on such a column.
    Columns of roles the kind does not have are dropped silently.

    Args:
        spec: Loaded configuration.
        columns: Every column of the current dataset.
        hints: Its usable columns (``column_hints`` of the dataset without ``skipped``).
        skipped: Columns the person chose to skip ("Skip columns", shared with Profile):
            they exist but are not offered, and the message says so.

    Returns:
        The adapted spec, and one message per column or filter that was removed.
    """
    messages: list[str] = []
    changes: dict[str, str | None] = {}
    roles = {role.field: role for role in ROLES[spec.kind]}
    for name in ("x", "y", "value", "group"):
        column = getattr(spec, name)
        if column is None:
            continue
        role = roles.get(name)
        if role is None:
            changes[name] = None
        elif column not in columns:
            messages.append(f"{role.label}: there is no column {column!r} in this dataset.")
            changes[name] = None
        elif column in skipped:
            messages.append(f"{role.label}: {column!r} is skipped (Skip columns).")
            changes[name] = None
        elif column not in hints.columns_for(role.accepts):
            messages.append(f"{role.label}: {column!r} cannot be used here.")
            changes[name] = None
    type_names = {cls: name for name, cls in FILTER_TYPES.items()}
    filters = []
    for f in spec.filters:
        if f.column not in columns:
            messages.append(f"Filter: there is no column {f.column!r} in this dataset.")
        elif f.column in skipped:
            messages.append(f"Filter: {f.column!r} is skipped (Skip columns).")
        elif hints.filter_type(f.column) != type_names[type(f)]:
            messages.append(f"Filter: {f.column!r} cannot be filtered this way.")
        else:
            filters.append(f)
    return replace(spec, **changes, filters=tuple(filters)), tuple(messages)
