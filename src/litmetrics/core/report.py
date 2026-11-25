"""Data quality report of a dataset, as a Markdown file: what the Issues tab of Profile shows,
with why each issue matters and what to do about it.

``issues_report`` renders ``templates/issues_report.md.j2`` from results the Profile page has
already computed (the base pass, the schema and the issues): it reads no row of the data, so it
is cheap (R-P6). The words that describe each kind of issue come from ``profile.ISSUE_INFO``.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path, PurePath
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from litmetrics.core.io.base import LoadedDataset
from litmetrics.core.profile import (
    ISSUE_INFO,
    SEVERITIES,
    BaseProfile,
    CorrelationMethod,
    Issue,
)
from litmetrics.core.schema import IDENTIFIER_MIN_DISTINCT, IDENTIFIER_MIN_ROWS, Schema

# Templates are files next to this module, so they ship with the installed package.
_TEMPLATES = Path(__file__).resolve().parent / "templates"
_TEMPLATE = "issues_report.md.j2"
# MIME type of the report, for the download.
MIME = "text/markdown"
# Characters with a meaning in Markdown, escaped in plain text (names, messages).
_MARKDOWN = re.compile(r"([\\`*_{}\[\]<>#|~])")
# Runs of backticks inside a name: inline code needs a longer fence around them.
_BACKTICKS = re.compile(r"`+")


@dataclass(frozen=True)
class ReportSettings:
    """How the issues were found, written in the "Method" section.

    Attributes:
        high_missing_pct: A column with more missing values than this (%) is an issue.
        high_correlation: A pair with a larger absolute correlation is an issue.
        correlation_method: Method of the correlation the issues come from.
        sample_rows: Rows of the sample the correlation was computed on; None if it used
            every row.
        seed: Seed of that sample.
        correlation_columns: Numeric columns the correlation compared.
        numeric_columns: Numeric non-constant columns of the dataset, compared or not.
    """

    high_missing_pct: float
    high_correlation: float
    correlation_method: CorrelationMethod
    sample_rows: int | None
    seed: int
    correlation_columns: int
    numeric_columns: int


def issues_report(
    dataset: LoadedDataset,
    base: BaseProfile,
    schema: Schema,
    issues: Sequence[Issue],
    settings: ReportSettings,
    *,
    memory: str,
    generated_at: datetime,
    version: str,
    part: str | None = None,
    skipped: Sequence[str] = (),
) -> str:
    """The data quality report of ``dataset`` as Markdown.

    Args:
        dataset: The dataset described, without the ``skipped`` columns.
        base: ``base_profile`` of ``dataset``.
        schema: Schema of ``dataset``.
        issues: ``detect_issues`` of the same data, in any order: the report sorts them by
            severity and groups those of one kind, keeping their order within it.
        settings: Thresholds and sampling the issues were found with.
        memory: Memory used by the dataset, already written for the user ("12.4 MB").
        generated_at: When the report is made.
        version: Version of LitMetrics.
        part: Table or sheet of a file with several, added to the title.
        skipped: Columns the user left out of the profile.
    """
    kinds = list(ISSUE_INFO)
    ranked = sorted(issues, key=lambda i: (SEVERITIES.index(i.info.severity), kinds.index(i.kind)))
    # One group per kind found, in the order of ``ranked``: what the kind means is written
    # once, then every case of it.
    groups: dict[str, dict[str, Any]] = {}
    for issue in ranked:
        info = issue.info
        group = groups.setdefault(
            issue.kind,
            {
                "title": info.title,
                "severity": info.severity,
                "why": info.why,
                "action": info.action,
                "items": [],
            },
        )
        group["items"].append({"columns": issue.columns, "message": issue.message})
    per_column: dict[str, int] = {}
    for issue in ranked:
        for name in issue.columns:
            per_column[name] = per_column.get(name, 0) + 1
    counts = [
        f"{n} {severity}"
        for severity in SEVERITIES
        if (n := sum(issue.info.severity == severity for issue in ranked))
    ]
    names = set(dataset.df.columns)
    unparsed = {c: n for c, n in dataset.unparsed.items() if c in names}
    changed = {c: text for c, text in dataset.changed.items() if c in names}
    # "Issues" and "Values changed on load" are only there when they have something: the
    # sections after them are numbered by what the report has.
    shown = ["dataset", "summary"]
    shown += ["issues"] if ranked else []
    shown += ["changes"] if unparsed or changed else []
    shown += ["columns", "method"]
    sections = {name: i for i, name in enumerate(shown, start=1)}
    title = f"{dataset.name} ({part})" if part else dataset.name
    by_column = base.by_column
    context: dict[str, Any] = {
        "version": version,
        "generated_at": generated_at.strftime("%Y-%m-%d %H:%M"),
        "dataset": {
            "title": title,
            "name": dataset.name,
            "format": dataset.format,
            "details": dict(dataset.details),
        },
        "base": base,
        "n_columns": len(schema.columns),
        "memory": memory,
        "skipped": tuple(skipped),
        "rows_with_missing_pct": (
            100 * base.rows_with_missing / base.n_rows if base.n_rows else 0.0
        ),
        "counts": counts,
        "issues": [
            {
                "severity": issue.info.severity,
                "title": issue.info.title,
                "columns": issue.columns,
                "message": issue.message,
            }
            for issue in ranked
        ],
        "groups": list(groups.values()),
        "unparsed": unparsed,
        "changed": changed,
        "sections": sections,
        "columns": [
            {
                "name": c.name,
                "semantic_type": str(c.semantic_type),
                "missing_pct": float(by_column.loc[c.name, "missing_pct"]),
                "unique": int(by_column.loc[c.name, "unique"]),
                "issue_count": per_column.get(c.name, 0),
            }
            for c in schema.columns
        ],
        "settings": settings,
        "identifier": {
            "min_rows": IDENTIFIER_MIN_ROWS,
            "min_distinct_pct": IDENTIFIER_MIN_DISTINCT * 100,
        },
    }
    return _environment().get_template(_TEMPLATE).render(context)


def report_file_name(source_name: str, part: str | None = None) -> str:
    """File name of the report: ``titanic.csv`` gives ``titanic_issues.md``, and the table
    ``orders`` of ``shop.db`` gives ``shop_orders_issues.md``.
    """
    stem = PurePath(source_name).stem or "dataset"
    if part:
        stem = f"{stem}_{part}"
    return f"{stem}_issues.md"


@cache
def _environment() -> Environment:
    """The Jinja environment of the templates of this package, built once.

    No autoescape: the output is Markdown, escaped by the filters where it holds names.
    ``StrictUndefined`` makes a name missing from the context raise instead of printing nothing.
    """
    env = Environment(
        loader=FileSystemLoader(_TEMPLATES),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )
    env.filters |= {"code": _code, "cell": _cell, "text": _text, "num": _num, "pct": _pct}
    return env


def _cell(value: object) -> str:
    """``value`` as text that stays in one table cell: ``|`` escaped, line breaks as spaces."""
    return " ".join(str(value).splitlines()).replace("|", "\\|")


def _text(value: object) -> str:
    """``value`` as plain text: Markdown characters escaped, line breaks as spaces."""
    return _MARKDOWN.sub(r"\\\1", " ".join(str(value).splitlines()))


def _code(value: object, cell: bool = True) -> str:
    """``value`` as inline code; with ``cell``, safe inside a table cell.

    A name with backticks gets a fence one backtick longer than its longest run, with spaces
    so a backtick at either end is not read as part of the fence. In a cell ``|`` is escaped
    even in code: GitHub splits table cells on it before reading the code span. Elsewhere the
    backslash would be shown.
    """
    name = _cell(value) if cell else " ".join(str(value).splitlines())
    longest = max((len(run) for run in _BACKTICKS.findall(name)), default=0)
    if not longest:
        return f"`{name}`"
    fence = "`" * (longest + 1)
    return f"{fence} {name} {fence}"


def _num(value: int) -> str:
    return f"{int(value):,}"


def _pct(value: float) -> str:
    return f"{float(value):.1f}"
