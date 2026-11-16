"""Integration tests of the data path without the UI: file bytes -> profile.

Each test runs the same steps the app runs for an upload, with the real modules and no
mocks: ``formats.detect_options`` -> ``formats.load`` -> ``schema.prepare_dataset`` ->
``profile`` / ``columns``. The unit tests check each step alone; these check that the steps
agree with each other, which is where format-specific surprises show up.
"""

import numpy as np
import pandas as pd
import pytest
from fixtures import files, samples

from litmetrics.core import columns as cols
from litmetrics.core.io import formats
from litmetrics.core.io.base import LoadedDataset, Source
from litmetrics.core.profile import (
    IssueKind,
    base_profile,
    correlation_matrix,
    detect_issues,
    top_correlated_pairs,
)
from litmetrics.core.schema import Schema, SemanticType, prepare_dataset

T = SemanticType


def _read(name: str, data: bytes) -> tuple[LoadedDataset, Schema]:
    """What the Load page does with an upload: detect, load, type. Raises ``ReadError``."""
    source = Source(data, name, name)
    return prepare_dataset(formats.load(source, formats.detect_options(source)))


def _survey_rows() -> list[list]:
    """``survey_frame`` as the rows of a sheet: header first, dates as dates, NaN as empty."""
    df = samples.survey_frame()
    rows = [
        [r.respondent, r.region, None if np.isnan(r.rating) else r.rating, r.answered.date()]
        for r in df.itertuples()
    ]
    return [list(df.columns), *rows]


# The same survey written by the library of each format.
_SURVEY_FILES = {
    "survey.csv": lambda: files.csv(samples.survey_frame()),
    "survey.parquet": lambda: files.parquet(samples.survey_frame()),
    "survey.feather": lambda: files.feather_file(samples.survey_frame()),
    "survey.db": lambda: files.sqlite_tables({"survey": samples.survey_frame()}),
    "survey.xlsx": lambda: files.xlsx({"survey": _survey_rows()}),
}


@pytest.mark.parametrize("name", list(_SURVEY_FILES))
def test_same_data_gives_the_same_profile_in_every_format(name):
    """The survey profiles the same whatever format it was saved in.

    Case: a format stores something differently (dates as text in CSV and SQLite, ints as
    floats when a column has gaps, booleans as 0/1) and the schema or the counts drift, so the
    same data gets another profile depending on the file it came in.
    Affects: core/io/formats.py, every loader, core/io/sniffing.py, core/io/arrow_table.py,
    core/schema.py, core/profile.py (base_profile), core/columns.py (numeric/datetime_stats)
    How: reads ``survey_frame`` written in each format and compares types, missing and distinct
    counts, the statistics of ``rating`` and the range of ``answered`` with the original frame.
    """
    dataset, schema = _read(name, _SURVEY_FILES[name]())
    reference = samples.survey_frame()
    assert [(c.name, c.semantic_type) for c in schema.columns] == [
        ("respondent", T.NUMERIC),
        ("region", T.CATEGORICAL),
        ("rating", T.NUMERIC),
        ("answered", T.DATETIME),
    ]
    assert dataset.unparsed.get("answered", 0) == 0
    counts = ["missing", "unique", "repeated"]
    pd.testing.assert_frame_equal(
        base_profile(dataset.df).by_column[counts], base_profile(reference).by_column[counts]
    )
    rating = cols.numeric_stats(dataset.df, ["rating"], 1000).loc[0]
    expected = cols.numeric_stats(reference, ["rating"], 1000).loc[0]
    for stat in ["count", "mean", "std", "median", "min", "max"]:
        assert rating[stat] == pytest.approx(expected[stat]), stat
    dates = cols.datetime_stats(dataset.df, ["answered"], {}).loc[0]
    assert (dates["min"], dates["max"]) == (pd.Timestamp("2024-03-01"), pd.Timestamp("2024-04-09"))


def test_european_csv_is_profiled_as_numbers():
    """A ``;`` file with ``,`` decimals ends up as a numeric column, not as identifiers.

    Case: without the detected decimal mark, "0,5" stays text and, all distinct, the schema
    calls the price column an identifier: no statistics at all.
    Affects: core/io/sniffing.py (detect_decimal), core/io/csv_loader.py, core/schema.py
    How: reads ``EUROPEAN_CSV`` and checks the type, dtype and a value of ``precio``.
    """
    dataset, schema = _read("eu.csv", samples.EUROPEAN_CSV)
    assert schema.type_of("precio") is T.NUMERIC
    assert dataset.df["precio"].dtype == "float64"
    assert dataset.df["precio"].iloc[1] == pytest.approx(1.5)
    assert dataset.details["decimal mark"] == "Comma (,)"


def test_mixed_excel_column_is_numeric_with_its_text_counted():
    """A numeric Excel column with one stray word stays numeric; the word is counted.

    Case: the Excel loader makes mixed columns text (to be safe) and the schema never gets the
    chance to see the column is 97% numbers.
    Affects: core/io/excel_loader.py (mixed columns), core/schema.py (apply_schema)
    How: reads 29 numbers and "unknown" from a sheet and checks type and unparsed count.
    """
    values = [i + 0.5 for i in range(29)] + ["unknown"]
    data = files.xlsx({"S": [["price"], *([v] for v in values)]})
    dataset, schema = _read("book.xlsx", data)
    assert schema.type_of("price") is T.NUMERIC
    assert dataset.unparsed == {"price": 1}


def test_excel_time_column_keeps_its_values():
    """Times of day read from Excel keep every value after typing.

    Case: the loader turns times into text and the schema then parses that text as dates of
    today, or drops them as unparsed.
    Affects: core/io/excel_loader.py (time columns), core/schema.py (infer_semantic_type)
    How: reads 30 times from a sheet and checks the type is not a date and nothing is lost.
    """
    from datetime import time

    rows = [[time(8 + i % 9, 30)] for i in range(30)]
    dataset, schema = _read("book.xlsx", files.xlsx({"S": [["start"], *rows]}))
    assert schema.type_of("start") is not T.DATETIME
    assert dataset.df["start"].notna().all()
    assert dataset.unparsed == {}


def test_issues_of_a_file_are_found_end_to_end():
    """Every problem written into ``QUALITY_CSV`` is reported by the Issues tab's logic.

    Case: a step changes the data under the next one (e.g. typing turns the constant column into
    a category the issue detector skips), so a problem in the file is not reported.
    Affects: core/io (CSV path), core/schema.py, core/profile.py (base_profile,
    correlation_matrix, detect_issues)
    How: reads ``QUALITY_CSV`` and runs the issue detection as ui/pages/profile.py does.
    """
    dataset, schema = _read("quality.csv", samples.QUALITY_CSV)
    numeric = schema.names_of(cols.NUMERIC_TYPES)
    corr = correlation_matrix(dataset.df[list(numeric)], "pearson", 10, 1000)
    issues = detect_issues(
        base_profile(dataset.df),
        schema,
        top_correlated_pairs(corr.matrix),
        high_missing_pct=10.0,
        high_correlation=0.9,
    )
    kinds = {i.kind for i in issues}
    assert {IssueKind.DUPLICATE_ROWS, IssueKind.CONSTANT, IssueKind.HIGH_CORRELATION} <= kinds
    assert (IssueKind.HIGH_MISSING, ("score",)) in {(i.kind, i.columns) for i in issues}
