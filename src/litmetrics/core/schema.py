"""Semantic type of each column: what the data means, beyond its pandas dtype.

pandas only knows how values are stored (int64, object...). The rest of the app needs to know
what they mean: a numeric column can be plotted on an axis, an identifier should not be, a
categorical column can be used to group, etc. Here that is guessed with simple heuristics.

Once guessed, ``apply_schema`` converts the columns to the dtype of their type, once, when the
file is loaded (R-L2 in docs/performance.md): dates and numbers stored as text become
``datetime64`` and ``float64``/``int64``, and text categories and booleans become ``category``.
Nothing after loading has to parse them again.

Cost on large files: the checks that parse values (numbers, dates, spaces) look at a random
sample of each column (``INFER_SAMPLE``), not at its first rows, which in a sorted file are not
like the rest. Conversions parse each distinct value once (``_by_unique``): dates repeat, and
``pd.to_datetime`` on 3 M text dates took 3.9 s against 0.1 s for the 1,500 distinct ones.
"""

import dataclasses
import re
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from pandas.tseries.api import guess_datetime_format

from litmetrics.core.io.base import LoadedDataset
from litmetrics.core.sampling import sample

# A text column is categorical if it has at most this many distinct values...
CATEGORICAL_MAX_UNIQUE = 50
# ...and they are at most this fraction of its non-missing values (so values repeat).
CATEGORICAL_MAX_RATIO = 0.5
# A text column with all values distinct is an identifier only from this many rows on.
# In a tiny file, all values being distinct says nothing.
IDENTIFIER_MIN_ROWS = 20
# Share of distinct values (over the non-missing ones) from which a column counts as a key.
# Below 1 because dirty files repeat some ids, usually through duplicated rows.
IDENTIFIER_MIN_DISTINCT = 0.95
# Random values checked to decide whether a text column holds dates or numbers.
INFER_SAMPLE = 200
# Random values checked to decide whether a text column has spaces (identifiers have almost
# none). Checking every value took 227 ms on 3 M codes; the share is stable long before that.
SPACES_SAMPLE = 2_000
# Fraction of those values that must be numbers for a text column to be numeric. Below 1
# because dirty files write some numbers as words or with units ("cincuenta", "41 años").
NUMERIC_MIN_PARSED = 0.95
# Fraction of those values that must parse as dates.
DATETIME_MIN_PARSED = 0.95
# When both day/month orders read the values ("01/02/2024"), read the day first (European
# files). Dates that start with the year ("2024-01-02") always use year-month-day.
DATE_DAYFIRST = True

# Lowercase values accepted as booleans in a text column ("si"/"sí" for Spanish files).
_BOOLEAN_TOKENS = frozenset({"true", "false", "yes", "no", "y", "n", "t", "f", "si", "sí"})
# Column names that look like an id: "id", "user_id", "ID", "userId"; not "paid" or "video".
# "Unnamed: 0" is the index that pandas writes with ``to_csv`` and reads back as a column.
_ID_NAME = re.compile(r"(?:^|[^a-zA-Z])[iI][dD](?:$|[^a-zA-Z])|[a-z]Id$|^Unnamed: \d+$")


class SemanticType(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    # Free text, or anything that does not fit the other types.
    TEXT = "text"
    # One distinct value per row (keys, codes). Not useful for statistics or charts.
    IDENTIFIER = "identifier"


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    pandas_dtype: str
    semantic_type: SemanticType


@dataclass(frozen=True)
class Schema:
    """Semantic type of every column of a dataset, in the same order as the DataFrame."""

    columns: tuple[ColumnSchema, ...]

    @cached_property
    def _types(self) -> dict[str, SemanticType]:
        # cached_property writes the instance __dict__ directly, so it works on a frozen
        # dataclass. It is not a field: equality, hashing and cache keys ignore it.
        return {c.name: c.semantic_type for c in self.columns}

    def type_of(self, name: str) -> SemanticType:
        """Semantic type of column ``name``. Raise ``KeyError`` if there is no such column."""
        return self._types[name]

    def names_of(self, kinds: Iterable[SemanticType]) -> tuple[str, ...]:
        """Names of the columns whose semantic type is one of ``kinds``, in dataset order.

        Always the same tuple for the same schema and kinds, so it can be part of a cache key.
        """
        wanted = set(kinds)
        return tuple(c.name for c in self.columns if c.semantic_type in wanted)


def infer_schema(df: pd.DataFrame) -> Schema:
    """Guess the semantic type of every column of ``df``."""
    return Schema(
        tuple(
            ColumnSchema(name, str(series.dtype), infer_semantic_type(series))
            for name, series in df.items()
        )
    )


def infer_semantic_type(series: pd.Series) -> SemanticType:
    """Guess the semantic type of one column.

    The checks go from the most to the least specific, and the first match wins:
    first the dtype pandas already gave the column, then the content of text columns.
    """
    # 1. The dtype is already enough.
    if ptypes.is_bool_dtype(series):
        return SemanticType.BOOLEAN
    if ptypes.is_datetime64_any_dtype(series):
        return SemanticType.DATETIME
    if ptypes.is_numeric_dtype(series):
        # Integers are numbers unless the name says id and the values (almost) never repeat.
        if (
            ptypes.is_integer_dtype(series)
            and _ID_NAME.search(str(series.name))
            and mostly_distinct(series.nunique(), series.notna().sum())
        ):
            return SemanticType.IDENTIFIER
        return SemanticType.NUMERIC
    if isinstance(series.dtype, pd.CategoricalDtype):
        return SemanticType.CATEGORICAL

    # 2. Text column (object/string dtype): look at the values.
    values = series.dropna().astype(str)
    if values.empty:
        return SemanticType.TEXT
    unique = values.nunique()
    # Lowercase the (at most two) distinct values, not every row.
    if unique <= 2 and {v.lower() for v in values.unique()} <= _BOOLEAN_TOKENS:
        return SemanticType.BOOLEAN
    if _looks_numeric(values):
        return SemanticType.NUMERIC
    if _looks_like_datetime(values):
        return SemanticType.DATETIME
    # (Almost) all distinct and (almost) no spaces: codes like "A-001" or emails, not sentences.
    # Dirty files have a few broken values ("con espacio@gmail.com"): same margin as repeats.
    if (
        len(values) >= IDENTIFIER_MIN_ROWS
        and mostly_distinct(unique, len(values))
        and sample(values, SPACES_SAMPLE).str.contains(r"\s").mean() <= 1 - IDENTIFIER_MIN_DISTINCT
    ):
        return SemanticType.IDENTIFIER
    if unique <= CATEGORICAL_MAX_UNIQUE and unique <= CATEGORICAL_MAX_RATIO * len(values):
        return SemanticType.CATEGORICAL
    return SemanticType.TEXT


def drop_columns(
    df: pd.DataFrame, schema: Schema, names: Iterable[str]
) -> tuple[pd.DataFrame, Schema]:
    """``df`` and ``schema`` without the columns ``names``. ``df`` is not changed."""
    skip = set(names)
    if not skip:
        return df, schema
    kept = tuple(c for c in schema.columns if c.name not in skip)
    return df[[c.name for c in kept]], Schema(kept)


def mostly_distinct(unique: int, n_values: int) -> bool:
    """Whether ``unique`` distinct values among ``n_values`` non-missing ones look like a key."""
    return n_values > 0 and unique >= IDENTIFIER_MIN_DISTINCT * n_values


def _looks_numeric(values: pd.Series) -> bool:
    """Say whether a text column holds numbers, apart from a few dirty values.

    The column was read as text because some values are not numbers ("cincuenta", "41 años",
    "26.809,81 €"). A column with leading zeros ("01234") holds codes such as postal codes,
    which lose their zeros as numbers: it stays text.
    """
    picked = sample(values, INFER_SAMPLE)
    if pd.to_numeric(picked, errors="coerce").notna().mean() < NUMERIC_MIN_PARSED:
        return False
    # Every value, not the sample: a postal code column may have only a couple of them.
    return not values.str.match(r"\s*0\d").any()


def _looks_like_datetime(values: pd.Series) -> bool:
    """Say whether almost all of a random sample of a text column parses as dates."""
    picked = sample(values, INFER_SAMPLE)
    # Every date has three groups of digits ("2024-01-02") or a 4-digit year next to other text
    # ("Jan 2024", "2024-01"). Values without that shape count as not dates even if pandas
    # would parse them: words ("today"), bare numbers ("2021"), ranges ("10-20"). The check
    # also skips the slow parse for most text columns.
    groups = picked.str.count(r"\d+")
    year = picked.str.contains(r"\d{4}") & ~picked.str.fullmatch(r"\d+")
    shaped = (groups >= 3) | year
    if shaped.mean() < DATETIME_MIN_PARSED:
        return False
    # format="mixed" parses each value on its own and warns about it; the warning is expected.
    # utc=True: without it, values with different UTC offsets raise instead of parsing.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(picked[shaped], errors="coerce", format="mixed", utc=True)
    return parsed.notna().sum() / len(picked) >= DATETIME_MIN_PARSED


# Converting the columns


@dataclass(frozen=True, eq=False)
class TypedFrame:
    """Result of ``apply_schema``.

    Attributes:
        df: The data, with every column in the dtype of its semantic type.
        schema: The schema, with ``pandas_dtype`` updated to the new dtypes.
        unparsed: For each date or numeric column that was text, values that were not dates
            or numbers and became missing.
    """

    df: pd.DataFrame
    schema: Schema
    unparsed: Mapping[str, int]


def apply_schema(df: pd.DataFrame, schema: Schema) -> TypedFrame:
    """Convert each column of ``df`` to the dtype of its semantic type. ``df`` is not changed.

    - Dates: ``datetime64`` without timezone. Text is parsed with a fixed format when one fits
      a random sample of the values (faster and stricter than guessing each value); values
      that are not dates become missing and are counted. Dates with a timezone are converted to UTC.
    - Numbers stored as text: ``float64`` (``int64`` if all are integers and none is missing).
      Values that are not numbers become missing and are counted.
    - Categorical text and boolean text ("yes"/"no"): ``category``, which makes counting and
      grouping faster (14 ms against 87 ms per ``value_counts`` on 3 M rows) and uses less
      memory. Boolean text keeps its own labels instead of becoming True/False.
    - Everything else is kept as it is.
    """
    # Shallow copy: with copy-on-write, assigning a column here never changes ``df``.
    typed = df.copy(deep=False)
    unparsed: dict[str, int] = {}
    for column in schema.columns:
        name, kind = column.name, column.semantic_type
        series = df[name]
        if kind is SemanticType.DATETIME:
            dates = _to_dates(series)
            if not ptypes.is_datetime64_any_dtype(series):
                unparsed[name] = int(series.notna().sum() - dates.notna().sum())
            typed[name] = dates
        elif kind is SemanticType.NUMERIC and not ptypes.is_numeric_dtype(series):
            numbers = _by_unique(series, lambda u: pd.to_numeric(u, errors="coerce"))
            unparsed[name] = int(series.notna().sum() - numbers.notna().sum())
            typed[name] = numbers
        elif _is_text_category(series, kind):
            typed[name] = series.astype("category")
    columns = tuple(
        dataclasses.replace(c, pandas_dtype=str(typed[c.name].dtype)) for c in schema.columns
    )
    return TypedFrame(typed, Schema(columns), unparsed)


def _is_text_category(series: pd.Series, kind: SemanticType) -> bool:
    """Whether ``series`` is text that ``apply_schema`` stores as ``category``."""
    if isinstance(series.dtype, pd.CategoricalDtype) or ptypes.is_bool_dtype(series):
        return False
    return kind in (SemanticType.CATEGORICAL, SemanticType.BOOLEAN)


def _by_unique(series: pd.Series, convert: Callable[[pd.Series], pd.Series]) -> pd.Series:
    """``convert`` applied to each distinct value of ``series`` once, then spread to every row.

    Same result as ``convert(series)``: missing values stay missing, and integers stay
    ``int64`` only when nothing is missing. On columns where every value is different it
    costs one ``factorize`` (75 ms on 3 M rows) more than converting directly.
    """
    codes, uniques = pd.factorize(series)
    converted = convert(pd.Series(uniques)).to_numpy()
    missing = codes < 0
    if missing.any():
        if converted.dtype.kind in "iub":
            converted = converted.astype("float64")
        # Missing rows point at one more slot, holding the missing value of the dtype.
        na = np.array(["NaT"] if converted.dtype.kind == "M" else [np.nan], dtype=converted.dtype)
        converted = np.append(converted, na)
        codes = np.where(missing, converted.size - 1, codes)
    return pd.Series(converted[codes], index=series.index, name=series.name)


def _to_dates(series: pd.Series) -> pd.Series:
    """``series`` as timezone-naive ``datetime64``. Values that are not dates become missing."""
    if ptypes.is_datetime64_any_dtype(series):
        dates = series
    else:
        fmt, dayfirst = _date_format(series)

        # utc=True: a column mixing UTC offsets cannot be parsed otherwise. Values without an
        # offset are taken as UTC, so dropping the timezone below gives them back unchanged.
        def parse(values: pd.Series) -> pd.Series:
            parsed = pd.to_datetime(
                values, errors="coerce", format=fmt, dayfirst=dayfirst, utc=True
            )
            return parsed.dt.tz_convert(None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dates = _by_unique(series, parse)
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)
    return dates


def _date_format(series: pd.Series) -> tuple[str, bool]:
    """Format that reads a random sample of the values, and whether the day goes first.

    One value is guessed in both day/month orders, and the order that reads more of the
    sample wins (``DATE_DAYFIRST`` on a tie). That fixed format reads the whole column
    in one order: "mixed" alone would read "01/02/2024" as January and "13/02/2024" as
    February. If no format reads enough values, it returns ``"mixed"`` with the day/month
    order of the best guess.
    """
    picked = sample(series.dropna().astype(str), INFER_SAMPLE)
    if picked.empty:
        return "mixed", False
    # guess_datetime_format warns about ambiguous day/month orders; the scores below decide.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        month_first = guess_datetime_format(picked.iloc[0])
        day_first = guess_datetime_format(picked.iloc[0], dayfirst=True)
        # A date that starts with the year is year-month-day, never year-day-month.
        prefer_day = DATE_DAYFIRST and not (month_first or "").startswith("%Y")
        ordered = [day_first, month_first] if prefer_day else [month_first, day_first]
        candidates = list(dict.fromkeys(f for f in ordered if f is not None))
        if not candidates:
            return "mixed", prefer_day
        scores = [
            pd.to_datetime(picked, errors="coerce", format=f, utc=True).notna().mean()
            for f in candidates
        ]
    # max keeps the first of equal scores, so the preferred order wins ties.
    best = max(range(len(candidates)), key=lambda i: scores[i])
    fmt = candidates[best]
    dayfirst = fmt.find("%d") < fmt.find("%m") if "%d" in fmt and "%m" in fmt else prefer_day
    if scores[best] >= DATETIME_MIN_PARSED:
        return fmt, dayfirst
    return "mixed", dayfirst


def prepare_dataset(dataset: LoadedDataset) -> tuple[LoadedDataset, Schema]:
    """Infer the schema of a freshly read dataset and convert its columns to match it.

    Returns:
        The dataset with converted columns (and its ``unparsed``), and its schema.
    """
    result = apply_schema(dataset.df, infer_schema(dataset.df))
    typed = dataclasses.replace(dataset, df=result.df, unparsed=result.unparsed)
    return typed, result.schema
