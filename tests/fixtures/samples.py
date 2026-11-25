"""Named test subjects: the datasets and files the tests read, and why each one looks as it does.

Each subject is built in code, so what it contains is visible here and in review. Bytes are
module constants (cheap, built once); DataFrames are returned by functions, so a test that
changes one never changes it for the next test.
"""

from datetime import date

import numpy as np
import pandas as pd
import pyarrow as pa

from fixtures import files
from litmetrics.core.schema import ColumnSchema, Schema, SemanticType
from litmetrics.ui.charts import ChartColors

# ---------------------------------------------------------------------------------------------
# CSV files


# 10 rows, 2 integer columns. One of two small, different files to fill the datasets in memory.
CSV_A = b"a,b\n" + b"".join(b"%d,%d\n" % (i, i * 2) for i in range(10))
# 20 rows, 3 integer columns. The other small file.
CSV_B = b"x,y,z\n" + b"".join(b"%d,%d,%d\n" % (i, i, i) for i in range(20))

# 60 rows: an id, an int column, a categorical one, and a float column with missing values
# (every 7th row). A plain, valid file used wherever "a CSV" is needed.
PEOPLE_CSV = b"id,age,city,score\n" + b"".join(
    b"%d,%d,%s,%s\n"
    % (i, 20 + i % 30, (b"Lugo", b"Vigo", b"Ourense")[i % 3], b"" if i % 7 == 0 else b"%.1f" % i)
    for i in range(60)
)

# 6 rows with one of each data-quality problem the Profile reports: a missing value, a
# duplicate row (the last two), a constant column and two correlated columns (id and age).
QUALITY_CSV = (
    b"id,age,city,score,const\n1,20,Lugo,1.5,7\n2,21,Vigo,,7\n3,22,Ourense,4.5,7\n"
    b"4,23,Lugo,6.0,7\n5,24,Vigo,7.5,7\n5,24,Vigo,7.5,7\n"
)

# 30 rows written as a Spanish spreadsheet exports them: ``;`` separator and ``,`` decimals.
# Read with the default options, ``precio`` would be text and look like an identifier.
EUROPEAN_CSV = b"precio;ciudad\n" + b"".join(b"%d,5;Lugo\n" % i for i in range(30))


def _orders_rows() -> list[bytes]:
    rows = [b"%d,2024-03-%02d,%d.5" % (i, i % 28 + 1, 10 + i) for i in range(60)]
    rows[4] = b"4,2024-03-05,12 kg"
    rows[29] = b"29,someday,abc"
    return rows


# 60 orders with values the app cannot read, as typed by hand: ``price`` has "12 kg" (row 5)
# and "abc" (row 30), ``date`` has "someday" (row 30). Both columns still pass the inference
# thresholds (95%), so they are typed and those 3 values become empty: 2 rows affected.
ORDERS_WITH_TYPOS_CSV = b"id,date,price\n" + b"\n".join(_orders_rows()) + b"\n"

# A file with only blanks: the smallest file that cannot be read.
EMPTY_CSV = b"   \n"


def _sales_rows() -> list[str]:
    rows = [
        "2024-01-{:02d};{};{},{:02d};{};{};España".format(
            i % 28 + 1,
            ("Coruña", "Ourense", "Lugo", "Vigo")[i % 4],
            100 + 7 * i,
            (13 * i) % 100,
            "" if i % 4 else f"{5 * (i % 3 + 1)},0",
            1 + i % 6,
        )
        for i in range(40)
    ]
    return rows + rows[-1:]


# A sales export as a Spanish Excel saves it: Windows encoding (cp1252), ``;`` separator,
# ``,`` decimals, accents in values and header. 41 rows with one problem of each kind:
# the last row is a duplicate, ``descuento`` misses 3 of 4 values (above the 50% threshold)
# and ``país`` is constant.
SALES_EXPORT_CSV = (
    "fecha;ciudad;importe;descuento;unidades;país\n" + "\n".join(_sales_rows()) + "\n"
).encode("cp1252")


# ---------------------------------------------------------------------------------------------
# Arrow-based files (Parquet, Feather)


def people() -> pd.DataFrame:
    """3 rows of id (int), name (text), score (float) and born (date), one missing in the last
    three: the types every columnar format must keep."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Ana", "Luis", None],
            "score": [1.5, None, 3.0],
            "born": pd.to_datetime(["2000-01-02", "1999-03-04", None]),
        }
    )


PEOPLE_PARQUET = files.parquet(people())
PEOPLE_FEATHER = files.feather_file(people())


def people_with_photo_parquet() -> bytes:
    """``PEOPLE_CSV`` as a Parquet file plus a binary ``photo`` column (60 rows, 5 columns).

    The binary column is one the app cannot show as is: it is replaced by its size.
    """
    import io

    table = pa.Table.from_pandas(pd.read_csv(io.BytesIO(PEOPLE_CSV)), preserve_index=False)
    table = table.append_column("photo", pa.array([b"\x00" * 10] * table.num_rows))
    return files.parquet(table)


def people_feather_60() -> bytes:
    """``PEOPLE_CSV`` as a Feather file (60 rows, 4 columns)."""
    import io

    return files.feather_file(pd.read_csv(io.BytesIO(PEOPLE_CSV)))


# ---------------------------------------------------------------------------------------------
# SQLite databases

# Every kind of object a database may hold: a table (people, 2 rows, 4 columns), a table with a
# name that needs quoting, a view, and a table without rows.
SQLITE_SHOWCASE = files.sqlite(
    "CREATE TABLE people (id INTEGER, name TEXT, score REAL, born TEXT)",
    "INSERT INTO people VALUES (1, 'Ana', 1.5, '2000-01-02'), (2, 'Luis', NULL, '1999-03-04')",
    'CREATE TABLE "odd ""name""" (x INTEGER)',
    'INSERT INTO "odd ""name""" VALUES (7)',
    "CREATE VIEW adults AS SELECT name FROM people",
    "CREATE TABLE empty (x INTEGER)",
)

# Two tables, created ``b`` first: listed and read in name order, not creation order.
SQLITE_AB = files.sqlite(
    "CREATE TABLE b (x INTEGER)",
    "INSERT INTO b VALUES (2)",
    "CREATE TABLE a (y INTEGER)",
    "INSERT INTO a VALUES (1)",
)

# A table whose untyped column ``code`` holds integers, text and NULL: SQLite allows it, and a
# typed format (Parquet, Feather) cannot write it as it comes from pandas (``object``).
SQLITE_MIXED = files.sqlite(
    "CREATE TABLE t (id INTEGER, code)",
    "INSERT INTO t VALUES (1, 10), (2, 'A-7'), (3, NULL), (4, 2.5)",
)

# A database with two tables of different sizes, to see which one is open:
# ``cities`` (3 rows, 1 column) and ``people`` (60 rows, 4 columns).
TOWN_DB = files.sqlite_tables(
    {
        "cities": pd.DataFrame({"name": ["Lugo", "Vigo", "Ourense"]}),
        "people": pd.DataFrame(
            {
                "id": range(60),
                "age": [20 + i % 30 for i in range(60)],
                "city": [("Lugo", "Vigo")[i % 2] for i in range(60)],
                "score": [i * 1.5 for i in range(60)],
            }
        ),
    }
)


def versioned_db(rows: int) -> bytes:
    """A database with tables ``a`` and ``b`` of ``rows`` rows each.

    Two calls with different ``rows`` are two versions of the same file: same name and tables,
    different content.
    """
    frame = pd.DataFrame({"x": range(rows)})
    return files.sqlite_tables({"a": frame, "b": frame})


# ---------------------------------------------------------------------------------------------
# Excel workbooks

# Every kind of sheet: a typed table (Sales: int, text, date, bool and float with a missing
# value), a hidden sheet (Archive), an empty one (Blank) and a small one (Notes).
WORKBOOK = files.xlsx(
    {
        "Sales": [
            ["id", "city", "when", "paid", "amount"],
            [1, "Lugo", date(2024, 1, 2), True, 10.5],
            [2, "Vigo", date(2024, 3, 4), False, None],
            [3, "Ourense", date(2024, 5, 6), True, 7],
        ],
        "Archive": [["code"], ["a"]],
        "Blank": [],
        "Notes": [["x", "y"], [1, 2], [3, 4]],
    },
    hidden=["Archive"],
)

# Two readable sheets: Sales (2 rows, 2 columns) and Notes (1 row, 1 column).
TWO_SHEETS = files.xlsx({"Sales": [["a", "b"], [1, 2], [3, 4]], "Notes": [["n"], ["x"]]})

# A readable sheet and an empty one, whose read fails.
SHEET_AND_BLANK = files.xlsx({"Sales": [["a"], [1]], "Blank": []})


# ---------------------------------------------------------------------------------------------
# DataFrames for core/profile.py and core/columns.py


def quality_frame() -> pd.DataFrame:
    """5 rows: ``x`` with one missing value and perfectly correlated with ``y``, a constant
    column and a text column with a missing value and a repeated one."""
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, np.nan],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0],
            "const": [7, 7, 7, 7, 7],
            "city": ["a", "b", None, "a", "a"],
        }
    )


def typed_frame() -> pd.DataFrame:
    """40 rows, one column of each semantic type, each with something the Columns tab reports.

    - ``value``: 1..38, one far outlier (1000) and one missing value.
    - ``rating``: 4 distinct values, 10 times each (one bar per value).
    - ``city``: categorical where ``Madrid`` and ``madrid `` are the same value written twice.
    - ``flag``: booleans. ``when``: 40 consecutive days.
    - ``note``: text with blanks (``"  "``) and missing values. ``code``: distinct ids.
    """
    n = 40
    return pd.DataFrame(
        {
            "value": [*range(1, 39), 1000.0, np.nan],
            "rating": [1, 2, 3, 4] * 10,
            "city": pd.Categorical(["Madrid", "madrid ", "Lugo", "Lugo"] * 10),
            "flag": [True, False] * 20,
            "when": pd.date_range("2024-01-01", periods=n, freq="D"),
            "note": ["hello world", "  ", "hi", None] * 10,
            "code": [f"A-{i:03d}" for i in range(n)],
        }
    )


def schema_of(**types: SemanticType) -> Schema:
    """A schema with the given semantic type per column (the pandas dtype is left empty)."""
    return Schema(tuple(ColumnSchema(name, "", kind) for name, kind in types.items()))


def typed_schema() -> Schema:
    """The semantic types of ``typed_frame``, as the schema inference would give them."""
    t = SemanticType
    return schema_of(
        value=t.NUMERIC,
        rating=t.NUMERIC,
        city=t.CATEGORICAL,
        flag=t.BOOLEAN,
        when=t.DATETIME,
        note=t.TEXT,
        code=t.IDENTIFIER,
    )


def frame_with_missing(seed: int = 0) -> pd.DataFrame:
    """3,000 random rows, with missing values in ``x`` (10%), ``y`` (30%) and ``big`` (20%).

    ``y`` correlates with ``x``; ``big`` is ``x`` scaled around 1.7e12: summing such values
    loses precision unless they are centred first, a known trap of fast correlation code.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=3_000)
    df = pd.DataFrame(
        {
            "x": x,
            "y": 2 * x + rng.normal(size=3_000),
            "z": rng.normal(size=3_000),
            "big": 1.7e12 + x * 1e3 + rng.normal(size=3_000),
        }
    )
    for name, share in (("x", 0.1), ("y", 0.3), ("big", 0.2)):
        df.loc[rng.random(3_000) < share, name] = np.nan
    return df


def every_type_frame(n: int = 60, extra_numeric: int = 0) -> pd.DataFrame:
    """``n`` rows with one column of each semantic type and the hard cases of the Profile page.

    - ``id``, ``age``, ``income``: numbers; ``score`` misses every 7th value.
    - ``city``: ``Lugo`` and ``lugo `` written two ways; ``active``: yes / no text.
    - ``joined``: dates as text; ``valid_until``: placeholder dates (9999-12-31, 0001-01-01)
      that do not fit in ``datetime64[ns]``.
    - ``comment``: free text; ``empty``: all missing; ``pair``: two values only, so its
      skewness is undefined (NaN must never reach the screen).
    - ``x0``, ``x1``...: ``extra_numeric`` random columns, for a large correlation heatmap.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "id": range(n),
            "age": rng.integers(18, 80, n),
            "score": np.where(np.arange(n) % 7 == 0, np.nan, rng.normal(50, 10, n)),
            "income": rng.normal(3000, 500, n).round(),
            "city": rng.choice(["Lugo", "lugo ", "Vigo"], n),
            "active": rng.choice(["yes", "no"], n),
            "joined": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
            "valid_until": [
                "9999-12-31",
                "0001-01-01",
                *pd.date_range("2025-01-01", periods=n - 2, freq="D").astype(str),
            ],
            "comment": [f"comment number {i}" for i in range(n)],
            "empty": [None] * n,
            "pair": [1.0, 5.0] + [np.nan] * (n - 2),
        }
    )
    return df.assign(**{f"x{i}": rng.normal(0, 1, n) for i in range(extra_numeric)})


def survey_frame() -> pd.DataFrame:
    """40 rows of a survey, written the same way to every format by the integration tests.

    Holds what every format can store: an integer id, a categorical text column, a float column
    with missing values (every 5th) and a date column. Dates are stored as each format stores
    them (text in CSV and SQLite, native dates elsewhere).
    """
    n = 40
    return pd.DataFrame(
        {
            "respondent": range(1, n + 1),
            "region": [("North", "South", "East", "West")[i % 4] for i in range(n)],
            "rating": [np.nan if i % 5 == 0 else 1.0 + (i % 9) / 2 for i in range(n)],
            "answered": pd.date_range("2024-03-01", periods=n, freq="D"),
        }
    )


# ---------------------------------------------------------------------------------------------
# Charts

# Any valid colors: chart builders only place the colors they are given; the theme chooses them.
CHART_COLORS = ChartColors(
    accent="#4F46E5",
    accent_light="#A29DF0",
    highlight="#EA580C",
    outlier="#DC2626",
    median="#EA580C",
    mean="#059669",
    background="#FFFFFF",
    emphasis="#111827",
    diverging=(
        "#2563EB",
        "rgba(37, 99, 235, 0.45)",
        "#EEEEEE",
        "rgba(220, 38, 38, 0.45)",
        "#DC2626",
    ),
)


# ---------------------------------------------------------------------------------------------
# Visualization


def chart_frame(n: int = 3_000) -> pd.DataFrame:
    """``n`` rows, untyped (as read from a CSV), with what the chart builder must tell apart.

    - ``a`` and ``b``: two correlated numbers (the best scatter pair); ``a`` misses every 17th.
    - ``plan`` (3 values) and ``churn`` (2): categories; ``city`` has 60 values, which the
      schema reads as free text and the page must not offer.
    - ``date``: dates as text every 6 hours, with one placeholder ``9999-12-31`` written in the
      same format (so it is read as a date) that would squash a time axis.
    - ``empty``: all missing; ``row``: 1..n without "id" in its name (a row number, not a
      measure); ``id``: 0..n-1 (an identifier for the schema).
    - ``income``: integers almost all distinct over a wide range: a measure, not a row number.
    """
    rng = np.random.default_rng(1)
    a = rng.normal(50, 10, n)
    dates = pd.date_range("2020-01-01", periods=n, freq="6h").astype(str).to_numpy()
    dates[0] = "9999-12-31 00:00:00"
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "a": a,
            "b": a * 2 + rng.normal(0, 5, n),
            "plan": rng.choice(["basic", "standard", "premium"], n),
            "churn": rng.choice(["yes", "no"], n),
            "city": [f"city {i}" for i in rng.integers(0, 60, n)],
            "date": dates,
            "empty": np.full(n, np.nan),
            "row": np.arange(1, n + 1),
            "income": rng.integers(10_000, 90_000, n),
        }
    )
    df.loc[::17, "a"] = np.nan
    return df


def chart_csv() -> bytes:
    """``chart_frame`` as the CSV file a user uploads."""
    return chart_frame().to_csv(index=False).encode()
