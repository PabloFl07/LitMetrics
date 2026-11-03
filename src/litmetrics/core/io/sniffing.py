"""Detect how a CSV file is written: separator, encoding and header.

Everything here is a guess. Each function returns a safe default (comma, header present)
when the data does not give a clear answer, and leaves error reporting to the loader.
"""

import codecs
import csv
import io
import re
from dataclasses import dataclass
from itertools import islice, zip_longest

# Share of values that must agree for a column to count as numeric (or as text).
# Below it the column is mixed and says nothing about the header.
_TYPE_THRESHOLD = 0.9
# Bytes looked at to guess the separator and the header. Enough for many rows of a wide file.
SAMPLE_BYTES = 64 * 1024
# Rows compared against the first one to decide whether it is a header.
SAMPLE_ROWS = 50
# Chunk size when checking the encoding of the whole file.
_DECODE_CHUNK = 1024 * 1024

# Separators that are tried, with the name shown to the user.
SEPARATORS = {",": "Comma (,)", ";": "Semicolon (;)", "\t": "Tab", "|": "Pipe (|)"}

# Integers and decimals with "." or "," as decimal mark, with optional sign and exponent:
# "12", "-3.5", "3,5", ".5", "1e-3". Thousands separators ("1.000.000") are not supported.
_NUMBER = re.compile(r"[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?")


@dataclass(frozen=True)
class CsvOptions:
    """How to read a CSV file. Frozen so it can be hashed and used in cache keys."""

    separator: str = ","
    encoding: str = "utf-8"
    has_header: bool = True


def detect_options(data: bytes) -> CsvOptions:
    """Guess the options. The encoding is checked on the whole file, the rest on its first bytes.

    The encoding needs the whole file because a single bad byte near the end is enough for the
    read to fail. The separator and the header are clear from the first rows.
    """
    encoding = detect_encoding(data)
    sample = data[:SAMPLE_BYTES]
    text = sample.decode(encoding, errors="replace")
    if len(data) > SAMPLE_BYTES:
        # The last line of the sample is probably cut in half.
        text = text[: text.rfind("\n") + 1] or text
    separator = detect_separator(text)
    return CsvOptions(separator, encoding, detect_header(text, separator))


def detect_encoding(data: bytes) -> str:
    """Return the first encoding that decodes the whole file, from most to least strict."""
    # A BOM states the encoding explicitly. "utf-8-sig" also removes it from the first column name.
    if data.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    # Windows exports (Excel) use cp1252; latin-1 decodes any byte, so it is the last resort.
    for encoding in ("utf-8", "cp1252"):
        if _decodes_as(data, encoding):
            return encoding
    return "latin-1"


def _decodes_as(data: bytes, encoding: str) -> bool:
    try:
        data.decode(encoding)
    except UnicodeDecodeError:
        return False
    return True


def detect_separator(text: str) -> str:
    """Pick one of ``SEPARATORS``. Fall back to comma (e.g. a file with a single column)."""
    try:
        return csv.Sniffer().sniff(text, delimiters="".join(SEPARATORS)).delimiter
    except csv.Error:
        return ","


def detect_header(text: str, separator: str) -> bool:
    """Say whether the first row holds column names.

    csv.Sniffer.has_header fails on files where every column is text, so this uses a simpler rule:
    the first row is data only if it has a number and every value matches the type of its column.

    Examples:
        "age,city / 30,Paris"   -> header ("age" is text, its column is numeric)
        "30,Paris / 41,Rome"    -> no header (types match and the row has a number)
        "name,city / Ana,Paris" -> header (no number to tell them apart, so assume a header)
    """
    try:
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=separator)
        rows = list(islice(reader, SAMPLE_ROWS))
    except csv.Error:
        # Not readable as CSV (e.g. a binary file); the loader reports the real problem.
        return True
    if len(rows) < 2:
        return True
    first, rest = rows[0], rows[1:]
    first_has_number = False
    # zip_longest turns the rows into columns; short rows count as empty cells.
    for value, column in zip(first, zip_longest(*rest, fillvalue=""), strict=False):
        column_is_number = _column_is_number(column)
        # Empty cells and mixed columns say nothing about the type.
        if column_is_number is None or not value.strip():
            continue
        if _is_number(value) != column_is_number:
            return True
        first_has_number = first_has_number or column_is_number
    return not first_has_number


def _column_is_number(values) -> bool | None:
    """True if most non-empty values are numbers, False if most are text, None if mixed or empty."""
    values = [v for v in values if v.strip()]
    if not values:
        return None
    share = sum(map(_is_number, values)) / len(values)
    if share >= _TYPE_THRESHOLD:
        return True
    if share <= 1 - _TYPE_THRESHOLD:
        return False
    return None


def _is_number(value: str) -> bool:
    return _NUMBER.fullmatch(value.strip()) is not None
