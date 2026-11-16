"""Unit tests of ``core/io/sniffing.py``: detecting how a CSV is written.

Users never choose separator, encoding, header or decimal mark: the app guesses them. A wrong
guess reads the whole file wrong (one column, mangled accents, numbers as text), so each
detector is tested on the cases that fool simple heuristics.
"""

import codecs

import pytest

from litmetrics.core.io import sniffing
from litmetrics.core.io.sniffing import CsvOptions, detect_options


def _rows(text: str, separator: str = ",") -> list[list[str]]:
    return sniffing._sample_rows(text, separator)


# Header


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("age,city\n30,Paris\n", True),
        ("30,Paris\n41,Rome\n", False),
        ("name,city\nAna,Paris\n", True),
        ("a,b\n", True),
    ],
    ids=["text-over-numbers", "numbers-only", "text-over-text", "single-row"],
)
def test_detect_header(text, expected):
    """A first row is a header when it looks different from the rest, or is alone.

    Case: a file of only text is read without header (names become a data row), or a file of
    only numbers loses its first row as a header.
    Affects: core/io/sniffing.py (detect_header)
    How: the examples of the docstring of ``detect_header`` and a file of one row.
    """
    assert sniffing.detect_header(_rows(text)) is expected


# Separator


@pytest.mark.parametrize("separator", [";", "\t", "|"])
def test_detect_separator(separator):
    """Semicolon, tab and pipe are found when they split every row the same way.

    Case: a semicolon file is read with commas, giving one column per row.
    Affects: core/io/sniffing.py (detect_separator)
    How: builds 10 rows of 3 fields with each separator.
    """
    text = "".join(separator.join(row) + "\n" for row in [["a", "b", "c"], ["1", "2", "3"]] * 5)
    assert sniffing.detect_separator(text) == separator


def test_detect_separator_single_column_falls_back_to_comma():
    """A file of one column has no separator to find: the comma is used.

    Case: a stray character (a space, a dot in the values) is taken as the separator.
    Affects: core/io/sniffing.py (detect_separator)
    How: detects the separator of a list of names.
    """
    assert sniffing.detect_separator("name\nAna\nLuis\n") == ","


# Encoding


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"a,b\n1,2\n", "utf-8"),
        ("ciudad\nMálaga\n".encode(), "utf-8"),
        (codecs.BOM_UTF8 + b"a\n", "utf-8-sig"),
        ("a\n".encode("utf-16"), "utf-16"),
        # 0x81 is undefined in cp1252: only latin-1 decodes any byte.
        (b"a\x81\n", "latin-1"),
    ],
    ids=["ascii", "utf8", "utf8-bom", "utf16", "latin1"],
)
def test_detect_encoding(data, expected):
    """The encoding is found from BOMs, then by trying UTF-8, cp1252 and latin-1.

    Case: a BOM is kept as a character, UTF-16 is decoded as garbage, or a byte undefined in
    cp1252 raises instead of falling back.
    Affects: core/io/sniffing.py (detect_encoding)
    How: detects the encoding of short byte strings, one per case.
    """
    assert sniffing.detect_encoding(data) == expected


def test_detect_encoding_cp1252_byte_at_the_end(monkeypatch):
    """A Windows byte far into the file is still found.

    Case: only the start of the file is checked, so a cp1252 accent in row 10,000 later fails
    to decode or comes out mangled.
    Affects: core/io/sniffing.py (detect_encoding, _DECODE_CHUNK)
    How: makes chunks 8 bytes long and puts the cp1252 ``á`` in a later chunk.
    """
    monkeypatch.setattr(sniffing, "_DECODE_CHUNK", 8)
    data = b"ciudad\n" * 10 + "Málaga\n".encode("cp1252")
    assert sniffing.detect_encoding(data) == "cp1252"


def test_detect_encoding_multibyte_char_split_between_chunks(monkeypatch):
    """A UTF-8 character split by a chunk boundary is not taken for an error.

    Case: decoding chunk by chunk breaks a 2-byte character and wrongly falls back to cp1252.
    Affects: core/io/sniffing.py (detect_encoding)
    How: makes chunks 1 byte long and detects ``Málaga`` in UTF-8.
    """
    monkeypatch.setattr(sniffing, "_DECODE_CHUNK", 1)
    assert sniffing.detect_encoding("Málaga\n".encode()) == "utf-8"


# Decimal mark


@pytest.mark.parametrize(
    ("text", "separator", "expected"),
    [
        ("precio;ciudad\n0,5;Lugo\n1,25;Vigo\n", ";", ","),
        ("precio;ciudad\n0.5;Lugo\n1.25;Vigo\n", ";", "."),
        ("n;ciudad\n1;Lugo\n2;Vigo\n", ";", "."),
        ("a,b\n0,5\n", ",", "."),
    ],
    ids=["comma-decimals", "dot-decimals", "integers-only", "comma-is-separator"],
)
def test_detect_decimal(text, separator, expected):
    """The comma is the decimal mark only when values use it and it is not the separator.

    Case: European numbers stay text; or a comma-separated file is read with comma decimals.
    Affects: core/io/sniffing.py (detect_decimal)
    How: detects the mark in files with comma decimals, dot decimals, integers, and commas as
    separator.
    """
    assert sniffing.detect_decimal(_rows(text, separator), separator) == expected


def test_detect_options_european_file():
    """All four options are found together on a typical Spanish export.

    Case: one detector depends on another (decimal on separator) and runs in the wrong order.
    Affects: core/io/sniffing.py (detect_options)
    How: detects the options of a ``;`` file with ``,`` decimals and a header.
    """
    data = b"precio;ciudad\n0,5;Lugo\n1,25;Vigo\n3;Ourense\n"
    assert detect_options(data) == CsvOptions(";", "utf-8", True, ",")
