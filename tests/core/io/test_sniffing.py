import codecs

import pytest

from litmetrics.core.io import sniffing
from litmetrics.core.io.sniffing import CsvOptions, detect_options


def _rows(text: str, separator: str = ",") -> list[list[str]]:
    return sniffing._sample_rows(text, separator)


# The three examples of the docstring of detect_header.
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("age,city\n30,Paris\n", True),
        ("30,Paris\n41,Rome\n", False),
        ("name,city\nAna,Paris\n", True),
    ],
)
def test_detect_header(text, expected):
    assert sniffing.detect_header(_rows(text)) is expected


def test_detect_header_single_row_assumes_a_header():
    assert sniffing.detect_header(_rows("a,b\n"))


def test_detect_separator_single_column_falls_back_to_comma():
    assert sniffing.detect_separator("name\nAna\nLuis\n") == ","


@pytest.mark.parametrize("separator", [";", "\t", "|"])
def test_detect_separator(separator):
    text = "".join(separator.join(row) + "\n" for row in [["a", "b", "c"], ["1", "2", "3"]] * 5)
    assert sniffing.detect_separator(text) == separator


def test_detect_encoding_ascii_is_utf8():
    assert sniffing.detect_encoding(b"a,b\n1,2\n") == "utf-8"


def test_detect_encoding_utf8():
    assert sniffing.detect_encoding("ciudad\nMálaga\n".encode()) == "utf-8"


def test_detect_encoding_boms():
    assert sniffing.detect_encoding(codecs.BOM_UTF8 + b"a\n") == "utf-8-sig"
    assert sniffing.detect_encoding("a\n".encode("utf-16")) == "utf-16"


def test_detect_encoding_cp1252_byte_at_the_end(monkeypatch):
    # Small chunks, so the bad byte is in a later chunk than the first one.
    monkeypatch.setattr(sniffing, "_DECODE_CHUNK", 8)
    data = b"ciudad\n" * 10 + "Málaga\n".encode("cp1252")
    assert sniffing.detect_encoding(data) == "cp1252"


def test_detect_encoding_multibyte_char_split_between_chunks(monkeypatch):
    # "á" is two bytes in UTF-8; a chunk boundary between them must not look like an error.
    monkeypatch.setattr(sniffing, "_DECODE_CHUNK", 1)
    assert sniffing.detect_encoding("Málaga\n".encode()) == "utf-8"


def test_detect_encoding_any_byte_is_latin1():
    # 0x81 is undefined in cp1252.
    assert sniffing.detect_encoding(b"a\x81\n") == "latin-1"


@pytest.mark.parametrize(
    ("text", "separator", "expected"),
    [
        ("precio;ciudad\n0,5;Lugo\n1,25;Vigo\n", ";", ","),
        ("precio;ciudad\n0.5;Lugo\n1.25;Vigo\n", ";", "."),
        # Integers say nothing: the default.
        ("n;ciudad\n1;Lugo\n2;Vigo\n", ";", "."),
        # The comma is the separator, so it cannot be the decimal mark.
        ("a,b\n0,5\n", ",", "."),
    ],
)
def test_detect_decimal(text, separator, expected):
    assert sniffing.detect_decimal(_rows(text, separator), separator) == expected


def test_detect_options_european_file():
    data = b"precio;ciudad\n0,5;Lugo\n1,25;Vigo\n3;Ourense\n"
    assert detect_options(data) == CsvOptions(";", "utf-8", True, ",")
