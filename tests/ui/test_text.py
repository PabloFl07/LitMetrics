import pytest

from litmetrics.ui import text


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("sales_2024", r"sales\_2024"),
        ("*draft*", r"\*draft\*"),
        ("cost $5 to $9", r"cost \$5 to \$9"),
        (":red[alert]", r"\:red\[alert\]"),
        (":material/home:", r"\:material/home\:"),
        ("a<b>", r"a\<b\>"),
        (r"C:\data", r"C\:\\data"),
        ("plain name (v2).db", "plain name (v2).db"),
    ],
)
def test_md_escapes_what_markdown_would_read(name, expected):
    assert text.md(name) == expected


def test_md_accepts_any_value():
    assert text.md(2024) == "2024"


@pytest.mark.parametrize(
    ("n", "expected"),
    [(512, "512 B"), (1536, "1.5 KB"), (5 * 1024**2, "5.0 MB"), (3 * 1024**3, "3.0 GB")],
)
def test_format_bytes(n, expected):
    assert text.format_bytes(n) == expected
