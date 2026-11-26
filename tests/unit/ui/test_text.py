"""Unit tests of ``ui/text.py``: text shown to the user.

Names of files, tables and columns come from the user's data and are drawn as Markdown;
unescaped, ``sales_2024`` turns into italics and ``:red[...]`` into colored text.
"""

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
    ids=["underscore", "asterisks", "dollars", "color", "icon", "html", "backslash", "plain"],
)
def test_md_escapes_what_markdown_would_read(name, expected):
    """Every character Streamlit's Markdown would interpret is escaped, and nothing else.

    Case: a name is shown in italics, as LaTeX (``$``), colored, as an icon, or as HTML.
    Affects: ui/text.py (md), used wherever a page draws a user's name as Markdown
    How: escapes one name per Markdown feature and a plain one.
    """
    assert text.md(name) == expected


def test_md_accepts_any_value():
    """Non-string names (a column called 2024) are escaped as their text.

    Case: ``md`` raises on an integer column name.
    Affects: ui/text.py (md)
    How: escapes the integer 2024.
    """
    assert text.md(2024) == "2024"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("*price*", "`*price*`"),
        ("a`b", "`` a`b ``"),
        ("x``y`", "``` x``y` ```"),
        ("`quoted`", "`` `quoted` ``"),
        ("two\nlines", "`two lines`"),
    ],
    ids=["plain", "one-backtick", "longest-run", "ends", "line-break"],
)
def test_code_keeps_names_with_backticks_in_one_span(name, expected):
    """A name in inline code shows whole, even when it holds backticks.

    Case: a column named ``a`b`` closes the code span early and the rest of the line is read
    as Markdown (Issues list, spelling variants of the inspector).
    Affects: ui/text.py (code)
    How: wraps names with no backtick, one, a run of two, backticks at both ends and a line
    break, and checks the fence is longer than any run inside.
    """
    assert text.code(name) == expected


@pytest.mark.parametrize(
    ("n", "expected"),
    [(512, "512 B"), (1536, "1.5 KB"), (5 * 1024**2, "5.0 MB"), (3 * 1024**3, "3.0 GB")],
)
def test_format_bytes(n, expected):
    """Sizes use the largest unit that keeps the number above 1, in powers of 1024.

    Case: sizes shown as "0.0 GB" or in powers of 1000 (not what the OS shows).
    Affects: ui/text.py (format_bytes)
    How: formats one size per unit.
    """
    assert text.format_bytes(n) == expected
