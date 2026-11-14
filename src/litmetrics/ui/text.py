"""Text shown to the user: sizes, and names that must not be read as Markdown.

Names that come from the user's files (file, table, sheet and column names) can hold any
character. Streamlit draws ``st.markdown``, ``st.caption`` and widget labels as Markdown, where
``*draft*`` turns italic, ``$x$`` turns into a formula and ``:red[x]`` turns red: ``md``
escapes them so they show as written.
"""

import re

# Characters that change how a name is drawn inside a line: emphasis (``*``, ``_``, ``~``),
# code (`````), links (``[``, ``]``), formulas (``$``), HTML (``<``, ``>``), the backslash
# itself, and the colon of Streamlit's directives: ``:red[x]`` loses its ``:red`` even with
# the brackets escaped, and ``:material/home:`` turns into an icon. Others (``(``, ``.``,
# ``-``, ``#``) only mean something at the start of a line or after a link, and names are
# never there. CommonMark lets a backslash escape any of them.
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_~\[\]$<>:])")


def md(name: object) -> str:
    """``name`` as Markdown that shows it exactly as written."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", str(name))


def format_bytes(n: int) -> str:
    """Size in the largest unit that keeps it above 1: "512 B", "1.5 MB"."""
    size = float(n)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
