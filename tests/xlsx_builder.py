"""Builds small .xlsx workbooks in memory for the tests, without an Excel writer library.

An .xlsx file is a zip of XML parts. This writes the minimum Excel and calamine accept:
numbers, text (inline strings), booleans, dates and times (numbers with the built-in date and
time formats).
"""

import io
import zipfile
from collections.abc import Mapping, Sequence
from datetime import date, time
from xml.sax.saxutils import escape

type Cell = str | int | float | bool | date | time | None
type Rows = Sequence[Sequence[Cell]]

# Day 0 of Excel's date numbers (1900 system, with its fake 1900-02-29).
_EXCEL_EPOCH = date(1899, 12, 30)


def xlsx(sheets: Mapping[str, Rows], hidden: Sequence[str] = ()) -> bytes:
    """Bytes of a workbook with one sheet per item of ``sheets``, in order.

    Args:
        sheets: Sheet name -> rows; the first row usually holds the column names.
        hidden: Names of the sheets marked as hidden.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        n = len(sheets)
        z.writestr("[Content_Types].xml", _content_types(n))
        z.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="xl/workbook.xml" Type="http://schemas.openxmlformats'
            '.org/officeDocument/2006/relationships/officeDocument"/></Relationships>',
        )
        z.writestr("xl/workbook.xml", _workbook(list(sheets), hidden))
        z.writestr("xl/_rels/workbook.xml.rels", _workbook_rels(n))
        z.writestr("xl/styles.xml", _STYLES)
        for i, rows in enumerate(sheets.values(), start=1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", _sheet(rows))
    return buffer.getvalue()


def _content_types(n: int) -> str:
    sheets = "".join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/'
        'vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, n + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package'
        '.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-'
        'officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-'
        f'officedocument.spreadsheetml.styles+xml"/>{sheets}</Types>'
    )


def _workbook(names: list[str], hidden: Sequence[str]) -> str:
    sheets = "".join(
        f'<sheet name="{escape(name, {chr(34): "&quot;"})}" sheetId="{i}" r:id="rId{i}"'
        + (' state="hidden"' if name in hidden else "")
        + "/>"
        for i, name in enumerate(names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheets}</sheets></workbook>"
    )


def _workbook_rels(n: int) -> str:
    rels = "".join(
        f'<Relationship Id="rId{i}" Target="worksheets/sheet{i}.xml" Type="http://schemas.'
        'openxmlformats.org/officeDocument/2006/relationships/worksheet"/>'
        for i in range(1, n + 1)
    )
    rels += (
        f'<Relationship Id="rId{n + 1}" Target="styles.xml" Type="http://schemas.'
        'openxmlformats.org/officeDocument/2006/relationships/styles"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{rels}</Relationships>"
    )


# Style 0: general; style 1: built-in date format 14 (m/d/yyyy); style 2: time format 21
# (h:mm:ss).
_STYLES = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    '<fonts count="1"><font/></fonts><fills count="1"><fill/></fills>'
    '<borders count="1"><border/></borders><cellStyleXfs count="1"><xf/></cellStyleXfs>'
    '<cellXfs count="3"><xf numFmtId="0"/><xf numFmtId="14" applyNumberFormat="1"/>'
    '<xf numFmtId="21" applyNumberFormat="1"/></cellXfs>'
    "</styleSheet>"
)


def _sheet(rows: Rows) -> str:
    body = "".join(
        f'<row r="{r}">' + "".join(_cell(r, c, v) for c, v in enumerate(row)) + "</row>"
        for r, row in enumerate(rows, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{body}</sheetData></worksheet>"
    )


def _cell(row: int, col: int, value: Cell) -> str:
    ref = f"{chr(ord('A') + col)}{row}"
    match value:
        case None:
            return ""
        case bool():
            return f'<c r="{ref}" t="b"><v>{int(value)}</v></c>'
        case int() | float():
            return f'<c r="{ref}"><v>{value}</v></c>'
        case time():
            # A time is the fraction of a day.
            seconds = value.hour * 3600 + value.minute * 60 + value.second
            return f'<c r="{ref}" s="2"><v>{seconds / 86400}</v></c>'
        case date():
            return f'<c r="{ref}" s="1"><v>{(value - _EXCEL_EPOCH).days}</v></c>'
        case _:
            return f'<c r="{ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'
