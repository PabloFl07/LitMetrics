"""Checks of the test suite itself: the rules of the "Tests" section of AGENTS.md that a
machine can check, so they hold for every test added later."""

import ast
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parents[1]
TEST_FILES = sorted(TESTS.rglob("test_*.py"))


def _id(path: Path) -> str:
    return str(path.relative_to(TESTS))


@pytest.mark.parametrize("path", TEST_FILES, ids=_id)
def test_every_test_is_explained(path):
    """Every test module has a docstring, and every test says its case, scope and steps.

    Case: a test is added without saying which bug it looks for or what it touches, so nobody
    can tell later whether it still earns its place.
    Affects: every file under tests/ (convention of AGENTS.md, section "Tests")
    How: parses each test file and checks the module docstring and, for each ``test_*``
    function, a docstring with ``Case:``, ``Affects:`` and ``How:``.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert ast.get_docstring(tree), "the module has no docstring"
    unexplained = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        if not all(k in (ast.get_docstring(node) or "") for k in ("Case:", "Affects:", "How:"))
    ]
    assert not unexplained, f"tests without Case / Affects / How: {unexplained}"


@pytest.mark.parametrize("path", TEST_FILES, ids=_id)
def test_tests_live_in_a_level_folder(path):
    """Every test file is under ``unit/``, ``integration/`` or ``acceptance/``.

    Case: a test file left at the top of tests/ gets no level marker, so ``-m`` filters and the
    default run (which skips acceptance) treat it wrongly.
    Affects: tests/ layout, tests/conftest.py (pytest_collection_modifyitems)
    How: checks the first folder of each test file's path.
    """
    assert path.relative_to(TESTS).parts[0] in {"unit", "integration", "acceptance"}
