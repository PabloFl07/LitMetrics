"""litmetrics package. ``main`` is the ``litmetrics`` command declared in pyproject.toml."""

import os
import sys
from collections.abc import MutableMapping
from pathlib import Path

# Project root: theme files are found from here (``StyleConfig.themes_dir``).
_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Start the Streamlit app, as ``streamlit run src/litmetrics/app.py`` would.

    ``--style <n>`` picks a style variant (``theme_<n>.toml`` and its stylesheet); without it,
    the ``LITMETRICS_STYLE`` environment variable is used, and then variant 0. Other
    command-line arguments are passed on to ``streamlit run``
    (e.g. ``uv run litmetrics --style 2 --server.port 8502``).

    The theme is always passed as an absolute ``--theme.base``: Streamlit only reads
    ``.streamlit/config.toml`` from the directory the app starts in, so without it the app
    started from any other directory has no theme and every chart fails.
    """
    # Imported here: this module runs on any ``import litmetrics...``, and ``core`` must stay
    # importable without loading Streamlit.
    from streamlit.web import cli

    args = streamlit_args(sys.argv[1:], os.environ)
    app = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app), *args]
    sys.exit(cli.main())


def streamlit_args(argv: list[str], environ: MutableMapping[str, str]) -> list[str]:
    """Arguments for ``streamlit run``: ``argv`` without ``--style``, plus the theme.

    Sets ``LITMETRICS_STYLE`` in ``environ`` to the chosen variant: the app process reads it to
    load the stylesheet that goes with the theme.

    Raises:
        SystemExit: The variant is not a number or has no theme file.
    """
    from litmetrics.config import StyleConfig

    value, rest = _pop_style(argv)
    if value == "":
        sys.exit("--style needs a variant number, e.g. --style 2.")
    variant = value or environ.get(StyleConfig.env_var) or str(StyleConfig.default_variant)
    theme = _ROOT / StyleConfig.themes_dir / f"theme_{variant}.toml"
    if not variant.isdigit() or not theme.is_file():
        sys.exit(f"Unknown style {variant!r}: there is no {theme}.")
    environ[StyleConfig.env_var] = variant
    # A command-line option wins over the base set in .streamlit/config.toml.
    return [f"--theme.base={theme}", *rest]


def _pop_style(args: list[str]) -> tuple[str | None, list[str]]:
    """Value of ``--style N`` / ``--style=N`` in ``args``, and ``args`` without it.

    ``--style`` without a value gives ``""``; no ``--style`` gives None.
    """
    rest: list[str] = []
    value: str | None = None
    it = iter(args)
    for arg in it:
        if arg == "--style":
            value = next(it, "")
        elif arg.startswith("--style="):
            value = arg.removeprefix("--style=")
        else:
            rest.append(arg)
    return value, rest
