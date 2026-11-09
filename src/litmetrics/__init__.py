"""litmetrics package. ``main`` is the ``litmetrics`` command declared in pyproject.toml."""

import os
import sys
from pathlib import Path

# Project root: theme files are found from here (``StyleConfig.themes_dir``).
_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Start the Streamlit app, as ``streamlit run src/litmetrics/app.py`` would.

    ``--style <n>`` picks a style variant (``theme_<n>.toml`` and its stylesheet); without it,
    the ``LITMETRICS_STYLE`` environment variable is used, and then variant 0. Other
    command-line arguments are passed on to ``streamlit run``
    (e.g. ``uv run litmetrics --style 2 --server.port 8502``).
    """
    # Imported here: this module runs on any ``import litmetrics...``, and ``core`` must stay
    # importable without loading Streamlit.
    from streamlit.web import cli

    from litmetrics.config import StyleConfig

    args = sys.argv[1:]
    variant, args = _pop_style(args)
    variant = variant or os.environ.get(StyleConfig.env_var)
    if variant is not None:
        theme = _ROOT / StyleConfig.themes_dir / f"theme_{variant}.toml"
        if not variant.isdigit() or not theme.is_file():
            sys.exit(f"Unknown style {variant!r}: there is no {theme}.")
        # The app process reads the variable to load the matching stylesheet.
        os.environ[StyleConfig.env_var] = variant
        # A command-line option wins over the base set in .streamlit/config.toml.
        args = [f"--theme.base={theme}", *args]

    app = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app), *args]
    sys.exit(cli.main())


def _pop_style(args: list[str]) -> tuple[str | None, list[str]]:
    """Value of ``--style N`` / ``--style=N`` in ``args``, and ``args`` without it."""
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
