"""litmetrics package. ``main`` is the ``litmetrics`` command declared in pyproject.toml."""

import sys
from pathlib import Path


def main() -> None:
    """Start the Streamlit app, as ``streamlit run src/litmetrics/app.py`` would.

    Extra command-line arguments are passed on to ``streamlit run``
    (e.g. ``uv run litmetrics --server.port 8502``).
    """
    # Imported here: this module runs on any ``import litmetrics...``, and ``core`` must stay
    # importable without loading Streamlit.
    from streamlit.web import cli

    app = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app), *sys.argv[1:]]
    sys.exit(cli.main())
