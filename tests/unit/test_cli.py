"""Unit tests of the ``litmetrics`` command (``litmetrics/__init__.py``).

The command passes its arguments on to ``streamlit run`` plus the theme and the upload limit,
which Streamlit would otherwise only read from ``.streamlit/config.toml`` in the folder the app
starts in.
"""

import tomllib
from pathlib import Path

import pytest

from litmetrics import _pop_style, streamlit_args
from litmetrics.config import LoadConfig, StyleConfig

ROOT = Path(__file__).resolve().parents[2]


def _theme(args: list[str]) -> Path:
    """The theme file passed in ``--theme.base``."""
    (base,) = [a.removeprefix("--theme.base=") for a in args if a.startswith("--theme.base=")]
    return Path(base)


@pytest.mark.parametrize(
    ("argv", "value", "rest"),
    [
        (["--style", "2", "--server.port", "1"], "2", ["--server.port", "1"]),
        (["--style=3"], "3", []),
        (["--style"], "", []),
        (["--server.port", "1"], None, ["--server.port", "1"]),
    ],
    ids=["separate", "equals", "no-value", "absent"],
)
def test_pop_style(argv, value, rest):
    """``--style`` is taken out in both spellings; the rest is passed on in order.

    Case: ``--style=3`` is passed to Streamlit (which rejects it), or the next option is eaten.
    Affects: litmetrics/__init__.py (_pop_style)
    How: pops the option from each spelling and from arguments without it.
    """
    assert _pop_style(argv) == (value, rest)


def test_theme_is_always_absolute_even_without_style():
    """Without ``--style``, the default variant is passed as an absolute existing path.

    Case: the app started outside the project root finds no theme and every chart fails.
    Affects: litmetrics/__init__.py (streamlit_args), config.py (StyleConfig)
    How: builds the arguments without style and checks the theme path, the environment
    variable for the app process, and that other arguments are kept at the end.
    """
    environ: dict[str, str] = {}
    args = streamlit_args(["--server.port", "1"], environ)
    theme = _theme(args)
    assert theme.is_absolute() and theme.is_file()
    assert theme.name == f"theme_{StyleConfig.default_variant}.toml"
    assert environ[StyleConfig.env_var] == str(StyleConfig.default_variant)
    assert args[-2:] == ["--server.port", "1"]


def test_style_option_wins_over_the_environment():
    """``--style`` wins over ``LITMETRICS_STYLE``, and the environment is updated to match.

    Case: the theme and the stylesheet come from different variants.
    Affects: litmetrics/__init__.py (streamlit_args)
    How: passes ``--style 2`` with the variable set to 1.
    """
    environ = {StyleConfig.env_var: "1"}
    assert _theme(streamlit_args(["--style", "2"], environ)).name == "theme_2.toml"
    assert environ[StyleConfig.env_var] == "2"


def test_environment_picks_the_variant():
    """Without ``--style``, ``LITMETRICS_STYLE`` picks the variant.

    Case: the variable is ignored and the default is always used.
    Affects: litmetrics/__init__.py (streamlit_args)
    How: sets the variable to 3 and checks the theme.
    """
    assert _theme(streamlit_args([], {StyleConfig.env_var: "3"})).name == "theme_3.toml"


@pytest.mark.parametrize("argv", [["--style", "99"], ["--style", "x"], ["--style"]])
def test_unknown_style_exits(argv):
    """An unknown, non-numeric or missing variant stops the command with a message.

    Case: Streamlit starts without a theme and fails later with an unclear error.
    Affects: litmetrics/__init__.py (streamlit_args)
    How: passes each bad value and expects ``SystemExit``.
    """
    with pytest.raises(SystemExit):
        streamlit_args(argv, {})


def test_upload_limit_is_passed_on():
    """The upload limit of the config is passed on the command line.

    Case: an installed app does not read .streamlit/config.toml and rejects files over 200 MB.
    Affects: litmetrics/__init__.py (streamlit_args), config.py (LoadConfig.max_upload_mb)
    How: builds the arguments and looks for the option.
    """
    assert f"--server.maxUploadSize={LoadConfig.max_upload_mb}" in streamlit_args([], {})


def test_upload_limit_given_by_the_user_wins():
    """A limit given by the user is kept and not duplicated.

    Case: two ``--server.maxUploadSize`` options, the user's silently overridden.
    Affects: litmetrics/__init__.py (streamlit_args)
    How: passes a limit of 50 and checks it is the only one.
    """
    args = streamlit_args(["--server.maxUploadSize=50"], {})
    assert [a for a in args if a.startswith("--server.maxUploadSize")] == [
        "--server.maxUploadSize=50"
    ]


def test_config_toml_has_the_same_upload_limit():
    """``config.toml`` and ``LoadConfig`` agree on the upload limit.

    Case: ``streamlit run`` (config.toml) and ``litmetrics`` (LoadConfig) accept different sizes.
    Affects: .streamlit/config.toml, config.py (LoadConfig.max_upload_mb)
    How: reads the limit from config.toml and compares.
    """
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["server"]["maxUploadSize"] == LoadConfig.max_upload_mb


def test_themes_are_inside_the_package():
    """Theme files ship inside the package.

    Case: themes live at the repository root, so an installed ``litmetrics`` has none.
    Affects: litmetrics/__init__.py (_PACKAGE), ui/themes/
    How: checks the theme passed is under ``src/litmetrics``.
    """
    assert _theme(streamlit_args([], {})).is_relative_to(ROOT / "src" / "litmetrics")
