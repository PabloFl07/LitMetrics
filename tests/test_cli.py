"""``litmetrics`` command: arguments passed on to ``streamlit run`` (``litmetrics.__init__``)."""

import tomllib
from pathlib import Path

import pytest

from litmetrics import _pop_style, streamlit_args
from litmetrics.config import LoadConfig, StyleConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("argv", "value", "rest"),
    [
        (["--style", "2", "--server.port", "1"], "2", ["--server.port", "1"]),
        (["--style=3"], "3", []),
        (["--style"], "", []),
        (["--server.port", "1"], None, ["--server.port", "1"]),
    ],
)
def test_pop_style(argv, value, rest):
    assert _pop_style(argv) == (value, rest)


def _theme(args: list[str]) -> Path:
    (base,) = [a.removeprefix("--theme.base=") for a in args if a.startswith("--theme.base=")]
    return Path(base)


def test_theme_is_always_absolute_even_without_style():
    # Without it, the app started outside the project root has no theme and charts fail.
    environ: dict[str, str] = {}
    args = streamlit_args(["--server.port", "1"], environ)
    theme = _theme(args)
    assert theme.is_absolute() and theme.is_file()
    assert theme.name == f"theme_{StyleConfig.default_variant}.toml"
    assert environ[StyleConfig.env_var] == str(StyleConfig.default_variant)
    assert args[-2:] == ["--server.port", "1"]


def test_style_option_wins_over_the_environment():
    environ = {StyleConfig.env_var: "1"}
    assert _theme(streamlit_args(["--style", "2"], environ)).name == "theme_2.toml"
    assert environ[StyleConfig.env_var] == "2"


def test_environment_picks_the_variant():
    assert _theme(streamlit_args([], {StyleConfig.env_var: "3"})).name == "theme_3.toml"


@pytest.mark.parametrize("argv", [["--style", "99"], ["--style", "x"], ["--style"]])
def test_unknown_style_exits(argv):
    with pytest.raises(SystemExit):
        streamlit_args(argv, {})


def test_upload_limit_is_passed_on():
    # An installed app does not read .streamlit/config.toml: the limit goes on the command line.
    args = streamlit_args([], {})
    assert f"--server.maxUploadSize={LoadConfig.max_upload_mb}" in args


def test_upload_limit_given_by_the_user_wins():
    args = streamlit_args(["--server.maxUploadSize=50"], {})
    assert [a for a in args if a.startswith("--server.maxUploadSize")] == [
        "--server.maxUploadSize=50"
    ]


def test_config_toml_has_the_same_upload_limit():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["server"]["maxUploadSize"] == LoadConfig.max_upload_mb


def test_themes_are_inside_the_package():
    # So an installed ``litmetrics`` finds them, not only a clone of the repository.
    theme = _theme(streamlit_args([], {}))
    assert theme.is_relative_to(ROOT / "src" / "litmetrics")
