"""Draws ECharts charts with the look of the page. The only module that imports
``streamlit_echarts``.

Pages build the option with ``ui/charts.py`` (colors from ``theme.chart_colors()``) and draw it
with ``show``. ``show`` lays the shared look underneath (series palette, font, transparent
background) and keeps the ``"streamlit"`` theme of ``st_echarts``, which follows light / dark
mode for text, axes and tooltips in the browser.
"""

from typing import Any

from streamlit_echarts import st_echarts

from litmetrics.ui import charts, theme


def show(option: dict[str, Any], *, key: str, height: str, **kwargs: Any) -> Any:
    """Draw ``option`` with the theme of the page and return what ``st_echarts`` returns.

    Args:
        key: Unique key of the chart. Required: without it the chart remounts and replays its
            animation on every rerun.
        height: CSS height, such as ``"300px"``.
        kwargs: Other arguments of ``st_echarts`` (``events``, ``on_select``...).
    """
    return st_echarts(
        charts.themed(option, theme.chart_theme()),
        height=height,
        key=key,
        theme="streamlit",
        **kwargs,
    )
