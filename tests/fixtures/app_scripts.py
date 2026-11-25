"""Scripts run by ``AppTest`` and helpers to find widgets in what they drew.

``AppTest.from_function`` runs the *source* of a function as a Streamlit script, so each
script here is self-contained: it imports what it needs inside its body. Never import
``litmetrics.ui.pages`` at module level: ``streamlit_echarts`` fails outside a running app.

Pages are functions registered with ``st.navigation``, which ``AppTest.switch_page`` cannot
open, so the scripts draw a page function directly, with the steps app.py runs around it.
"""

from streamlit.testing.v1 import AppTest

# Seconds AppTest waits for one script run. Generous: the first run imports pandas and pyarrow.
TIMEOUT = 30


def run_app() -> None:
    """What app.py does on each rerun, with the Load page as the page.

    - The first run of each test empties the datasets in memory, which are shared by the
      whole process (``st.cache_resource``), so tests do not see each other's files.
    - ``session_state["away"] = True`` draws no page, as if the user were on another page:
      the widgets of the Load page lose their state, as with ``st.navigation``.
    - ``session_state["reads"]`` counts the calls to ``cache.load_dataset``: the tests check
      that a file is not read again on reruns.
    """
    import streamlit as st

    from litmetrics.ui import cache, reading, sidebar, state
    from litmetrics.ui.pages.load import load_page

    state.init_state()
    if "started" not in st.session_state:
        st.session_state["started"] = True
        st.session_state["reads"] = 0
        cache.clear_all()
    read = cache.load_dataset

    def counting(*args, **kwargs):
        st.session_state["reads"] += 1
        return read(*args, **kwargs)

    cache.load_dataset = counting
    try:
        reading.read_chosen_part()
        if not st.session_state.get("away", False):
            load_page()
        sidebar.draw()
    finally:
        cache.load_dataset = read


def run_page(module: str, func: str, with_dataset: bool) -> None:
    """Draw page ``func`` of ``litmetrics.ui.pages.<module>``, with ``QUALITY_CSV`` already
    loaded as the active dataset when ``with_dataset``."""
    import importlib

    from fixtures.samples import QUALITY_CSV
    from litmetrics.core.io.base import Source, compute_dataset_id
    from litmetrics.core.io.csv_loader import CsvLoader
    from litmetrics.core.io.sniffing import detect_options
    from litmetrics.core.schema import prepare_dataset
    from litmetrics.ui import state

    state.init_state()
    if with_dataset and state.get_dataset() is None:
        options = detect_options(QUALITY_CSV)
        dataset, schema = prepare_dataset(
            CsvLoader().load(Source(QUALITY_CSV, "sample.csv", "s"), options)
        )
        state.set_dataset(dataset, compute_dataset_id(QUALITY_CSV, options), "sample")
        state.set_schema(schema)
    page = importlib.import_module("litmetrics.ui.pages." + module)
    getattr(page, func)()


def run_profile(
    tab: str,
    view: str | None = None,
    inspect: str | None = None,
    chart: str | None = None,
    skipped: tuple[str, ...] = (),
    extra_numeric: int = 0,
) -> None:
    """Load ``samples.every_type_frame`` and draw part of the Profile page.

    ``tab`` and ``view`` are set through the widget keys before the page draws them. With
    ``inspect`` (a column name), only the inspector of that column is drawn, since AppTest
    cannot select a row of a table; ``chart`` picks the chart of a numeric column. ``skipped``
    columns are left out of the profile. ``extra_numeric`` adds random numeric columns.
    """
    import importlib

    import streamlit as st

    from fixtures.samples import every_type_frame
    from litmetrics.core.io.base import LoadedDataset
    from litmetrics.core.schema import prepare_dataset
    from litmetrics.ui import state

    state.init_state()
    if state.get_dataset() is None:
        df = every_type_frame(extra_numeric=extra_numeric)
        dataset, schema = prepare_dataset(LoadedDataset(df, "people.csv", "CSV"))
        state.set_dataset(dataset, "people", "people")
        state.set_schema(schema)
        state.set_skipped_columns(skipped)
    page = importlib.import_module("litmetrics.ui.pages.profile")
    if inspect:
        if chart is not None:
            st.session_state[state.widget_key("inspect_numeric_chart")] = chart
        page._inspector(page._active(), inspect)
        return
    st.session_state[state.widget_key("profile_tab")] = tab
    if view is not None:
        st.session_state[state.widget_key("columns_view")] = view
    page.profile_page()


def run_visualize() -> None:
    """Draw the Visualization page on ``samples.chart_frame``, typed as the Load page does it.

    ``session_state["other_dataset"] = True`` makes the next run load another dataset (the
    first 500 rows), as if the user had loaded a new file.
    """
    import importlib

    import streamlit as st

    from fixtures.samples import chart_frame
    from litmetrics.core.io.base import LoadedDataset
    from litmetrics.core.schema import prepare_dataset
    from litmetrics.ui import state

    state.init_state()
    other = st.session_state.get("other_dataset", False)
    dataset_id = "chart-other" if other else "chart"
    if state.get_dataset_id() != dataset_id:
        frame = chart_frame(500) if other else chart_frame()
        dataset, schema = prepare_dataset(LoadedDataset(frame, "chart.csv", "CSV"))
        state.set_dataset(dataset, dataset_id, dataset_id)
        state.set_schema(schema)
    importlib.import_module("litmetrics.ui.pages.visualize").visualize_page()


# ---------------------------------------------------------------------------------------------
# Helpers to read what a script drew


def start_app() -> AppTest:
    """``run_app`` after its first run: the Load page, empty."""
    return AppTest.from_function(run_app, default_timeout=TIMEOUT).run()


def errors(at: AppTest) -> list[str]:
    """Exceptions raised by the script, as text, for the message of a failed assertion."""
    return [str(e.value) for e in at.exception]


def metrics(at: AppTest) -> list[str]:
    """Values of the metrics drawn, in order (the Load page starts with rows, columns, format)."""
    return [m.value for m in at.metric]


def captions(at: AppTest) -> str:
    """Every caption drawn, joined by spaces."""
    return " ".join(c.value for c in at.caption)


def button(at: AppTest, key: str):
    """The button with ``key``. Raises StopIteration if there is none."""
    return next(b for b in at.button if b.key == key)


def buttons(at: AppTest, prefix: str) -> list:
    """Buttons whose key starts with ``prefix``, in the order drawn."""
    return [b for b in at.button if b.key and b.key.startswith(prefix)]


def memory_open_buttons(at: AppTest) -> list:
    """Open buttons of the datasets in memory (not of the tables of the open file)."""
    return [b for b in buttons(at, "open_") if not b.key.startswith("open_part")]


def parts_expander(at: AppTest):
    """Expander with the tables (or sheets) of the open file. Raises StopIteration if none."""
    return next(e for e in at.expander if e.label.startswith(("Table: ", "Sheet: ")))


def has_parts_expander(at: AppTest) -> bool:
    """Whether the Load page lists the tables or sheets of an open file."""
    return any(e.label.startswith(("Table", "Sheet")) for e in at.expander)


def open_part(at: AppTest, name: str) -> AppTest:
    """Click the Open button of table (or sheet) ``name`` in the Load page, and rerun."""
    rows = buttons(at, "open_part_")
    names = [
        m.value.split("**")[1] for m in parts_expander(at).markdown if m.value.startswith("**")
    ]
    return rows[names.index(name)].click().run()


def upload(at: AppTest, name: str, data: bytes) -> AppTest:
    """Upload ``data`` as file ``name`` in the Load page, and rerun."""
    mime = "text/csv" if name.endswith(".csv") else "application/octet-stream"
    return at.file_uploader[0].upload(name, data, mime).run()
