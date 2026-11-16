"""Test subjects shared by every level of the suite: data, builders, app scripts and helpers.

- ``files``: builders of file bytes in each format the app reads (CSV, SQLite, Parquet,
  Feather; ``xlsx_builder`` for Excel).
- ``samples``: the named datasets and files the tests use, each with what it contains and why.
- ``app_scripts``: scripts run by ``AppTest`` (pages drawn as app.py draws them) and helpers to
  find widgets in the result.
- ``browser``: helpers of the acceptance tests to drive the app in Playwright.

Imported as ``from fixtures import samples`` (``pythonpath = ["tests"]`` in pyproject.toml).
"""
