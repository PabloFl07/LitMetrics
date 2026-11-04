@AGENTS.md

## Claude Code

- Skills in `.claude/skills/`: `verify` (run before declaring work done and before each commit),
  `new-page`, `add-chart` (streamlit-echarts API in `add-chart/echarts-api.md`), `add-loader`,
  `streamlit-api`, `streamlit-extras` (decide extra vs native before building any new UI).
- Path-scoped rules in `.claude/rules/` load when you touch `ui/`, `core/` or `tests/`.
- Hooks format and lint every edited `.py` file and block edits to `uv.lock` and `.venv/`.
  If a hook reports ruff errors, fix them before moving on.
- For a new page or feature that touches several modules, plan first and name the files you
  will change.
