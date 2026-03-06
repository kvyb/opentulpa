# Quality Gate

Run from repository root.

## Standard Command

```bash
UV_CACHE_DIR=.opentulpa/.uv-cache \
  uv run python skills/opentulpa-engineering/scripts/quality_gate.py --smoke
```

## What It Runs

1. `uv run ruff check src/opentulpa scripts/manager.py pyproject.toml README.md tulpa_stuff/README.md` (existing files only)
2. `python3 -m compileall src/opentulpa`
3. `uv run pytest -q`
4. Smoke checks for runtime/API/Telegram when `--smoke` is set

## Exit Policy

- Any lint/compile failure: fail.
- Any test failure: fail.
- `pytest` exit code `5` (no tests collected):
  - warn and continue by default,
  - fail when `--strict-tests` is set.
- Any smoke failure: fail.

## Why `UV_CACHE_DIR`

Use local cache path (`.opentulpa/.uv-cache`) to avoid permission failures in restricted/sandboxed environments.
