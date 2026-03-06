---
name: routine-schedule-composer
description: Compose reliable `routine_create` payloads for reminders and automations. Use when creating/updating/deleting schedules, especially when trigger-time execution instructions must be distinct from user-facing notifications, when converting natural language time requests into schedule strings, or when troubleshooting silent/notify behavior.
---

# Routine Schedule Composer

## Overview

Compose schedule payloads where `instruction` is the single source of truth for schedule-time execution.
Prevent common failures such as empty instructions, natural-language implementation commands, and incorrect one-time schedule formats.

## Workflow

1. Decide schedule shape:
   - Use local ISO datetime for one-time reminders (`2026-03-03T19:30:00+08:00`).
   - Use cron for recurring jobs (`0 */3 * * *`).
2. Build `routine_create` payload:
   - `instruction`: second-person schedule-time scratchpad (`You must ...`) with scripts/files/keys source and required output.
   - `implementation_command`: concrete executable + args.
   - Path style: keep command paths relative to `working_dir` (default `tulpa_stuff`), e.g. `python3 tg_login.py` not `python3 tulpa_stuff/tg_login.py`.
   - `notify_user`: `true` by default unless user explicitly wants silent runs.
3. Verify before sending:
   - `instruction` is non-empty and action-oriented.
   - `instruction` includes required scripts, files/paths, and credential source references.
   - `implementation_command` is shell-like, not prose.
4. Enforce claim discipline:
   - If bootstrap-now was requested, run it and verify output before claiming success.
   - If bootstrap was not run, explicitly say schedule is created but data/output is not initialized yet.
   - Do not state concrete fetched facts unless they came from tool outputs in this run.

## Field Rules

- `instruction`: Describe exactly what must be executed at trigger time.
- `cleanup_paths`: Add deterministic file paths when automation creates files that should be deleted with the routine.

## Instruction Style

- Start with `You must ...`.
- Add concrete steps and dependencies (scripts, files/paths, key source, APIs).
- State expected output and where it should be written/sent.
- State failure behavior (what to log/return if blocked).
- Distinguish `executed now` from `scheduled for future runs` in final user response.

## Examples

### Silent log updater

```json
{
  "name": "Log Current Time",
  "schedule": "0 */3 * * *",
  "instruction": "You must append the current timestamp (ISO-8601 UTC) to tulpa_stuff/logtimes.md using scripts/logtime.py. If the file is missing, create it. Return a one-line execution summary and include any error.",
  "implementation_command": "python3 scripts/logtime.py",
  "notify_user": false
}
```

### Notifying world-news digest

```json
{
  "name": "World News Digest",
  "schedule": "0 */3 * * *",
  "instruction": "You must run scripts/worldnews.py to fetch top world headlines and append a concise bullet summary to tulpa_stuff/worldnewslog.md. Read NEWS_API_KEY from environment (do not print the value). If the API fails, log the error and return a failure summary.",
  "implementation_command": "python3 scripts/worldnews.py",
  "notify_user": true
}
```
